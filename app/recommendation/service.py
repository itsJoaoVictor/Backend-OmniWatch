import asyncio
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.core.database import AsyncSessionLocal
from app.recommendation.models import UserRecommendation, UserDismissedRecommendation
from app.recommendation.orchestrator import (
    get_personalized_recommendations,
    get_upcoming_recommendations,
    get_persona_recommendations,
)

logger = logging.getLogger(__name__)

# Conjunto em memória para evitar recálculos concorrentes para o mesmo usuário
_active_generation_users: Set[str] = set()

# Rastreamento de obras distintas recentes e tarefas de debounce por usuário
_user_recent_media_ids: Dict[str, Set[int]] = {}
_user_debounce_tasks: Dict[str, asyncio.Task] = {}

DEFAULT_COOLDOWN_SECONDS = 1200  # 20 minutos de cooldown padrão
MAX_STALE_SECONDS = 86400        # 24 horas para expiração natural

def _parse_uuid(uid: Any) -> uuid.UUID:
    if isinstance(uid, uuid.UUID):
        return uid
    return uuid.UUID(str(uid))

async def get_user_recommendations_record(
    db: AsyncSession, user_id: str
) -> Optional[UserRecommendation]:
    """Busca o registro de recomendações do usuário no banco de dados."""
    u_uuid = _parse_uuid(user_id)
    stmt = select(UserRecommendation).where(UserRecommendation.user_id == u_uuid)
    result = await db.execute(stmt)
    return result.scalars().first()

async def mark_user_recommendations_stale(
    db: AsyncSession, user_id: str
) -> None:
    """Marca o registro de recomendações do usuário como desatualizado (is_stale=True)."""
    try:
        record = await get_user_recommendations_record(db, user_id)
        if record:
            record.is_stale = True
            await db.commit()
        else:
            new_record = UserRecommendation(
                user_id=_parse_uuid(user_id),
                explore_items=[],
                upcoming_items={},
                personas_items=[],
                is_stale=True,
                last_generated_at=None,
            )
            db.add(new_record)
            await db.commit()
    except Exception as e:
        logger.error(f"[RECS-SERVICE] Erro ao marcar recomendações como stale para {user_id}: {e}")

async def eject_media_from_user_recommendations(
    db: AsyncSession, user_id: str, tmdb_id: int
) -> bool:
    """
    Remove imediatamente a obra com tmdb_id de todas as recomendações salvas
    do usuário (explore, upcoming e personas), garantindo que ela suma
    instantaneamente da tela /explore sem esperar nenhum recálculo.
    """
    try:
        record = await get_user_recommendations_record(db, user_id)
        if not record:
            return False

        modified = False

        # 1. Remove de explore_items
        if record.explore_items:
            orig_len = len(record.explore_items)
            filtered_explore = [
                it for it in record.explore_items
                if it.get("id") != tmdb_id and it.get("tmdb_id") != tmdb_id
            ]
            if len(filtered_explore) != orig_len:
                record.explore_items = filtered_explore
                flag_modified(record, "explore_items")
                modified = True

        # 2. Remove de upcoming_items
        if record.upcoming_items:
            up_movies = record.upcoming_items.get("movies", [])
            up_tv = record.upcoming_items.get("tv", [])
            new_movies = [it for it in up_movies if it.get("id") != tmdb_id and it.get("tmdb_id") != tmdb_id]
            new_tv = [it for it in up_tv if it.get("id") != tmdb_id and it.get("tmdb_id") != tmdb_id]
            if len(new_movies) != len(up_movies) or len(new_tv) != len(up_tv):
                record.upcoming_items = {"movies": new_movies, "tv": new_tv}
                flag_modified(record, "upcoming_items")
                modified = True

        # 3. Remove de personas_items
        if record.personas_items:
            new_personas = []
            for persona in record.personas_items:
                recs = persona.get("recommendations", [])
                filtered_recs = [
                    it for it in recs
                    if it.get("id") != tmdb_id and it.get("tmdb_id") != tmdb_id
                ]
                p_copy = dict(persona)
                p_copy["recommendations"] = filtered_recs
                new_personas.append(p_copy)
            record.personas_items = new_personas
            flag_modified(record, "personas_items")
            modified = True

        if modified:
            record.is_stale = True
            await db.commit()
            logger.info(f"[RECS-SERVICE] Obra tmdb_id={tmdb_id} ejetada imediatamente das recomendações de {user_id}.")

        return modified
    except Exception as e:
        logger.error(f"[RECS-SERVICE] Erro ao ejetar obra {tmdb_id} das recomendações de {user_id}: {e}")
        return False

def notify_user_media_action(
    user_id: str, tmdb_id: int, debounce_seconds: int = 35
) -> None:
    """
    Rastreia ações do usuário por obra para detecção inteligente:
    - Se o usuário interagir com 2 ou mais obras DIFERENTES no intervalo recente:
      agenda um recálculo com debounce curto de 35 segundos (force=True),
      garantindo que se ele adicionar múltiplos filmes em sequência,
      o sistema espere 35s após o último clique e atualize tudo com o novo gosto!
    - Se for apenas 1 obra (ex.: marcando vários episódios da mesma série),
      apenas agenda com o cooldown padrão longo para não sobrecarregar TMDB/CPU.
    """
    user_id_str = str(user_id)
    if user_id_str not in _user_recent_media_ids:
        _user_recent_media_ids[user_id_str] = set()

    _user_recent_media_ids[user_id_str].add(tmdb_id)
    distinct_count = len(_user_recent_media_ids[user_id_str])

    # Cancela debounce anterior se houver
    if user_id_str in _user_debounce_tasks:
        task = _user_debounce_tasks.pop(user_id_str)
        if not task.done():
            task.cancel()

    # Se atingiu 2 ou mais obras distintas adicionadas/alteradas
    if distinct_count >= 2:
        logger.info(
            f"[RECS-SERVICE] Usuário {user_id_str} adicionou {distinct_count} obras distintas. "
            f"Agendando recálculo com debounce inteligente de {debounce_seconds}s..."
        )

        async def _debounced_refresh():
            try:
                await asyncio.sleep(debounce_seconds)
                logger.info(f"[RECS-SERVICE] Debounce expirado para {user_id_str}. Executando recálculo com force=True...")
                _user_recent_media_ids.pop(user_id_str, None)
                await compute_and_save_user_recommendations(user_id_str, force=True)
            except asyncio.CancelledError:
                pass
            finally:
                _user_debounce_tasks.pop(user_id_str, None)

        try:
            loop = asyncio.get_running_loop()
            _user_debounce_tasks[user_id_str] = loop.create_task(_debounced_refresh())
        except RuntimeError:
            pass
    else:
        # Apenas 1 obra alterada até agora: agenda com cooldown padrão longo
        schedule_recommendation_computation(user_id_str, force=False, cooldown_seconds=DEFAULT_COOLDOWN_SECONDS)

_user_generation_events: Dict[str, asyncio.Event] = {}

async def compute_and_save_user_recommendations(
    user_id: str,
    force: bool = False,
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
) -> Optional[Dict[str, Any]]:
    """
    Executa o pipeline completo de recomendações (Explore, Upcoming e Personas)
    e persiste os resultados diretamente na tabela user_recommendations.
    
    Respeita controle de concorrência e cooldown para evitar sobrecarga de CPU e TMDB.
    """
    user_id_str = str(user_id)
    if user_id_str in _active_generation_users:
        logger.info(f"[RECS-SERVICE] Usuário {user_id_str} já possui recálculo em andamento. Aguardando conclusão...")
        event = _user_generation_events.get(user_id_str)
        if event:
            try:
                await asyncio.wait_for(event.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                pass
        async with AsyncSessionLocal() as session:
            record = await get_user_recommendations_record(session, user_id_str)
            if record:
                return {
                    "explore_items": record.explore_items,
                    "upcoming_items": record.upcoming_items,
                    "personas_items": record.personas_items,
                }
        return None

    _active_generation_users.add(user_id_str)
    event = asyncio.Event()
    _user_generation_events[user_id_str] = event

    try:
        async with AsyncSessionLocal() as session:
            record = await get_user_recommendations_record(session, user_id_str)
            now = datetime.now(timezone.utc)

            # Verifica cooldown se não for forçado e se já houver dados válidos
            if not force and record and record.last_generated_at:
                elapsed = (now - record.last_generated_at).total_seconds()
                has_data = bool(record.explore_items or record.personas_items)
                if elapsed < cooldown_seconds and has_data:
                    logger.info(
                        f"[RECS-SERVICE] Usuário {user_id_str} em cooldown ({elapsed:.0f}s / {cooldown_seconds}s). Mantendo is_stale=True."
                    )
                    record.is_stale = True
                    await session.commit()
                    return {
                        "explore_items": record.explore_items,
                        "upcoming_items": record.upcoming_items,
                        "personas_items": record.personas_items,
                    }

            logger.info(f"[RECS-SERVICE] Iniciando geração completa de recomendações para o usuário {user_id_str}...")

            try:
                explore_data = await get_personalized_recommendations(session, user_id_str, top_k=50)
            except Exception as e:
                logger.error(f"[RECS-SERVICE] Erro ao gerar explore_recommendations para {user_id_str}: {e}")
                explore_data = record.explore_items if record else []

            try:
                upcoming_data = await get_upcoming_recommendations(session, user_id_str, top_k_per_type=20)
            except Exception as e:
                logger.error(f"[RECS-SERVICE] Erro ao gerar upcoming_recommendations para {user_id_str}: {e}")
                upcoming_data = record.upcoming_items if record else {}

            try:
                personas_data = await get_persona_recommendations(session, user_id_str, top_k_per_persona=15)
            except Exception as e:
                logger.error(f"[RECS-SERVICE] Erro ao gerar persona_recommendations para {user_id_str}: {e}")
                personas_data = record.personas_items if record else []

            # Salva na tabela
            if record is None:
                record = UserRecommendation(
                    user_id=_parse_uuid(user_id_str),
                    explore_items=explore_data,
                    upcoming_items=upcoming_data,
                    personas_items=personas_data,
                    is_stale=False,
                    last_generated_at=now,
                )
                session.add(record)
            else:
                record.explore_items = explore_data
                record.upcoming_items = upcoming_data
                record.personas_items = personas_data
                record.is_stale = False
                record.last_generated_at = now
                flag_modified(record, "explore_items")
                flag_modified(record, "upcoming_items")
                flag_modified(record, "personas_items")

            await session.commit()
            logger.info(f"[RECS-SERVICE] Recomendações atualizadas e persistidas com sucesso no banco para {user_id_str}.")

            return {
                "explore_items": record.explore_items,
                "upcoming_items": record.upcoming_items,
                "personas_items": record.personas_items,
            }

    except Exception as e:
        logger.exception(f"[RECS-SERVICE] Erro crítico ao processar recomendações para {user_id_str}: {e}")
        return None
    finally:
        _active_generation_users.discard(user_id_str)
        evt = _user_generation_events.pop(user_id_str, None)
        if evt:
            evt.set()

def schedule_recommendation_computation(
    user_id: str,
    force: bool = False,
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
) -> None:
    """Dispara o recálculo em segundo plano via asyncio task de forma desacoplada."""
    user_id_str = str(user_id)
    if user_id_str in _active_generation_users:
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(
            compute_and_save_user_recommendations(
                user_id=user_id_str,
                force=force,
                cooldown_seconds=cooldown_seconds,
            )
        )
    except RuntimeError:
        asyncio.run(
            compute_and_save_user_recommendations(
                user_id=user_id_str,
                force=force,
                cooldown_seconds=cooldown_seconds,
            )
        )

async def run_recommendations_daily_sync_loop(interval_hours: int = 24):
    """
    Loop periódico de 24 horas que atualiza as recomendações de todos os usuários
    no banco de dados em segundo plano, garantindo que mesmo usuários inativos
    tenham recomendações frescas com novidades do TMDB todos os dias.
    """
    from app.users.models import User
    logger.info(f"[RECS-CRON] Iniciando rotina diária de recomendações (intervalo: {interval_hours}h)...")
    
    # Aguarda 60s após startup para não disputar recursos com a inicialização
    await asyncio.sleep(60)

    while True:
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(User.id))
                user_ids = [str(uid) for uid in result.scalars().all()]

            logger.info(f"[RECS-CRON] Sincronização diária disparada para {len(user_ids)} usuários.")
            for uid in user_ids:
                try:
                    await compute_and_save_user_recommendations(uid, force=True)
                    # Pausa de 3s entre usuários para respeitar rate-limit do TMDB
                    await asyncio.sleep(3)
                except Exception as err:
                    logger.warning(f"[RECS-CRON] Erro ao sincronizar recomendações para usuário {uid}: {err}")

            logger.info("[RECS-CRON] Ciclo diário de recomendações finalizado com sucesso.")
        except asyncio.CancelledError:
            logger.info("[RECS-CRON] Loop de recomendações cancelado.")
            break
        except Exception as e:
            logger.error(f"[RECS-CRON] Erro inesperado na rotina diária de recomendações: {e}")

        # Aguarda 24 horas
        await asyncio.sleep(interval_hours * 3600)

async def dismiss_recommendation(
    db: AsyncSession,
    user_id: str,
    tmdb_id: int,
    media_type: str,
    title: str = "",
    poster_path: Optional[str] = None,
    days_snooze: int = 180,
) -> UserDismissedRecommendation:
    """
    Registra uma obra como dispensada ('Não tenho interesse') com Snooze de 6 a 12 meses.
    Tenta capturar o embedding da obra para proteção semântica fina e ejeta
    a obra imediatamente do carrossel do usuário.
    """
    u_uuid = _parse_uuid(user_id)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=days_snooze)

    # 1. Tenta recuperar embedding da mídia se já estiver no banco
    embedding = None
    try:
        from app.media.models import Media
        stmt = select(Media.embedding, Media.title, Media.poster_path).where(
            Media.tmdb_id == tmdb_id,
            Media.media_type == media_type,
        )
        res = await db.execute(stmt)
        row = res.first()
        if row:
            embedding = row[0]
            if not title and row[1]:
                title = row[1]
            if not poster_path and row[2]:
                poster_path = row[2]
    except Exception as e:
        logger.warning(f"[RECS-SERVICE] Falha ao buscar embedding de mídia local {tmdb_id}: {e}")

    # 2. Busca se já existe registro de dispensa anterior para atualizar
    stmt = select(UserDismissedRecommendation).where(
        UserDismissedRecommendation.user_id == u_uuid,
        UserDismissedRecommendation.tmdb_id == tmdb_id,
        UserDismissedRecommendation.media_type == media_type,
    )
    res = await db.execute(stmt)
    dismissed = res.scalars().first()

    if dismissed:
        dismissed.expires_at = expires_at
        dismissed.created_at = now
        if embedding:
            dismissed.embedding = embedding
        if title:
            dismissed.title = title
        if poster_path:
            dismissed.poster_path = poster_path
    else:
        dismissed = UserDismissedRecommendation(
            user_id=u_uuid,
            tmdb_id=tmdb_id,
            media_type=media_type,
            title=title or "Sem título",
            poster_path=poster_path,
            embedding=embedding,
            expires_at=expires_at,
            created_at=now,
        )
        db.add(dismissed)

    await db.commit()
    await db.refresh(dismissed)

    # 3. Ejetar imediatamente dos carrosséis salvos
    await eject_media_from_user_recommendations(db, user_id, tmdb_id)

    logger.info(f"[RECS-SERVICE] Obra {tmdb_id} dispensada pelo usuário {user_id} com snooze até {expires_at}.")
    return dismissed

async def undismiss_recommendation(
    db: AsyncSession,
    user_id: str,
    tmdb_id: int,
    media_type: Optional[str] = None,
) -> bool:
    """
    Remove uma obra da lista de dispensadas, permitindo que ela volte a ser
    recomendada no futuro caso o perfil do usuário tenha afinidade.
    """
    u_uuid = _parse_uuid(user_id)
    stmt = select(UserDismissedRecommendation).where(
        UserDismissedRecommendation.user_id == u_uuid,
        UserDismissedRecommendation.tmdb_id == tmdb_id,
    )
    if media_type:
        stmt = stmt.where(UserDismissedRecommendation.media_type == media_type)

    res = await db.execute(stmt)
    records = res.scalars().all()
    if not records:
        return False

    for r in records:
        await db.delete(r)

    await db.commit()
    logger.info(f"[RECS-SERVICE] Obra {tmdb_id} restaurada da lista de dispensadas para usuário {user_id}.")
    return True

async def get_active_dismissed_tmdb_ids(
    db: AsyncSession,
    user_id: str,
) -> Set[int]:
    """Retorna o conjunto de tmdb_ids ativos com dispensa vigente (dentro do prazo de snooze)."""
    try:
        u_uuid = _parse_uuid(user_id)
        now = datetime.now(timezone.utc)
        stmt = select(UserDismissedRecommendation.tmdb_id).where(
            UserDismissedRecommendation.user_id == u_uuid,
            UserDismissedRecommendation.expires_at > now,
        )
        res = await db.execute(stmt)
        return set(res.scalars().all())
    except Exception as e:
        logger.error(f"[RECS-SERVICE] Erro ao buscar tmdb_ids dispensados para {user_id}: {e}")
        return set()

async def get_user_dismissed_records(
    db: AsyncSession,
    user_id: str,
    limit: int = 100,
) -> List[UserDismissedRecommendation]:
    """Retorna todos os registros de dispensas ativas do usuário para a tela de configurações/perfil."""
    u_uuid = _parse_uuid(user_id)
    now = datetime.now(timezone.utc)
    stmt = (
        select(UserDismissedRecommendation)
        .where(
            UserDismissedRecommendation.user_id == u_uuid,
            UserDismissedRecommendation.expires_at > now,
        )
        .order_by(UserDismissedRecommendation.created_at.desc())
        .limit(limit)
    )
    res = await db.execute(stmt)
    return list(res.scalars().all())

def calculate_semantic_dismiss_penalty(
    candidate_embedding: Optional[List[float]],
    dismissed_records: List[UserDismissedRecommendation],
) -> float:
    """
    Calcula uma penalidade semântica suave (-15% a -25%) para candidatos que tenham
    altíssima similaridade cosseno (>= 0.75) com obras dispensadas pelo usuário.
    Aplica decaimento temporal conforme a data de expiração se aproxima.
    """
    if not candidate_embedding or not dismissed_records:
        return 0.0

    from app.recommendation.embeddings import calculate_cosine_similarity

    max_penalty = 0.0
    now = datetime.now(timezone.utc)

    for item in dismissed_records:
        if not item.embedding:
            continue

        sim = calculate_cosine_similarity(candidate_embedding, item.embedding)
        if sim >= 0.75:
            # Fator de decaimento temporal: quanto mais recente, mais próximo de 1.0; perto de expirar, chega a 0.2
            total_duration = (item.expires_at - item.created_at).total_seconds() or 1.0
            remaining = (item.expires_at - now).total_seconds()
            time_factor = max(0.2, min(1.0, remaining / total_duration))

            # Penalidade proporcional entre 0.75 e 1.0 (escala de 15 a 25 pontos percentuais)
            norm_sim = (sim - 0.75) / 0.25  # 0.0 a 1.0
            penalty = (15.0 + norm_sim * 10.0) * time_factor
            if penalty > max_penalty:
                max_penalty = penalty

    return round(max_penalty, 1)

