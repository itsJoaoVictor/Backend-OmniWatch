import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.core.database import AsyncSessionLocal
from app.recommendation.models import UserRecommendation
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
        logger.info(f"[RECS-SERVICE] Usuário {user_id_str} já possui recálculo em andamento. Ignorando concorrência.")
        return None

    _active_generation_users.add(user_id_str)

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
