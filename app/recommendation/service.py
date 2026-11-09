import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.core.database import AsyncSessionLocal
from app.core.cache import recommendations_seen_cache
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

def sanitize_recommendation_item(item: Any) -> Any:
    """
    Remove vetores de embedding e metadados internos pesados de uma obra candidata,
    garantindo complexidade O(1) por item e payload ultraleve (~1.5 KB em vez de ~90 KB).
    Elimina o vazamento de memória e sobrecarga de CPU do Samsung Galaxy A9 ao serializar JSON.
    """
    if not isinstance(item, dict):
        return item

    # Campos internos e volumosos que NUNCA devem ser persistidos no JSONB de exibição
    # nem serializados na resposta HTTP para o frontend:
    item.pop("embedding", None)
    item.pop("_candidate_vector", None)
    item.pop("candidate_vector", None)
    item.pop("_best_persona_id", None)
    item.pop("_ml_score", None)
    item.pop("_rrf_score", None)
    item.pop("_channels", None)
    item.pop("cast", None)
    item.pop("crew", None)
    item.pop("softcore", None)
    item.pop("adult", None)
    return item

def sanitize_recommendation_list(items: Optional[List[Any]]) -> List[Any]:
    """
    Sanitiza uma lista de obras em O(N) in-place com O(1) de espaço auxiliar.
    """
    if not items or not isinstance(items, list):
        return items or []
    for it in items:
        sanitize_recommendation_item(it)
    return items

def check_is_bloated(items: Any) -> bool:
    """
    Verificação O(1) por amostragem para detectar se o registro persistido contém embeddings legados.
    """
    if not isinstance(items, list) or not items:
        return False
    sample = items[:3]
    for it in sample:
        if isinstance(it, dict) and ("embedding" in it or "_candidate_vector" in it or "cast" in it or "crew" in it):
            return True
    return False

async def cleanup_bloated_user_record(user_id: str) -> None:
    """
    Higieniza registros legados no PostgreSQL que foram gravados com embeddings volumosos.
    Reduz a linha JSONB de ~15MB para ~200KB no banco de forma transparente e assíncrona,
    aliviando disco, I/O e conexões do PostgreSQL no Galaxy A9.
    """
    try:
        async with AsyncSessionLocal() as session:
            record = await get_user_recommendations_record(session, user_id)
            if not record:
                return
            modified = False
            if check_is_bloated(record.explore_items):
                sanitize_recommendation_list(record.explore_items)
                flag_modified(record, "explore_items")
                modified = True
            if check_is_bloated(record.recent_vibe_items):
                sanitize_recommendation_list(record.recent_vibe_items)
                flag_modified(record, "recent_vibe_items")
                modified = True
            if record.personas_items and isinstance(record.personas_items, list):
                p_mod = False
                for p in record.personas_items:
                    if isinstance(p, dict) and p.get("recommendations") and check_is_bloated(p["recommendations"]):
                        sanitize_recommendation_list(p["recommendations"])
                        p_mod = True
                if p_mod:
                    flag_modified(record, "personas_items")
                    modified = True
            if record.upcoming_items and isinstance(record.upcoming_items, dict):
                u_mod = False
                for k in ("movies", "tv", "series"):
                    if check_is_bloated(record.upcoming_items.get(k)):
                        sanitize_recommendation_list(record.upcoming_items[k])
                        u_mod = True
                if u_mod:
                    flag_modified(record, "upcoming_items")
                    modified = True
            if modified:
                await session.commit()
                logger.info(f"[RECS-SERVICE] Registro legado de {user_id} higienizado e reduzido no PostgreSQL com sucesso.")
    except Exception as e:
        logger.warning(f"[RECS-SERVICE] Falha não-bloqueante na higienização de registro legado: {e}")

def _parse_uuid(uid: Any) -> uuid.UUID:
    if isinstance(uid, uuid.UUID):
        return uid
    return uuid.UUID(str(uid))

def check_etag_match(client_etag: Optional[str], current_etag: str) -> bool:
    """
    Verifica se o cabeçalho If-None-Match enviado pelo cliente corresponde
    ao ETag atual do recurso em conformidade com a RFC 7232.
    Lida com tags fracas (W/), aspas duplas e listas separadas por vírgula.
    """
    if not client_etag or not current_etag:
        return False
    curr_clean = current_etag.strip().lstrip("W/").strip('"')
    client_tags = [t.strip().lstrip("W/").strip('"') for t in client_etag.split(",")]
    return curr_clean in client_tags or "*" in client_tags

def generate_recommendations_etag(
    user_id: str,
    record: Optional[Any],
    seen_ids: Optional[Set[int]] = None,
    section: str = "explore",
    extra: str = ""
) -> str:
    """
    Gera um ETag HTTP determinístico e ultrarrápido em O(1) baseado no estado
    de geração das recomendações do usuário no PostgreSQL.
    Permite responder com HTTP 304 Not Modified (payload de 0 bytes) sem serializar JSON nem
    gastar CPU e conexões do Galaxy A9 caso nada tenha mudado.

    Se o worker em segundo plano gerar novas recomendações (last_generated_at),
    se itens forem ejetados/dispensados (updated_at) ou se novas obras forem adicionadas
    à lista (seen_ids), o hash se altera imediatamente, forçando a revalidação (HTTP 200).
    """
    if not record:
        return f'"{hashlib.md5(f"{user_id}:{section}:empty:{extra}".encode()).hexdigest()}"'

    last_gen = (
        record.last_generated_at.isoformat()
        if getattr(record, "last_generated_at", None)
        else "none"
    )
    updated = (
        record.updated_at.isoformat()
        if getattr(record, "updated_at", None)
        else "none"
    )
    is_stale = "1" if getattr(record, "is_stale", False) else "0"

    seen_count = len(seen_ids) if seen_ids else 0
    seen_sample = ""
    if seen_ids:
        # Soma determinística de uma amostra dos IDs para hash em O(1)
        seen_sample = str(sum(list(seen_ids)[:15]))

    raw = f"{user_id}:{section}:{last_gen}:{updated}:{is_stale}:{seen_count}:{seen_sample}:{extra}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return f'"{digest}"'

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
                recent_vibe_items=[],
                impressions={},
                is_stale=True,
                last_generated_at=None,
            )
            db.add(new_record)
            await db.commit()
    except Exception as e:
        logger.error(f"[RECS-SERVICE] Erro ao marcar recomendações como stale para {user_id}: {e}")

async def eject_media_from_user_recommendations(
    db: AsyncSession,
    user_id: str,
    tmdb_id: int,
    media_type: Optional[str] = None
) -> bool:
    """
    Remove imediatamente a obra com chave canônica (media_type, tmdb_id)
    de todas as recomendações salvas do usuário (explore, upcoming e personas),
    garantindo que ela suma instantaneamente sem colidir filmes e séries com mesmo ID.
    Se media_type for None, remove por tmdb_id (compatibilidade retroativa).
    """
    try:
        record = await get_user_recommendations_record(db, user_id)
        if not record:
            return False

        modified = False

        def _matches(item: Dict[str, Any]) -> bool:
            it_id = item.get("id") or item.get("tmdb_id")
            if it_id != tmdb_id:
                return False
            if media_type is not None:
                it_mtype = item.get("media_type")
                if it_mtype and it_mtype != media_type:
                    return False
            return True

        # 1. Remove de explore_items
        if record.explore_items:
            orig_len = len(record.explore_items)
            filtered_explore = [it for it in record.explore_items if not _matches(it)]
            if len(filtered_explore) != orig_len:
                record.explore_items = filtered_explore
                flag_modified(record, "explore_items")
                modified = True

        # 2. Remove de upcoming_items
        if record.upcoming_items:
            up_movies = record.upcoming_items.get("movies", [])
            up_tv = record.upcoming_items.get("tv", [])
            new_movies = [
                it for it in up_movies
                if not (it.get("id") == tmdb_id or it.get("tmdb_id") == tmdb_id)
                or (media_type is not None and media_type != "movie")
            ]
            new_tv = [
                it for it in up_tv
                if not (it.get("id") == tmdb_id or it.get("tmdb_id") == tmdb_id)
                or (media_type is not None and media_type != "tv")
            ]
            if len(new_movies) != len(up_movies) or len(new_tv) != len(up_tv):
                record.upcoming_items = {"movies": new_movies, "tv": new_tv}
                flag_modified(record, "upcoming_items")
                modified = True

        # 3. Remove de personas_items
        if record.personas_items:
            new_personas = []
            for persona in record.personas_items:
                recs = persona.get("recommendations", [])
                filtered_recs = [it for it in recs if not _matches(it)]
                p_copy = dict(persona)
                p_copy["recommendations"] = filtered_recs
                new_personas.append(p_copy)
            record.personas_items = new_personas
            flag_modified(record, "personas_items")
            modified = True

        # 4. Remove de recent_vibe_items
        if getattr(record, "recent_vibe_items", None):
            orig_len = len(record.recent_vibe_items)
            filtered_vibe = [it for it in record.recent_vibe_items if not _matches(it)]
            if len(filtered_vibe) != orig_len:
                record.recent_vibe_items = filtered_vibe
                flag_modified(record, "recent_vibe_items")
                modified = True

        # 5. Remove de impressions se presente
        if getattr(record, "impressions", None):
            imp_copy = dict(record.impressions)
            keys_to_del = [k for k in imp_copy if (k == str(tmdb_id) or k.endswith(f":{tmdb_id}"))]
            if keys_to_del:
                for k in keys_to_del:
                    imp_copy.pop(k, None)
                record.impressions = imp_copy
                flag_modified(record, "impressions")
                modified = True

        if modified:
            record.is_stale = True
            await db.commit()
            recommendations_seen_cache.delete_prefix(str(user_id))
            type_info = f" ({media_type})" if media_type else ""
            logger.info(f"[RECS-SERVICE] Obra tmdb_id={tmdb_id}{type_info} ejetada imediatamente das recomendações de {user_id}.")

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
    recommendations_seen_cache.delete_prefix(user_id_str)
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
                    "recent_vibe_items": getattr(record, "recent_vibe_items", []) or [],
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
            # (0 <= elapsed protege contra clock skew quando last_generated_at ficou no futuro)
            if not force and record and record.last_generated_at:
                elapsed = (now - record.last_generated_at).total_seconds()
                has_data = bool(record.explore_items or record.personas_items)
                if 0 <= elapsed < cooldown_seconds and has_data:
                    logger.info(
                        f"[RECS-SERVICE] Usuário {user_id_str} em cooldown ({elapsed:.0f}s / {cooldown_seconds}s). Mantendo is_stale=True."
                    )
                    record.is_stale = True
                    await session.commit()
                    return {
                        "explore_items": record.explore_items,
                        "upcoming_items": record.upcoming_items,
                        "personas_items": record.personas_items,
                        "recent_vibe_items": getattr(record, "recent_vibe_items", []) or [],
                    }

            logger.info(f"[RECS-SERVICE] Iniciando geração completa de recomendações para o usuário {user_id_str}...")

            current_impressions = dict(record.impressions or {}) if record else {}

            try:
                explore_data = await get_personalized_recommendations(
                    session, user_id_str, top_k=50, impressions_map=current_impressions
                )
            except Exception as e:
                logger.error(f"[RECS-SERVICE] Erro ao gerar explore_recommendations para {user_id_str}: {e}")
                explore_data = record.explore_items if record else []

            _explore_keys = {(it.get("media_type", "movie"), it.get("id")) for it in (explore_data or []) if isinstance(it, dict)}

            try:
                from app.recommendation.orchestrator import get_recent_vibe_recommendations
                recent_vibe_data = await get_recent_vibe_recommendations(session, user_id_str, top_k=25, exclude_keys=_explore_keys)
            except Exception as e:
                logger.error(f"[RECS-SERVICE] Erro ao gerar recent_vibe_recommendations para {user_id_str}: {e}")
                recent_vibe_data = getattr(record, "recent_vibe_items", []) if record else []

            _vibe_keys = {(it.get("media_type", "movie"), it.get("id")) for it in (recent_vibe_data or []) if isinstance(it, dict)}
            _cascade_excluded_keys = _explore_keys | _vibe_keys

            try:
                upcoming_data = await get_upcoming_recommendations(session, user_id_str, top_k_per_type=25)
            except Exception as e:
                logger.error(f"[RECS-SERVICE] Erro ao gerar upcoming_recommendations para {user_id_str}: {e}")
                upcoming_data = record.upcoming_items if record else {}

            try:
                personas_data = await get_persona_recommendations(session, user_id_str, top_k_per_persona=25, exclude_keys=_cascade_excluded_keys)
            except Exception as e:
                logger.error(f"[RECS-SERVICE] Erro ao gerar persona_recommendations para {user_id_str}: {e}")
                personas_data = record.personas_items if record else []

            # Atualiza mapa de impressões para os itens da vitrine explore (top 30)
            new_impressions = dict(current_impressions)
            for it in (explore_data or [])[:30]:
                it_type = it.get("media_type", "movie")
                it_id = it.get("id") or it.get("tmdb_id")
                if it_id:
                    k = f"{it_type}:{it_id}"
                    new_impressions[k] = new_impressions.get(k, 0) + 1

            # Limita tamanho do dicionário de impressões para não inflar o JSONB indefinidamente
            if len(new_impressions) > 250:
                sorted_imps = sorted(
                    ((k, v) for k, v in new_impressions.items() if k != "_algo_version"),
                    key=lambda x: x[1],
                    reverse=True,
                )
                new_impressions = dict(sorted_imps[:200])
            new_impressions["_algo_version"] = 5

            # Verifica elegibilidade real de Vibe Recente (completed/watching <= 30 dias)
            from app.recommendation.profile_manager import user_has_recent_vibe_activity
            has_recent_vibe_eligible = await user_has_recent_vibe_activity(session, user_id_str, max_age_days=30.0)

            # Sanitização defensiva em O(N) para garantir que NENHUM embedding seja gravado
            # na tabela user_recommendations (reduzindo a linha JSONB de ~15MB para ~200KB no PostgreSQL)
            # Protege contra sobrescrita por resultado vazio (ex.: falha transitoria do TMDB)
            if record is not None:
                if not explore_data and record.explore_items:
                    logger.warning(f"[RECS-SERVICE] explore vazio para {user_id_str}; mantendo versao anterior.")
                    explore_data = record.explore_items
                if has_recent_vibe_eligible and not recent_vibe_data and getattr(record, "recent_vibe_items", None):
                    recent_vibe_data = record.recent_vibe_items
                if not personas_data and record.personas_items:
                    personas_data = record.personas_items
                _up_empty = not isinstance(upcoming_data, dict) or not (
                    upcoming_data.get("movies") or upcoming_data.get("series") or upcoming_data.get("tv")
                )
                _old_up = record.upcoming_items
                if _up_empty and isinstance(_old_up, dict) and (_old_up.get("movies") or _old_up.get("series") or _old_up.get("tv")):
                    logger.warning(f"[RECS-SERVICE] upcoming vazio para {user_id_str}; mantendo versao anterior.")
                    upcoming_data = _old_up

            sanitize_recommendation_list(explore_data)
            sanitize_recommendation_list(recent_vibe_data)
            if isinstance(upcoming_data, dict):
                sanitize_recommendation_list(upcoming_data.get("movies"))
                sanitize_recommendation_list(upcoming_data.get("series") or upcoming_data.get("tv"))
            if isinstance(personas_data, list):
                for p in personas_data:
                    if isinstance(p, dict):
                        sanitize_recommendation_list(p.get("recommendations"))

            # Verifica se o usuário possui itens na biblioteca e se alguma seção crítica ficou vazia
            from app.tracking.models import UserListItem
            has_user_items = bool(
                (await session.execute(select(UserListItem.id).where(UserListItem.user_id == _parse_uuid(user_id_str)).limit(1))).scalar_one_or_none()
            )
            final_up_empty = not isinstance(upcoming_data, dict) or not (
                upcoming_data.get("movies") or upcoming_data.get("series") or upcoming_data.get("tv")
            )
            should_remain_stale = bool(
                has_user_items and (not explore_data or final_up_empty or (has_recent_vibe_eligible and not recent_vibe_data))
            )

            # Salva na tabela
            if record is None:
                record = UserRecommendation(
                    user_id=_parse_uuid(user_id_str),
                    explore_items=explore_data,
                    upcoming_items=upcoming_data,
                    personas_items=personas_data,
                    recent_vibe_items=recent_vibe_data,
                    impressions=new_impressions,
                    is_stale=should_remain_stale,
                    last_generated_at=now,
                )
                session.add(record)
            else:
                record.explore_items = explore_data
                record.upcoming_items = upcoming_data
                record.personas_items = personas_data
                record.recent_vibe_items = recent_vibe_data
                record.impressions = new_impressions
                record.is_stale = should_remain_stale
                record.last_generated_at = now
                flag_modified(record, "explore_items")
                flag_modified(record, "upcoming_items")
                flag_modified(record, "personas_items")
                flag_modified(record, "recent_vibe_items")
                flag_modified(record, "impressions")

            await session.commit()
            logger.info(f"[RECS-SERVICE] Recomendações atualizadas e persistidas com sucesso no banco para {user_id_str}.")

            return {
                "explore_items": record.explore_items,
                "upcoming_items": record.upcoming_items,
                "personas_items": record.personas_items,
                "recent_vibe_items": record.recent_vibe_items,
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
    No boot inicial (t=6s), cura prioritariamente registros ausentes, stale ou incompletos.
    """
    from app.users.models import User
    from app.tracking.models import UserListItem
    logger.info(f"[RECS-CRON] Iniciando rotina diária de recomendações (intervalo: {interval_hours}h)...")
    
    await asyncio.sleep(6)
    first_boot_pass = True

    while True:
        try:
            now = datetime.now(timezone.utc)
            async with AsyncSessionLocal() as session:
                if first_boot_pass:
                    # No boot, foca apenas em usuários com itens na lista cujo cache está ausente, expirado, stale ou incompleto
                    active_res = await session.execute(select(UserListItem.user_id).distinct())
                    candidate_uids = [str(r[0]) for r in active_res.all() if r[0]]
                    recs_res = await session.execute(select(UserRecommendation))
                    recs_by_uid = {str(r.user_id): r for r in recs_res.scalars().all()}

                    from app.recommendation.profile_manager import user_has_recent_vibe_activity, rebuild_user_profile
                    user_ids = []
                    needs_profile_rebuild = set()
                    for uid in candidate_uids:
                        rec = recs_by_uid.get(uid)
                        if rec is None or rec.is_stale or not rec.explore_items:
                            user_ids.append(uid)
                            continue
                        if (rec.impressions or {}).get("_algo_version") != 5:
                            user_ids.append(uid)
                            needs_profile_rebuild.add(uid)
                            continue
                        if not getattr(rec, "recent_vibe_items", None):
                            if await user_has_recent_vibe_activity(session, uid, max_age_days=30.0):
                                user_ids.append(uid)
                                continue
                        up = rec.upcoming_items if isinstance(rec.upcoming_items, dict) else {}
                        if not (up.get("movies") or up.get("tv") or up.get("series")):
                            user_ids.append(uid)
                            continue
                        if rec.last_generated_at is None:
                            user_ids.append(uid)
                            continue
                        elapsed = (now - rec.last_generated_at).total_seconds()
                        if elapsed > interval_hours * 3600 or elapsed < -60:
                            user_ids.append(uid)
                else:
                    result = await session.execute(select(User.id))
                    user_ids = [str(uid) for uid in result.scalars().all()]
                    needs_profile_rebuild = set()

            if user_ids:
                logger.info(
                    f"[RECS-CRON] {'Heal de boot' if first_boot_pass else 'Sincronização diária'} disparado para {len(user_ids)} usuários."
                )
                for uid in user_ids:
                    try:
                        if uid in needs_profile_rebuild:
                            async with AsyncSessionLocal() as prof_session:
                                await rebuild_user_profile(prof_session, _parse_uuid(uid))
                        await compute_and_save_user_recommendations(uid, force=True)
                        await asyncio.sleep(4)
                    except Exception as err:
                        logger.warning(f"[RECS-CRON] Erro ao sincronizar recomendações para usuário {uid}: {err}")
                logger.info("[RECS-CRON] Ciclo de recomendações finalizado com sucesso.")
            else:
                logger.info("[RECS-CRON] Todos os registros de recomendações já estão completos no boot.")

            first_boot_pass = False
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

    # 3. Ejetar imediatamente dos carrosséis salvos com chave canônica (media_type, tmdb_id)
    await eject_media_from_user_recommendations(db, user_id, tmdb_id, media_type=media_type)

    logger.info(f"[RECS-SERVICE] Obra {media_type} tmdb_id={tmdb_id} dispensada pelo usuário {user_id} com snooze até {expires_at}.")
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
    recommendations_seen_cache.delete_prefix(str(user_id))
    logger.info(f"[RECS-SERVICE] Obra {media_type or ''} tmdb_id={tmdb_id} restaurada da lista de dispensadas para usuário {user_id}.")
    return True

async def get_active_dismissed_media_keys(
    db: AsyncSession,
    user_id: str,
) -> Set[Tuple[str, int]]:
    """
    Retorna o conjunto de chaves canônicas (media_type, tmdb_id) ativas com dispensa vigente
    (dentro do prazo de snooze), prevenindo colisão entre filmes e séries com mesmo id.
    """
    try:
        u_uuid = _parse_uuid(user_id)
        now = datetime.now(timezone.utc)
        stmt = select(UserDismissedRecommendation.media_type, UserDismissedRecommendation.tmdb_id).where(
            UserDismissedRecommendation.user_id == u_uuid,
            UserDismissedRecommendation.expires_at > now,
        )
        res = await db.execute(stmt)
        return {(row[0], row[1]) for row in res.all()}
    except Exception as e:
        logger.error(f"[RECS-SERVICE] Erro ao buscar chaves dispensadas para {user_id}: {e}")
        return set()

async def get_active_dismissed_tmdb_ids(
    db: AsyncSession,
    user_id: str,
) -> Set[int]:
    """Retorna o conjunto de tmdb_ids ativos com dispensa vigente (dentro do prazo de snooze)."""
    keys = await get_active_dismissed_media_keys(db, user_id)
    return {k[1] for k in keys}

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
        if isinstance(item, dict):
            item_emb = item.get("embedding")
            expires_at = item.get("expires_at")
            created_at = item.get("created_at")
        else:
            try:
                item_emb = getattr(item, "embedding", None)
                expires_at = getattr(item, "expires_at", None)
                created_at = getattr(item, "created_at", None)
            except Exception:
                continue

        if not item_emb:
            continue

        sim = calculate_cosine_similarity(candidate_embedding, item_emb)
        if sim >= 0.75:
            # Fator de decaimento temporal: quanto mais recente, mais próximo de 1.0; perto de expirar, chega a 0.2
            if expires_at and created_at:
                total_duration = (expires_at - created_at).total_seconds() or 1.0
                remaining = (expires_at - now).total_seconds()
                time_factor = max(0.2, min(1.0, remaining / total_duration))
            else:
                time_factor = 1.0

            # Penalidade proporcional entre 0.75 e 1.0 (escala de 15 a 25 pontos percentuais)
            norm_sim = (sim - 0.75) / 0.25  # 0.0 a 1.0
            penalty = (15.0 + norm_sim * 10.0) * time_factor
            if penalty > max_penalty:
                max_penalty = penalty

    return round(max_penalty, 1)


async def refresh_explore_media_batch(
    db: AsyncSession,
    user_id: str,
    media_type: str = "movie",
    exclude_ids: Optional[List[int]] = None,
    limit: int = 24,
) -> Dict[str, Any]:
    """
    Renova de forma síncrona, ultraleve (< 25ms) e sem chamadas externas ao TMDB
    o pool de recomendações de um formato ('movie' ou 'tv') após o usuário fechar
    o ciclo de lotes na tela Explorar.

    Complexidade:
    - O(U + E) para montar conjuntos hash de exclusão (vistos, dispensados e ciclo atual)
    - O(M + K log K) na projeção vetorial SIMD em memória (_CACHED_MATRIX, K <= 48)
    - O(K) constante no reranqueamento OmniMatch v3 + LightGBM + MMR
    """
    from app.users.models import User
    from app.tracking.services import get_user_list
    from app.recommendation.vector_search import search_semantic_candidates
    from app.recommendation.collaborative import fetch_collaborative_candidates
    from app.recommendation.orchestrator import (
        build_candidate_vector,
        apply_mmr_reranking,
        reciprocal_rank_fusion,
    )
    from app.recommendation.matcher import compute_omnimatch_score
    from app.recommendation.ranker import LightGBMRanker

    u_uuid = _parse_uuid(user_id)
    user_res = await db.execute(select(User).where(User.id == u_uuid))
    user = user_res.scalars().first()

    record = await get_user_recommendations_record(db, user_id)
    existing_explore = list(record.explore_items or []) if record else []
    other_type_items = [it for it in existing_explore if it.get("media_type") != media_type]
    current_type_items = [it for it in existing_explore if it.get("media_type") == media_type]

    if not user or not user.feature_vector:
        return {"items": sanitize_recommendation_list(current_type_items[:limit]), "refreshed": False}

    user_list = await get_user_list(db, user_id)
    seen_media_keys: Set[Tuple[str, int]] = {
        (it.media.media_type, it.media.tmdb_id) for it in user_list if getattr(it, "media", None)
    }
    dismissed_keys = await get_active_dismissed_media_keys(db, user_id)
    dismissed_records = await get_user_dismissed_records(db, user_id)
    dismissed_snapshots = [
        {
            "embedding": getattr(r, "embedding", None),
            "expires_at": getattr(r, "expires_at", None),
            "created_at": getattr(r, "created_at", None),
        }
        for r in dismissed_records
    ]

    exclude_set: Set[int] = {int(x) for x in (exclude_ids or []) if x}
    cycle_excluded_keys: Set[Tuple[str, int]] = {(media_type, tid) for tid in exclude_set}
    all_excluded = seen_media_keys | dismissed_keys | cycle_excluded_keys

    # Snapshots leves do histórico para tags explicativas ("Porque você favoritou...")
    user_item_snapshots = []
    for it in user_list:
        m = getattr(it, "media", None)
        if not m:
            continue
        ep_list = getattr(it, "__dict__", {}).get("episodes") or []
        ep_rewatches = sum((getattr(ep, "rewatch_count", 0) or 0) for ep in ep_list)
        user_item_snapshots.append({
            "title": getattr(m, "title", None),
            "media_type": getattr(m, "media_type", "movie"),
            "embedding": getattr(m, "embedding", None),
            "media_obj": m,
            "rating": getattr(it, "rating", None),
            "status": getattr(it, "status", None),
            "is_favorite": getattr(it, "is_favorite", False),
            "rewatch_count": getattr(it, "rewatch_count", 0) or 0,
            "ep_rewatches": ep_rewatches,
        })

    # 1. Busca semântica ANN em memória filtrada por media_type (zero chamadas HTTP)
    fetch_k = max(36, limit * 2)
    sem_cands = await search_semantic_candidates(
        db=db,
        query_embedding=user.embedding,
        excluded_keys=all_excluded,
        limit=fetch_k,
        media_type_filter=media_type,
    )

    collab_cands: List[Dict[str, Any]] = []
    try:
        collab_cands = await fetch_collaborative_candidates(
            db=db,
            target_user_id=str(user_id),
            seen_media_keys=all_excluded,
            limit=36,
            media_type_filter=media_type,
        )
    except Exception:
        collab_cands = []

    channels = []
    if sem_cands:
        channels.append(("semantic", sem_cands, 1.0))
    if collab_cands:
        channels.append(("collaborative", collab_cands, 1.15))

    candidates = reciprocal_rank_fusion(channels, k_constant=60, top_n=48) if channels else []

    user_taste_clusters = user.taste_clusters or []
    active_personas = user_taste_clusters if len(user_taste_clusters) > 1 else None
    lgbm_ranker = LightGBMRanker()
    current_impressions = dict(record.impressions or {}) if record else {}

    # Registra impressão nos itens que o usuário acabou de rotacionar
    for tid in exclude_set:
        imp_k = f"{media_type}:{tid}"
        current_impressions[imp_k] = current_impressions.get(imp_k, 0) + 1

    scored: List[Dict[str, Any]] = []
    for c in candidates:
        c_vec = build_candidate_vector(c)
        c["_candidate_vector"] = c_vec
        cand_emb = c.get("embedding")

        omni = compute_omnimatch_score(
            user_vector=user.feature_vector,
            candidate_vector=c_vec,
            user_embedding=user.embedding,
            candidate_embedding=cand_emb,
            active_personas=active_personas,
            vote_average=float(c.get("vote_average") or 0.0),
            vote_count=int(c.get("vote_count") or 0),
            user_item_count=len(user_list),
            candidate=c,
            user_items=user_item_snapshots,
        )
        score = omni["match_score"]
        tags = omni["match_tags"]

        if lgbm_ranker.is_ready():
            try:
                score = lgbm_ranker.predict_score(
                    user=user,
                    candidate=c,
                    cand_emb=cand_emb,
                    base_omnimatch=score,
                    user_items=user_item_snapshots,
                    active_personas=active_personas,
                )
            except Exception:
                pass

        if dismissed_snapshots and cand_emb:
            pen = calculate_semantic_dismiss_penalty(cand_emb, dismissed_snapshots)
            if pen > 0:
                score = max(0.0, round(score - pen, 1))

        imp_cnt = current_impressions.get(f"{media_type}:{c.get('id')}", 0)
        if imp_cnt >= 2:
            score = max(0.0, round(score - min(15.0, (imp_cnt - 1) * 3.0), 1))

        c["match_score"] = score
        c["match_tags"] = tags
        c["is_preliminary"] = omni.get("is_preliminary", False)
        scored.append(c)

    selected = apply_mmr_reranking(scored, lambda_param=0.62, top_k=limit) if scored else []
    fresh_count = len(selected)

    # Fallback gracioso caso o catálogo local ainda tenha poucos itens inéditos fora de exclude_set:
    # completa até `limit` rotacionando itens anteriores não vistos e agenda descoberta TMDB assíncrona
    if len(selected) < min(8, len(current_type_items)):
        selected_ids = {it.get("id") for it in selected}
        base_seen_ids = {k[1] for k in (seen_media_keys | dismissed_keys) if k[0] == media_type}
        rotated_prev = [
            it for it in reversed(current_type_items)
            if it.get("id") not in selected_ids and it.get("id") not in base_seen_ids
        ]
        selected.extend(rotated_prev[: max(0, limit - len(selected))])
        schedule_recommendation_computation(str(user_id), force=False)

    sanitize_recommendation_list(selected)

    if record is not None:
        if len(current_impressions) > 250:
            sorted_imps = sorted(current_impressions.items(), key=lambda x: x[1], reverse=True)
            current_impressions = dict(sorted_imps[:200])

        merged_explore = selected + other_type_items if media_type == "movie" else other_type_items + selected
        record.explore_items = merged_explore
        record.impressions = current_impressions
        record.updated_at = datetime.now(timezone.utc)
        flag_modified(record, "explore_items")
        flag_modified(record, "impressions")
        await db.commit()

    return {
        "items": selected,
        "refreshed": fresh_count >= 4,
        "fresh_count": fresh_count,
    }


