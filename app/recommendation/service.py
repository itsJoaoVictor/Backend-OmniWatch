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
            # Se ainda não existe registro, cria um vazio marcado como stale
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
        logger.info(f"[RECS-SERVICE] Usuário {user_id_str} já possui recálculo em andamento. Ignorando solicitação concorrente.")
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

            # Executa os pipelines em paralelo ou ordenados com a sessão
            # Nota: para evitar conflitos na mesma sessão assíncrona, executamos sequencialmente
            # ou com sessões dedicadas se necessário. Execução sequencial aqui é muito limpa e segura:
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
        # Se não houver event loop em execução na thread atual
        asyncio.run(
            compute_and_save_user_recommendations(
                user_id=user_id_str,
                force=force,
                cooldown_seconds=cooldown_seconds,
            )
        )
