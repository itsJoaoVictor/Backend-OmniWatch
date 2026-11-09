import os
import json
import uuid
import logging
import asyncio
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete

logger = logging.getLogger(__name__)

try:
    from pywebpush import webpush, WebPushException
    PYWEBPUSH_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as err:
    webpush = None
    WebPushException = Exception
    PYWEBPUSH_AVAILABLE = False
    logger.warning(
        f"pywebpush não está disponível no ambiente Python atual: {err}. "
        "As notificações push estarão desabilitadas até a execução de 'poetry install'."
    )

from app.core.database import AsyncSessionLocal
from app.notifications.models import PushSubscription

# Chaves VAPID para Web Push Notifications.
# A chave privada NUNCA deve ser incluída no código-fonte e deve ser configurada
# via variável de ambiente VAPID_PRIVATE_KEY no arquivo .env do servidor.
DEFAULT_VAPID_PUBLIC_KEY = "BCu50BtTZ5m557ZRl68C1114ZY7pwkRAzdXFDdEcPFQRI5E2udRK9ko0XkyOdcO4tDnTId3xUqcR9nFjRjqixho"
DEFAULT_VAPID_CLAIMS_EMAIL = "mailto:jvgf98@gmail.com"

VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY", DEFAULT_VAPID_PUBLIC_KEY)
raw_priv = os.getenv("VAPID_PRIVATE_KEY", "")
VAPID_PRIVATE_KEY = raw_priv.replace("\\n", "\n") if "\\n" in raw_priv else raw_priv
VAPID_CLAIMS_EMAIL = os.getenv("VAPID_CLAIMS_EMAIL", DEFAULT_VAPID_CLAIMS_EMAIL)

_cached_vapid = None


def get_vapid_instance():
    """
    Retorna o objeto Vapid singleton para assinatura das notificações.
    Carrega o Vapid de forma resiliente a partir de VAPID_PRIVATE_KEY no .env,
    suportando chave bruta base64/base64url de 32 bytes, string PEM ou arquivo no disco.
    """
    global _cached_vapid
    if _cached_vapid is not None:
        return _cached_vapid

    if not PYWEBPUSH_AVAILABLE:
        return None

    if not VAPID_PRIVATE_KEY or not VAPID_PRIVATE_KEY.strip():
        logger.warning(
            "VAPID_PRIVATE_KEY não está configurada no ambiente (.env). "
            "O envio de notificações push externas estará desabilitado até ser definida."
        )
        return None

    try:
        from py_vapid import Vapid

        key_str = VAPID_PRIVATE_KEY.strip()

        # 1. Verifica se aponta para arquivo existente no disco
        if os.path.isfile(key_str):
            _cached_vapid = Vapid.from_file(key_str)
            return _cached_vapid

        # 2. Se for formato PEM (começa com -----BEGIN)
        if "-----BEGIN" in key_str:
            _cached_vapid = Vapid.from_pem(key_str.encode("utf-8"))
            return _cached_vapid

        # 3. Se for string raw base64 / base64url de 32 bytes ou der
        _cached_vapid = Vapid.from_string(key_str)
        return _cached_vapid
    except Exception as e:
        logger.error(f"Erro ao inicializar chave VAPID privada: {e}", exc_info=True)
        return None


async def send_web_push_notification(
    db: AsyncSession,
    user_id: uuid.UUID,
    title: str,
    body: str,
    url: str = "/my-list?tab=recommendations",
    tag: Optional[str] = None,
    icon: Optional[str] = None,
    image: Optional[str] = None,
) -> tuple[int, Optional[str]]:
    """
    Envia uma notificação Web Push para todos os aparelhos registrados de um usuário.
    Limpa assinaturas inválidas (404/410) automaticamente.
    Retorna uma tupla (número de envios bem sucedidos, última mensagem de erro se houver).
    """
    if not PYWEBPUSH_AVAILABLE or not webpush:
        logger.warning("pywebpush não está instalado no ambiente. Web push ignorado com segurança.")
        return 0, "Biblioteca pywebpush não disponível no servidor."

    vapid_obj = get_vapid_instance()
    if not vapid_obj:
        logger.error("Vapid instance não pôde ser inicializada. Web push cancelado.")
        return 0, "Chave VAPID privada inválida ou com formato não suportado."

    res = await db.execute(
        select(PushSubscription).where(PushSubscription.user_id == user_id)
    )
    subscriptions = res.scalars().all()
    if not subscriptions:
        return 0, "Nenhum aparelho ativo encontrado para este usuário."

    payload_dict = {
        "title": title,
        "body": body,
        "icon": icon or "/icon-192x192.png",
        "badge": "/icon-192x192.png",
        "data": {
            "url": url,
        },
        "tag": tag or f"omniwatch-{uuid.uuid4().hex[:8]}"
    }
    if image:
        payload_dict["image"] = image

    payload = json.dumps(payload_dict)

    success_count = 0
    expired_endpoints = []
    last_error: Optional[str] = None

    def _sync_webpush(sub_info: Dict[str, Any]):
        return webpush(
            subscription_info=sub_info,
            data=payload,
            vapid_private_key=vapid_obj,
            vapid_claims={"sub": VAPID_CLAIMS_EMAIL},
            timeout=8
        )

    for sub in subscriptions:
        sub_info = {
            "endpoint": sub.endpoint,
            "keys": {
                "p256dh": sub.p256dh,
                "auth": sub.auth
            }
        }
        try:
            # webpush faz requisição HTTP síncrona, executamos em thread pool para não bloquear o loop assíncrono
            response = await asyncio.to_thread(_sync_webpush, sub_info)
            status_code = getattr(response, "status_code", 200)
            if status_code in (200, 201, 202):
                success_count += 1
            else:
                last_error = f"Status {status_code}: {getattr(response, 'text', '')}"
                logger.warning(f"Web Push retornou status inesperado para {sub.endpoint}: {last_error}")
        except WebPushException as ex:
            last_error = f"{ex}"
            logger.warning(f"Erro ao enviar Web Push para {sub.endpoint}: {ex}")
            # Se a inscrição expirou ou o app foi desinstalado (404 / 410)
            resp = getattr(ex, "response", None)
            if resp is not None and getattr(resp, "status_code", None) in (404, 410):
                expired_endpoints.append(sub.endpoint)
        except Exception as e:
            last_error = str(e)
            logger.error(f"Erro genérico no envio de Web Push para {sub.endpoint}: {e}", exc_info=True)

    # Remove inscrições expiradas
    if expired_endpoints:
        await db.execute(
            delete(PushSubscription).where(PushSubscription.endpoint.in_(expired_endpoints))
        )
        await db.commit()

    return success_count, last_error


def trigger_web_push_background(
    user_id: uuid.UUID,
    title: str,
    body: str,
    url: str = "/my-list?tab=recommendations",
    tag: Optional[str] = None,
    icon: Optional[str] = None,
    image: Optional[str] = None,
):
    """
    Dispara o envio de Web Push em uma task assíncrona em background com sessão própria.
    """
    async def _runner():
        async with AsyncSessionLocal() as session:
            try:
                await send_web_push_notification(
                    session,
                    user_id=user_id,
                    title=title,
                    body=body,
                    url=url,
                    tag=tag,
                    icon=icon,
                    image=image
                )
            except Exception as e:
                logger.error(f"Falha na task de background Web Push: {e}")

    asyncio.create_task(_runner())
