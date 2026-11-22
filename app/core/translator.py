import asyncio
import hashlib
import logging
import re
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx
from sqlalchemy import bindparam, text

from app.core.config import settings
from app.core.database import AsyncSessionLocal

logger = logging.getLogger(__name__)

TRANSLATE_MAX_CONCURRENCY = 5
MEMORY_CACHE_MAX_SIZE = 4096
MAX_CHUNK_CHARS = 3500

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

_clients_by_loop: Dict[int, httpx.AsyncClient] = {}
_semaphores_by_loop: Dict[int, asyncio.Semaphore] = {}
_inflight_by_loop: Dict[int, Dict[str, asyncio.Future]] = {}
_memory_cache: "OrderedDict[str, str]" = OrderedDict()

_SYSTEM_TRANSLATE_PROMPT = (
    "Você é um tradutor profissional especializado em cinema e séries. "
    "Traduza a sinopse fornecida para Português do Brasil (pt-BR) de forma natural, fiel e fluida. "
    "Retorne APENAS o texto traduzido, sem aspas extras, sem comentários e sem explicações."
)


def _current_loop_id() -> int:
    try:
        return id(asyncio.get_running_loop())
    except RuntimeError:
        return 0


def _get_translate_client() -> httpx.AsyncClient:
    """Retorna um AsyncClient reutilizável vinculado ao event loop atual."""
    loop_id = _current_loop_id()
    client = _clients_by_loop.get(loop_id)
    if client is None or client.is_closed:
        stale = [k for k, c in _clients_by_loop.items() if c.is_closed]
        for k in stale:
            _clients_by_loop.pop(k, None)
            _semaphores_by_loop.pop(k, None)
            _inflight_by_loop.pop(k, None)
        client = httpx.AsyncClient(
            timeout=12.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
        _clients_by_loop[loop_id] = client
    return client


def _get_translate_semaphore() -> asyncio.Semaphore:
    loop_id = _current_loop_id()
    sem = _semaphores_by_loop.get(loop_id)
    if sem is None:
        sem = asyncio.Semaphore(TRANSLATE_MAX_CONCURRENCY)
        _semaphores_by_loop[loop_id] = sem
    return sem


def _get_inflight_map() -> Dict[str, asyncio.Future]:
    loop_id = _current_loop_id()
    inflight = _inflight_by_loop.get(loop_id)
    if inflight is None:
        inflight = {}
        _inflight_by_loop[loop_id] = inflight
    return inflight


def clear_memory_cache() -> None:
    """Limpa o cache LRU em memória (útil em testes)."""
    _memory_cache.clear()


def _mem_get(key: str) -> Optional[str]:
    val = _memory_cache.get(key)
    if val is not None:
        _memory_cache.move_to_end(key)
    return val


def _mem_set(key: str, value: str) -> None:
    _memory_cache[key] = value
    _memory_cache.move_to_end(key)
    while len(_memory_cache) > MEMORY_CACHE_MAX_SIZE:
        _memory_cache.popitem(last=False)


def is_non_latin(s: Optional[str]) -> bool:
    """Verifica se uma string possui caracteres CJK (japonês/chinês/coreano)."""
    if not s:
        return False
    cjk = sum(
        1
        for ch in s
        if "\u4e00" <= ch <= "\u9fff"
        or "\u3040" <= ch <= "\u30ff"
        or "\uac00" <= ch <= "\ud7af"
        or "\u3000" <= ch <= "\u303f"
    )
    return cjk >= 2 or cjk > len(s) * 0.15


def normalize_source_lang(lang: Optional[str]) -> str:
    """Normaliza códigos ISO de idioma vindos do TMDB para uso nos provedores de tradução."""
    if not lang:
        return "auto"
    cleaned = lang.strip()
    if not cleaned:
        return "auto"
    lower = cleaned.lower()
    if lower in ("xx", "und", "auto"):
        return "auto"
    if lower in ("cn", "zh", "zh-cn", "zh-sg"):
        return "zh-CN"
    if lower in ("zh-tw", "zh-hk"):
        return "zh-TW"
    if lower == "pt-br":
        return "pt-BR"
    if lower == "pt-pt":
        return "pt-PT"
    if "-" in lower:
        return lower.split("-", 1)[0]
    return lower


def make_cache_key(text_value: str, source_lang: str = "auto", target_lang: str = "pt-BR") -> str:
    norm_sl = normalize_source_lang(source_lang)
    raw = f"{target_lang}|{norm_sl}|{text_value.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _db_get_translation(text_hash: str) -> Optional[str]:
    try:
        async with AsyncSessionLocal() as session:
            res = await session.execute(
                text("SELECT translated_text FROM translation_cache WHERE text_hash = :h LIMIT 1"),
                {"h": text_hash},
            )
            row = res.first()
            if row and row[0]:
                return str(row[0])
    except Exception as exc:
        logger.debug(f"[Translator] Falha ao ler cache no banco ({text_hash[:8]}): {exc}")
    return None


async def _db_get_translations_bulk(text_hashes: List[str]) -> Dict[str, str]:
    if not text_hashes:
        return {}
    unique_hashes = list(dict.fromkeys(text_hashes))
    try:
        stmt = text(
            "SELECT text_hash, translated_text FROM translation_cache WHERE text_hash IN :hashes"
        ).bindparams(bindparam("hashes", expanding=True))
        async with AsyncSessionLocal() as session:
            res = await session.execute(stmt, {"hashes": unique_hashes})
            return {str(row[0]): str(row[1]) for row in res.all() if row[0] and row[1]}
    except Exception as exc:
        logger.debug(f"[Translator] Falha ao ler cache em lote no banco: {exc}")
    return {}


async def _db_save_translation(
    text_hash: str,
    source_lang: str,
    target_lang: str,
    source_text: str,
    translated_text: str,
    provider: str,
) -> None:
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO translation_cache (
                        text_hash, source_lang, target_lang, source_text, translated_text, provider
                    )
                    VALUES (:h, :sl, :tl, :src, :dst, :prov)
                    ON CONFLICT (text_hash) DO NOTHING
                    """
                ),
                {
                    "h": text_hash,
                    "sl": source_lang[:16],
                    "tl": target_lang[:16],
                    "src": source_text,
                    "dst": translated_text,
                    "prov": provider[:32],
                },
            )
            await session.commit()
    except Exception as exc:
        logger.debug(f"[Translator] Falha ao salvar cache no banco ({text_hash[:8]}): {exc}")


def _is_valid_translation(result: Optional[str], target_lang: str) -> bool:
    if not result:
        return False
    cleaned = result.strip()
    if not cleaned:
        return False
    lower = cleaned[:200].lower()
    if "<html" in lower or "<!doctype" in lower:
        return False
    if target_lang.lower().startswith("pt") and is_non_latin(cleaned):
        return False
    return True


async def _translate_google_clients5(
    client: httpx.AsyncClient,
    text_value: str,
    sl: str,
    target_lang: str,
) -> Optional[str]:
    try:
        resp = await client.post(
            "https://clients5.google.com/translate_a/t",
            params={"client": "dict-chrome-ex", "sl": sl, "tl": target_lang},
            data={"q": text_value},
            headers={"User-Agent": BROWSER_USER_AGENT},
            timeout=6.0,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
        if not isinstance(payload, list) or not payload:
            return None

        parts: List[str] = []
        for item in payload:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, list) and item and isinstance(item[0], str):
                parts.append(item[0])
        joined = "".join(parts).strip()
        if _is_valid_translation(joined, target_lang):
            return joined
    except Exception as exc:
        logger.debug(f"[Translator] google_clients5 erro: {type(exc).__name__}: {exc}")
    return None


async def _translate_google_gtx(
    client: httpx.AsyncClient,
    text_value: str,
    sl: str,
    target_lang: str,
) -> Optional[str]:
    try:
        resp = await client.post(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": sl, "tl": target_lang, "dt": "t"},
            data={"q": text_value},
            headers={"User-Agent": BROWSER_USER_AGENT},
            timeout=5.0,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
        if not isinstance(payload, list) or not payload or not isinstance(payload[0], list):
            return None
        joined = "".join(
            seg[0]
            for seg in payload[0]
            if isinstance(seg, list) and seg and isinstance(seg[0], str)
        ).strip()
        if _is_valid_translation(joined, target_lang):
            return joined
    except Exception as exc:
        logger.debug(f"[Translator] google_gtx erro: {type(exc).__name__}: {exc}")
    return None


async def _translate_mymemory(
    client: httpx.AsyncClient,
    text_value: str,
    sl: str,
    target_lang: str,
) -> Optional[str]:
    if sl == "auto" or len(text_value) > 450:
        return None
    try:
        resp = await client.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text_value, "langpair": f"{sl}|{target_lang}"},
            headers={"User-Agent": BROWSER_USER_AGENT},
            timeout=5.0,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
        if not isinstance(payload, dict) or payload.get("responseStatus") != 200:
            return None
        translated = ((payload.get("responseData") or {}).get("translatedText") or "").strip()
        upper = translated.upper()
        if "MYMEMORY WARNING" in upper or "INVALID LANGUAGE PAIR" in upper:
            return None
        if _is_valid_translation(translated, target_lang):
            return translated
    except Exception as exc:
        logger.debug(f"[Translator] mymemory erro: {type(exc).__name__}: {exc}")
    return None


def _strip_wrapping_quotes(translated: str, original: str) -> str:
    stripped_orig = original.strip()
    if len(translated) >= 2:
        for q_open, q_close in (('"', '"'), ("'", "'"), ("“", "”")):
            if (
                translated.startswith(q_open)
                and translated.endswith(q_close)
                and not (stripped_orig.startswith(q_open) and stripped_orig.endswith(q_close))
            ):
                return translated[1:-1].strip()
    return translated


async def _translate_openrouter_llm(
    client: httpx.AsyncClient,
    text_value: str,
    sl: str,
    target_lang: str,
) -> Optional[str]:
    if not settings.OPENAI_API_KEY:
        return None
    url = f"{settings.OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://omniwatch.app",
        "X-Title": "OmniWatch",
    }
    lang_hint = f" (idioma de origem: {sl})" if sl and sl != "auto" else ""
    payload = {
        "model": settings.OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_TRANSLATE_PROMPT},
            {"role": "user", "content": f"Traduza para {target_lang}{lang_hint}:\n\n{text_value}"},
        ],
        "temperature": 0.1,
        "max_tokens": max(256, min(1200, len(text_value) * 2)),
    }
    try:
        resp = await client.post(url, json=payload, headers=headers, timeout=18.0)
        if resp.status_code != 200:
            return None
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return None
        raw = ((choices[0].get("message") or {}).get("content") or "").strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = _strip_wrapping_quotes(raw, text_value)
        if _is_valid_translation(raw, target_lang):
            return raw
    except Exception as exc:
        logger.debug(f"[Translator] openrouter_llm erro: {type(exc).__name__}: {exc}")
    return None


async def _translate_chunk_via_providers(
    text_value: str,
    source_lang: str,
    target_lang: str,
) -> Tuple[Optional[str], Optional[str]]:
    sl = normalize_source_lang(source_lang)
    client = _get_translate_client()

    translated = await _translate_google_clients5(client, text_value, sl, target_lang)
    if translated:
        return translated, "google_clients5"

    translated = await _translate_google_gtx(client, text_value, sl, target_lang)
    if translated:
        return translated, "google_gtx"

    translated = await _translate_mymemory(client, text_value, sl, target_lang)
    if translated:
        return translated, "mymemory"

    translated = await _translate_openrouter_llm(client, text_value, sl, target_lang)
    if translated:
        return translated, "openrouter_llm"

    return None, None


def _split_into_chunks(text_value: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    stripped = text_value.strip()
    if len(stripped) <= max_chars:
        return [stripped]

    paragraphs = re.split(r"\n{2,}", stripped)
    chunks: List[str] = []
    current = ""

    def _flush_sentence_parts(part: str) -> None:
        nonlocal current
        sentences = re.split(r"(?<=[.!?])\s+", part)
        for sent in sentences:
            if len(sent) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                for i in range(0, len(sent), max_chars):
                    chunks.append(sent[i : i + max_chars])
            elif not current:
                current = sent
            elif len(current) + 1 + len(sent) <= max_chars:
                current = f"{current} {sent}"
            else:
                chunks.append(current)
                current = sent

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            _flush_sentence_parts(para)
        elif not current:
            current = para
        elif len(current) + 2 + len(para) <= max_chars:
            current = f"{current}\n\n{para}"
        else:
            chunks.append(current)
            current = para

    if current:
        chunks.append(current)
    return chunks or [stripped]


async def _translate_text_via_providers(
    text_value: str,
    source_lang: str,
    target_lang: str,
) -> Tuple[Optional[str], Optional[str]]:
    chunks = _split_into_chunks(text_value, MAX_CHUNK_CHARS)
    if len(chunks) == 1:
        return await _translate_chunk_via_providers(chunks[0], source_lang, target_lang)

    translated_parts: List[str] = []
    used_provider: Optional[str] = None
    for chunk in chunks:
        tr, prov = await _translate_chunk_via_providers(chunk, source_lang, target_lang)
        if not tr:
            return None, None
        translated_parts.append(tr)
        if used_provider is None:
            used_provider = prov
    return "\n\n".join(translated_parts), used_provider


async def translate_to_pt_br(text_value: Optional[str], source_lang: Optional[str] = None) -> str:
    """
    Traduz um texto para pt-BR usando provedor gratuito primeiro e LLM OpenRouter como fallback,
    com cache em memória LRU, deduplicação in-flight e persistência no PostgreSQL.
    Nunca grava falhas em cache; em falha total retorna o texto original limpo.
    """
    if not text_value or not text_value.strip():
        return ""

    cleaned = text_value.strip()
    norm_sl = normalize_source_lang(source_lang)
    if norm_sl == "pt-BR":
        return cleaned

    target_lang = "pt-BR"
    cache_key = make_cache_key(cleaned, source_lang=norm_sl, target_lang=target_lang)

    cached = _mem_get(cache_key)
    if cached is not None:
        return cached

    inflight = _get_inflight_map()
    existing_future = inflight.get(cache_key)
    if existing_future is not None:
        return await asyncio.shield(existing_future)

    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
    inflight[cache_key] = future

    try:
        db_hit = await _db_get_translation(cache_key)
        if db_hit:
            _mem_set(cache_key, db_hit)
            if not future.done():
                future.set_result(db_hit)
            return db_hit

        sem = _get_translate_semaphore()
        async with sem:
            translated, provider = await _translate_text_via_providers(
                cleaned, source_lang=norm_sl, target_lang=target_lang
            )

        if translated and provider:
            _mem_set(cache_key, translated)
            await _db_save_translation(
                text_hash=cache_key,
                source_lang=norm_sl,
                target_lang=target_lang,
                source_text=cleaned,
                translated_text=translated,
                provider=provider,
            )
            if not future.done():
                future.set_result(translated)
            return translated

        logger.warning(
            f"[Translator] Todos os provedores falharam para {norm_sl}->{target_lang}; mantendo texto original."
        )
        if not future.done():
            future.set_result(cleaned)
        return cleaned
    except BaseException as exc:
        if not future.done():
            future.set_exception(exc)
        raise
    finally:
        if inflight.get(cache_key) is future:
            inflight.pop(cache_key, None)


async def translate_batch_to_pt_br(items: List[Tuple[str, Optional[str]]]) -> List[str]:
    """Traduz uma lista de (texto, source_lang) pré-carregando acertos do banco em lote."""
    if not items:
        return []

    target_lang = "pt-BR"
    hashes_to_query: List[str] = []
    for raw_text, src_lang in items:
        if not raw_text or not raw_text.strip():
            continue
        norm_sl = normalize_source_lang(src_lang)
        if norm_sl == "pt-BR":
            continue
        key = make_cache_key(raw_text, source_lang=norm_sl, target_lang=target_lang)
        if _mem_get(key) is None:
            hashes_to_query.append(key)

    if hashes_to_query:
        db_hits = await _db_get_translations_bulk(hashes_to_query)
        for k, v in db_hits.items():
            _mem_set(k, v)

    results = await asyncio.gather(
        *(translate_to_pt_br(raw_text, source_lang=src_lang) for raw_text, src_lang in items)
    )
    return list(results)


def _extract_translations_list(translations_payload: Any) -> List[Dict[str, Any]]:
    if isinstance(translations_payload, list):
        return [t for t in translations_payload if isinstance(t, dict)]
    if isinstance(translations_payload, dict):
        inner = translations_payload.get("translations")
        if isinstance(inner, list):
            return [t for t in inner if isinstance(t, dict)]
    return []


def extract_best_field_from_translations(
    translations_payload: Any,
    field: str = "overview",
    original_language: Optional[str] = None,
) -> Tuple[str, Optional[str], bool]:
    """
    Seleciona o melhor texto para `field` (ex: overview, tagline, biography) em `translations` do TMDB.
    Retorna `(text_value, source_lang, needs_translation)`.
    """
    entries = _extract_translations_list(translations_payload)
    valid: List[Tuple[str, str, str]] = []
    for entry in entries:
        data = entry.get("data")
        if not isinstance(data, dict):
            continue
        val = (data.get(field) or "").strip()
        if not val:
            continue
        iso639 = (entry.get("iso_639_1") or "").strip().lower()
        iso3166 = (entry.get("iso_3166_1") or "").strip().upper()
        valid.append((iso639, iso3166, val))

    if not valid:
        return "", None, False

    orig_norm = (original_language or "").strip().lower()

    # 1. pt-BR (não CJK)
    for iso639, iso3166, val in valid:
        if iso639 == "pt" and iso3166 == "BR" and not is_non_latin(val):
            return val, "pt-BR", False

    # 2. Se idioma original é português, qualquer pt não-CJK
    if orig_norm == "pt":
        for iso639, _, val in valid:
            if iso639 == "pt" and not is_non_latin(val):
                return val, "pt", False

    # 3. en-US
    for iso639, iso3166, val in valid:
        if iso639 == "en" and iso3166 == "US":
            return val, "en", True

    # 4. Qualquer en
    for iso639, _, val in valid:
        if iso639 == "en":
            return val, "en", True

    # 5. Idioma original da obra
    if orig_norm:
        for iso639, _, val in valid:
            if iso639 == orig_norm:
                if orig_norm == "pt" and not is_non_latin(val):
                    return val, "pt", False
                return val, orig_norm, True

    # 6. Qualquer pt restante (ex: pt-PT)
    for iso639, _, val in valid:
        if iso639 == "pt" and not is_non_latin(val):
            return val, "pt", False

    # 7. Qualquer outra tradução disponível, priorizando idiomas latinos próximos
    preferred_langs = ("es", "fr", "it", "de")
    for pref in preferred_langs:
        for iso639, _, val in valid:
            if iso639 == pref:
                return val, iso639, True

    iso639, _, val = valid[0]
    return val, (iso639 or "auto"), True


def extract_best_overview_from_translations(
    translations_payload: Any,
    original_language: Optional[str] = None,
) -> Tuple[str, Optional[str], bool]:
    """
    Seleciona a melhor sinopse disponível em `translations` do TMDB.
    Retorna `(overview_text, source_lang, needs_translation)`.
    """
    return extract_best_field_from_translations(
        translations_payload,
        field="overview",
        original_language=original_language,
    )


def extract_latin_title_from_translations(
    translations_payload: Any,
    field: str = "title",
) -> Optional[str]:
    """Tenta extrair título/nome em alfabeto latino (en-US/en) do bloco `translations` do TMDB."""
    entries = _extract_translations_list(translations_payload)
    candidates: List[Tuple[str, str, str]] = []
    for entry in entries:
        data = entry.get("data")
        if not isinstance(data, dict):
            continue
        val = (data.get(field) or "").strip()
        if not val or is_non_latin(val):
            continue
        iso639 = (entry.get("iso_639_1") or "").strip().lower()
        iso3166 = (entry.get("iso_3166_1") or "").strip().upper()
        candidates.append((iso639, iso3166, val))

    for iso639, iso3166, val in candidates:
        if iso639 == "en" and iso3166 == "US":
            return val
    for iso639, _, val in candidates:
        if iso639 == "en":
            return val
    return None


async def resolve_text_field_pt_br(
    current_value: Optional[str],
    field: str,
    translations_payload: Any = None,
    original_language: Optional[str] = None,
    fallback_en_value: Optional[str] = None,
) -> str:
    """
    Garante um campo textual (`overview`, `tagline`, `biography`, etc.) em pt-BR:
    - Mantém se já existir em texto latino.
    - Caso contrário, busca em `translations` ou `fallback_en_value` e traduz automaticamente para pt-BR.
    """
    if current_value and current_value.strip() and not is_non_latin(current_value):
        return current_value.strip()

    val_text, src_lang, needs_tr = extract_best_field_from_translations(
        translations_payload,
        field=field,
        original_language=original_language,
    )
    if val_text:
        if not needs_tr:
            return val_text
        return await translate_to_pt_br(val_text, source_lang=src_lang)

    if fallback_en_value and fallback_en_value.strip():
        return await translate_to_pt_br(fallback_en_value.strip(), source_lang="en")

    return ""


async def resolve_overview_pt_br(
    current_overview: Optional[str],
    translations_payload: Any = None,
    original_language: Optional[str] = None,
    fallback_en_overview: Optional[str] = None,
) -> str:
    """
    Garante uma sinopse em pt-BR:
    - Mantém se já existir em texto latino.
    - Caso contrário, busca em `translations` (en-US/en/original/outros) ou `fallback_en_overview`
      e traduz automaticamente para pt-BR.
    """
    return await resolve_text_field_pt_br(
        current_overview,
        field="overview",
        translations_payload=translations_payload,
        original_language=original_language,
        fallback_en_value=fallback_en_overview,
    )


async def localize_tmdb_list_items_pt_br(
    items: List[Dict[str, Any]],
    fetch_en_items: Callable[[], Awaitable[List[Dict[str, Any]]]],
) -> None:
    """
    Para listas de resultados TMDB (search/trending), verifica itens movie/tv sem título latino
    ou sem sinopse em pt-BR; quando necessário, executa um único fallback en-US e traduz em lote.
    """
    pending: List[Tuple[int, str, str, bool, bool]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        mtype = item.get("media_type")
        if not mtype and ("title" in item or "name" in item):
            mtype = "movie" if "title" in item else "tv"
        if mtype not in ("movie", "tv"):
            continue
        title_key = "title" if mtype == "movie" else "name"
        t_val = (item.get(title_key) or "").strip()
        ov_val = (item.get("overview") or "").strip()
        needs_title = not t_val or is_non_latin(t_val)
        needs_ov = not ov_val or is_non_latin(ov_val)
        if needs_title or needs_ov:
            pending.append((idx, mtype, title_key, needs_title, needs_ov))

    if not pending:
        return

    try:
        en_items = await fetch_en_items()
    except Exception:
        en_items = []

    if not en_items:
        return

    en_by_key: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    for en_it in en_items:
        if not isinstance(en_it, dict) or not en_it.get("id"):
            continue
        en_type = en_it.get("media_type") or ("movie" if "title" in en_it else "tv")
        en_by_key[(en_type, en_it.get("id"))] = en_it

    ovs_to_translate: List[Tuple[int, str]] = []
    for idx, mtype, title_key, needs_title, needs_ov in pending:
        item = items[idx]
        en_it = en_by_key.get((mtype, item.get("id")))
        if not en_it:
            continue
        if needs_title:
            en_title = (
                (en_it.get("title") if mtype == "movie" else en_it.get("name"))
                or en_it.get("title")
                or en_it.get("name")
                or ""
            ).strip()
            if en_title and not is_non_latin(en_title):
                item[title_key] = en_title
        if needs_ov:
            en_ov = (en_it.get("overview") or "").strip()
            if en_ov:
                ovs_to_translate.append((idx, en_ov))

    if ovs_to_translate:
        translated_ovs = await translate_batch_to_pt_br([(txt, "en") for _, txt in ovs_to_translate])
        for (idx, _), tr_ov in zip(ovs_to_translate, translated_ovs):
            items[idx]["overview"] = tr_ov
