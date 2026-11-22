import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from app.core.cache import match_insight_cache
from app.core.config import settings

logger = logging.getLogger(__name__)

INSIGHT_SYSTEM_PROMPT = """Você é um curador editorial de filmes e séries do OmniWatch escrevendo em Português do Brasil (pt-BR).
Sua tarefa é escrever 1 único parágrafo curto (2 a 3 frases, entre 200 e 380 caracteres, no máximo ~65 palavras) explicando de forma humana e natural por que a obra combina com o perfil do usuário.

Regras obrigatórias:
1. Explique brevemente a dinâmica/proposta da obra usando APENAS os fatos da sinopse e dos gêneros fornecidos (ZERO spoilers, ZERO invenção de enredo).
2. Conecte essa proposta às obras da biblioteca do usuário fornecidas em "Obras similares" e destaque quais pilares e faceta principal sustentam o match.
3. Se houver fatores em "O que segurou a nota", cite em poucas palavras o principal ponto que equilibrou a pontuação.
4. Você PODE usar **negrito** apenas para destacar títulos de obras ou o nome da faceta principal.
5. PROIBIDO usar títulos, listas/bullet points, raciocínio (<think>), jargões técnicos (como embedding, cosseno, vetor, prior, algoritmo, score esparso) ou qualquer texto em inglês."""

_LEAK_TOKENS = (
    "the user",
    "based on the provided",
    "here is",
    "let's",
    "therefore",
    "as an ai",
    "in summary",
)

_PILLAR_IMPORTANCE_WEIGHTS = {
    "semantic": 1.45,
    "genres": 1.30,
    "themes": 1.00,
    "talent": 0.95,
    "language": 0.55,
}

_PILLAR_HUMAN_NAMES = {
    "semantic": "atmosfera narrativa",
    "genres": "gêneros",
    "themes": "temas",
    "talent": "elenco e direção",
    "language": "origem cultural",
}

_inflight_insight_tasks: Dict[str, asyncio.Task] = {}


def _strip_outer_quotes(text: str) -> str:
    if len(text) >= 2 and (
        (text.startswith('"') and text.endswith('"'))
        or (text.startswith("'") and text.endswith("'"))
        or (text.startswith("“") and text.endswith("”"))
    ):
        return text[1:-1].strip()
    return text


def _sanitize_insight_text(raw: Optional[str]) -> Optional[str]:
    """Limpa raciocínios, cercas markdown e prefixos, validando idioma e tamanho."""
    if not raw or not isinstance(raw, str):
        return None

    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    text = _strip_outer_quotes(text)
    text = re.sub(
        r"^(?:análise|resumo|insight|explicação|veredito)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = _strip_outer_quotes(text)

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 60 or len(text) > 600:
        return None

    lowered = text.lower()
    if any(tok in lowered for tok in _LEAK_TOKENS):
        return None

    return text


def build_insight_cache_key(
    user_id: str,
    media_type: str,
    tmdb_id: int,
    breakdown: Dict[str, Any],
) -> str:
    """Gera a chave de cache vinculada ao usuário, obra, score arredondado e âncoras."""
    score_int = int(round(float(breakdown.get("score") or 0)))
    anchors = breakdown.get("anchors") or []
    anchor_sig = ",".join(str(a.get("tmdb_id", "")) for a in anchors[:3] if a.get("tmdb_id") is not None)
    return f"{user_id}:{media_type}:{tmdb_id}:{score_int}:{anchor_sig}"


def _evict_oldest_insights_if_needed() -> None:
    if len(match_insight_cache._cache) <= 1000:
        return
    ordered_keys = sorted(
        match_insight_cache._cache.keys(),
        key=lambda k: match_insight_cache._cache[k].get("timestamp", 0),
    )
    for key in ordered_keys[:200]:
        match_insight_cache._cache.pop(key, None)


def build_deterministic_editorial_insight(
    title: str,
    media_type: str,
    genres: List[str],
    breakdown: Dict[str, Any],
) -> str:
    """Gera um parágrafo editorial natural em pt-BR quando a LLM não está disponível."""
    kind_label = "A série" if media_type == "tv" else "O filme"
    clean_genres = [str(g).strip() for g in (genres or []) if g][:3]
    genres_phrase = (
        f" combina {', '.join(clean_genres[:-1])} e {clean_genres[-1]}"
        if len(clean_genres) > 1
        else (f" aposta em {clean_genres[0]}" if clean_genres else " traz uma proposta alinhada ao seu gosto")
    )

    anchors = breakdown.get("anchors") or []
    if len(anchors) >= 2:
        anchors_phrase = (
            f', dialogando diretamente com **{anchors[0]["title"]}** e **{anchors[1]["title"]}**'
        )
    elif len(anchors) == 1:
        anchors_phrase = f', em sintonia com **{anchors[0]["title"]}** na sua biblioteca'
    else:
        anchors_phrase = ""

    sentence_1 = f"{kind_label} **{title}**{genres_phrase}{anchors_phrase}."

    pillars = breakdown.get("pillars") or []
    weighted_pillars = sorted(
        [p for p in pillars if p.get("key") in _PILLAR_IMPORTANCE_WEIGHTS and p.get("percent", 0) >= 60],
        key=lambda p: p.get("percent", 0) * _PILLAR_IMPORTANCE_WEIGHTS.get(p["key"], 0.5),
        reverse=True,
    )
    pillar_bits = [
        f"{_PILLAR_HUMAN_NAMES[p['key']]} ({p['percent']}%)"
        for p in weighted_pillars[:2]
    ]

    personas = breakdown.get("personas") or []
    primary_persona = next((p for p in personas if p.get("is_primary")), None)
    persona_phrase = (
        f" dentro da sua faceta **{primary_persona['name']}**"
        if primary_persona and primary_persona.get("name")
        else ""
    )

    score_int = int(round(float(breakdown.get("score") or 0)))
    if pillar_bits:
        sentence_2_core = (
            f"Os {score_int}% de compatibilidade se apoiam principalmente em "
            f"{' e '.join(pillar_bits)}{persona_phrase}"
        )
    else:
        sentence_2_core = f"Os {score_int}% refletem o equilíbrio atual do seu perfil{persona_phrase}"

    restraints = breakdown.get("restraints") or []
    if score_int < 90 and restraints:
        top_r = restraints[0]
        sentence_2 = f"{sentence_2_core}, moderados por {str(top_r.get('label', '')).lower()}."
    else:
        sentence_2 = f"{sentence_2_core}."

    return f"{sentence_1} {sentence_2}"


def _build_insight_user_prompt(
    title: str,
    media_type: str,
    overview: str,
    genres: List[str],
    breakdown: Dict[str, Any],
) -> str:
    kind = "Série" if media_type == "tv" else "Filme"
    clean_overview = (overview or "").strip()[:500] or "Sinopse não informada."
    clean_genres = ", ".join(str(g) for g in (genres or [])[:5]) or "Não informados"

    personas = breakdown.get("personas") or []
    primary_persona = next((p for p in personas if p.get("is_primary")), None)
    persona_str = primary_persona.get("name") if primary_persona else "Perfil geral"

    pillars_lines = []
    for p in breakdown.get("pillars") or []:
        chips_str = f" [{', '.join(p['chips'][:4])}]" if p.get("chips") else ""
        pillars_lines.append(
            f"- {p.get('label')} ({p.get('impact_label', '')}): {p.get('percent')}%{chips_str}"
        )

    anchors_lines = []
    for a in breakdown.get("anchors") or []:
        shared = f" | em comum: {', '.join(a['shared_genres'])}" if a.get("shared_genres") else ""
        anchors_lines.append(
            f"- {a.get('title')} ({a.get('reason_label')}, {a.get('similarity')}% similar{shared})"
        )

    boosts_lines = [
        f"- {b.get('label')} ({b.get('detail')})"
        for b in (breakdown.get("boosts") or [])
    ]
    restraints_lines = [
        f"- {r.get('label')} ({r.get('detail')})"
        for r in (breakdown.get("restraints") or [])
    ]

    score = breakdown.get("score")
    base_score = breakdown.get("base_score", score)
    net_adj = breakdown.get("net_adjustment", 0)

    sections = [
        f"Obra: {title} ({kind})",
        f"Gêneros: {clean_genres}",
        f"Sinopse: {clean_overview}",
        f"Compatibilidade final: {score}% (base {base_score}%, ajuste líquido {net_adj:+}%)",
        f"Faceta principal do usuário: {persona_str}",
        "Pilares:\n" + ("\n".join(pillars_lines) if pillars_lines else "- Sem pilares"),
        "Obras similares na biblioteca do usuário:\n" + ("\n".join(anchors_lines) if anchors_lines else "- Nenhuma âncora direta"),
    ]
    if boosts_lines:
        sections.append("O que puxou a nota:\n" + "\n".join(boosts_lines))
    if restraints_lines:
        sections.append("O que segurou a nota:\n" + "\n".join(restraints_lines))

    sections.append(
        "Escreva o parágrafo editorial curto em pt-BR conectando a dinâmica da obra, "
        "as semelhanças com a biblioteca e os principais pilares."
    )
    return "\n\n".join(sections)


async def _run_insight_generation(
    cache_key: str,
    title: str,
    media_type: str,
    overview: str,
    genres: List[str],
    breakdown: Dict[str, Any],
) -> Dict[str, Any]:
    api_key = (settings.OPENAI_API_KEY or "").strip()
    if not api_key:
        fallback_text = build_deterministic_editorial_insight(
            title=title,
            media_type=media_type,
            genres=genres,
            breakdown=breakdown,
        )
        match_insight_cache.set(cache_key, {"insight": fallback_text, "source": "fallback"})
        _evict_oldest_insights_if_needed()
        return {"insight": fallback_text, "source": "fallback", "cached": False}

    url = f"{settings.OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://omniwatch.app",
        "X-Title": "OmniWatch",
    }
    payload = {
        "model": settings.OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": INSIGHT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_insight_user_prompt(
                    title=title,
                    media_type=media_type,
                    overview=overview,
                    genres=genres,
                    breakdown=breakdown,
                ),
            },
        ],
        "temperature": 0.45,
        "max_tokens": 260,
    }

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.post(url, json=payload, headers=headers, timeout=12.0)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                raw_content = ((choices[0].get("message") or {}).get("content")) if choices else None
                sanitized = _sanitize_insight_text(raw_content)
                if sanitized:
                    match_insight_cache.set(cache_key, {"insight": sanitized, "source": "llm"})
                    _evict_oldest_insights_if_needed()
                    return {"insight": sanitized, "source": "llm", "cached": False}
                logger.warning("[MatchInsight] Resposta LLM rejeitada na sanitização.")
            else:
                logger.warning(f"[MatchInsight] LLM retornou HTTP {resp.status_code}")
    except Exception as exc:
        logger.warning(f"[MatchInsight] Falha ao gerar insight via LLM ({type(exc).__name__}): {exc}")

    fallback_text = build_deterministic_editorial_insight(
        title=title,
        media_type=media_type,
        genres=genres,
        breakdown=breakdown,
    )
    return {"insight": fallback_text, "source": "fallback", "cached": False}


async def generate_match_editorial_insight(
    user_id: str,
    tmdb_id: int,
    media_type: str,
    title: str,
    overview: str,
    genres: List[str],
    breakdown: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Gera (ou recupera do cache TTL de 24h) um parágrafo editorial curto em pt-BR
    explicando a dinâmica da obra, pilares e semelhanças com a biblioteca.
    """
    cache_key = build_insight_cache_key(
        user_id=str(user_id),
        media_type=media_type,
        tmdb_id=tmdb_id,
        breakdown=breakdown,
    )
    cached, is_stale = match_insight_cache.get_with_status(cache_key)
    if cached and not is_stale and cached.get("insight"):
        return {
            "insight": cached["insight"],
            "source": cached.get("source", "llm"),
            "cached": True,
        }

    existing_task = _inflight_insight_tasks.get(cache_key)
    if existing_task is not None and not existing_task.done():
        res = await existing_task
        return dict(res)

    task = asyncio.create_task(
        _run_insight_generation(
            cache_key=cache_key,
            title=title,
            media_type=media_type,
            overview=overview,
            genres=genres,
            breakdown=breakdown,
        )
    )
    _inflight_insight_tasks[cache_key] = task
    try:
        res = await task
        return dict(res)
    finally:
        if _inflight_insight_tasks.get(cache_key) is task:
            _inflight_insight_tasks.pop(cache_key, None)
