import asyncio
import logging
import httpx
from uuid import UUID
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Set, Tuple, Iterable
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, or_
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.universes.models import (
    Universe,
    UniverseItem,
    UserUniverse,
    UserUniverseSuggestion,
    UserUniverseIgnoredItem,
)
from app.universes.schemas import (
    UniverseFollowResponse,
    UniverseStatusResponse,
    UserUniverseDetailResponse,
    UniverseItemResponse,
    UniverseSuggestionResponse,
    UniverseFollowRequest,
    ToggleIgnoreItemRequest,
    UpdateUniverseSettingsRequest,
)
from app.media.models import Media
from app.tracking.models import UserListItem
from app.tracking.schemas import UserListItemCreate
from app.tracking.services import add_to_list, invalidate_user_stats_cache
from app.core.utils import is_date_released
from app.core.tmdb_client import get_tmdb_client, tmdb_get_json
from app.core.translator import translate_batch_to_pt_br, translate_to_pt_br

logger = logging.getLogger(__name__)

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "OmniWatch/1.0 (https://omniwatch.local; contact@omniwatch.local)",
}

# Controle de scans em background por usuário (debounce)
_pending_universe_scans: Dict[str, asyncio.Task] = {}


class _TMDBHelper:
    async def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        req_params: Dict[str, Any] = {"language": "pt-BR"}
        if params:
            req_params.update(params)
        return await tmdb_get_json(endpoint, params=req_params, timeout=12.0)

    async def get_collection_details(self, collection_id: int) -> Optional[Dict[str, Any]]:
        return await self._get(f"/collection/{collection_id}", params={"language": "pt-BR"})


tmdb_client = _TMDBHelper()


async def _sparql_post(query: str, timeout: float = 25.0, max_attempts: int = 3) -> List[Dict[str, Any]]:
    """Executa uma query SPARQL no Wikidata via POST com otimizador explícito e retry contra 429/timeout."""
    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.post(
                    WIKIDATA_SPARQL_URL,
                    data={"query": query, "format": "json"},
                    headers=WIKIDATA_HEADERS,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("results", {}).get("bindings", [])
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts - 1:
                    delay = 1.5 * (2 ** attempt)
                    logger.warning(f"[Wikidata-RETRY] HTTP {resp.status_code} (tentativa {attempt + 1}/{max_attempts}) — aguardando {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                logger.warning(f"Wikidata SPARQL retornou HTTP {resp.status_code}")
                return []
        except (httpx.TimeoutException, httpx.TransportError, httpx.RequestError) as e:
            if attempt < max_attempts - 1:
                delay = 1.5 * (2 ** attempt)
                logger.warning(f"[Wikidata-RETRY] {type(e).__name__} (tentativa {attempt + 1}/{max_attempts}) — aguardando {delay:.1f}s")
                await asyncio.sleep(delay)
                continue
            logger.warning(f"Falha ao consultar Wikidata SPARQL: {type(e).__name__}: {e}")
            return []
        except Exception as e:
            logger.warning(f"Falha ao consultar Wikidata SPARQL: {e}")
            return []
    return []


async def _wikidata_step1_find_candidates(
    movie_ids: List[int],
    tv_ids: List[int],
) -> Dict[str, Dict[str, Any]]:
    """
    Passo 1: Mapeia filmes (P4947) e séries (P4983) da lista do usuário para
    entidades de universo/franquia no Wikidata em lotes rápidos (~0.6s por lote).
    """
    candidates: Dict[str, Dict[str, Any]] = {}
    desc_lang_by_qid: Dict[str, str] = {}
    batch_size = 45

    async def run_seed_batch(ids: List[int], prop: str, media_type: str):
        if not ids:
            return
        values_str = " ".join(f'"{i}"' for i in ids)
        query = f"""
        SELECT DISTINCT ?seedTmdb ?universe ?universeLabel ?universeDesc ?uType WHERE {{
          hint:Query hint:optimizer "None" .
          VALUES ?seedTmdb {{ {values_str} }}
          ?seedItem wdt:{prop} ?seedTmdb .
          ?seedItem (wdt:P179|wdt:P361|^wdt:P527)+ ?universe .
          ?universe wdt:P31 ?uType .
          VALUES ?uType {{ wd:Q196600 wd:Q559618 wd:Q17517379 wd:Q111466168 wd:Q21198342 }}
          SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "pt-br,pt,en".
            ?universe rdfs:label ?universeLabel .
            ?universe schema:description ?universeDesc .
          }}
        }}
        """
        rows = await _sparql_post(query, timeout=20.0)
        for r in rows:
            u_uri = r.get("universe", {}).get("value", "")
            if not u_uri:
                continue
            qid = u_uri.rsplit("/", 1)[-1]
            label = r.get("universeLabel", {}).get("value", qid)
            desc_obj = r.get("universeDesc") or {}
            desc = (desc_obj.get("value") or "").strip()
            desc_lang = (desc_obj.get("xml:lang") or "").strip().lower()
            u_type = r.get("uType", {}).get("value", "").rsplit("/", 1)[-1]
            seed_id = int(r["seedTmdb"]["value"])

            if qid not in candidates:
                candidates[qid] = {
                    "universe_key": qid,
                    "name": label,
                    "overview": desc,
                    "u_types": set(),
                    "seed_matches": set(),
                    "members": {},
                    "child_collection_ids": set(),
                }
                if desc_lang:
                    desc_lang_by_qid[qid] = desc_lang
            elif not candidates[qid].get("overview") and desc:
                candidates[qid]["overview"] = desc
                if desc_lang:
                    desc_lang_by_qid[qid] = desc_lang
            candidates[qid]["u_types"].add(u_type)
            candidates[qid]["seed_matches"].add((seed_id, media_type))

    tasks = []
    for start in range(0, len(movie_ids), batch_size):
        tasks.append(run_seed_batch(movie_ids[start : start + batch_size], "P4947", "movie"))
    for start in range(0, len(tv_ids), batch_size):
        tasks.append(run_seed_batch(tv_ids[start : start + batch_size], "P4983", "tv"))

    # Executa em grupos de até 3 queries concorrentes para respeitar limites do Wikidata
    for i in range(0, len(tasks), 3):
        await asyncio.gather(*tasks[i : i + 3])

    descs_to_translate = [
        (qid, cand["overview"], desc_lang_by_qid[qid])
        for qid, cand in candidates.items()
        if cand.get("overview")
        and desc_lang_by_qid.get(qid)
        and not desc_lang_by_qid[qid].startswith("pt")
    ]
    if descs_to_translate:
        translated_descs = await translate_batch_to_pt_br(
            [(txt, lang) for _, txt, lang in descs_to_translate]
        )
        for (qid, _, _), tr_desc in zip(descs_to_translate, translated_descs):
            candidates[qid]["overview"] = tr_desc

    return candidates


async def _wikidata_step2_expand_members(candidates: Dict[str, Dict[str, Any]]) -> None:
    """
    Passo 2: Para cada universo candidato (QID), busca todos os filmes (P4947)
    e séries (P4983) pertencentes a ele no Wikidata.
    """
    if not candidates:
        return

    qids = [qid for qid in candidates.keys() if qid.startswith("Q")]
    batch_size = 15

    async def expand_batch(batch: List[str]):
        u_values = " ".join(f"wd:{q}" for q in batch)
        query = f"""
        SELECT DISTINCT ?universe ?tmdbMovie ?tmdbTv WHERE {{
          hint:Query hint:optimizer "None" .
          VALUES ?universe {{ {u_values} }}
          ?member (wdt:P179|wdt:P361|^wdt:P527)+ ?universe .
          {{ ?member wdt:P4947 ?tmdbMovie . }}
          UNION
          {{ ?member wdt:P4983 ?tmdbTv . }}
        }}
        """
        rows = await _sparql_post(query, timeout=25.0)
        for r in rows:
            qid = r.get("universe", {}).get("value", "").rsplit("/", 1)[-1]
            if qid not in candidates:
                continue
            if "tmdbMovie" in r and r["tmdbMovie"]["value"].isdigit():
                mid = int(r["tmdbMovie"]["value"])
                candidates[qid]["members"][(mid, "movie")] = {"tmdb_id": mid, "media_type": "movie"}
            if "tmdbTv" in r and r["tmdbTv"]["value"].isdigit():
                tid = int(r["tmdbTv"]["value"])
                candidates[qid]["members"][(tid, "tv")] = {"tmdb_id": tid, "media_type": "tv"}

    tasks = [expand_batch(qids[i : i + batch_size]) for i in range(0, len(qids), batch_size)]
    for i in range(0, len(tasks), 3):
        await asyncio.gather(*tasks[i : i + 3])


GENERIC_TROPE_KEYWORDS = {
    "alternate universe",
    "parallel universe",
    "pocket universe",
    "mirror universe",
    "virtual universe",
    "fictional universe",
    "shared universe",
    "cinematic universe",
    "expanded universe",
    "extended universe",
    "multiple universes",
    "miss universe",
    "universe",
    "universo paralelo",
    "universo alternativo",
}

RECAP_KEYWORDS = {
    "recap",
    "compilation",
    "compilado",
    "compilação",
    "arco da",
    "arco de",
    "arco do",
    "arco dos",
    "arco ",
    "arcos",
    "arc compilation",
    "special edition",
    "edição especial",
    "edição comemorativa",
    "決戦編",
    "総集編",
    "潜入編",
    "recapitulação",
    "swordsmith village",
    "hashira training",
    "vínculo de irmãos",
    "district decisive battle",
    "district infiltration",
    "special edition swordsmith village",
    "special edition hashira training",
    "arc part",
    "the movie: recap",
    "recapeamento",
    "omnibus",
}

RECAP_OVERVIEW_MARKERS = (
    "reconta os primeiros episódios",
    "filme de compilação",
    "compilação dos episódios",
    "episódios compilados",
    "resumo da temporada",
    "compilation of episodes",
    "recap of",
    "recaps the",
)


def _check_is_recap(
    title: str,
    overview: Optional[str] = None,
    keywords: Optional[List[Any]] = None,
    media_type: str = "movie",
) -> bool:
    """
    Camada de Filtro Preventivo de Recaps/Compilações em Universos:
    Detecta se uma obra é um filme recap ou compilação de episódios da TV para
    evitar que ela entre e polua a linha do tempo ou distorça o progresso do universo.
    """
    if media_type != "movie":
        return False

    t_lower = (title or "").lower()
    o_lower = (overview or "").lower()

    # 1. Verifica keywords oficiais do TMDB
    if keywords:
        for k in keywords:
            name = (k.get("name") if isinstance(k, dict) else str(k)).strip().lower()
            if name in ("edited from tv series", "tv compilation", "recap"):
                return True
            if name == "compilation" and any(
                tag in (t_lower + " " + o_lower)
                for tag in ("anime", "demon", "kimetsu", "episódio", "episodio", "episode", "season", "temporada", "arco")
            ):
                return True

    # 2. Verifica palavras-chave e nomes de arcos no título
    for kw in RECAP_KEYWORDS:
        if kw in t_lower:
            return True

    # 3. Verifica marcadores de compilação na sinopse/overview
    for marker in RECAP_OVERVIEW_MARKERS:
        if marker in o_lower:
            return True

    return False


def _is_non_latin(s: str) -> bool:
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



async def _fetch_tmdb_keyword_members(
    kw_id: int,
    max_pages: int = 4,
) -> Dict[Tuple[int, str], Dict[str, Any]]:
    """
    Busca filmes e séries associados a uma keyword do TMDB com paginação controlada
    (até `max_pages` páginas por tipo) para cobrir universos grandes sem cortar em 20 itens.
    """
    members: Dict[Tuple[int, str], Dict[str, Any]] = {}

    def _ingest_results(results: List[Dict[str, Any]], media_type: str) -> None:
        for item in results:
            tid = item.get("id")
            if not tid:
                continue
            if media_type == "movie":
                title = item.get("title") or item.get("original_title") or f"Filme #{tid}"
                release_date = item.get("release_date") or None
            else:
                title = item.get("name") or item.get("original_name") or f"Série #{tid}"
                release_date = item.get("first_air_date") or None
            if _check_is_recap(title, item.get("overview"), None, media_type):
                continue
            members[(tid, media_type)] = {
                "tmdb_id": tid,
                "media_type": media_type,
                "title": title,
                "release_date": release_date,
                "poster_path": item.get("poster_path"),
                "backdrop_path": item.get("backdrop_path"),
            }

    movies_p1, tv_p1 = await asyncio.gather(
        tmdb_client._get("/discover/movie", params={"with_keywords": str(kw_id), "language": "pt-BR", "page": 1}),
        tmdb_client._get("/discover/tv", params={"with_keywords": str(kw_id), "language": "pt-BR", "page": 1}),
    )

    extra_tasks: List[Tuple[str, Any]] = []
    if movies_p1 and "results" in movies_p1:
        _ingest_results(movies_p1["results"], "movie")
        total_m_pages = min(int(movies_p1.get("total_pages") or 1), max_pages)
        for p in range(2, total_m_pages + 1):
            extra_tasks.append((
                "movie",
                tmdb_client._get("/discover/movie", params={"with_keywords": str(kw_id), "language": "pt-BR", "page": p}),
            ))

    if tv_p1 and "results" in tv_p1:
        _ingest_results(tv_p1["results"], "tv")
        total_t_pages = min(int(tv_p1.get("total_pages") or 1), max_pages)
        for p in range(2, total_t_pages + 1):
            extra_tasks.append((
                "tv",
                tmdb_client._get("/discover/tv", params={"with_keywords": str(kw_id), "language": "pt-BR", "page": p}),
            ))

    if extra_tasks:
        extra_resps = await asyncio.gather(*(coro for _, coro in extra_tasks))
        for (mtype, _), resp in zip(extra_tasks, extra_resps):
            if resp and "results" in resp:
                _ingest_results(resp["results"], mtype)

    return members


def _normalize_source_keys(primary_key: str, extra_keys: Optional[Iterable[str]] = None) -> List[str]:
    """Normaliza, deduplica e ordena chaves secundárias (aliases) de um universo, excluindo a chave primária."""
    cleaned: Set[str] = set()
    for k in (extra_keys or []):
        if isinstance(k, str):
            ks = k.strip()
            if ks and ks != primary_key:
                cleaned.add(ks)
    return sorted(cleaned)


def _should_merge_universes(
    pairs_a: Set[Tuple[int, str]],
    key_a: str,
    pairs_b: Set[Tuple[int, str]],
    key_b: str,
) -> bool:
    """
    Decide se dois universos representam a mesma franquia/universo com base na
    sobreposição real de obras (Coeficiente de Sobreposição Szymkiewicz-Simpson),
    sem depender de comparação de nomes.
    """
    if key_a == key_b:
        return True
    if not pairs_a or not pairs_b:
        return False

    intersection = len(pairs_a & pairs_b)
    if intersection < 2:
        return False

    min_size = min(len(pairs_a), len(pairs_b))
    overlap = intersection / min_size

    is_cross_source = (
        (key_a.startswith("Q") and key_b.startswith("kw:"))
        or (key_b.startswith("Q") and key_a.startswith("kw:"))
    )
    if is_cross_source:
        return (intersection >= 3 and overlap >= 0.35) or (intersection >= 2 and overlap >= 0.50)

    return intersection >= 2 and overlap >= 0.50


def _canonical_rank(ukey: str, member_count: int, is_followed: bool = False) -> Tuple[int, int, int, str]:
    """
    Critério de escolha do universo canônico em um cluster:
    1. Universo já seguido pelo usuário
    2. Fonte Wikidata (Q...) sobre TMDB Keyword (kw:...)
    3. Maior número de obras
    4. Desempate determinístico pela chave
    """
    return (
        1 if is_followed else 0,
        1 if ukey.startswith("Q") else 0,
        member_count,
        ukey,
    )


def _merge_member_entry(
    existing: Optional[Dict[str, Any]],
    incoming: Dict[str, Any],
) -> Dict[str, Any]:
    """Combina metadados de uma mesma obra (tmdb_id, media_type) preservando os campos mais completos."""
    if not existing:
        return dict(incoming)
    merged = dict(existing)
    inc_title = incoming.get("title")
    cur_title = merged.get("title")
    if inc_title and (not cur_title or _is_non_latin(cur_title)) and not _is_non_latin(inc_title):
        merged["title"] = inc_title
    elif not cur_title and inc_title:
        merged["title"] = inc_title

    for field in ("release_date", "poster_path", "backdrop_path", "collection_name", "overview"):
        if not merged.get(field) and incoming.get(field):
            merged[field] = incoming[field]

    if not merged.get("keywords_list") and incoming.get("keywords_list"):
        merged["keywords_list"] = incoming["keywords_list"]
    return merged


def _merge_overlapping_candidates(
    candidates: Dict[str, Dict[str, Any]],
    followed_keys: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Agrupa candidatos que representam o mesmo universo (por sobreposição real de obras
    ou compartilhamento de source_keys) usando Union-Find e funde seus membros, coleções
    e chaves secundárias em um único candidato canônico por cluster.
    """
    if len(candidates) <= 1:
        for ukey, cand in candidates.items():
            cand["source_keys"] = _normalize_source_keys(ukey, cand.get("source_keys"))
        return candidates

    keys = list(candidates.keys())
    followed_set = followed_keys or set()
    parent: Dict[str, str] = {k: k for k in keys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pairs_by_key: Dict[str, Set[Tuple[int, str]]] = {}
    all_keys_by_cand: Dict[str, Set[str]] = {}
    for k, cand in candidates.items():
        members_dict = cand.get("members") or {}
        pairs_by_key[k] = set(members_dict.keys())
        all_keys_by_cand[k] = {k, *(_normalize_source_keys(k, cand.get("source_keys")))}

    for i in range(len(keys)):
        ki = keys[i]
        for j in range(i + 1, len(keys)):
            kj = keys[j]
            if (all_keys_by_cand[ki] & all_keys_by_cand[kj]) or _should_merge_universes(
                pairs_by_key[ki], ki, pairs_by_key[kj], kj
            ):
                union(ki, kj)

    clusters: Dict[str, List[str]] = {}
    for k in keys:
        root = find(k)
        clusters.setdefault(root, []).append(k)

    merged_candidates: Dict[str, Dict[str, Any]] = {}
    for cluster_keys in clusters.values():
        if len(cluster_keys) == 1:
            k = cluster_keys[0]
            cand = candidates[k]
            cand["source_keys"] = _normalize_source_keys(k, cand.get("source_keys"))
            merged_candidates[k] = cand
            continue

        canonical_key = max(
            cluster_keys,
            key=lambda k: _canonical_rank(k, len(pairs_by_key[k]), is_followed=(k in followed_set)),
        )
        canonical = dict(candidates[canonical_key])
        merged_members: Dict[Tuple[int, str], Dict[str, Any]] = {
            pair: dict(info) for pair, info in (canonical.get("members") or {}).items()
        }
        merged_seeds: Set[Tuple[int, str]] = set(canonical.get("seed_matches") or set())
        merged_cols: Set[int] = set(canonical.get("child_collection_ids") or set())
        merged_types: Set[str] = set(canonical.get("u_types") or set())
        absorbed_keys: Set[str] = set(canonical.get("source_keys") or [])

        for other_key in cluster_keys:
            if other_key == canonical_key:
                continue
            other = candidates[other_key]
            absorbed_keys.add(other_key)
            absorbed_keys.update(other.get("source_keys") or [])
            merged_seeds.update(other.get("seed_matches") or set())
            merged_cols.update(other.get("child_collection_ids") or set())
            merged_types.update(other.get("u_types") or set())

            for pair, info in (other.get("members") or {}).items():
                merged_members[pair] = _merge_member_entry(merged_members.get(pair), info)

            if not canonical.get("overview") and other.get("overview"):
                canonical["overview"] = other["overview"]

        canonical["universe_key"] = canonical_key
        canonical["members"] = merged_members
        canonical["seed_matches"] = merged_seeds
        canonical["child_collection_ids"] = merged_cols
        canonical["u_types"] = merged_types
        canonical["source_keys"] = _normalize_source_keys(canonical_key, absorbed_keys)
        merged_candidates[canonical_key] = canonical

        logger.info(
            f"🔗 [Universes-Merge] Cluster unido em '{canonical.get('name')}' ({canonical_key}) "
            f"com aliases {canonical['source_keys']} -> {len(merged_members)} obras totais."
        )

    return merged_candidates


async def _discover_keyword_universes_for_items(
    unresolved_items: List[Media],
    followed_keys: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Fallback via Keywords nomeadas do TMDB para itens que não encontraram universo no Wikidata
    (ex.: 'marvel cinematic universe (mcu)', 'the conjuring universe', 'dc extended universe').
    Ignora tags genéricas de enredo como 'alternate universe' ou 'parallel universe'.
    """
    kw_candidates: Dict[str, Dict[str, Any]] = {}
    semaphore = asyncio.Semaphore(5)

    async def check_item_keywords(media: Media):
        endpoint = f"/{media.media_type}/{media.tmdb_id}/keywords"
        async with semaphore:
            data = await tmdb_client._get(endpoint)
        if not data:
            return
        kw_list = data.get("keywords") or data.get("results") or []
        for kw in kw_list:
            kw_id = kw.get("id")
            kw_name = (kw.get("name") or "").strip()
            kw_lower = kw_name.lower()
            if not kw_id or not kw_name or kw_lower in GENERIC_TROPE_KEYWORDS:
                continue
            if any(
                phrase in kw_lower
                for phrase in ("cinematic universe", "extended universe", "shared universe", "conjuring universe", "monsterverse")
            ):
                ukey = f"kw:{kw_id}"
                if ukey in followed_keys:
                    continue
                if ukey not in kw_candidates:
                    display_name = kw_name.title()
                    kw_candidates[ukey] = {
                        "universe_key": ukey,
                        "source_keys": [],
                        "keyword_id": kw_id,
                        "name": display_name,
                        "overview": f"Universo compartilhado identificado pela tag '{display_name}' no TMDB.",
                        "u_types": {"TMDB_KEYWORD"},
                        "seed_matches": set(),
                        "members": {},
                        "child_collection_ids": set(),
                    }
                kw_candidates[ukey]["seed_matches"].add((media.tmdb_id, media.media_type))

    await asyncio.gather(*(check_item_keywords(m) for m in unresolved_items[:25]))

    for ukey, cand in list(kw_candidates.items()):
        kw_id = cand["keyword_id"]
        cand["members"] = await _fetch_tmdb_keyword_members(kw_id, max_pages=4)

    return kw_candidates


def _is_candidate_potentially_valid(
    cand: Dict[str, Any],
    known_movie_collections: Dict[int, Optional[int]],
) -> bool:
    """Pré-filtro rápido antes de consultar detalhes no TMDB."""
    members = cand.get("members", {})
    if len(members) < 2:
        return False

    movie_ids = [tid for (tid, mtype) in members.keys() if mtype == "movie"]
    tv_ids = [tid for (tid, mtype) in members.keys() if mtype == "tv"]

    if len(tv_ids) >= 1 and len(members) >= 2:
        return True

    known_cols = {
        known_movie_collections.get(mid)
        for mid in movie_ids
        if known_movie_collections.get(mid) and known_movie_collections.get(mid, 0) > 0
    }
    if len(known_cols) >= 2:
        return True

    unresolved_cols = sum(1 for mid in movie_ids if known_movie_collections.get(mid) is None)
    if len(movie_ids) >= 3 and unresolved_cols >= 1:
        return True

    return False


def _is_enriched_universe_valid(
    enriched_members: List[Dict[str, Any]],
    child_col_ids: Set[int],
) -> bool:
    """
    Validação definitiva pós-enriquecimento TMDB (Anti-Duplicação contra Coleções simples):
    Só é um Universo se:
    1. Tiver Filmes + Séries (movies >= 1 e tv >= 1) ou múltiplas Séries (tv >= 2, ex: Breaking Bad + Better Call Saul), OU
    2. Tiver >= 2 Coleções de filmes distintas no TMDB (ex: Invocação do Mal + Annabelle + A Freira, ou Harry Potter + Animais Fantásticos).
    """
    if len(enriched_members) < 2:
        return False

    tvs = [m for m in enriched_members if m.get("media_type") == "tv"]

    if len(tvs) >= 1 and len(enriched_members) >= 2:
        return True

    if len(child_col_ids) >= 2:
        return True

    return False


async def _enrich_universe_members(
    db: AsyncSession,
    cand: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Enriquece os membros descobertos de um universo com metadados do banco local e do TMDB,
    incluindo expansão de coleções filhas (child_collection_ids).
    """
    members_map: Dict[Tuple[int, str], Dict[str, Any]] = dict(cand.get("members", {}))
    child_col_ids: Set[int] = set(cand.get("child_collection_ids", set()))
    semaphore = asyncio.Semaphore(16)

    # 1. Preencher dados já existentes na tabela local Media (evita chamadas HTTP desnecessárias)
    all_tmdb_ids = [tid for (tid, _) in members_map.keys()]
    if all_tmdb_ids:
        stmt = select(Media).where(Media.tmdb_id.in_(all_tmdb_ids))
        res = await db.execute(stmt)
        for m in res.scalars().all():
            key = (m.tmdb_id, m.media_type)
            if key in members_map:
                if _check_is_recap(m.title, None, m.keywords, m.media_type):
                    members_map.pop(key, None)
                    continue
                entry = members_map[key]
                if m.title and not _is_non_latin(m.title):
                    entry["title"] = entry.get("title") or m.title
                entry["release_date"] = entry.get("release_date") or m.release_date
                entry["poster_path"] = entry.get("poster_path") or m.poster_path
                entry["backdrop_path"] = entry.get("backdrop_path") or m.backdrop_path
                entry["collection_name"] = entry.get("collection_name") or m.collection_name
                entry["keywords_list"] = m.keywords or []
                if m.media_type == "movie" and m.collection_tmdb_id and m.collection_tmdb_id > 0:
                    child_col_ids.add(m.collection_tmdb_id)

    # 2. Buscar no TMDB os itens que ainda não possuem título/poster ou cujo título está em japonês/CJK
    missing_keys = [k for k, v in members_map.items() if not v.get("title") or _is_non_latin(v.get("title", ""))]

    async def fetch_missing_item(key: Tuple[int, str]):
        tid, mtype = key
        async with semaphore:
            data = await tmdb_client._get(f"/{mtype}/{tid}", params={"language": "pt-BR"})
        title_cand = data.get("title") or data.get("name") if data else None
        if not data or not title_cand or _is_non_latin(title_cand):
            async with semaphore:
                en_data = await tmdb_client._get(f"/{mtype}/{tid}", params={"language": "en-US"})
            if en_data and (en_data.get("title") or en_data.get("name")):
                if not data:
                    data = en_data
                else:
                    data["title"] = en_data.get("title") or en_data.get("name")
                    data["name"] = en_data.get("title") or en_data.get("name")
                    if not data.get("overview") and en_data.get("overview"):
                        data["overview"] = en_data.get("overview")
        if not data:
            return key, None
        col_info = data.get("belongs_to_collection")
        col_name = col_info.get("name") if isinstance(col_info, dict) else None
        if isinstance(col_info, dict) and col_info.get("id"):
            child_col_ids.add(col_info["id"])
        final_title = data.get("title") or data.get("name") or data.get("original_title") or data.get("original_name") or f"Item #{tid}"
        kw_data = data.get("keywords") or data.get("keywords_list")
        if isinstance(kw_data, dict):
            kw_list = kw_data.get("keywords") or kw_data.get("results") or []
        elif isinstance(kw_data, list):
            kw_list = kw_data
        else:
            kw_list = []
        is_recap = _check_is_recap(final_title, data.get("overview"), kw_list, mtype)
        if is_recap:
            return key, None
        return key, {
            "tmdb_id": tid,
            "media_type": mtype,
            "title": final_title,
            "release_date": data.get("release_date") or data.get("first_air_date") or None,
            "poster_path": data.get("poster_path"),
            "backdrop_path": data.get("backdrop_path"),
            "collection_name": col_name,
            "is_recap": False,
        }

    if missing_keys:
        fetched = await asyncio.gather(*(fetch_missing_item(k) for k in missing_keys[:80]))
        for key, info in fetched:
            if info:
                members_map[key] = info
            else:
                members_map.pop(key, None)

    # 3. Expandir coleções filhas do TMDB (ex: trilogias dentro de um universo) para garantir 100% dos filmes
    async def fetch_child_col(col_id: int):
        async with semaphore:
            return col_id, await tmdb_client.get_collection_details(col_id)

    if child_col_ids:
        col_results = await asyncio.gather(*(fetch_child_col(cid) for cid in list(child_col_ids)[:20]))
        for cid, col_data in col_results:
            if not col_data or "parts" not in col_data:
                continue
            col_name = col_data.get("name")
            for part in col_data.get("parts", []):
                pid = part.get("id")
                if not pid:
                    continue
                part_title = part.get("title") or part.get("original_title") or f"Filme #{pid}"
                if _check_is_recap(part_title, part.get("overview"), None, "movie"):
                    continue
                key = (pid, "movie")
                if key not in members_map:
                    members_map[key] = {
                        "tmdb_id": pid,
                        "media_type": "movie",
                        "title": part_title,
                        "release_date": part.get("release_date") or None,
                        "poster_path": part.get("poster_path"),
                        "backdrop_path": part.get("backdrop_path"),
                        "collection_name": col_name,
                        "is_recap": False,
                    }
                elif not members_map[key].get("collection_name") and col_name:
                    members_map[key]["collection_name"] = col_name

    # 4. Pós-verificação: Garante que NENHUMA obra adicionada pelas coleções filhas fique em japonês/CJK
    post_cjk_keys = [
        k for k, v in members_map.items()
        if not v.get("title") or _is_non_latin(v.get("title", ""))
    ]
    if post_cjk_keys:
        post_fetched = await asyncio.gather(*(fetch_missing_item(k) for k in post_cjk_keys[:80]))
        for key, info in post_fetched:
            if info:
                members_map[key] = info
            else:
                members_map.pop(key, None)

    cand["child_collection_ids"] = child_col_ids

    # 5. Filtro preventivo estrito: Remove qualquer recap/compilação remanescente e ordena por data
    cleaned_members = [
        v for v in members_map.values()
        if not _check_is_recap(
            v.get("title", ""),
            v.get("overview"),
            v.get("keywords_list") or v.get("keywords"),
            v.get("media_type", "movie"),
        )
    ]
    cleaned_members.sort(key=lambda x: x.get("release_date") or "9999-12-31")
    return cleaned_members


def _extract_member_pairs_from_json(members: Optional[List[Dict[str, Any]]]) -> Set[Tuple[int, str]]:
    pairs: Set[Tuple[int, str]] = set()
    for m in (members or []):
        tid = m.get("tmdb_id")
        if tid:
            pairs.add((int(tid), m.get("media_type", "movie")))
    return pairs


def _merge_enriched_member_lists(
    lists: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Une listas de membros enriquecidos (dicts), deduplicando por (tmdb_id, media_type) e ordenando por data."""
    merged: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for member_list in lists:
        for m in (member_list or []):
            tid = m.get("tmdb_id")
            if not tid:
                continue
            mtype = m.get("media_type", "movie")
            if _check_is_recap(
                m.get("title", ""),
                m.get("overview"),
                m.get("keywords_list") or m.get("keywords"),
                mtype,
            ):
                continue
            pair = (int(tid), mtype)
            merged[pair] = _merge_member_entry(merged.get(pair), m)

    result = list(merged.values())
    result.sort(key=lambda x: x.get("release_date") or "9999-12-31")
    return result


def _count_distinct_collections(members: List[Dict[str, Any]]) -> int:
    cols = {
        (m.get("collection_name") or "").strip().lower()
        for m in members
        if m.get("media_type", "movie") == "movie" and (m.get("collection_name") or "").strip()
    }
    return len(cols)


async def _absorb_members_into_followed_universe(
    db: AsyncSession,
    universe: Universe,
    incoming_key: str,
    incoming_source_keys: Optional[Iterable[str]],
    incoming_members: List[Dict[str, Any]],
) -> bool:
    """
    Absorve uma chave secundária (alias) e eventuais obras faltantes em um Universo já seguido,
    evitando que uma sugestão duplicada seja criada para o usuário.
    """
    changed = False
    new_source_keys = _normalize_source_keys(
        universe.universe_key,
        [*(universe.source_keys or []), incoming_key, *(incoming_source_keys or [])],
    )
    if new_source_keys != (universe.source_keys or []):
        universe.source_keys = new_source_keys
        changed = True

    existing_pairs = {(it.tmdb_id, it.media_type) for it in universe.items}
    added_any = False
    for m in incoming_members:
        tid = m.get("tmdb_id")
        if not tid:
            continue
        mtype = m.get("media_type", "movie")
        pair = (int(tid), mtype)
        if pair in existing_pairs:
            continue
        if _check_is_recap(m.get("title", ""), m.get("overview"), m.get("keywords_list") or m.get("keywords"), mtype):
            continue
        u_item = UniverseItem(
            universe_id=universe.id,
            tmdb_id=int(tid),
            media_type=mtype,
            title=m.get("title") or f"Item #{tid}",
            release_date=m.get("release_date"),
            poster_path=m.get("poster_path"),
            backdrop_path=m.get("backdrop_path"),
            collection_name=m.get("collection_name"),
        )
        db.add(u_item)
        universe.items.append(u_item)
        existing_pairs.add(pair)
        added_any = True
        changed = True

    if added_any:
        universe.items_count = len(universe.items)
        universe.movies_count = sum(1 for it in universe.items if it.media_type == "movie")
        universe.tv_count = sum(1 for it in universe.items if it.media_type == "tv")
        distinct_cols = {
            (it.collection_name or "").strip().lower()
            for it in universe.items
            if it.media_type == "movie" and (it.collection_name or "").strip()
        }
        universe.collections_count = max(universe.collections_count or 0, len(distinct_cols))
        logger.info(
            f"🔗 [Universes-Absorb] Universo seguido '{universe.name}' ({universe.universe_key}) "
            f"absorveu '{incoming_key}' -> total agora: {universe.items_count} obras."
        )

    return changed


async def _consolidate_user_universe_suggestions(
    db: AsyncSession,
    user_id: UUID,
    suggestions_by_key: Dict[str, UserUniverseSuggestion],
    followed_universes: List[Universe],
    user_media_by_pair: Optional[Dict[Tuple[int, str], Media]] = None,
) -> bool:
    """
    Consolida sugestões duplicadas já persistidas no banco para o usuário:
    1. Se uma sugestão coincidir (por chave, alias ou sobreposição real de obras) com um universo
       já seguido, absorve seus itens e alias no universo seguido e remove a sugestão duplicada.
    2. Se duas ou mais sugestões representarem o mesmo universo (ex.: Wikidata Q... + TMDB kw:...),
       funde todos os membros, coleções e source_keys na sugestão canônica e remove as duplicadas.
    Retorna True se houve qualquer alteração no banco.
    """
    changed = False

    followed_keys_all: Set[str] = set()
    followed_pairs_map: List[Tuple[Universe, Set[Tuple[int, str]]]] = []
    for u in followed_universes:
        followed_keys_all.add(u.universe_key)
        followed_keys_all.update(u.source_keys or [])
        followed_pairs_map.append((u, {(it.tmdb_id, it.media_type) for it in u.items}))

    # Passo 1: Verifica sobreposição contra universos já seguidos
    for ukey, sugg in list(suggestions_by_key.items()):
        sugg_all_keys = {ukey, *(sugg.source_keys or [])}
        sugg_pairs = _extract_member_pairs_from_json(sugg.discovered_members)

        matched_followed: Optional[Universe] = None
        for u, u_pairs in followed_pairs_map:
            u_all_keys = {u.universe_key, *(u.source_keys or [])}
            if (sugg_all_keys & u_all_keys) or _should_merge_universes(u_pairs, u.universe_key, sugg_pairs, ukey):
                matched_followed = u
                break

        if matched_followed is not None:
            await _absorb_members_into_followed_universe(
                db,
                matched_followed,
                ukey,
                sugg.source_keys,
                sugg.discovered_members or [],
            )
            await db.delete(sugg)
            suggestions_by_key.pop(ukey, None)
            changed = True

    if len(suggestions_by_key) <= 1:
        return changed

    # Passo 2: Agrupa sugestões equivalentes entre si via Union-Find
    keys = list(suggestions_by_key.keys())
    parent: Dict[str, str] = {k: k for k in keys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pairs_by_key: Dict[str, Set[Tuple[int, str]]] = {}
    all_keys_by_sugg: Dict[str, Set[str]] = {}
    for k, sugg in suggestions_by_key.items():
        pairs_by_key[k] = _extract_member_pairs_from_json(sugg.discovered_members)
        all_keys_by_sugg[k] = {k, *(_normalize_source_keys(k, sugg.source_keys))}

    for i in range(len(keys)):
        ki = keys[i]
        for j in range(i + 1, len(keys)):
            kj = keys[j]
            if (all_keys_by_sugg[ki] & all_keys_by_sugg[kj]) or _should_merge_universes(
                pairs_by_key[ki], ki, pairs_by_key[kj], kj
            ):
                union(ki, kj)

    clusters: Dict[str, List[str]] = {}
    for k in keys:
        clusters.setdefault(find(k), []).append(k)

    for cluster_keys in clusters.values():
        if len(cluster_keys) <= 1:
            continue

        canonical_key = max(
            cluster_keys,
            key=lambda k: _canonical_rank(k, len(pairs_by_key[k]), is_followed=False),
        )
        canonical = suggestions_by_key[canonical_key]
        other_suggs = [suggestions_by_key[k] for k in cluster_keys if k != canonical_key]

        merged_members = _merge_enriched_member_lists(
            [canonical.discovered_members or [], *(s.discovered_members or [] for s in other_suggs)]
        )
        absorbed_keys: Set[str] = set(canonical.source_keys or [])
        all_dismissed = bool(canonical.is_dismissed)
        for s in other_suggs:
            absorbed_keys.add(s.universe_key)
            absorbed_keys.update(s.source_keys or [])
            all_dismissed = all_dismissed and bool(s.is_dismissed)
            if not canonical.poster_path and s.poster_path:
                canonical.poster_path = s.poster_path
            if not canonical.backdrop_path and s.backdrop_path:
                canonical.backdrop_path = s.backdrop_path
            if not canonical.overview and s.overview:
                canonical.overview = s.overview

        movies_count = sum(1 for x in merged_members if x.get("media_type", "movie") == "movie")
        tv_count = sum(1 for x in merged_members if x.get("media_type") == "tv")
        collections_count = max(
            canonical.collections_count or 0,
            *(s.collections_count or 0 for s in other_suggs),
            _count_distinct_collections(merged_members),
        )

        canonical.discovered_members = merged_members
        canonical.total_items = len(merged_members)
        canonical.movies_count = movies_count
        canonical.tv_count = tv_count
        canonical.collections_count = collections_count
        canonical.source_keys = _normalize_source_keys(canonical_key, absorbed_keys)
        canonical.is_dismissed = all_dismissed

        if user_media_by_pair is not None:
            matched_titles: List[str] = []
            for m in merged_members:
                mpair = (m.get("tmdb_id"), m.get("media_type", "movie"))
                if mpair in user_media_by_pair:
                    u_media = user_media_by_pair[mpair]
                    if u_media.title not in matched_titles:
                        matched_titles.append(u_media.title)
                    u_media.universe_key = canonical_key
                    u_media.universe_name = canonical.name
            canonical.matched_titles = matched_titles
            canonical.items_in_list = len(matched_titles)
        else:
            merged_titles: List[str] = list(canonical.matched_titles or [])
            for s in other_suggs:
                for t in (s.matched_titles or []):
                    if t not in merged_titles:
                        merged_titles.append(t)
            canonical.matched_titles = merged_titles
            canonical.items_in_list = len(merged_titles)

        for s in other_suggs:
            await db.delete(s)
            suggestions_by_key.pop(s.universe_key, None)

        changed = True
        logger.info(
            f"🔗 [Universes-Consolidate] Sugestões duplicadas fundidas em '{canonical.name}' "
            f"({canonical_key}, aliases={canonical.source_keys}) -> {canonical.total_items} obras."
        )

    return changed


async def generate_user_universe_suggestions(
    db: AsyncSession,
    user_id: UUID,
    limit: int = 20,
) -> List[UniverseSuggestionResponse]:
    """
    Escaneia a lista do usuário em background para descobrir sugestões de Universos:
    - Ignora imediatamente filmes/séries que já pertencem a Universos seguidos pelo usuário.
    - Ignora mídias já verificadas anteriormente que não pertencem a nenhum universo (universe_key == 'none').
    - Consulta Wikidata (2-step SPARQL) + TMDB Collections/Keywords apenas para mídias inéditas.
    - Funde universos equivalentes (por sobreposição real de obras) e persiste em `user_universe_suggestions`.
    """
    # 1. Busca universos já seguidos pelo usuário e todos os seus itens (para exclusão imediata)
    followed_stmt = (
        select(Universe)
        .join(UserUniverse, UserUniverse.universe_id == Universe.id)
        .options(selectinload(Universe.items))
        .where(UserUniverse.user_id == user_id)
    )
    followed_res = await db.execute(followed_stmt)
    followed_universes = followed_res.scalars().all()

    followed_keys: Set[str] = set()
    followed_media_pairs: Set[Tuple[int, str]] = set()
    followed_pairs_by_universe: List[Tuple[Universe, Set[Tuple[int, str]]]] = []
    for u in followed_universes:
        followed_keys.add(u.universe_key)
        followed_keys.update(u.source_keys or [])
        u_pairs: Set[Tuple[int, str]] = set()
        for item in u.items:
            pair = (item.tmdb_id, item.media_type)
            followed_media_pairs.add(pair)
            u_pairs.add(pair)
        followed_pairs_by_universe.append((u, u_pairs))

    # Remove sugestões residuais de universos (ou aliases) que o usuário já segue
    if followed_keys:
        await db.execute(
            delete(UserUniverseSuggestion).where(
                UserUniverseSuggestion.user_id == user_id,
                UserUniverseSuggestion.universe_key.in_(followed_keys),
            )
        )
        await db.commit()

    # 2. Busca todos os itens na lista do usuário (Filmes e Séries)
    list_stmt = (
        select(UserListItem, Media)
        .join(Media, UserListItem.media_id == Media.id)
        .where(UserListItem.user_id == user_id)
    )
    list_res = await db.execute(list_stmt)
    rows = list_res.all()

    if not rows:
        return []

    user_media_by_pair: Dict[Tuple[int, str], Media] = {}
    known_movie_collections: Dict[int, Optional[int]] = {}
    unresolved_media: List[Media] = []

    for _, media in rows:
        pair = (media.tmdb_id, media.media_type)
        user_media_by_pair[pair] = media
        if media.media_type == "movie":
            known_movie_collections[media.tmdb_id] = media.collection_tmdb_id

        # REGRA DE OURO: Se a mídia já faz parte de um universo seguido pelo usuário, ignora completamente!
        if pair in followed_media_pairs or (media.universe_key and media.universe_key in followed_keys):
            continue

        # Se ainda não foi escaneada (universe_key is None), separa para descoberta
        if media.universe_key is None:
            unresolved_media.append(media)

    # 3. Fast-Path Local DB: Verifica se alguma mídia pendente já pertence a um UniverseItem ou Universe conhecido no banco
    if unresolved_media:
        unresolved_tmdbs = [m.tmdb_id for m in unresolved_media]
        existing_ui_stmt = (
            select(UniverseItem, Universe)
            .join(Universe, UniverseItem.universe_id == Universe.id)
            .where(UniverseItem.tmdb_id.in_(unresolved_tmdbs))
        )
        existing_ui_res = await db.execute(existing_ui_stmt)
        pair_to_existing_universe: Dict[Tuple[int, str], Tuple[str, str]] = {}
        for ui, univ in existing_ui_res.all():
            pair_to_existing_universe[(ui.tmdb_id, ui.media_type)] = (univ.universe_key, univ.name)

        still_unresolved: List[Media] = []
        db_updated_fast = False
        for m in unresolved_media:
            pair = (m.tmdb_id, m.media_type)
            if pair in pair_to_existing_universe:
                ukey, uname = pair_to_existing_universe[pair]
                m.universe_key = ukey
                m.universe_name = uname
                db_updated_fast = True
            else:
                still_unresolved.append(m)

        if db_updated_fast:
            await db.commit()
        unresolved_media = still_unresolved

    # 4. Descoberta Híbrida Externa (Wikidata 2-Step + TMDB Keywords) apenas para itens ainda não resolvidos
    discovered_candidates: Dict[str, Dict[str, Any]] = {}

    existing_sugg_stmt = select(UserUniverseSuggestion).where(UserUniverseSuggestion.user_id == user_id)
    existing_sugg_res = await db.execute(existing_sugg_stmt)
    existing_suggestions_by_key: Dict[str, UserUniverseSuggestion] = {
        s.universe_key: s for s in existing_sugg_res.scalars().all()
    }

    existing_alias_to_canonical: Dict[str, str] = {}
    for k, s in existing_suggestions_by_key.items():
        existing_alias_to_canonical[k] = k
        for alias_k in (s.source_keys or []):
            existing_alias_to_canonical[alias_k] = k

    items_needing_discovery: Dict[Tuple[int, str], Media] = {
        (m.tmdb_id, m.media_type): m for m in unresolved_media
    }
    for pair, m in user_media_by_pair.items():
        if pair in followed_media_pairs:
            continue
        if m.universe_key and m.universe_key not in ("none", "") and m.universe_key not in followed_keys:
            if m.universe_key not in existing_alias_to_canonical:
                items_needing_discovery[pair] = m

    if items_needing_discovery:
        movie_seed_ids = [tid for (tid, mtype) in items_needing_discovery.keys() if mtype == "movie"]
        tv_seed_ids = [tid for (tid, mtype) in items_needing_discovery.keys() if mtype == "tv"]

        wd_candidates = await _wikidata_step1_find_candidates(movie_seed_ids, tv_seed_ids)
        for fkey in list(followed_keys):
            wd_candidates.pop(fkey, None)

        if wd_candidates:
            await _wikidata_step2_expand_members(wd_candidates)
            discovered_candidates.update(wd_candidates)

        matched_seed_pairs: Set[Tuple[int, str]] = set()
        for cand in discovered_candidates.values():
            matched_seed_pairs.update(cand.get("seed_matches", set()))
            matched_seed_pairs.update(cand.get("members", {}).keys())

        unmatched_by_wd = [
            m for pair, m in items_needing_discovery.items() if pair not in matched_seed_pairs and m.universe_key is None
        ]
        if unmatched_by_wd:
            kw_candidates = await _discover_keyword_universes_for_items(unmatched_by_wd, followed_keys)
            discovered_candidates.update(kw_candidates)

        # Pré-filtro de candidatos
        valid_candidates: Dict[str, Dict[str, Any]] = {}
        for ukey, cand in discovered_candidates.items():
            if ukey in followed_keys:
                continue
            if _is_candidate_potentially_valid(cand, known_movie_collections):
                valid_candidates[ukey] = cand

        # Fusão inteligente por sobreposição real de obras (une Wikidata + TMDB Keywords / sub-universos)
        valid_candidates = _merge_overlapping_candidates(valid_candidates, followed_keys=followed_keys)

        # Desambiguação Hierárquica: Prioriza o universo mais amplo para cada obra
        seed_to_best_ukey: Dict[Tuple[int, str], str] = {}
        for ukey, cand in sorted(
            valid_candidates.items(),
            key=lambda kv: len(kv[1].get("members", {})),
            reverse=True,
        ):
            all_cand_pairs = set(cand.get("members", {}).keys()) | set(cand.get("seed_matches", set()))
            for pair in all_cand_pairs:
                if pair in user_media_by_pair and pair not in seed_to_best_ukey:
                    seed_to_best_ukey[pair] = ukey

        chosen_ukeys = set(seed_to_best_ukey.values())
        discovered_candidates = {ukey: cand for ukey, cand in valid_candidates.items() if ukey in chosen_ukeys}

    # 5. Enriquece e valida definitivamente cada candidato antes de salvar em UserUniverseSuggestion
    confirmed_ukeys: Set[str] = set()
    for ukey, cand in discovered_candidates.items():
        cand_source_keys = _normalize_source_keys(ukey, cand.get("source_keys"))
        cand_all_keys = {ukey, *cand_source_keys}
        if cand_all_keys & followed_keys:
            continue

        canonical_existing_key = existing_alias_to_canonical.get(ukey, ukey)
        existing_sugg = existing_suggestions_by_key.get(canonical_existing_key)
        if existing_sugg and existing_sugg.is_dismissed:
            confirmed_ukeys.add(canonical_existing_key)
            continue

        enriched_members = await _enrich_universe_members(db, cand)
        if not _is_enriched_universe_valid(enriched_members, cand.get("child_collection_ids", set())):
            continue

        # Verifica pós-enriquecimento se o candidato coincide com algum universo já seguido
        enriched_pairs = _extract_member_pairs_from_json(enriched_members)
        matched_followed_u: Optional[Universe] = None
        for fu, fu_pairs in followed_pairs_by_universe:
            if _should_merge_universes(fu_pairs, fu.universe_key, enriched_pairs, ukey):
                matched_followed_u = fu
                break

        if matched_followed_u is not None:
            await _absorb_members_into_followed_universe(
                db,
                matched_followed_u,
                ukey,
                cand_source_keys,
                enriched_members,
            )
            followed_keys.add(ukey)
            followed_keys.update(cand_source_keys)
            confirmed_ukeys.add(matched_followed_u.universe_key)
            continue

        confirmed_ukeys.add(ukey)

        matched_titles: List[str] = []
        for member in enriched_members:
            mpair = (member["tmdb_id"], member["media_type"])
            if mpair in user_media_by_pair:
                u_media = user_media_by_pair[mpair]
                if u_media.title not in matched_titles:
                    matched_titles.append(u_media.title)
                u_media.universe_key = existing_sugg.universe_key if existing_sugg else ukey
                u_media.universe_name = existing_sugg.name if existing_sugg else cand["name"]

        if not matched_titles:
            continue

        movies_count = sum(1 for x in enriched_members if x.get("media_type") == "movie")
        tv_count = sum(1 for x in enriched_members if x.get("media_type") == "tv")
        collections_count = max(
            len(cand.get("child_collection_ids", set())),
            _count_distinct_collections(enriched_members),
        )

        poster_path = next((x.get("poster_path") for x in enriched_members if x.get("poster_path")), None)
        backdrop_path = next((x.get("backdrop_path") for x in enriched_members if x.get("backdrop_path")), None)
        overview = cand.get("overview") or f"Universo expandido contendo {movies_count} filme(s) e {tv_count} série(s)."

        if existing_sugg:
            merged_members = _merge_enriched_member_lists(
                [existing_sugg.discovered_members or [], enriched_members]
            )
            merged_source_keys = _normalize_source_keys(
                existing_sugg.universe_key,
                [*(existing_sugg.source_keys or []), ukey, *cand_source_keys],
            )
            existing_sugg.source_keys = merged_source_keys
            if existing_sugg.universe_key == ukey:
                existing_sugg.name = cand["name"]
                existing_sugg.overview = overview
            elif not existing_sugg.overview:
                existing_sugg.overview = overview
            existing_sugg.poster_path = poster_path or existing_sugg.poster_path
            existing_sugg.backdrop_path = backdrop_path or existing_sugg.backdrop_path
            existing_sugg.total_items = len(merged_members)
            existing_sugg.movies_count = sum(1 for x in merged_members if x.get("media_type", "movie") == "movie")
            existing_sugg.tv_count = sum(1 for x in merged_members if x.get("media_type") == "tv")
            existing_sugg.collections_count = max(
                existing_sugg.collections_count or 0,
                collections_count,
                _count_distinct_collections(merged_members),
            )
            existing_sugg.discovered_members = merged_members
            recalc_titles: List[str] = []
            for member in merged_members:
                mpair = (member["tmdb_id"], member.get("media_type", "movie"))
                if mpair in user_media_by_pair:
                    t = user_media_by_pair[mpair].title
                    if t not in recalc_titles:
                        recalc_titles.append(t)
            existing_sugg.items_in_list = len(recalc_titles)
            existing_sugg.matched_titles = recalc_titles
        else:
            new_sugg = UserUniverseSuggestion(
                user_id=user_id,
                universe_key=ukey,
                source_keys=cand_source_keys,
                name=cand["name"],
                overview=overview,
                poster_path=poster_path,
                backdrop_path=backdrop_path,
                total_items=len(enriched_members),
                movies_count=movies_count,
                tv_count=tv_count,
                collections_count=collections_count,
                items_in_list=len(matched_titles),
                matched_titles=matched_titles,
                discovered_members=enriched_members,
                is_dismissed=False,
            )
            db.add(new_sugg)
            existing_suggestions_by_key[ukey] = new_sugg

    # 5.5 Consolidação global das sugestões do usuário (mescla duplicatas de scans diferentes e contra seguidos)
    await _consolidate_user_universe_suggestions(
        db,
        user_id,
        existing_suggestions_by_key,
        followed_universes,
        user_media_by_pair=user_media_by_pair,
    )

    # Marca mídias verificadas que não pertencem a nenhum universo válido como 'none' para não re-escanear
    active_confirmed_keys = confirmed_ukeys | set(existing_suggestions_by_key.keys()) | followed_keys
    for m in unresolved_media:
        if m.universe_key is None or m.universe_key not in active_confirmed_keys:
            m.universe_key = "none"
            m.universe_name = None

    # 6. Atualiza contagens e avalia reativação inteligente condicional de sugestões dispensadas
    for ukey, sugg in list(existing_suggestions_by_key.items()):
        if ukey in followed_keys:
            await db.delete(sugg)
            continue

        if sugg.discovered_members:
            matched_titles = []
            current_media_types: Set[str] = set()
            current_col_ids: Set[int] = set()
            for member in sugg.discovered_members:
                mpair = (member.get("tmdb_id"), member.get("media_type", "movie"))
                if mpair in user_media_by_pair:
                    media_obj = user_media_by_pair[mpair]
                    t = media_obj.title
                    if t not in matched_titles:
                        matched_titles.append(t)
                    current_media_types.add(media_obj.media_type)
                    if media_obj.collection_tmdb_id:
                        current_col_ids.add(media_obj.collection_tmdb_id)

            if not matched_titles:
                await db.delete(sugg)
                continue

            sugg.items_in_list = len(matched_titles)
            sugg.matched_titles = matched_titles

            if sugg.is_dismissed:
                # Regras de Reativação Inteligente Condicional por Afinidade Aumentada:
                dismissed_count = sugg.dismissed_items_count or 0
                dismissed_types = set(sugg.dismissed_media_types or [])
                dismissed_cols = set(sugg.dismissed_collection_ids or [])
                current_count = len(matched_titles)

                reactivated = False
                reactivation_msg = None

                # Regra 1: Expansão Transmídia (ex: tinha apenas filmes e agora começou a assistir séries ou especiais)
                new_types = current_media_types - dismissed_types
                if dismissed_types and new_types:
                    reactivated = True
                    if "tv" in new_types:
                        reactivation_msg = "Notamos que você começou a acompanhar séries deste universo."
                    elif "movie" in new_types:
                        reactivation_msg = "Notamos que você começou a acompanhar filmes deste universo."
                    else:
                        reactivation_msg = "Você começou a explorar novos formatos deste universo."

                # Regra 2: Expansão de Saga / Coleção conectada (ex: assistiu filmes de uma nova coleção dentro do universo)
                elif dismissed_cols and (current_col_ids - dismissed_cols):
                    reactivated = True
                    reactivation_msg = "Você começou a assistir outras sagas conectadas a este universo."

                # Regra 3: Transição Baixa Afinidade -> Alta Afinidade (tinha <= 1 obra e agora tem >= 3)
                elif dismissed_count <= 1 and current_count >= 3:
                    reactivated = True
                    reactivation_msg = "Seu engajamento com esta franquia aumentou."

                if reactivated:
                    sugg.is_dismissed = False
                    sugg.is_reactivated = True
                    sugg.reactivation_reason = reactivation_msg
                    logger.info(
                        f"✨ [Universes] Sugestão '{sugg.name}' reativada para o usuário {user_id}: {reactivation_msg}"
                    )

    await db.commit()

    # 7. Retorna sugestões ativas ordenadas por relevância (mais itens na lista do usuário primeiro)
    final_stmt = (
        select(UserUniverseSuggestion)
        .where(
            UserUniverseSuggestion.user_id == user_id,
            UserUniverseSuggestion.is_dismissed == False,
        )
        .order_by(
            UserUniverseSuggestion.items_in_list.desc(),
            UserUniverseSuggestion.total_items.desc(),
        )
        .limit(limit)
    )
    final_res = await db.execute(final_stmt)
    suggestions = final_res.scalars().all()
    return [UniverseSuggestionResponse.model_validate(s) for s in suggestions]


async def get_cached_user_universe_suggestions(
    db: AsyncSession,
    user_id: UUID,
    limit: int = 20,
) -> List[UniverseSuggestionResponse]:
    """
    Retorna imediatamente as sugestões já cacheadas em `user_universe_suggestions`.
    Consolida localmente eventuais duplicatas (sem chamadas HTTP externas) e,
    se não houver sugestões cacheadas mas o usuário possuir itens ainda não escaneados,
    agenda um scan em background.
    """
    followed_stmt = (
        select(Universe)
        .join(UserUniverse, UserUniverse.universe_id == Universe.id)
        .options(selectinload(Universe.items))
        .where(UserUniverse.user_id == user_id)
    )
    followed_res = await db.execute(followed_stmt)
    followed_universes = followed_res.scalars().all()
    followed_keys: Set[str] = set()
    for u in followed_universes:
        followed_keys.add(u.universe_key)
        followed_keys.update(u.source_keys or [])

    all_sugg_stmt = select(UserUniverseSuggestion).where(
        UserUniverseSuggestion.user_id == user_id,
        UserUniverseSuggestion.is_dismissed == False,
    )
    all_sugg_res = await db.execute(all_sugg_stmt)
    all_suggs = all_sugg_res.scalars().all()

    if all_suggs:
        sugg_map = {s.universe_key: s for s in all_suggs}
        # Consolidação rápida em memória (0 chamadas externas) para auto-healing imediato de duplicatas
        changed = await _consolidate_user_universe_suggestions(
            db,
            user_id,
            sugg_map,
            followed_universes,
            user_media_by_pair=None,
        )
        if changed:
            # Recalcula matched_titles/items_in_list com base na lista real do usuário para as sugestões afetadas
            user_list_res = await db.execute(
                select(Media.tmdb_id, Media.media_type, Media.title)
                .join(UserListItem, UserListItem.media_id == Media.id)
                .where(UserListItem.user_id == user_id)
            )
            user_title_by_pair = {(r[0], r[1]): r[2] for r in user_list_res.all()}
            for s in sugg_map.values():
                if s.discovered_members:
                    matched_titles = []
                    for m in s.discovered_members:
                        pair = (m.get("tmdb_id"), m.get("media_type", "movie"))
                        if pair in user_title_by_pair:
                            t = user_title_by_pair[pair]
                            if t not in matched_titles:
                                matched_titles.append(t)
                    s.matched_titles = matched_titles
                    s.items_in_list = len(matched_titles)
            await db.commit()

        active_suggs = [
            s for s in sugg_map.values()
            if not s.is_dismissed and s.universe_key not in followed_keys
        ]
        active_suggs.sort(key=lambda s: (s.items_in_list or 0, s.total_items or 0), reverse=True)
        suggestions = active_suggs[:limit]
    else:
        suggestions = []

    if not suggestions:
        # Verifica se há itens com universe_key == None para agendar scan em background
        unscanned_stmt = (
            select(Media.id)
            .join(UserListItem, UserListItem.media_id == Media.id)
            .where(UserListItem.user_id == user_id, Media.universe_key.is_(None))
            .limit(1)
        )
        unscanned_res = await db.execute(unscanned_stmt)
        if unscanned_res.scalars().first() is not None:
            schedule_universe_suggestions_scan(str(user_id), debounce_seconds=0.5)

    return [UniverseSuggestionResponse.model_validate(s) for s in suggestions]


async def dismiss_universe_suggestion(db: AsyncSession, user_id: UUID, universe_key: str) -> bool:
    """
    Marca uma sugestão de universo como dispensada (is_dismissed = True) para o usuário,
    registrando snapshot do estado de consumo (quantidade, tipos de mídia e coleções)
    para viabilizar a reativação inteligente condicional por afinidade aumentada.
    """
    stmt = select(UserUniverseSuggestion).where(
        UserUniverseSuggestion.user_id == user_id,
        UserUniverseSuggestion.universe_key == universe_key,
    )
    res = await db.execute(stmt)
    sugg = res.scalars().first()
    if not sugg:
        return False

    sugg.is_dismissed = True
    sugg.is_reactivated = False
    sugg.reactivation_reason = None
    sugg.dismissed_items_count = sugg.items_in_list

    # Snapshot dos tipos de mídia e coleções presentes na lista do usuário no momento da dispensa
    media_types: Set[str] = set()
    col_ids: Set[int] = set()
    if sugg.discovered_members:
        member_pairs = {(m.get("tmdb_id"), m.get("media_type", "movie")) for m in sugg.discovered_members}
        user_items_stmt = (
            select(Media)
            .join(UserListItem, UserListItem.media_id == Media.id)
            .where(UserListItem.user_id == user_id)
        )
        ui_res = await db.execute(user_items_stmt)
        for m in ui_res.scalars().all():
            if (m.tmdb_id, m.media_type) in member_pairs:
                media_types.add(m.media_type)
                if m.collection_tmdb_id:
                    col_ids.add(m.collection_tmdb_id)

    sugg.dismissed_media_types = list(media_types)
    sugg.dismissed_collection_ids = list(col_ids)
    await db.commit()
    return True


async def restore_universe_suggestion(db: AsyncSession, user_id: UUID, universe_key: str) -> bool:
    """Restaura uma sugestão dispensada, tornando-a ativa novamente para o usuário."""
    stmt = select(UserUniverseSuggestion).where(
        UserUniverseSuggestion.user_id == user_id,
        UserUniverseSuggestion.universe_key == universe_key,
    )
    res = await db.execute(stmt)
    sugg = res.scalars().first()
    if not sugg:
        return False

    sugg.is_dismissed = False
    sugg.is_reactivated = False
    sugg.reactivation_reason = None
    sugg.dismissed_items_count = 0
    sugg.dismissed_media_types = []
    sugg.dismissed_collection_ids = []
    await db.commit()
    return True


async def get_dismissed_universe_suggestions(
    db: AsyncSession,
    user_id: UUID,
    limit: int = 50,
) -> List[UniverseSuggestionResponse]:
    """Retorna as sugestões que foram dispensadas pelo usuário."""
    followed_stmt = (
        select(Universe.universe_key, Universe.source_keys)
        .join(UserUniverse, UserUniverse.universe_id == Universe.id)
        .where(UserUniverse.user_id == user_id)
    )
    followed_res = await db.execute(followed_stmt)
    followed_keys: Set[str] = set()
    for ukey, skeys in followed_res.all():
        if ukey:
            followed_keys.add(ukey)
        for sk in (skeys or []):
            followed_keys.add(sk)

    stmt = (
        select(UserUniverseSuggestion)
        .where(
            UserUniverseSuggestion.user_id == user_id,
            UserUniverseSuggestion.is_dismissed == True,
        )
        .order_by(UserUniverseSuggestion.updated_at.desc())
        .limit(limit)
    )
    res = await db.execute(stmt)
    suggestions = [s for s in res.scalars().all() if s.universe_key not in followed_keys]
    return [UniverseSuggestionResponse.model_validate(s) for s in suggestions]


async def _discover_single_universe_by_key(
    db: AsyncSession,
    universe_key: str,
    source_keys: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Busca a estrutura completa de um universo pela sua `universe_key` principal
    (QID do Wikidata como 'Q19785923' ou Keyword do TMDB como 'kw:180547') e,
    quando houver `source_keys` absorvidas, consulta também essas fontes secundárias
    fazendo a união dos membros antes do enriquecimento.
    """
    secondary_keys = _normalize_source_keys(universe_key, source_keys)
    cand: Dict[str, Any] = {
        "universe_key": universe_key,
        "source_keys": secondary_keys,
        "name": universe_key,
        "overview": None,
        "members": {},
        "child_collection_ids": set(),
    }

    if universe_key.startswith("kw:"):
        kw_id_str = universe_key.split(":", 1)[1]
        if not kw_id_str.isdigit():
            return None
        kw_id = int(kw_id_str)
        kw_info, kw_members = await asyncio.gather(
            tmdb_client._get(f"/keyword/{kw_id}"),
            _fetch_tmdb_keyword_members(kw_id, max_pages=4),
        )
        kw_name = (kw_info.get("name") if kw_info else f"Universo #{kw_id}").title()
        cand["name"] = kw_name
        cand["overview"] = f"Universo compartilhado '{kw_name}' no TMDB."
        cand["members"].update(kw_members)
    elif universe_key.startswith("Q"):
        label_query = f"""
        SELECT ?universeLabel ?universeDesc WHERE {{
          BIND(wd:{universe_key} AS ?universe)
          SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "pt-br,pt,en".
            ?universe rdfs:label ?universeLabel .
            ?universe schema:description ?universeDesc .
          }}
        }}
        """
        label_task = _sparql_post(label_query, timeout=12.0)
        expand_task = _wikidata_step2_expand_members({universe_key: cand})
        label_rows, _ = await asyncio.gather(label_task, expand_task)

        if label_rows:
            cand["name"] = label_rows[0].get("universeLabel", {}).get("value", universe_key)
            desc_obj = label_rows[0].get("universeDesc") or {}
            desc_val = (desc_obj.get("value") or "").strip()
            desc_lang = (desc_obj.get("xml:lang") or "").strip().lower()
            if desc_val and desc_lang and not desc_lang.startswith("pt"):
                desc_val = await translate_to_pt_br(desc_val, source_lang=desc_lang)
            cand["overview"] = desc_val or None
    else:
        return None

    # Consulta fontes secundárias (aliases absorvidos) e faz a união dos membros
    secondary_wd_cands: Dict[str, Dict[str, Any]] = {}
    secondary_kw_ids: List[int] = []
    for skey in secondary_keys:
        if skey.startswith("Q"):
            secondary_wd_cands[skey] = {"universe_key": skey, "members": {}}
        elif skey.startswith("kw:"):
            kw_str = skey.split(":", 1)[1]
            if kw_str.isdigit():
                secondary_kw_ids.append(int(kw_str))

    secondary_tasks: List[Any] = []
    if secondary_wd_cands:
        secondary_tasks.append(_wikidata_step2_expand_members(secondary_wd_cands))
    for kw_id in secondary_kw_ids:
        secondary_tasks.append(_fetch_tmdb_keyword_members(kw_id, max_pages=4))

    if secondary_tasks:
        sec_results = await asyncio.gather(*secondary_tasks, return_exceptions=True)
        for wd_cand in secondary_wd_cands.values():
            for pair, info in (wd_cand.get("members") or {}).items():
                cand["members"][pair] = _merge_member_entry(cand["members"].get(pair), info)
        for res_item in sec_results:
            if isinstance(res_item, dict):
                for pair, info in res_item.items():
                    cand["members"][pair] = _merge_member_entry(cand["members"].get(pair), info)

    enriched = await _enrich_universe_members(db, cand)
    cand["enriched_members"] = enriched
    return cand


async def get_or_create_universe(
    db: AsyncSession,
    universe_key: str,
    user_id: Optional[UUID] = None,
) -> Optional[Universe]:
    """
    Obtém um Universo do banco (pela chave principal ou por alias em `source_keys`)
    ou o cria a partir do cache de sugestões (`UserUniverseSuggestion`)
    ou consultando Wikidata + TMDB.
    """
    stmt = (
        select(Universe)
        .options(selectinload(Universe.items))
        .where(Universe.universe_key == universe_key)
    )
    res = await db.execute(stmt)
    universe = res.scalars().first()
    if universe:
        return universe

    # Resolve se `universe_key` já foi absorvida como alias em `source_keys` de algum Universe canônico
    all_u_res = await db.execute(select(Universe).options(selectinload(Universe.items)))
    for existing_u in all_u_res.scalars().all():
        if universe_key in (existing_u.source_keys or []):
            return existing_u

    # Tenta reaproveitar os dados já descobertos em UserUniverseSuggestion (instantâneo!)
    sugg_stmt = select(UserUniverseSuggestion).where(UserUniverseSuggestion.universe_key == universe_key)
    if user_id:
        sugg_stmt = sugg_stmt.where(UserUniverseSuggestion.user_id == user_id)
    sugg_res = await db.execute(sugg_stmt)
    sugg = sugg_res.scalars().first()

    if not sugg and user_id:
        # Tenta sugestão de qualquer usuário para o mesmo universe_key
        any_sugg_res = await db.execute(
            select(UserUniverseSuggestion).where(UserUniverseSuggestion.universe_key == universe_key).limit(1)
        )
        sugg = any_sugg_res.scalars().first()

    if sugg and sugg.discovered_members:
        name = sugg.name
        overview = sugg.overview
        poster_path = sugg.poster_path
        backdrop_path = sugg.backdrop_path
        members = sugg.discovered_members
        collections_count = sugg.collections_count
        source_keys = _normalize_source_keys(universe_key, sugg.source_keys)
    else:
        cand = await _discover_single_universe_by_key(db, universe_key)
        if not cand or not cand.get("enriched_members"):
            return None
        members = cand["enriched_members"]
        name = cand["name"]
        movies_c = sum(1 for x in members if x.get("media_type") == "movie")
        tv_c = sum(1 for x in members if x.get("media_type") == "tv")
        overview = cand.get("overview") or f"Universo expandido contendo {movies_c} filme(s) e {tv_c} série(s)."
        poster_path = next((x.get("poster_path") for x in members if x.get("poster_path")), None)
        backdrop_path = next((x.get("backdrop_path") for x in members if x.get("backdrop_path")), None)
        collections_count = len(cand.get("child_collection_ids", set()))
        source_keys = _normalize_source_keys(universe_key, cand.get("source_keys"))

    movies_count = sum(1 for x in members if x.get("media_type", "movie") == "movie")
    tv_count = sum(1 for x in members if x.get("media_type") == "tv")

    universe = Universe(
        universe_key=universe_key,
        source_keys=source_keys,
        name=name,
        overview=overview,
        poster_path=poster_path,
        backdrop_path=backdrop_path,
        items_count=len(members),
        movies_count=movies_count,
        tv_count=tv_count,
        collections_count=collections_count,
    )
    db.add(universe)
    await db.flush()

    # Vincula Media local se já existir
    all_tmdbs = [m["tmdb_id"] for m in members if m.get("tmdb_id")]
    local_media_map: Dict[Tuple[int, str], Media] = {}
    if all_tmdbs:
        m_res = await db.execute(select(Media).where(Media.tmdb_id.in_(all_tmdbs)))
        for m in m_res.scalars().all():
            local_media_map[(m.tmdb_id, m.media_type)] = m
            m.universe_key = universe_key
            m.universe_name = name

    seen_pairs: Set[Tuple[int, str]] = set()
    for m in members:
        tid = m.get("tmdb_id")
        mtype = m.get("media_type", "movie")
        if not tid or (tid, mtype) in seen_pairs:
            continue
        if _check_is_recap(m.get("title", ""), m.get("overview"), m.get("keywords_list") or m.get("keywords"), mtype):
            continue
        seen_pairs.add((tid, mtype))
        local_m = local_media_map.get((tid, mtype))

        u_item = UniverseItem(
            universe_id=universe.id,
            media_id=local_m.id if local_m else None,
            tmdb_id=tid,
            media_type=mtype,
            title=m.get("title") or (local_m.title if local_m else f"Item #{tid}"),
            release_date=m.get("release_date") or (local_m.release_date if local_m else None),
            poster_path=m.get("poster_path") or (local_m.poster_path if local_m else None),
            backdrop_path=m.get("backdrop_path") or (local_m.backdrop_path if local_m else None),
            collection_name=m.get("collection_name") or (local_m.collection_name if local_m else None),
        )
        db.add(u_item)

    await db.commit()

    stmt = (
        select(Universe)
        .options(selectinload(Universe.items))
        .where(Universe.id == universe.id)
    )
    res = await db.execute(stmt)
    return res.scalars().first()


async def _background_enrich_universe_follow(user_id_str: str, media_ids: List[UUID]):
    """
    Executa em background para não bloquear a resposta HTTP do usuário:
    1. Enriquece metadados TMDB detalhados (elenco, diretores, gêneros, keywords, runtime, collections).
    2. Gera embeddings semânticos para o sistema de recomendação.
    3. Atualiza o perfil de preferências do usuário.
    4. Garante cache de imagem local.
    5. Dispara recomputação de sugestões.
    """
    try:
        from app.core.database import AsyncSessionLocal
        from app.details.services import fetch_movie_details, fetch_tv_details
        from app.recommendation.embeddings import build_text_for_embedding, generate_embedding
        from app.recommendation.profile_manager import update_user_profile
        from app.images.router import ensure_image_cached
        from app.media_collections.services import schedule_collection_suggestions_scan

        async with AsyncSessionLocal() as session:
            stmt = select(Media).where(Media.id.in_(media_ids))
            res = await session.execute(stmt)
            medias = res.scalars().all()

            for media in medias:
                try:
                    # Enriquecimento de metadados se faltar dados
                    if not media.genres or not media.directors or not media.main_cast or not media.runtime:
                        if media.media_type == "movie":
                            tmdb_data = await fetch_movie_details(media.tmdb_id)
                            if tmdb_data:
                                media.runtime = tmdb_data.runtime or media.runtime or 0
                                media.original_language = tmdb_data.original_language or media.original_language
                                if not media.genres and tmdb_data.genres:
                                    media.genres = [{"id": g.id, "name": g.name} for g in tmdb_data.genres]
                                if not media.directors and tmdb_data.credits and tmdb_data.credits.crew:
                                    media.directors = [
                                        c.name
                                        for c in tmdb_data.credits.crew
                                        if getattr(c, "job", "") in ("Director", "Creator", "Series Director")
                                    ]
                                if not media.main_cast and tmdb_data.credits and tmdb_data.credits.cast:
                                    media.main_cast = [c.name for c in tmdb_data.credits.cast]
                                if hasattr(tmdb_data, "keywords") and tmdb_data.keywords:
                                    media.keywords = tmdb_data.keywords
                                media.cast_ids = tmdb_data.cast_ids or []
                                media.crew_ids = tmdb_data.crew_ids or []
                                if hasattr(tmdb_data, "belongs_to_collection") and tmdb_data.belongs_to_collection:
                                    media.collection_tmdb_id = tmdb_data.belongs_to_collection.id
                                    media.collection_name = tmdb_data.belongs_to_collection.name
                        else:
                            tmdb_data = await fetch_tv_details(media.tmdb_id)
                            if tmdb_data:
                                if tmdb_data.episode_run_time and len(tmdb_data.episode_run_time) > 0:
                                    media.runtime = tmdb_data.episode_run_time[0]
                                else:
                                    media.runtime = 45
                                media.original_language = tmdb_data.original_language or media.original_language
                                if not media.genres and tmdb_data.genres:
                                    media.genres = [{"id": g.id, "name": g.name} for g in tmdb_data.genres]
                                if not media.directors and tmdb_data.credits and tmdb_data.credits.crew:
                                    media.directors = [
                                        c.name
                                        for c in tmdb_data.credits.crew
                                        if getattr(c, "job", "") in ("Director", "Creator", "Series Director")
                                    ]
                                if not media.main_cast and tmdb_data.credits and tmdb_data.credits.cast:
                                    media.main_cast = [c.name for c in tmdb_data.credits.cast]
                                if hasattr(tmdb_data, "keywords") and tmdb_data.keywords:
                                    media.keywords = tmdb_data.keywords
                                media.cast_ids = tmdb_data.cast_ids or []
                                media.crew_ids = tmdb_data.crew_ids or []

                    # Embedding semântico em background se ainda não possuir
                    if not media.embedding:
                        dummy_dict = {
                            "title": media.title,
                            "media_type": media.media_type,
                            "genres": media.genres,
                            "crew": media.directors,
                            "cast": media.main_cast,
                            "keywords": media.keywords,
                        }
                        emb_text = build_text_for_embedding(dummy_dict)
                        media.embedding = await generate_embedding(emb_text)

                    # Poster local cache
                    if media.poster_path:
                        asyncio.create_task(ensure_image_cached(media.poster_path, "w342"))

                except Exception as ex:
                    logger.warning(f"Erro ao enriquecer media {media.tmdb_id} em background: {ex}")

            await session.commit()

            # Atualiza perfil do usuário para recomendações
            try:
                user_uuid = UUID(user_id_str)
                for media in medias:
                    uli_res = await session.execute(
                        select(UserListItem).where(
                            UserListItem.user_id == user_uuid,
                            UserListItem.media_id == media.id,
                        )
                    )
                    uli = uli_res.scalars().first()
                    if uli:
                        await update_user_profile(session, user_id_str, uli, media)
            except Exception as ex:
                logger.warning(f"Erro ao atualizar perfil do usuário {user_id_str} em background: {ex}")

        # Agenda scans de coleções e universos com debounce
        schedule_collection_suggestions_scan(user_id_str, debounce_seconds=4)
        schedule_universe_suggestions_scan(user_id_str, debounce_seconds=4.0)

    except Exception as e:
        logger.error(f"Erro geral no background enrich do follow_universe para {user_id_str}: {e}")


async def follow_universe(
    db: AsyncSession,
    user_id: UUID,
    universe_key: str,
    auto_add: bool = True,
    ignored_items: Optional[List[Tuple[int, str]]] = None,
) -> Optional[UniverseFollowResponse]:
    """
    Segue um Universo e adiciona à lista do usuário apenas os filmes e séries selecionados
    (ignorando os desmarcados pelo usuário).
    """
    universe = await get_or_create_universe(db, universe_key, user_id=user_id)
    if not universe:
        return None

    # Verifica se já segue
    uc_stmt = select(UserUniverse).where(
        UserUniverse.user_id == user_id,
        UserUniverse.universe_id == universe.id,
    )
    uc_res = await db.execute(uc_stmt)
    user_universe = uc_res.scalars().first()

    if not user_universe:
        user_universe = UserUniverse(
            user_id=user_id,
            universe_id=universe.id,
            auto_add=auto_add,
        )
        db.add(user_universe)
    else:
        user_universe.auto_add = auto_add

    # Persiste os itens expressamente ignorados pelo usuário neste universo
    existing_ign_stmt = select(UserUniverseIgnoredItem).where(
        UserUniverseIgnoredItem.user_id == user_id,
        UserUniverseIgnoredItem.universe_id == universe.id,
    )
    existing_ign_res = await db.execute(existing_ign_stmt)
    pre_ignored = {(x.tmdb_id, x.media_type) for x in existing_ign_res.scalars().all()}

    ignored_set: Set[Tuple[int, str]] = set(ignored_items or []).union(pre_ignored)
    await db.execute(
        delete(UserUniverseIgnoredItem).where(
            UserUniverseIgnoredItem.user_id == user_id,
            UserUniverseIgnoredItem.universe_id == universe.id,
        )
    )
    for tid, mtype in ignored_set:
        db.add(
            UserUniverseIgnoredItem(
                user_id=user_id,
                universe_id=universe.id,
                tmdb_id=tid,
                media_type=mtype,
            )
        )

    # Remove da tabela de sugestões para este usuário (chave principal + aliases absorvidos)
    keys_to_remove = {universe_key, universe.universe_key, *(universe.source_keys or [])}
    await db.execute(
        delete(UserUniverseSuggestion).where(
            UserUniverseSuggestion.user_id == user_id,
            UserUniverseSuggestion.universe_key.in_(keys_to_remove),
        )
    )
    await db.commit()

    # Busca itens já na lista do usuário (escopo restrito aos itens deste universo O(K))
    universe_tmdb_ids = [it.tmdb_id for it in universe.items]
    existing_user_pairs: Dict[Tuple[int, str], Media] = {}
    if universe_tmdb_ids:
        list_stmt = (
            select(UserListItem, Media)
            .join(Media, UserListItem.media_id == Media.id)
            .where(
                UserListItem.user_id == user_id,
                Media.tmdb_id.in_(universe_tmdb_ids),
            )
        )
        list_res = await db.execute(list_stmt)
        existing_user_pairs = {
            (media.tmdb_id, media.media_type): media for _, media in list_res.all()
        }

    items_added = 0
    items_already_in_list = 0
    items_to_add: List[UniverseItem] = []

    sorted_items = sorted(universe.items, key=lambda x: x.release_date or "9999-12-31")
    for item in sorted_items:
        pair = (item.tmdb_id, item.media_type)
        if pair in existing_user_pairs:
            items_already_in_list += 1
            media_obj = existing_user_pairs[pair]
            if not item.media_id:
                item.media_id = media_obj.id
            if not media_obj.universe_key:
                media_obj.universe_key = universe.universe_key
                media_obj.universe_name = universe.name
            continue

        if pair in ignored_set:
            # Não auto-adiciona obras desmarcadas/ignoradas pelo usuário
            continue

        items_to_add.append(item)

    new_media_ids_to_enrich: List[UUID] = []
    if items_to_add:
        tmdbs = [it.tmdb_id for it in items_to_add]
        m_stmt = select(Media).where(Media.tmdb_id.in_(tmdbs))
        m_res = await db.execute(m_stmt)
        existing_medias: Dict[Tuple[int, str], Media] = {
            (m.tmdb_id, m.media_type): m for m in m_res.scalars().all()
        }

        new_medias: List[Media] = []
        for it in items_to_add:
            pair = (it.tmdb_id, it.media_type)
            media_obj = existing_medias.get(pair)
            if not media_obj:
                media_obj = Media(
                    tmdb_id=it.tmdb_id,
                    media_type=it.media_type,
                    title=it.title,
                    poster_path=it.poster_path,
                    backdrop_path=it.backdrop_path,
                    release_date=it.release_date,
                    collection_name=it.collection_name,
                    universe_key=universe.universe_key,
                    universe_name=universe.name,
                )
                db.add(media_obj)
                existing_medias[pair] = media_obj
                new_medias.append(media_obj)
            else:
                if not media_obj.universe_key:
                    media_obj.universe_key = universe.universe_key
                    media_obj.universe_name = universe.name

        if new_medias:
            await db.flush()

        for it in items_to_add:
            pair = (it.tmdb_id, it.media_type)
            media_obj = existing_medias[pair]
            if not it.media_id:
                it.media_id = media_obj.id

            is_rel = is_date_released(it.release_date)
            final_status = "plan_to_watch" if is_rel else "upcoming"

            user_list_item = UserListItem(
                user_id=user_id,
                media_id=media_obj.id,
                status=final_status,
            )
            db.add(user_list_item)
            items_added += 1
            new_media_ids_to_enrich.append(media_obj.id)

    await db.commit()
    invalidate_user_stats_cache(str(user_id))

    if new_media_ids_to_enrich:
        asyncio.create_task(_background_enrich_universe_follow(str(user_id), new_media_ids_to_enrich))

    return UniverseFollowResponse(
        success=True,
        message=f"Seguindo '{universe.name}'! {items_added} obras adicionadas à sua lista.",
        universe_id=universe.id,
        universe_key=universe.universe_key,
        name=universe.name,
        items_added=items_added,
        items_already_in_list=items_already_in_list,
        ignored_count=len(ignored_set),
    )


async def unfollow_universe(db: AsyncSession, user_id: UUID, universe_key: str) -> bool:
    """Deixa de seguir um universo e limpa itens ignorados."""
    stmt = (
        select(UserUniverse)
        .join(Universe, UserUniverse.universe_id == Universe.id)
        .where(
            UserUniverse.user_id == user_id,
            Universe.universe_key == universe_key,
        )
    )
    res = await db.execute(stmt)
    uc = res.scalars().first()
    if not uc:
        return False

    await db.execute(
        delete(UserUniverseIgnoredItem).where(
            UserUniverseIgnoredItem.user_id == user_id,
            UserUniverseIgnoredItem.universe_id == uc.universe_id,
        )
    )
    await db.delete(uc)
    await db.commit()
    return True


async def toggle_universe_item_ignored(
    db: AsyncSession,
    user_id: UUID,
    universe_key: str,
    tmdb_id: int,
    media_type: str,
    is_ignored: bool,
) -> bool:
    """Alterna se uma obra de um universo está ignorada para o usuário."""
    stmt = select(Universe).where(Universe.universe_key == universe_key)
    res = await db.execute(stmt)
    universe = res.scalars().first()
    if not universe:
        universe = await get_or_create_universe(db, universe_key, user_id=user_id)
        if not universe:
            return False

    ign_stmt = select(UserUniverseIgnoredItem).where(
        UserUniverseIgnoredItem.user_id == user_id,
        UserUniverseIgnoredItem.universe_id == universe.id,
        UserUniverseIgnoredItem.tmdb_id == tmdb_id,
        UserUniverseIgnoredItem.media_type == media_type,
    )
    ign_res = await db.execute(ign_stmt)
    existing = ign_res.scalars().first()

    if is_ignored and not existing:
        db.add(
            UserUniverseIgnoredItem(
                user_id=user_id,
                universe_id=universe.id,
                tmdb_id=tmdb_id,
                media_type=media_type,
            )
        )
        await db.commit()
        return True
    elif not is_ignored and existing:
        await db.delete(existing)
        await db.commit()
        return True
    return False


async def update_universe_settings(
    db: AsyncSession,
    user_id: UUID,
    universe_key: str,
    auto_add: bool,
) -> bool:
    """Atualiza configurações de acompanhamento do universo (como auto_add)."""
    stmt = (
        select(UserUniverse)
        .join(Universe, UserUniverse.universe_id == Universe.id)
        .where(
            UserUniverse.user_id == user_id,
            Universe.universe_key == universe_key,
        )
    )
    res = await db.execute(stmt)
    uu = res.scalars().first()
    if not uu:
        return False
    uu.auto_add = auto_add
    await db.commit()
    return True


async def get_universe_status(
    db: AsyncSession,
    user_id: UUID,
    universe_key: str,
) -> UniverseStatusResponse:
    """Retorna o status de acompanhamento de um universo pelo usuário."""
    stmt = (
        select(Universe)
        .options(selectinload(Universe.items))
        .where(Universe.universe_key == universe_key)
    )
    res = await db.execute(stmt)
    universe = res.scalars().first()

    if not universe:
        return UniverseStatusResponse(is_following=False)

    uc_stmt = select(UserUniverse).where(
        UserUniverse.user_id == user_id,
        UserUniverse.universe_id == universe.id,
    )
    uc_res = await db.execute(uc_stmt)
    uc = uc_res.scalars().first()

    ign_stmt = select(UserUniverseIgnoredItem).where(
        UserUniverseIgnoredItem.user_id == user_id,
        UserUniverseIgnoredItem.universe_id == universe.id,
    )
    ign_res = await db.execute(ign_stmt)
    ignored_count = len(ign_res.scalars().all())
    active_items = max(0, len(universe.items) - ignored_count)

    all_tmdbs = [item.tmdb_id for item in universe.items]
    watched_count = 0
    if all_tmdbs:
        w_stmt = (
            select(Media.tmdb_id, Media.media_type)
            .join(UserListItem, UserListItem.media_id == Media.id)
            .where(
                UserListItem.user_id == user_id,
                Media.tmdb_id.in_(all_tmdbs),
                or_(
                    UserListItem.status.in_(["completed", "watched"]),
                    UserListItem.rating > 0,
                ),
            )
        )
        w_res = await db.execute(w_stmt)
        watched_pairs = {(r[0], r[1]) for r in w_res.all()}
        watched_count = sum(1 for item in universe.items if (item.tmdb_id, item.media_type) in watched_pairs)

    return UniverseStatusResponse(
        is_following=uc is not None,
        auto_add=uc.auto_add if uc else True,
        total_items=len(universe.items),
        watched_items=watched_count,
        movies_count=universe.movies_count,
        tv_count=universe.tv_count,
        active_items=active_items,
        ignored_items_count=ignored_count,
        universe_id=universe.id,
    )


async def _build_universe_detail_response(
    db: AsyncSession,
    user_id: UUID,
    universe: Universe,
    created_at: datetime,
    user_items_by_pair: Optional[Dict[Tuple[int, str], Tuple[UserListItem, Media]]] = None,
    user_universe: Optional[UserUniverse] = None,
    ignored_pairs: Optional[Set[Tuple[int, str]]] = None,
) -> UserUniverseDetailResponse:
    """Monta o payload detalhado de um universo cruzando com status, notas e itens ignorados."""
    if user_items_by_pair is None:
        universe_tmdb_ids = [it.tmdb_id for it in universe.items]
        user_items_by_pair = {}
        if universe_tmdb_ids:
            list_stmt = (
                select(UserListItem, Media)
                .join(Media, UserListItem.media_id == Media.id)
                .where(
                    UserListItem.user_id == user_id,
                    Media.tmdb_id.in_(universe_tmdb_ids),
                )
            )
            list_res = await db.execute(list_stmt)
            user_items_by_pair = {
                (media.tmdb_id, media.media_type): (uli, media) for uli, media in list_res.all()
            }

    if ignored_pairs is None:
        ign_stmt = select(UserUniverseIgnoredItem).where(
            UserUniverseIgnoredItem.user_id == user_id,
            UserUniverseIgnoredItem.universe_id == universe.id,
        )
        ign_res = await db.execute(ign_stmt)
        ignored_pairs = {(x.tmdb_id, x.media_type) for x in ign_res.scalars().all()}

    if user_universe is not None:
        uu = user_universe
    else:
        uu_stmt = select(UserUniverse).where(
            UserUniverse.user_id == user_id,
            UserUniverse.universe_id == universe.id,
        )
        uu_res = await db.execute(uu_stmt)
        uu = uu_res.scalars().first()
    auto_add = uu.auto_add if uu else True

    watched_count = 0
    active_watched_count = 0
    ratings_sum = 0.0
    rated_count = 0
    items_resp: List[UniverseItemResponse] = []

    # Auto-healing assíncrono em background: Se algum item já persistido estiver com título em CJK/japonês, busca o título em inglês sem bloquear a resposta HTTP
    non_latin_items = [it for it in universe.items if _is_non_latin(it.title)]
    if non_latin_items:
        async def _bg_resolve_non_latin(u_id: UUID, items_to_fix: List[UniverseItem]):
            try:
                async with AsyncSessionLocal() as bg_db:
                    for u_item in items_to_fix:
                        try:
                            en_data = await tmdb_client._get(f"/{u_item.media_type}/{u_item.tmdb_id}", params={"language": "en-US"})
                            if en_data:
                                en_title = en_data.get("title") or en_data.get("name")
                                if en_title and not _is_non_latin(en_title):
                                    db_it_res = await bg_db.execute(select(UniverseItem).where(UniverseItem.id == u_item.id))
                                    db_it = db_it_res.scalars().first()
                                    if db_it:
                                        db_it.title = en_title
                                    m_stmt = select(Media).where(Media.tmdb_id == u_item.tmdb_id, Media.media_type == u_item.media_type)
                                    m_res = await bg_db.execute(m_stmt)
                                    m_obj = m_res.scalars().first()
                                    if m_obj and _is_non_latin(m_obj.title):
                                        m_obj.title = en_title
                        except Exception:
                            pass
                    await bg_db.commit()
            except Exception:
                pass

        asyncio.create_task(_bg_resolve_non_latin(universe.id, non_latin_items))

    canonical_items = [it for it in universe.items if not _check_is_recap(it.title, None, None, it.media_type)]
    sorted_items = sorted(canonical_items, key=lambda x: x.release_date or "9999-12-31")
    for item in sorted_items:
        pair = (item.tmdb_id, item.media_type)
        is_ignored = pair in ignored_pairs
        is_recap = False

        uli_tuple = user_items_by_pair.get(pair)
        status = uli_tuple[0].status if uli_tuple else None
        rating = uli_tuple[0].rating if uli_tuple else None
        media_id = uli_tuple[1].id if uli_tuple else item.media_id

        is_watched = status in ("completed", "watched") or (rating is not None and rating > 0)
        if is_watched:
            watched_count += 1
            if not is_ignored:
                active_watched_count += 1

        if rating is not None and rating > 0 and not is_ignored:
            ratings_sum += float(rating)
            rated_count += 1

        items_resp.append(
            UniverseItemResponse(
                id=item.id,
                tmdb_id=item.tmdb_id,
                media_type=item.media_type,
                title=item.title,
                release_date=item.release_date,
                poster_path=item.poster_path,
                backdrop_path=item.backdrop_path,
                collection_name=item.collection_name,
                media_id=media_id,
                status=status,
                rating=rating,
                is_ignored=is_ignored,
                is_recap=is_recap,
            )
        )

    total_items = len(sorted_items)
    ignored_items_count = len(ignored_pairs)
    active_items = max(0, total_items - ignored_items_count)

    personal_pct = (
        round((active_watched_count / active_items) * 100, 1)
        if active_items > 0
        else (100.0 if total_items > 0 and active_watched_count >= active_items else 0.0)
    )
    canonical_pct = round((watched_count / total_items) * 100, 1) if total_items > 0 else 0.0
    avg_rating = round(ratings_sum / rated_count, 1) if rated_count > 0 else None
    movies_count = sum(1 for x in sorted_items if x.media_type == "movie")
    tv_count = sum(1 for x in sorted_items if x.media_type == "tv")

    return UserUniverseDetailResponse(
        id=universe.id,
        universe_key=universe.universe_key,
        name=universe.name,
        overview=universe.overview,
        poster_path=universe.poster_path,
        backdrop_path=universe.backdrop_path,
        total_items=total_items,
        watched_items=watched_count,
        active_items=active_items,
        ignored_items_count=ignored_items_count,
        movies_count=movies_count,
        tv_count=tv_count,
        collections_count=universe.collections_count,
        completion_percentage=personal_pct,
        personal_completion_percentage=personal_pct,
        canonical_completion_percentage=canonical_pct,
        auto_add=auto_add,
        user_average_rating=avg_rating,
        rated_items_count=rated_count,
        items=items_resp,
        created_at=created_at,
    )


async def get_user_universes(
    db: AsyncSession,
    user_id: UUID,
) -> List[UserUniverseDetailResponse]:
    """Lista todos os universos seguidos pelo usuário com progresso e notas."""
    stmt = (
        select(UserUniverse, Universe)
        .join(Universe, UserUniverse.universe_id == Universe.id)
        .options(selectinload(Universe.items))
        .where(UserUniverse.user_id == user_id)
        .order_by(UserUniverse.created_at.desc())
    )
    res = await db.execute(stmt)
    rows = res.all()
    if not rows:
        return []

    # Coleta todos os TMDB IDs pertencentes aos universos acompanhados (escopo estrito O(I_u))
    all_universe_pairs = {
        (item.tmdb_id, item.media_type)
        for _, universe in rows if universe and universe.items
        for item in universe.items
    }
    all_tmdb_ids = {tmdb_id for tmdb_id, _ in all_universe_pairs}

    preloaded_user_items: Dict[Tuple[int, str], Tuple[UserListItem, Media]] = {}
    if all_tmdb_ids:
        list_stmt = (
            select(UserListItem, Media)
            .join(Media, UserListItem.media_id == Media.id)
            .where(
                UserListItem.user_id == user_id,
                Media.tmdb_id.in_(all_tmdb_ids),
            )
        )
        list_res = await db.execute(list_stmt)
        preloaded_user_items = {
            (media.tmdb_id, media.media_type): (uli, media)
            for uli, media in list_res.all()
            if (media.tmdb_id, media.media_type) in all_universe_pairs
        }

    # Pré-carrega itens ignorados de todos os universos acompanhados em 1 única query
    universe_ids = [universe.id for _, universe in rows if universe]
    ignored_map: Dict[UUID, Set[Tuple[int, str]]] = {}
    if universe_ids:
        ign_stmt = select(UserUniverseIgnoredItem).where(
            UserUniverseIgnoredItem.user_id == user_id,
            UserUniverseIgnoredItem.universe_id.in_(universe_ids),
        )
        ign_res = await db.execute(ign_stmt)
        for item in ign_res.scalars().all():
            ignored_map.setdefault(item.universe_id, set()).add((item.tmdb_id, item.media_type))

    results: List[UserUniverseDetailResponse] = []
    for uc, universe in rows:
        detail = await _build_universe_detail_response(
            db,
            user_id,
            universe,
            uc.created_at,
            user_items_by_pair=preloaded_user_items,
            user_universe=uc,
            ignored_pairs=ignored_map.get(universe.id, set()),
        )
        results.append(detail)
    return results


async def get_universe_details_public(
    db: AsyncSession,
    user_id: UUID,
    universe_key: str,
) -> Optional[UserUniverseDetailResponse]:
    """Retorna os detalhes completos de um universo (mesmo que o usuário ainda não o siga)."""
    universe = await get_or_create_universe(db, universe_key, user_id=user_id)
    if not universe:
        return None
    return await _build_universe_detail_response(db, user_id, universe, universe.created_at)


async def sync_universe_from_sources(
    db: AsyncSession,
    universe_key: str,
    user_id: Optional[UUID] = None,
) -> Optional[Universe]:
    """
    Sincroniza um universo com o Wikidata e TMDB para descobrir novos lançamentos/anúncios
    e adicioná-los automaticamente às listas dos seguidores com `auto_add=True`.
    """
    stmt = (
        select(Universe)
        .options(selectinload(Universe.items), selectinload(Universe.followers))
        .where(Universe.universe_key == universe_key)
    )
    res = await db.execute(stmt)
    universe = res.scalars().first()
    if not universe:
        return await get_or_create_universe(db, universe_key, user_id=user_id)

    cand = await _discover_single_universe_by_key(
        db,
        universe.universe_key,
        source_keys=list(universe.source_keys or []),
    )
    if not cand or not cand.get("enriched_members"):
        return universe

    members = cand["enriched_members"]
    existing_items_by_pair: Dict[Tuple[int, str], UniverseItem] = {
        (item.tmdb_id, item.media_type): item for item in universe.items
    }

    new_items: List[UniverseItem] = []
    valid_member_pairs = {(m["tmdb_id"], m.get("media_type", "movie")) for m in members}
    for pair, existing in list(existing_items_by_pair.items()):
        if pair not in valid_member_pairs:
            await db.delete(existing)
            if existing in universe.items:
                universe.items.remove(existing)

    for m in members:
        pair = (m["tmdb_id"], m.get("media_type", "movie"))
        if pair in existing_items_by_pair:
            existing = existing_items_by_pair[pair]
            existing.title = m.get("title") or existing.title
            existing.release_date = m.get("release_date") or existing.release_date
            existing.poster_path = m.get("poster_path") or existing.poster_path
            existing.backdrop_path = m.get("backdrop_path") or existing.backdrop_path
            existing.collection_name = m.get("collection_name") or existing.collection_name
        else:
            u_item = UniverseItem(
                universe_id=universe.id,
                tmdb_id=m["tmdb_id"],
                media_type=m.get("media_type", "movie"),
                title=m.get("title") or f"Item #{m['tmdb_id']}",
                release_date=m.get("release_date"),
                poster_path=m.get("poster_path"),
                backdrop_path=m.get("backdrop_path"),
                collection_name=m.get("collection_name"),
            )
            db.add(u_item)
            new_items.append(u_item)
            universe.items.append(u_item)

    universe.items_count = len(members)
    universe.movies_count = sum(1 for x in members if x.get("media_type", "movie") == "movie")
    universe.tv_count = sum(1 for x in members if x.get("media_type") == "tv")
    universe.collections_count = max(
        len(cand.get("child_collection_ids", set())),
        _count_distinct_collections(members),
    )
    # Atualiza as sugestões existentes deste universo caso estejam cacheadas
    sugg_res = await db.execute(
        select(UserUniverseSuggestion).where(UserUniverseSuggestion.universe_key == universe.universe_key)
    )
    for sugg in sugg_res.scalars().all():
        sugg.discovered_members = members
        sugg.total_items = len(members)
        sugg.movies_count = universe.movies_count
        sugg.tv_count = universe.tv_count
        sugg.collections_count = universe.collections_count
        sugg.source_keys = _normalize_source_keys(universe.universe_key, universe.source_keys)

    await db.commit()

    # Auto-add para seguidores caso existam novos itens ou caso user_id tenha solicitado sync manual
    follower_ids = [f.user_id for f in universe.followers if f.auto_add]
    if user_id and user_id not in follower_ids:
        # Se o usuário segue o universo, garante que todos os itens estejam sincronizados na lista dele
        is_f = any(f.user_id == user_id for f in universe.followers)
        if is_f:
            follower_ids.append(user_id)

    items_to_ensure = universe.items if user_id else new_items
    if follower_ids and items_to_ensure:
        for fid in follower_ids:
            ign_res = await db.execute(
                select(UserUniverseIgnoredItem.tmdb_id, UserUniverseIgnoredItem.media_type)
                .where(
                    UserUniverseIgnoredItem.user_id == fid,
                    UserUniverseIgnoredItem.universe_id == universe.id,
                )
            )
            u_ignored_pairs = {(r[0], r[1]) for r in ign_res.all()}

            u_list_res = await db.execute(
                select(Media.tmdb_id, Media.media_type)
                .join(UserListItem, UserListItem.media_id == Media.id)
                .where(UserListItem.user_id == fid)
            )
            u_existing_pairs = {(r[0], r[1]) for r in u_list_res.all()}
            u_to_add = [
                it for it in items_to_ensure
                if (it.tmdb_id, it.media_type) not in u_existing_pairs
                and (it.tmdb_id, it.media_type) not in u_ignored_pairs
            ]
            if u_to_add:
                tmdbs = [it.tmdb_id for it in u_to_add]
                m_stmt = select(Media).where(Media.tmdb_id.in_(tmdbs))
                m_res = await db.execute(m_stmt)
                existing_medias = {(m.tmdb_id, m.media_type): m for m in m_res.scalars().all()}

                new_medias = []
                for it in u_to_add:
                    pair = (it.tmdb_id, it.media_type)
                    media_obj = existing_medias.get(pair)
                    if not media_obj:
                        media_obj = Media(
                            tmdb_id=it.tmdb_id,
                            media_type=it.media_type,
                            title=it.title,
                            poster_path=it.poster_path,
                            backdrop_path=it.backdrop_path,
                            release_date=it.release_date,
                            collection_name=it.collection_name,
                            universe_key=universe.universe_key,
                            universe_name=universe.name,
                        )
                        db.add(media_obj)
                        existing_medias[pair] = media_obj
                        new_medias.append(media_obj)
                    else:
                        if not media_obj.universe_key:
                            media_obj.universe_key = universe.universe_key
                            media_obj.universe_name = universe.name

                if new_medias:
                    await db.flush()

                enriched_ids = []
                for it in u_to_add:
                    pair = (it.tmdb_id, it.media_type)
                    media_obj = existing_medias[pair]
                    if not it.media_id:
                        it.media_id = media_obj.id

                    is_rel = is_date_released(it.release_date)
                    final_status = "plan_to_watch" if is_rel else "upcoming"

                    user_list_item = UserListItem(
                        user_id=fid,
                        media_id=media_obj.id,
                        status=final_status,
                    )
                    db.add(user_list_item)
                    enriched_ids.append(media_obj.id)

                await db.commit()
                invalidate_user_stats_cache(str(fid))
                if enriched_ids:
                    asyncio.create_task(_background_enrich_universe_follow(str(fid), enriched_ids))

    return universe


async def _run_debounced_user_universe_scan(user_id_str: str, debounce_seconds: float = 2.0):
    try:
        if debounce_seconds > 0:
            await asyncio.sleep(debounce_seconds)
        async with AsyncSessionLocal() as db:
            await generate_user_universe_suggestions(db, UUID(user_id_str), limit=20)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Erro no scan em background de sugestões de universos para {user_id_str}: {e}")
    finally:
        _pending_universe_scans.pop(user_id_str, None)


def schedule_universe_suggestions_scan(user_id: str, debounce_seconds: float = 2.0):
    """Agenda uma atualização em background das sugestões de universos do usuário (com debounce)."""
    existing_task = _pending_universe_scans.get(user_id)
    if existing_task and not existing_task.done():
        existing_task.cancel()
    task = asyncio.create_task(_run_debounced_user_universe_scan(str(user_id), debounce_seconds))
    _pending_universe_scans[str(user_id)] = task


async def scan_all_users_universe_suggestions():
    """Varre as listas de todos os usuários ativos para pré-computar sugestões de universos."""
    async with AsyncSessionLocal() as db:
        stmt = select(UserListItem.user_id).distinct()
        res = await db.execute(stmt)
        user_ids = [row[0] for row in res.all() if row[0]]

    for uid in user_ids:
        try:
            async with AsyncSessionLocal() as db:
                await generate_user_universe_suggestions(db, uid, limit=20)
            await asyncio.sleep(1.0)
        except Exception as e:
            logger.error(f"Erro ao pré-computar universos para usuário {uid}: {e}")


async def run_universe_suggestions_loop(interval_hours: int = 6, initial_delay_seconds: int = 75):
    """Loop em background para manter as sugestões de universos atualizadas."""
    if initial_delay_seconds > 0:
        await asyncio.sleep(initial_delay_seconds)
    while True:
        try:
            logger.info("Iniciando varredura em background de sugestões de universos...")
            await scan_all_users_universe_suggestions()
            logger.info("Varredura de sugestões de universos concluída.")
        except Exception as e:
            logger.error(f"Erro no loop de sugestões de universos: {e}")
        await asyncio.sleep(interval_hours * 3600)


async def purge_recap_items_from_all_universes(db: AsyncSession) -> Dict[str, Any]:
    """
    Expurga preventivamente filmes de recap/compilação de séries de TV de todos os
    universos cadastrados no banco de dados, atualizando as contagens de itens e
    as sugestões cacheadas.
    """
    logger.info("🧹 [Universes] Iniciando varredura e expurgo de recaps de TV em universos...")
    res = await db.execute(select(Universe).options(selectinload(Universe.items)))
    universes = res.scalars().all()

    total_purged = 0
    purged_by_universe: Dict[str, List[str]] = {}

    for u in universes:
        recap_items: List[UniverseItem] = []
        for it in u.items:
            if _check_is_recap(it.title, None, None, it.media_type):
                recap_items.append(it)

        if not recap_items:
            continue

        purged_titles = [it.title for it in recap_items]
        purged_by_universe[u.name] = purged_titles
        total_purged += len(recap_items)

        for it in recap_items:
            # Remove referências de ignorados para não deixar registros órfãos
            await db.execute(
                delete(UserUniverseIgnoredItem).where(
                    UserUniverseIgnoredItem.universe_id == u.id,
                    UserUniverseIgnoredItem.tmdb_id == it.tmdb_id,
                    UserUniverseIgnoredItem.media_type == it.media_type,
                )
            )
            await db.delete(it)
            if it in u.items:
                u.items.remove(it)

        # Atualiza métricas do universo
        u.items_count = len(u.items)
        u.movies_count = sum(1 for it in u.items if it.media_type == "movie")
        u.tv_count = sum(1 for it in u.items if it.media_type == "tv")

        # Atualiza sugestões em cache
        sugg_res = await db.execute(
            select(UserUniverseSuggestion).where(UserUniverseSuggestion.universe_key == u.universe_key)
        )
        for sugg in sugg_res.scalars().all():
            if sugg.discovered_members:
                sugg.discovered_members = [
                    m for m in sugg.discovered_members
                    if not _check_is_recap(
                        m.get("title", ""),
                        m.get("overview"),
                        m.get("keywords_list") or m.get("keywords"),
                        m.get("media_type", "movie"),
                    )
                ]
            sugg.total_items = u.items_count
            sugg.movies_count = u.movies_count
            sugg.tv_count = u.tv_count

        logger.info(f"🧹 [Universes] Universo '{u.name}': expurgados {len(recap_items)} recaps de TV. Novo total: {u.items_count} obras.")

    await db.commit()
    logger.info(f"🧹 [Universes] Expurgo concluído com sucesso. Total de obras removidas: {total_purged}.")
    return {
        "total_purged": total_purged,
        "purged_by_universe": purged_by_universe,
    }


async def run_universe_sync_loop(interval_hours: int = 12, initial_delay_seconds: int = 135):
    """Loop periódico para sincronizar universos seguidos com novas estreias/anúncios."""
    if initial_delay_seconds > 0:
        await asyncio.sleep(initial_delay_seconds)

    # Executa expurgo preventivo no boot para auto-healing imediato do catálogo em produção
    try:
        async with AsyncSessionLocal() as db:
            await purge_recap_items_from_all_universes(db)
    except Exception as e:
        logger.warning(f"[Universes] Erro no expurgo inicial de recaps no boot: {e}")

    while True:
        try:
            async with AsyncSessionLocal() as db:
                stmt = select(Universe.universe_key).join(UserUniverse, UserUniverse.universe_id == Universe.id).distinct()
                res = await db.execute(stmt)
                keys = [r[0] for r in res.all() if r[0]]

            for ukey in keys:
                async with AsyncSessionLocal() as db:
                    await sync_universe_from_sources(db, ukey)
                await asyncio.sleep(2.0)
        except Exception as e:
            logger.error(f"Erro no loop de sincronização de universos: {e}")
        await asyncio.sleep(interval_hours * 3600)
