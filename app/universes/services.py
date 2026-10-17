import asyncio
import logging
import httpx
from uuid import UUID
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Set, Tuple
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
from app.tracking.services import add_to_list
from app.details.services import get_tmdb_client

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
        if not settings.TMDB_API_KEY:
            return None
        url = f"{settings.TMDB_BASE_URL}{endpoint}"
        req_params: Dict[str, Any] = {"language": "pt-BR"}
        if params:
            req_params.update(params)
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {settings.TMDB_API_KEY}",
        }
        try:
            client = get_tmdb_client()
            resp = await client.get(url, headers=headers, params=req_params, timeout=10.0)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"TMDB request failed for {endpoint}: {e}")
        return None

    async def get_collection_details(self, collection_id: int) -> Optional[Dict[str, Any]]:
        return await self._get(f"/collection/{collection_id}", params={"language": "pt-BR"})


tmdb_client = _TMDBHelper()


async def _sparql_post(query: str, timeout: float = 25.0) -> List[Dict[str, Any]]:
    """Executa uma query SPARQL no Wikidata via POST com otimizador explícito."""
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.post(
                WIKIDATA_SPARQL_URL,
                data={"query": query, "format": "json"},
                headers=WIKIDATA_HEADERS,
            )
            if resp.status_code != 200:
                logger.warning(f"Wikidata SPARQL retornou HTTP {resp.status_code}")
                return []
            data = resp.json()
            return data.get("results", {}).get("bindings", [])
    except Exception as e:
        logger.warning(f"Falha ao consultar Wikidata SPARQL: {e}")
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
            desc = r.get("universeDesc", {}).get("value", "")
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
}


def _check_is_recap(title: str, overview: Optional[str] = None) -> bool:
    """Detecta se uma obra é um filme recap ou compilação de episódios da TV."""
    t_lower = (title or "").lower()
    o_lower = (overview or "").lower()
    for kw in RECAP_KEYWORDS:
        if kw in t_lower or kw in o_lower:
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
        movies_resp, tv_resp = await asyncio.gather(
            tmdb_client._get("/discover/movie", params={"with_keywords": str(kw_id), "language": "pt-BR"}),
            tmdb_client._get("/discover/tv", params={"with_keywords": str(kw_id), "language": "pt-BR"}),
        )
        if movies_resp and "results" in movies_resp:
            for m in movies_resp["results"]:
                if m.get("id"):
                    cand["members"][(m["id"], "movie")] = {
                        "tmdb_id": m["id"],
                        "media_type": "movie",
                        "title": m.get("title") or m.get("original_title") or f"Filme #{m['id']}",
                        "release_date": m.get("release_date") or None,
                        "poster_path": m.get("poster_path"),
                        "backdrop_path": m.get("backdrop_path"),
                    }
        if tv_resp and "results" in tv_resp:
            for t in tv_resp["results"]:
                if t.get("id"):
                    cand["members"][(t["id"], "tv")] = {
                        "tmdb_id": t["id"],
                        "media_type": "tv",
                        "title": t.get("name") or t.get("original_name") or f"Série #{t['id']}",
                        "release_date": t.get("first_air_date") or None,
                        "poster_path": t.get("poster_path"),
                        "backdrop_path": t.get("backdrop_path"),
                    }

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
    semaphore = asyncio.Semaphore(8)

    # 1. Preencher dados já existentes na tabela local Media (evita chamadas HTTP desnecessárias)
    all_tmdb_ids = [tid for (tid, _) in members_map.keys()]
    if all_tmdb_ids:
        stmt = select(Media).where(Media.tmdb_id.in_(all_tmdb_ids))
        res = await db.execute(stmt)
        for m in res.scalars().all():
            key = (m.tmdb_id, m.media_type)
            if key in members_map:
                entry = members_map[key]
                if m.title and not _is_non_latin(m.title):
                    entry["title"] = entry.get("title") or m.title
                entry["release_date"] = entry.get("release_date") or m.release_date
                entry["poster_path"] = entry.get("poster_path") or m.poster_path
                entry["backdrop_path"] = entry.get("backdrop_path") or m.backdrop_path
                entry["collection_name"] = entry.get("collection_name") or m.collection_name
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
        is_recap = _check_is_recap(final_title, data.get("overview"))
        return key, {
            "tmdb_id": tid,
            "media_type": mtype,
            "title": final_title,
            "release_date": data.get("release_date") or data.get("first_air_date") or None,
            "poster_path": data.get("poster_path"),
            "backdrop_path": data.get("backdrop_path"),
            "collection_name": col_name,
            "is_recap": is_recap,
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
                key = (pid, "movie")
                part_title = part.get("title") or part.get("original_title") or f"Filme #{pid}"
                if key not in members_map:
                    members_map[key] = {
                        "tmdb_id": pid,
                        "media_type": "movie",
                        "title": part_title,
                        "release_date": part.get("release_date") or None,
                        "poster_path": part.get("poster_path"),
                        "backdrop_path": part.get("backdrop_path"),
                        "collection_name": col_name,
                        "is_recap": _check_is_recap(part_title),
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

    cand["child_collection_ids"] = child_col_ids

    # Ordena cronologicamente por data de lançamento
    enriched_list = list(members_map.values())
    enriched_list.sort(key=lambda x: x.get("release_date") or "9999-12-31")
    return enriched_list


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
    - Persiste as sugestões em `user_universe_suggestions`.
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

    followed_keys: Set[str] = {u.universe_key for u in followed_universes}
    followed_media_pairs: Set[Tuple[int, str]] = set()
    for u in followed_universes:
        for item in u.items:
            followed_media_pairs.add((item.tmdb_id, item.media_type))

    # Remove sugestões residuais de universos que o usuário já segue
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

    items_needing_discovery: Dict[Tuple[int, str], Media] = {
        (m.tmdb_id, m.media_type): m for m in unresolved_media
    }
    for pair, m in user_media_by_pair.items():
        if pair in followed_media_pairs:
            continue
        if m.universe_key and m.universe_key not in ("none", "") and m.universe_key not in followed_keys:
            if m.universe_key not in existing_suggestions_by_key:
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
        if ukey in followed_keys:
            continue

        existing_sugg = existing_suggestions_by_key.get(ukey)
        if existing_sugg and existing_sugg.is_dismissed:
            confirmed_ukeys.add(ukey)
            continue

        enriched_members = await _enrich_universe_members(db, cand)
        if not _is_enriched_universe_valid(enriched_members, cand.get("child_collection_ids", set())):
            continue

        confirmed_ukeys.add(ukey)

        matched_titles: List[str] = []
        for member in enriched_members:
            mpair = (member["tmdb_id"], member["media_type"])
            if mpair in user_media_by_pair:
                u_media = user_media_by_pair[mpair]
                if u_media.title not in matched_titles:
                    matched_titles.append(u_media.title)
                u_media.universe_key = ukey
                u_media.universe_name = cand["name"]

        if not matched_titles:
            continue

        movies_count = sum(1 for x in enriched_members if x.get("media_type") == "movie")
        tv_count = sum(1 for x in enriched_members if x.get("media_type") == "tv")
        collections_count = len(cand.get("child_collection_ids", set()))

        poster_path = next((x.get("poster_path") for x in enriched_members if x.get("poster_path")), None)
        backdrop_path = next((x.get("backdrop_path") for x in enriched_members if x.get("backdrop_path")), None)
        overview = cand.get("overview") or f"Universo expandido contendo {movies_count} filme(s) e {tv_count} série(s)."

        if existing_sugg:
            existing_sugg.name = cand["name"]
            existing_sugg.overview = overview
            existing_sugg.poster_path = poster_path or existing_sugg.poster_path
            existing_sugg.backdrop_path = backdrop_path or existing_sugg.backdrop_path
            existing_sugg.total_items = len(enriched_members)
            existing_sugg.movies_count = movies_count
            existing_sugg.tv_count = tv_count
            existing_sugg.collections_count = collections_count
            existing_sugg.items_in_list = len(matched_titles)
            existing_sugg.matched_titles = matched_titles
            existing_sugg.discovered_members = enriched_members
        else:
            new_sugg = UserUniverseSuggestion(
                user_id=user_id,
                universe_key=ukey,
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

    # Marca mídias verificadas que não pertencem a nenhum universo válido como 'none' para não re-escanear
    for m in unresolved_media:
        if m.universe_key is None or (m.universe_key not in confirmed_ukeys and m.universe_key not in followed_keys):
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
    Se não houver sugestões cacheadas mas o usuário possuir itens ainda não escaneados,
    agenda um scan em background.
    """
    followed_stmt = (
        select(Universe.universe_key)
        .join(UserUniverse, UserUniverse.universe_id == Universe.id)
        .where(UserUniverse.user_id == user_id)
    )
    followed_res = await db.execute(followed_stmt)
    followed_keys = set(followed_res.scalars().all())

    stmt = (
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
    res = await db.execute(stmt)
    suggestions = [s for s in res.scalars().all() if s.universe_key not in followed_keys]

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
        select(Universe.universe_key)
        .join(UserUniverse, UserUniverse.universe_id == Universe.id)
        .where(UserUniverse.user_id == user_id)
    )
    followed_res = await db.execute(followed_stmt)
    followed_keys = set(followed_res.scalars().all())

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
) -> Optional[Dict[str, Any]]:
    """
    Busca a estrutura completa de um universo diretamente pela sua `universe_key`
    (QID do Wikidata como 'Q19785923' ou Keyword do TMDB como 'kw:180547').
    """
    if universe_key.startswith("kw:"):
        kw_id_str = universe_key.split(":", 1)[1]
        if not kw_id_str.isdigit():
            return None
        kw_id = int(kw_id_str)
        kw_info = await tmdb_client._get(f"/keyword/{kw_id}")
        kw_name = (kw_info.get("name") if kw_info else f"Universo #{kw_id}").title()
        cand: Dict[str, Any] = {
            "universe_key": universe_key,
            "name": kw_name,
            "overview": f"Universo compartilhado '{kw_name}' no TMDB.",
            "members": {},
            "child_collection_ids": set(),
        }
        movies_resp, tv_resp = await asyncio.gather(
            tmdb_client._get("/discover/movie", params={"with_keywords": str(kw_id), "language": "pt-BR"}),
            tmdb_client._get("/discover/tv", params={"with_keywords": str(kw_id), "language": "pt-BR"}),
        )
        for m in (movies_resp or {}).get("results", []):
            if m.get("id"):
                cand["members"][(m["id"], "movie")] = {
                    "tmdb_id": m["id"],
                    "media_type": "movie",
                    "title": m.get("title") or m.get("original_title"),
                    "release_date": m.get("release_date"),
                    "poster_path": m.get("poster_path"),
                    "backdrop_path": m.get("backdrop_path"),
                }
        for t in (tv_resp or {}).get("results", []):
            if t.get("id"):
                cand["members"][(t["id"], "tv")] = {
                    "tmdb_id": t["id"],
                    "media_type": "tv",
                    "title": t.get("name") or t.get("original_name"),
                    "release_date": t.get("first_air_date"),
                    "poster_path": t.get("poster_path"),
                    "backdrop_path": t.get("backdrop_path"),
                }
        enriched = await _enrich_universe_members(db, cand)
        cand["enriched_members"] = enriched
        return cand

    # Caso padrão: QID do Wikidata (ex: Q19785923)
    if not universe_key.startswith("Q"):
        return None

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
    label_rows = await _sparql_post(label_query)
    name = universe_key
    overview = None
    if label_rows:
        name = label_rows[0].get("universeLabel", {}).get("value", universe_key)
        overview = label_rows[0].get("universeDesc", {}).get("value")

    cand = {
        "universe_key": universe_key,
        "name": name,
        "overview": overview,
        "members": {},
        "child_collection_ids": set(),
    }
    await _wikidata_step2_expand_members({universe_key: cand})
    enriched = await _enrich_universe_members(db, cand)
    cand["enriched_members"] = enriched
    return cand


async def get_or_create_universe(
    db: AsyncSession,
    universe_key: str,
    user_id: Optional[UUID] = None,
) -> Optional[Universe]:
    """
    Obtém um Universo do banco ou o cria a partir do cache de sugestões (`UserUniverseSuggestion`)
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

    movies_count = sum(1 for x in members if x.get("media_type", "movie") == "movie")
    tv_count = sum(1 for x in members if x.get("media_type") == "tv")

    universe = Universe(
        universe_key=universe_key,
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

    # Remove da tabela de sugestões para este usuário
    await db.execute(
        delete(UserUniverseSuggestion).where(
            UserUniverseSuggestion.user_id == user_id,
            UserUniverseSuggestion.universe_key == universe_key,
        )
    )
    await db.commit()

    # Busca itens já na lista do usuário
    list_stmt = (
        select(UserListItem, Media)
        .join(Media, UserListItem.media_id == Media.id)
        .where(UserListItem.user_id == user_id)
    )
    list_res = await db.execute(list_stmt)
    existing_user_pairs: Dict[Tuple[int, str], Media] = {
        (media.tmdb_id, media.media_type): media for _, media in list_res.all()
    }

    items_added = 0
    items_already_in_list = 0

    sorted_items = sorted(universe.items, key=lambda x: x.release_date or "9999-12-31")
    for item in sorted_items:
        pair = (item.tmdb_id, item.media_type)
        if pair in existing_user_pairs:
            items_already_in_list += 1
            media_obj = existing_user_pairs[pair]
            if not item.media_id:
                item.media_id = media_obj.id
            media_obj.universe_key = universe.universe_key
            media_obj.universe_name = universe.name
            continue

        if pair in ignored_set:
            # Não auto-adiciona obras desmarcadas/ignoradas pelo usuário
            continue

        try:
            item_schema = UserListItemCreate(
                tmdb_id=item.tmdb_id,
                media_type=item.media_type,
                title=item.title,
                poster_path=item.poster_path,
                backdrop_path=item.backdrop_path,
                release_date=item.release_date,
                status="plan_to_watch",
            )
            created_item = await add_to_list(db, str(user_id), item_schema)
            if created_item and created_item.media_id:
                item.media_id = created_item.media_id
                m_res = await db.execute(select(Media).where(Media.id == created_item.media_id))
                m_obj = m_res.scalars().first()
                if m_obj:
                    m_obj.universe_key = universe.universe_key
                    m_obj.universe_name = universe.name
            items_added += 1
        except Exception as e:
            logger.warning(f"Erro ao auto-adicionar item {item.tmdb_id} ({item.media_type}) do universo {universe.name}: {e}")

    await db.commit()

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
) -> UserUniverseDetailResponse:
    """Monta o payload detalhado de um universo cruzando com status, notas e itens ignorados."""
    list_stmt = (
        select(UserListItem, Media)
        .join(Media, UserListItem.media_id == Media.id)
        .where(UserListItem.user_id == user_id)
    )
    list_res = await db.execute(list_stmt)
    user_items_by_pair: Dict[Tuple[int, str], Tuple[UserListItem, Media]] = {
        (media.tmdb_id, media.media_type): (uli, media) for uli, media in list_res.all()
    }

    ign_stmt = select(UserUniverseIgnoredItem).where(
        UserUniverseIgnoredItem.user_id == user_id,
        UserUniverseIgnoredItem.universe_id == universe.id,
    )
    ign_res = await db.execute(ign_stmt)
    ignored_pairs = {(x.tmdb_id, x.media_type) for x in ign_res.scalars().all()}

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

    # Auto-healing: Se algum item já persistido estiver com título em CJK/japonês, busca o título em inglês e persiste
    non_latin_items = [it for it in universe.items if _is_non_latin(it.title)]
    if non_latin_items:
        async def resolve_item_en(u_item: UniverseItem):
            try:
                en_data = await tmdb_client._get(f"/{u_item.media_type}/{u_item.tmdb_id}", params={"language": "en-US"})
                if en_data:
                    en_title = en_data.get("title") or en_data.get("name")
                    if en_title and not _is_non_latin(en_title):
                        u_item.title = en_title
                        m_stmt = select(Media).where(Media.tmdb_id == u_item.tmdb_id, Media.media_type == u_item.media_type)
                        m_res = await db.execute(m_stmt)
                        m_obj = m_res.scalars().first()
                        if m_obj and _is_non_latin(m_obj.title):
                            m_obj.title = en_title
            except Exception as e:
                logger.warning(f"Falha ao buscar fallback em inglês para item {u_item.tmdb_id}: {e}")

        await asyncio.gather(*(resolve_item_en(it) for it in non_latin_items))
        try:
            await db.commit()
        except Exception:
            await db.rollback()

    sorted_items = sorted(universe.items, key=lambda x: x.release_date or "9999-12-31")
    for item in sorted_items:
        pair = (item.tmdb_id, item.media_type)
        is_ignored = pair in ignored_pairs
        is_recap = _check_is_recap(item.title)

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

    results: List[UserUniverseDetailResponse] = []
    for uc, universe in rows:
        detail = await _build_universe_detail_response(db, user_id, universe, uc.created_at)
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

    cand = await _discover_single_universe_by_key(db, universe_key)
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
    # Atualiza as sugestões existentes deste universo caso estejam cacheadas
    sugg_res = await db.execute(
        select(UserUniverseSuggestion).where(UserUniverseSuggestion.universe_key == universe_key)
    )
    for sugg in sugg_res.scalars().all():
        sugg.discovered_members = members
        sugg.total_items = len(members)
        sugg.movies_count = universe.movies_count
        sugg.tv_count = universe.tv_count
        sugg.collections_count = universe.collections_count

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
            for item in items_to_ensure:
                pair = (item.tmdb_id, item.media_type)
                if pair in u_existing_pairs or pair in u_ignored_pairs:
                    continue
                try:
                    schema = UserListItemCreate(
                        tmdb_id=item.tmdb_id,
                        media_type=item.media_type,
                        title=item.title,
                        poster_path=item.poster_path,
                        backdrop_path=item.backdrop_path,
                        release_date=item.release_date,
                        status="plan_to_watch",
                    )
                    await add_to_list(db, str(fid), schema)
                except Exception as e:
                    logger.warning(f"Erro ao sincronizar item {item.tmdb_id} do universo para usuário {fid}: {e}")

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


async def run_universe_suggestions_loop(interval_hours: int = 6):
    """Loop em background para manter as sugestões de universos atualizadas."""
    await asyncio.sleep(8)
    while True:
        try:
            logger.info("Iniciando varredura em background de sugestões de universos...")
            await scan_all_users_universe_suggestions()
            logger.info("Varredura de sugestões de universos concluída.")
        except Exception as e:
            logger.error(f"Erro no loop de sugestões de universos: {e}")
        await asyncio.sleep(interval_hours * 3600)


async def run_universe_sync_loop(interval_hours: int = 12):
    """Loop periódico para sincronizar universos seguidos com novas estreias/anúncios."""
    await asyncio.sleep(60)
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
