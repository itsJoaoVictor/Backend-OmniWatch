import uuid
from typing import List, Dict, Any, Set, Optional, Tuple, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np

from app.users.models import User
from app.tracking.models import UserListItem


async def find_similar_users(
    db: AsyncSession,
    target_user_id: str,
    n_neighbors: int = 5,
    min_similarity: float = 0.20
) -> List[Dict[str, Any]]:
    """
    Encontra os usuários mais parecidos com o usuário alvo com base em seus feature_vectors,
    usando DictVectorizer e NearestNeighbors (métrica cosine) do scikit-learn.

    Retorna uma lista ordenada com os usuários similares e suas métricas de afinidade.
    Garante isolamento: apenas ID e métricas gerais transitam.
    """
    try:
        target_uuid = uuid.UUID(target_user_id)
    except (ValueError, TypeError):
        return []

    # 1. Carrega todos os usuários que possuem feature_vector cadastrado
    result = await db.execute(
        select(User).where(User.feature_vector != None)
    )
    all_users = result.scalars().all()

    target_user = None
    peer_users = []

    for u in all_users:
        if u.id == target_uuid:
            target_user = u
        else:
            if u.feature_vector and len(u.feature_vector) > 0:
                peer_users.append(u)

    if not target_user or not target_user.feature_vector or not peer_users:
        return []

    # 2. Converte os dicionários esparsos heterogêneos em uma matriz comum
    # DictVectorizer(sparse=True) mapeia todas as chaves (ex: 'genre_ação', 'cast_chris_evans')
    # para colunas numéricas de forma altamente eficiente em memória.
    vectorizer = DictVectorizer(sparse=True)

    # Ajusta o vocabulário com todos os usuários (alvo + pares)
    all_vectors = [target_user.feature_vector] + [p.feature_vector for p in peer_users]
    matrix = vectorizer.fit_transform(all_vectors)

    target_matrix_vector = matrix[0]
    peer_matrix = matrix[1:]

    # 3. Treina o modelo NearestNeighbors com distância cosseno
    k = min(n_neighbors, len(peer_users))
    if k == 0:
        return []

    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(peer_matrix)

    distances, indices = nn.kneighbors(target_matrix_vector)

    similar_users = []
    for dist, idx in zip(distances[0], indices[0]):
        # Distância cosseno d varia de 0 (idêntico) a 2 (oposto).
        # Similaridade cosseno s = 1 - d
        similarity = float(1.0 - dist)

        if similarity >= min_similarity:
            peer = peer_users[idx]

            # Identifica interesses comuns de alto peso para explicabilidade/debug
            shared_interests = []
            target_vec = target_user.feature_vector
            peer_vec = peer.feature_vector

            for key, t_val in target_vec.items():
                if key in peer_vec and t_val > 0 and peer_vec[key] > 0:
                    clean_name = (
                        key.replace("genre_", "")
                        .replace("cast_", "")
                        .replace("director_", "")
                        .replace("keyword_", "")
                    )
                    score = min(t_val, peer_vec[key])
                    shared_interests.append((clean_name, score))

            shared_interests.sort(key=lambda x: x[1], reverse=True)
            top_shared = [x[0] for x in shared_interests[:5]]

            similar_users.append({
                "user_id": str(peer.id),
                "similarity": round(similarity, 3),
                "shared_interests": top_shared
            })

    similar_users.sort(key=lambda x: x["similarity"], reverse=True)
    return similar_users


async def fetch_collaborative_candidates(
    db: AsyncSession,
    target_user_id: str,
    seen_media_keys: Optional[Union[Set[Tuple[str, int]], Set[int]]] = None,
    limit: int = 15,
    min_similarity: float = 0.20,
    seen_tmdb_ids: Optional[Set[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Busca obras bem avaliadas ou concluídas por usuários com gostos parecidos,
    que o usuário alvo ainda não assistiu nem adicionou à lista.
    Usa chave canônica (media_type, tmdb_id) para evitar colisões entre filmes e séries.

    Garante isolamento total:
    - Nenhum dado pessoal, notas ou avaliações de outros usuários vazam.
    - Apenas os metadados da obra (Media) são retornados formatados como candidatos
      para o orquestrador ranquear com o perfil do próprio usuário.
    """
    effective_seen_keys = seen_media_keys if seen_media_keys is not None else (seen_tmdb_ids or set())

    similar_users = await find_similar_users(
        db, target_user_id, n_neighbors=5, min_similarity=min_similarity
    )

    if not similar_users:
        return []

    # Mapa de pesos por similaridade
    user_weight_map = {u["user_id"]: u["similarity"] for u in similar_users}
    peer_uuids = [uuid.UUID(u["user_id"]) for u in similar_users]

    # Busca os itens da lista desses usuários vizinhos que foram bem avaliados
    # (rating >= 3.5 ou status 'completed')
    result = await db.execute(
        select(UserListItem)
        .where(
            UserListItem.user_id.in_(peer_uuids),
            (UserListItem.rating >= 3.5) | (UserListItem.status == "completed")
        )
        .options(selectinload(UserListItem.media))
    )
    list_items = result.scalars().all()

    # Agrupa obras candidatas e calcula pontuação colaborativa inicial usando chave canônica
    candidate_scores: Dict[Tuple[str, int], float] = {}
    media_map: Dict[Tuple[str, int], Any] = {}

    for item in list_items:
        media = item.media
        if not media or not media.tmdb_id:
            continue

        m_key = (media.media_type or "movie", media.tmdb_id)

        # Não recomenda obras que o usuário alvo já assistiu/adicionou
        if m_key in effective_seen_keys or media.tmdb_id in effective_seen_keys:
            continue

        peer_sim = user_weight_map.get(str(item.user_id), 0.5)
        # Nota relativa (ou 0.8 se não tiver nota explícita mas estiver completed)
        rating_factor = (item.rating / 5.0) if item.rating else 0.8
        fav_factor = 1.35 if getattr(item, "is_favorite", False) else 1.0
        rewatch_cnt = getattr(item, "rewatch_count", 0) or 0
        rewatch_factor = (1.0 + min(0.30 * rewatch_cnt, 0.75)) if rewatch_cnt > 0 else 1.0
        item_score = peer_sim * rating_factor * fav_factor * rewatch_factor

        candidate_scores[m_key] = candidate_scores.get(m_key, 0.0) + item_score
        media_map[m_key] = media

    # Ordena pelo score colaborativo
    sorted_media_keys = sorted(
        candidate_scores.keys(),
        key=lambda k: candidate_scores[k],
        reverse=True
    )

    candidates = []
    for m_key in sorted_media_keys[:limit]:
        media = media_map[m_key]

        genre_ids = []
        if media.genres:
            for g in media.genres:
                if isinstance(g, dict) and "id" in g:
                    genre_ids.append(g["id"])

        candidate = {
            "id": media.tmdb_id,
            "title": media.title,
            "media_type": media.media_type,
            "poster_path": media.poster_path,
            "backdrop_path": media.backdrop_path,
            "genre_ids": genre_ids,
            "genres": media.genres or [],
            "cast": media.main_cast or [],
            "crew": media.directors or [],
            "keywords_list": media.keywords or [],
            "original_language": media.original_language or "",
            "collaborative": True,  # Identificador interno de origem colaborativa
            "collaborative_score": round(candidate_scores[m_key], 2)
        }
        candidates.append(candidate)

    return candidates
