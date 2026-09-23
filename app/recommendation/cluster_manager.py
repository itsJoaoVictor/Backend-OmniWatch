import json
import re
import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np
import httpx
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import settings
from app.users.models import User
from app.tracking.services import get_user_list
from app.recommendation.vector_builder import build_sparse_vector

logger = logging.getLogger(__name__)

GENRE_EMOJIS = {
    "ação": "🎬",
    "aventura": "🗺️",
    "action & adventure": "🗺️",
    "ficção científica": "🚀",
    "sci-fi & fantasy": "🚀",
    "animação": "🎨",
    "comédia": "😂",
    "drama": "🎭",
    "terror": "👻",
    "mistério": "🔍",
    "crime": "🕵️",
    "fantasia": "🧙‍♂️",
    "romance": "❤️",
    "thriller": "⚡",
    "família": "👨‍👩‍👧",
    "documentário": "📽️",
    "guerra": "⚔️",
    "história": "🏛️",
    "música": "🎵",
    "faroeste": "🤠"
}

PROMPT_SYSTEM = """Você é um curador editorial de cinema e streaming de altíssimo nível (estilo Netflix, MUBI, Criterion e Spotify).
Sua missão é dar um NOME CRIATIVO, INSPIRADOR e ELEGANTE para uma "Persona de Gosto" de um usuário, além de escolher 1 único emoji que melhor resuma essa vibe.

Diretrizes:
- O tom deve ser EDITORIAL, CINEMATOGRÁFICO e EVOCATIVO (ex: 'Aventura Cósmica & Heróis Improváveis', 'Noites Sombrias & Suspense Psicológico', 'Mundos Fantásticos & Shonen Épico', 'Pesadelos Reais & Terror Visceral').
- Evite nomes secos e burocráticos como apenas 'Ação e Aventura' ou 'Ficção Científica'.
- Dê também uma 'tagline' curta (uma frase de 5 a 10 palavras que define o sentimento desse grupo).
- Responda OBRIGATORIAMENTE em formato JSON válido com as seguintes chaves:
{
  "name": "Título Editorial Criativo",
  "emoji": "🎬",
  "tagline": "Frase curta sobre a vibe"
}
"""

def generate_persona_metadata(cluster_media: List[Any]) -> Dict[str, Any]:
    genre_counts: Dict[str, int] = {}
    for media in cluster_media:
        if media.genres:
            for g in media.genres:
                g_name = (g.get("name", "") if isinstance(g, dict) else str(g)).strip().lower()
                if g_name in ["action & adventure", "ação"]:
                    genre_counts["Ação"] = genre_counts.get("Ação", 0) + 1
                    genre_counts["Aventura"] = genre_counts.get("Aventura", 0) + 1
                elif g_name in ["sci-fi & fantasy", "ficção científica"]:
                    genre_counts["Ficção Científica"] = genre_counts.get("Ficção Científica", 0) + 1
                else:
                    norm = g_name.title()
                    genre_counts[norm] = genre_counts.get(norm, 0) + 1

    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    top_genres = [g[0] for g in sorted_genres[:3]]

    dominant_genre = top_genres[0].lower() if top_genres else "ação"
    emoji = GENRE_EMOJIS.get(dominant_genre, "✨")

    if len(top_genres) >= 2:
        g1, g2 = top_genres[0], top_genres[1]
        if "Animação" in [g1, g2] and any(x in ["Ação", "Aventura", "Fantasia"] for x in [g1, g2]):
            name = "Animes & Animação Épica"
            emoji = "🎨"
        elif "Comédia" in [g1, g2]:
            name = "Comédias & Sitcoms"
            emoji = "😂"
        elif "Mistério" in [g1, g2] or "Crime" in [g1, g2]:
            name = "Suspense & Mistério"
            emoji = "🔍"
        else:
            name = f"{g1} & {g2}"
    elif len(top_genres) == 1:
        name = f"Universo {top_genres[0]}"
    else:
        name = "Mix Personalizado"

    return {
        "name": name,
        "emoji": emoji,
        "top_genres": top_genres
    }

def build_cluster_feature_vector(cluster_media: List[Any]) -> Dict[str, float]:
    cluster_vec: Dict[str, float] = {}
    for media in cluster_media:
        m_vec = build_sparse_vector(media)
        for k, v in m_vec.items():
            cluster_vec[k] = cluster_vec.get(k, 0.0) + v

    if cluster_vec:
        max_val = max(cluster_vec.values())
        if max_val > 0:
            scale = 30.0 / max_val
            cluster_vec = {k: round(v * scale, 2) for k, v in cluster_vec.items()}

    return cluster_vec

def should_refresh_cluster_llm(new_cluster: Dict[str, Any], matched_old_cluster: Optional[Dict[str, Any]]) -> bool:
    """
    Inteligência de Economia de Tokens (Smart Cache):
    Só autoriza chamada à LLM se o cluster for genuinamente novo, se tiver passado
    o TTL de 14 dias, ou se houver 'mudança brusca' no núcleo de obras do cluster.
    """
    if not matched_old_cluster:
        return True
    
    if not matched_old_cluster.get("llm_enhanced"):
        return True
        
    # 1. TTL de Tempo (14 dias)
    generated_at = matched_old_cluster.get("llm_generated_at")
    if generated_at:
        try:
            gen_dt = datetime.fromisoformat(generated_at)
            if gen_dt.tzinfo is None:
                gen_dt = gen_dt.replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - gen_dt).days >= 14:
                return True
        except Exception:
            return True
    else:
        return True

    # 2. Mudança Brusca na Escala (acréscimo de mais de 35% de obras e ao menos 5 novas)
    old_count = matched_old_cluster.get("item_count", 0)
    new_count = new_cluster.get("item_count", 0)
    diff = abs(new_count - old_count)
    if diff >= 5 and (diff / max(old_count, 1)) >= 0.35:
        return True

    # 3. Mudança Brusca no Núcleo do Cluster (menos de 50% de sobreposição nos top títulos)
    old_samples = set(matched_old_cluster.get("sample_titles", []))
    new_samples = set(new_cluster.get("sample_titles", []))
    if old_samples and new_samples:
        overlap = len(old_samples.intersection(new_samples))
        if overlap / len(new_samples) < 0.5:
            return True

    # Caso contrário, mantém o cache existente com total segurança
    return False

async def fetch_llm_persona_title(persona: Dict[str, Any], client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    if not settings.OPENAI_API_KEY:
        return None

PROMPT_BATCH_SYSTEM = """Você é o Diretor Editorial de Recomendação do OmniWatch (estilo Netflix, Criterion e Spotify).
Sua missão é dar NOMES CRIATIVOS, INSPIRADORES e EMOJIS para TODAS as Personas de Gosto de um usuário.

REGRA DE OURO — CONTRASTE RADICAL & ZERO REPETIÇÃO:
- As personas representam facetas DIFERENTES da mesma pessoa. Elas DEVEM ter vocabulários, estilos e focos temáticos COMPLETAMENTE DISTINTOS entre si.
- É TERMINANTEMENTE PROIBIDO repetir palavras-chave (ex: se uma persona usar 'Aventura' ou 'Heróis', NENHUMA outra persona pode usar essas palavras!).
- CLUSTER DE ANIMES / ANIMAÇÃO: Se uma persona contiver animes ou animações (ex: One Piece, Demon Slayer, Dan Da Dan, Jujutsu Kaisen), você DEVE explicitamente destacar esse nicho cultural com termos como 'Universo Anime', 'Shonen Lendário' ou 'Animação Oriental'. Jamais rotule um cluster de anime como apenas ação/aventura hollywoodiana.
- CLUSTER DE FICÇÃO CIENTÍFICA / HERÓIS: Use termos como 'Odisseia Cósmica', 'Ficção Especulativa', 'Heróis & Universos Paralelos'.
- CLUSTER DE TERROR / SLASHER: Use termos como 'Pesadelos Sombrios', 'Horror Visceral', 'Tensão Psicológica'.

Responda OBRIGATORIAMENTE em JSON como uma lista de objetos:
[
  {
    "id": "persona_0",
    "name": "Título Editorial Criativo",
    "emoji": "🚀",
    "tagline": "Frase curta sobre a vibe"
  }
]
"""

async def fetch_batch_llm_persona_titles(personas: List[Dict[str, Any]], client: httpx.AsyncClient) -> Dict[str, Dict[str, Any]]:
    """
    Envia todas as personas do usuário juntas em um único prompt para a LLM,
    garantindo contraste semântico absoluto e zero repetição de títulos.
    """
    if not settings.OPENAI_API_KEY or not personas:
        return {}

    prompt_parts = ["Aqui estão as Personas de gosto do usuário para você nomear com contraste máximo:\n"]
    for p in personas:
        # Detecta sinal de Anime/Animação Oriental para avisar a LLM
        sample_str = " ".join(p.get("sample_titles", [])).lower()
        has_anime = any(w in sample_str for w in ["piece", "slayer", "dan da dan", "naruto", "jujutsu", "haikyu", "ghibli", "bleach", "dragon ball"]) or "Animação" in p.get("top_genres", [])
        note = " [DESTAQUE: Universo Anime / Animação Japonesa!]" if has_anime else ""

        prompt_parts.append(
            f"--- PERSONA ID: {p.get('id')} {note} ---\n"
            f"- Gêneros: {', '.join(p.get('top_genres', []))}\n"
            f"- Exemplos assistidos: {', '.join(p.get('sample_titles', [])[:6])}\n"
            f"- Total de obras: {p.get('item_count')}\n"
        )

    prompt_user = "\n".join(prompt_parts) + "\nCrie o título editorial, emoji e tagline para cada Persona com contraste máximo em JSON."

    payload = {
        "model": settings.OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": PROMPT_BATCH_SYSTEM},
            {"role": "user", "content": prompt_user}
        ],
        "temperature": 0.7,
        "max_tokens": 2500
    }

    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://omniwatch.app",
        "X-Title": "OmniWatch"
    }

    try:
        resp = await client.post(
            f"{settings.OPENAI_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=35.0
        )
        if resp.status_code != 200:
            logger.warning(f"OpenRouter LLM batch status {resp.status_code}: {resp.text}")
            return {}

        data = resp.json()
        choice = data["choices"][0]
        msg = choice.get("message", {})
        raw_content = (msg.get("content") or "").strip()
        reasoning_text = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()

        # Combina content e reasoning para garantir parsing mesmo se o modelo raciocinar muito
        combined_text = raw_content if len(raw_content) > 10 else f"{raw_content}\n{reasoning_text}"
        if not combined_text:
            return {}

        clean_content = combined_text.replace('“', '"').replace('”', '"').strip()
        results_by_id = {}

        # 1. Busca por lista JSON [...]
        start_idx = clean_content.find("[")
        end_idx = clean_content.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_candidate = clean_content[start_idx:end_idx+1]
            try:
                parsed_list = json.loads(json_candidate)
                if isinstance(parsed_list, list):
                    for item in parsed_list:
                        if isinstance(item, dict) and "id" in item:
                            results_by_id[item["id"]] = {
                                "name": item.get("name"),
                                "emoji": item.get("emoji", "✨"),
                                "tagline": item.get("tagline", "")
                            }
                    if len(results_by_id) == len(personas):
                        return results_by_id
            except Exception:
                pass

        # 2. Busca por objetos JSON individuais {"id": "persona_X", ...}
        matches = re.findall(r'\{\s*"id"\s*:\s*"(persona_\d+)"\s*,\s*"name"\s*:\s*"([^"]+)"\s*,\s*"emoji"\s*:\s*"([^"]+)"(?:\s*,\s*"tagline"\s*:\s*"([^"]*)")?', clean_content)
        for m in matches:
            results_by_id[m[0]] = {
                "name": m[1],
                "emoji": m[2],
                "tagline": m[3] if len(m) > 3 else ""
            }

        if len(results_by_id) >= len(personas):
            return results_by_id

        # 3. Fallback de texto estruturado (ex: Persona_0: name: "...", emoji: "...", tagline: "...")
        text_matches = re.findall(
            r'Persona[_\s]*(\d+)[:\s]+(?:[^\n]*\n)*?.*?name[:\s]+["\']?([^"\'\n\r]+)["\']?.*?emoji[:\s]+["\']?([^\s"\'\n\r]+)["\']?.*?tagline[:\s]+["\']?([^"\'\n\r]+)["\']?',
            clean_content,
            re.IGNORECASE | re.DOTALL
        )
        for tm in text_matches:
            p_id = f"persona_{tm[0]}"
            if p_id not in results_by_id:
                results_by_id[p_id] = {
                    "name": tm[1].strip(),
                    "emoji": tm[2].strip(),
                    "tagline": tm[3].strip()
                }

        return results_by_id
    except Exception as e:
        logger.warning(f"Erro ao gerar titulos em lote via LLM: {e}")
        return {}

async def refine_user_clusters_llm_task(user_id: str):
    """
    Tarefa de background assíncrona: enriquece todas as personas em lote com um único prompt.
    """
    from app.core.database import AsyncSessionLocal
    try:
        async with AsyncSessionLocal() as db:
            user_result = await db.execute(select(User).where(User.id == user_id))
            user = user_result.scalars().first()
            if not user or not user.taste_clusters:
                return

            clusters = list(user.taste_clusters)
            pending_clusters = [c for c in clusters if c.get("needs_llm")]

            if not pending_clusters:
                return

            async with httpx.AsyncClient() as client:
                llm_results = await fetch_batch_llm_persona_titles(clusters, client)

            if llm_results:
                for c in clusters:
                    p_id = c.get("id")
                    if p_id in llm_results:
                        meta = llm_results[p_id]
                        if meta.get("name"):
                            c["name"] = meta["name"]
                        if meta.get("emoji"):
                            c["emoji"] = meta["emoji"]
                        if meta.get("tagline"):
                            c["tagline"] = meta["tagline"]
                        c["llm_enhanced"] = True
                        c["llm_generated_at"] = datetime.now(timezone.utc).isoformat()
                    c.pop("needs_llm", None)

                user.taste_clusters = list(clusters)
                flag_modified(user, "taste_clusters")
                await db.commit()
                logger.info(f"[LLM PERSONAS] Lote de clusters do usuario {user_id} refinado com sucesso.")
            else:
                for c in clusters:
                    c.pop("needs_llm", None)
                user.taste_clusters = list(clusters)
                flag_modified(user, "taste_clusters")
                await db.commit()
    except Exception as e:
        logger.error(f"[LLM PERSONAS] Erro na background task de refinamento em lote: {e}")

async def compute_user_taste_clusters(db: AsyncSession, user_id: str) -> List[Dict[str, Any]]:
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalars().first()
    if not user:
        return []

    old_clusters = user.taste_clusters or []

    user_items = await get_user_list(db, user_id)
    valid_items = [
        it for it in user_items
        if it.media and (
            it.status in ["completed", "watching", "plan_to_watch"] or
            (it.rating is not None and it.rating >= 2.5)
        )
    ]

    if not valid_items:
        user.taste_clusters = []
        await db.commit()
        return []

    items_with_emb = [it for it in valid_items if it.media.embedding and len(it.media.embedding) == 4096]
    n_items = len(items_with_emb)

    if n_items < 3:
        all_media = [it.media for it in valid_items]
        meta = generate_persona_metadata(all_media)
        centroid = user.embedding or (items_with_emb[0].media.embedding if items_with_emb else None)
        single_cluster = [{
            "id": "persona_0",
            "name": meta["name"],
            "emoji": meta["emoji"],
            "tagline": "O seu universo personalizado em um único lugar.",
            "item_count": len(valid_items),
            "top_genres": meta["top_genres"],
            "sample_titles": [m.title for m in all_media[:5]],
            "centroid_embedding": centroid,
            "feature_vector": user.feature_vector or build_cluster_feature_vector(all_media),
            "llm_enhanced": False
        }]
        user.taste_clusters = single_cluster
        await db.commit()
        return single_cluster

    if n_items < 6:
        k = 1
    else:
        if n_items < 12:
            max_k = 2
        elif n_items < 30:
            max_k = 3
        elif n_items < 80:
            max_k = 4
        else:
            max_k = 5

        embeddings = np.array([it.media.embedding for it in items_with_emb])
        min_cluster_size = min(5, n_items // max_k)

        best_k = 2
        best_score = -1.0
        best_labels = None

        for cand_k in range(2, max_k + 1):
            km = KMeans(n_clusters=cand_k, random_state=42, n_init=10)
            labels_cand = km.fit_predict(embeddings)
            
            counts = [int(np.sum(labels_cand == c_idx)) for c_idx in range(cand_k)]
            if any(cnt < min_cluster_size for cnt in counts):
                continue

            score = float(silhouette_score(embeddings, labels_cand))

            # Principio da Parcimonia (+0.015)
            if best_labels is None or (score > best_score + 0.015):
                best_score = score
                best_k = cand_k
                best_labels = labels_cand

        k = best_k if best_labels is not None else 1
        labels = best_labels if best_labels is not None else np.zeros(n_items, dtype=int)

    clusters_list = []
    has_pending_llm = False

    for cluster_idx in range(k):
        indices = [i for i, lbl in enumerate(labels) if lbl == cluster_idx]
        if not indices:
            continue

        cluster_media = [items_with_emb[i].media for i in indices]
        meta = generate_persona_metadata(cluster_media)

        cluster_embs = embeddings[indices]
        centroid_vec = np.mean(cluster_embs, axis=0)
        norm = np.linalg.norm(centroid_vec)
        if norm > 0:
            centroid_vec = centroid_vec / norm

        feature_vector = build_cluster_feature_vector(cluster_media)
        sample_titles = [m.title for m in cluster_media[:5]]

        candidate_cluster = {
            "id": f"persona_{cluster_idx}",
            "name": meta["name"],
            "emoji": meta["emoji"],
            "tagline": "",
            "item_count": len(cluster_media),
            "top_genres": meta["top_genres"],
            "sample_titles": sample_titles,
            "centroid_embedding": centroid_vec.tolist(),
            "feature_vector": feature_vector,
            "llm_enhanced": False
        }

        # Pareamento inteligente com clusters antigos para reaproveitamento de cache
        best_old_match = None
        best_sim = -1.0
        for old_c in old_clusters:
            old_cent = old_c.get("centroid_embedding")
            if old_cent and len(old_cent) == 4096:
                sim = float(np.dot(centroid_vec, np.array(old_cent)))
                if sim > best_sim:
                    best_sim = sim
                    best_old_match = old_c

        # Se a similaridade com um cluster antigo for >= 0.80, considera o mesmo polo
        matched_old = best_old_match if best_sim >= 0.80 else None

        if should_refresh_cluster_llm(candidate_cluster, matched_old):
            # Precisa de geracao nova da LLM
            candidate_cluster["needs_llm"] = True
            has_pending_llm = True
        else:
            # Reutiliza o cache existente sem gastar NENHUM token
            candidate_cluster["name"] = matched_old.get("name", meta["name"])
            candidate_cluster["emoji"] = matched_old.get("emoji", meta["emoji"])
            candidate_cluster["tagline"] = matched_old.get("tagline", "")
            candidate_cluster["llm_enhanced"] = True
            candidate_cluster["llm_generated_at"] = matched_old.get("llm_generated_at")

        clusters_list.append(candidate_cluster)

    clusters_list.sort(key=lambda x: x["item_count"], reverse=True)
    for idx, c in enumerate(clusters_list):
        c["id"] = f"persona_{idx}"

    user.taste_clusters = clusters_list
    flag_modified(user, "taste_clusters")
    await db.commit()

    # Dispara a background task somente se algum cluster precisar de fato da LLM
    if has_pending_llm:
        try:
            asyncio.create_task(refine_user_clusters_llm_task(user_id))
        except Exception:
            pass

    return clusters_list
