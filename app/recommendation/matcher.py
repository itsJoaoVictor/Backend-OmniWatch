import math
from typing import Dict, List, Tuple, Any, Optional

def dot_product(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    return sum(v1.get(k, 0) * v2.get(k, 0) for k in set(v1) & set(v2))

def magnitude(v: Dict[str, float]) -> float:
    return math.sqrt(sum(val * val for val in v.values()))

def cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product(v1, v2) / (mag1 * mag2)

def calculate_match_score(user_vector: Dict[str, float], media_vector: Dict[str, float]) -> Tuple[float, List[str]]:
    """
    Calcula o Match Score percentual (0.0 a 100.0) com denominador fixo em 100.0
    e Bayesian Shrinkage Prior (mu_0 = 0.35) para categorias vazias (Ideia 4 + Anomalia 2.1.3).

    Distribuição exata dos 100.0 pontos:
    - Gêneros compatíveis: até 40.0 pts (prior vazio: 14.0 pts)
    - Talento (Elenco e Direção): até 25.0 pts (prior vazio: 8.75 pts)
    - Palavras-chave e Temas: até 20.0 pts (prior vazio: 7.0 pts)
    - Idioma Original (lang_*): até 10.0 pts (prior vazio / inglês neutro: 7.0 pts)
    - Formato Proporcional (type_*): até 5.0 pts (distribuição proporcional filme vs série)
    Total fixo: 40 + 25 + 20 + 10 + 5 = 100.0 pts.
    """
    if not user_vector or not media_vector:
        return 0.0, []

    total_score = 0.0

    # 1. Componente de Gêneros (peso 40.0)
    media_genres = {k: v for k, v in media_vector.items() if k.startswith("genre_")}
    if media_genres:
        user_genres = {k: v for k, v in user_vector.items() if k.startswith("genre_")}
        max_u_genre = max(abs(v) for v in user_genres.values()) if user_genres else 1.0
        if max_u_genre == 0:
            max_u_genre = 1.0

        genre_ratios = []
        for k in media_genres:
            u_val = user_vector.get(k, 0.0)
            ratio = (u_val / max_u_genre) if max_u_genre > 0 else 0.0
            genre_ratios.append(ratio)
        genre_ratios.sort(reverse=True)
        # Soft Top-r Genre Matching (Ideia 4): 0.50 para 1º, 0.30 para 2º, 0.20 para 3º
        pos_weights = [0.50, 0.30, 0.20]
        weighted_genre_ratio = 0.0
        weight_sum = 0.0
        for i, r in enumerate(genre_ratios[:3]):
            w = pos_weights[i]
            weighted_genre_ratio += r * w
            weight_sum += w
        if weight_sum > 0:
            genre_ratio = max(0.0, min(1.0, weighted_genre_ratio / weight_sum))
            total_score += genre_ratio * 40.0
    else:
        total_score += 0.35 * 40.0  # Shrinkage prior (14.0 pts)

    # 2. Componente de Formato Proporcional (Filme vs Série - peso 5.0)
    u_movie = max(0.0, user_vector.get("type_movie", 0.0))
    u_tv = max(0.0, user_vector.get("type_tv", 0.0))
    total_type = u_movie + u_tv
    is_movie = any(k == "type_movie" for k in media_vector)
    is_tv = any(k == "type_tv" for k in media_vector)
    if total_type > 0:
        if is_movie:
            type_ratio = u_movie / total_type
        elif is_tv:
            type_ratio = u_tv / total_type
        else:
            type_ratio = 0.5
        total_score += max(0.0, min(1.0, type_ratio)) * 5.0
    else:
        total_score += 0.5 * 5.0  # Neutro (2.5 pts)

    # 3. Componente de Talento: Elenco e Direção (peso 25.0)
    media_people = {k: v for k, v in media_vector.items() if k.startswith("cast_") or k.startswith("director_")}
    if media_people:
        user_people = {k: v for k, v in user_vector.items() if k.startswith("cast_") or k.startswith("director_")}
        max_u_person = max(abs(v) for v in user_people.values()) if user_people else 1.0
        if max_u_person == 0:
            max_u_person = 1.0

        matched_ratios = [
            max(0.0, min(1.0, user_vector[k] / max_u_person))
            for k in media_people
            if k in user_vector and user_vector[k] > 0
        ]
        matched_sum = sum(matched_ratios)
        if matched_sum > 0:
            # Saturação suave exponencial: 1 - exp(-1.2 * matched_sum)
            sat = 1.0 - math.exp(-1.2 * matched_sum)
            total_score += sat * 25.0
        else:
            total_score += 0.35 * 25.0  # Bayesian prior (8.75 pts)
    else:
        total_score += 0.35 * 25.0  # Bayesian prior (8.75 pts)

    # 4. Componente de Palavras-chave e Temas (peso 20.0)
    media_kw = {k: v for k, v in media_vector.items() if k.startswith("keyword_")}
    if media_kw:
        user_kw = {k: v for k, v in user_vector.items() if k.startswith("keyword_")}
        max_u_kw = max(abs(v) for v in user_kw.values()) if user_kw else 1.0
        if max_u_kw == 0:
            max_u_kw = 1.0

        matched_kw_ratios = [
            max(0.0, min(1.0, user_vector[k] / max_u_kw))
            for k in media_kw
            if k in user_vector and user_vector[k] > 0
        ]
        matched_kw_sum = sum(matched_kw_ratios)
        if matched_kw_sum > 0:
            # Saturação suave exponencial: 1 - exp(-1.2 * matched_kw_sum)
            sat_kw = 1.0 - math.exp(-1.2 * matched_kw_sum)
            total_score += sat_kw * 20.0
        else:
            total_score += 0.35 * 20.0  # Bayesian prior (7.0 pts)
    else:
        total_score += 0.35 * 20.0  # Bayesian prior (7.0 pts)

    # 5. Componente de Idioma Original (peso 10.0)
    media_langs = {k: v for k, v in media_vector.items() if k.startswith("lang_")}
    if media_langs:
        user_langs = {k: v for k, v in user_vector.items() if k.startswith("lang_")}
        max_u_lang = max(abs(v) for v in user_langs.values()) if user_langs else 1.0
        if max_u_lang == 0:
            max_u_lang = 1.0

        lang_match_sum = 0.0
        for k in media_langs:
            u_val = user_vector.get(k, 0.0)
            if u_val > 0:
                lang_match_sum += (u_val / max_u_lang)
            elif u_val < 0:
                lang_match_sum -= (abs(u_val) / max_u_lang)

        lang_ratio = max(0.0, min(1.0, lang_match_sum))
        if lang_ratio == 0.0:
            lang_ratio = 0.7 if "lang_en" in media_langs else 0.35
        total_score += lang_ratio * 10.0
    else:
        total_score += 0.7 * 10.0  # Prior neutro inglês (7.0 pts)

    score = round(max(0.0, min(100.0, total_score)), 1)

    # Identificar principais contribuintes para explicabilidade
    contributors = []
    for k in set(user_vector) & set(media_vector):
        contribution = user_vector[k] * media_vector[k]
        if contribution > 0:
            contributors.append((k, contribution))

    contributors.sort(key=lambda x: x[1], reverse=True)

    top_tags = []
    for k, _ in contributors[:3]:
        if k.startswith("genre_"):
            top_tags.append(f"🎬 {k.replace('genre_', '').title()}")
        elif k.startswith("cast_"):
            top_tags.append(f"🌟 Com {k.replace('cast_', '').title()}")
        elif k.startswith("director_"):
            top_tags.append(f"🎬 Dirigido por {k.replace('director_', '').title()}")
        elif k.startswith("type_"):
            pass
        elif k.startswith("lang_"):
            lang = k.replace("lang_", "").lower()
            if lang in NICHE_LANGUAGE_MAP:
                top_tags.append(NICHE_LANGUAGE_MAP[lang])
        else:
            clean_kw = k.replace("keyword_", "").lower()
            if clean_kw in CURATED_KEYWORDS_MAP:
                top_tags.append(CURATED_KEYWORDS_MAP[clean_kw])
            else:
                top_tags.append(f"🏷️ {clean_kw.title()}")

    return score, top_tags


def compute_omnimatch_score(
    user_vector: Dict[str, float],
    candidate_vector: Dict[str, float],
    user_embedding: Optional[List[float]] = None,
    candidate_embedding: Optional[List[float]] = None,
    target_persona: Optional[Dict[str, Any]] = None,
    active_personas: Optional[List[Dict[str, Any]]] = None,
    vote_average: float = 0.0,
    vote_count: int = 0,
    user_item_count: int = 0,
    candidate: Optional[Dict[str, Any]] = None,
    user_items: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Função canônica única de compatibilidade ("OmniMatch v3") servindo de forma unificada:
    /explore, /personas, /upcoming, /score/{media_type}/{tmdb_id} e /together.

    Componentes da Equação (Ideias 4, 5 e 11):
    1. S_sparse: Denominador fixo 100.0 com Soft Top-r Genre Matching e Saturação Suave
    2. S_dense: Similaridade de cosseno angular com embeddings de 4096D
    3. Potencial Latente Sinérgico:
       phi = 0.55 * S_dense + 0.30 * S_sparse + 0.15 * sqrt(max(0, S_dense * S_sparse))
    4. Delta Comportamental: Afeto explícito (+4.0 se favorito) e rewatch
    5. Curadoria Anti-Bomba (IMDb Weighted Rating):
       WR = (v*R + 150*6.5)/(v + 150). Se WR < 5.3, teto proporcional de qualidade
    6. Amortecimento de Cold-Start: n_u / (n_u + 8) quando n_u < 8 ('Match Preliminar')
    7. Multi-Persona: seleção precisa do polo temático mais aderente
    """
    from app.recommendation.embeddings import calculate_cosine_similarity

    matched_persona = None
    best_persona_id = "general"

    if target_persona:
        p_vec = target_persona.get("feature_vector") or user_vector
        p_emb = target_persona.get("centroid_embedding") or user_embedding
        matched_persona = target_persona
        best_persona_id = target_persona.get("id")

        aff_score, tags = calculate_match_score(p_vec, candidate_vector)
        if p_emb and candidate_embedding:
            sem_sim = calculate_cosine_similarity(p_emb, candidate_embedding)
            sem_score = sem_sim * 100.0
            phi = 0.55 * sem_score + 0.30 * aff_score + 0.15 * math.sqrt(max(0.0, sem_score * aff_score))
            if sem_sim >= 0.60 and "✨ Sintonia Semântica" not in tags:
                tags.append("✨ Sintonia Semântica")
        else:
            phi = aff_score
    elif active_personas and len(active_personas) > 1:
        best_phi = -1.0
        best_tags = []
        for p in active_personas:
            p_vec = p.get("feature_vector") or user_vector
            p_emb = p.get("centroid_embedding") or user_embedding
            p_aff, p_tags = calculate_match_score(p_vec, candidate_vector)
            if p_emb and candidate_embedding:
                sem_sim = calculate_cosine_similarity(p_emb, candidate_embedding)
                sem_score = sem_sim * 100.0
                p_phi = 0.55 * sem_score + 0.30 * p_aff + 0.15 * math.sqrt(max(0.0, sem_score * p_aff))
                if sem_sim >= 0.60 and "✨ Sintonia Semântica" not in p_tags:
                    p_tags.append("✨ Sintonia Semântica")
            else:
                p_phi = p_aff

            if p_phi > best_phi:
                best_phi = p_phi
                best_tags = p_tags
                matched_persona = p
                best_persona_id = p.get("id")

        phi = best_phi
        tags = best_tags
    else:
        aff_score, tags = calculate_match_score(user_vector, candidate_vector)
        if user_embedding and candidate_embedding:
            sem_sim = calculate_cosine_similarity(user_embedding, candidate_embedding)
            sem_score = sem_sim * 100.0
            phi = 0.55 * sem_score + 0.30 * aff_score + 0.15 * math.sqrt(max(0.0, sem_score * aff_score))
            if sem_sim >= 0.60 and "✨ Sintonia Semântica" not in tags:
                tags.append("✨ Sintonia Semântica")
        else:
            phi = aff_score

    # Delta Comportamental e Tags Explicativas
    rich_tags = tags
    if candidate:
        rich_tags = build_rich_explanation_tags(
            candidate=candidate,
            user_vector=user_vector,
            user_items=user_items or [],
            cand_emb=candidate_embedding,
            target_persona=matched_persona,
            fallback_tags=tags
        )
        if any(t.startswith("❤️") for t in rich_tags):
            phi = min(100.0, phi + 4.0)

        rewatch_pts, rewatch_tag = calculate_rewatch_boost(candidate_vector, candidate_embedding, user_items or [])
        if rewatch_pts > 0:
            phi = min(100.0, phi + rewatch_pts)
            if rewatch_tag and not any("reassistiu" in t for t in rich_tags):
                rich_tags = ([rewatch_tag] + rich_tags)[:3]

    # Gate Anti-Bomba (IMDb Weighted Rating)
    va = float(vote_average or 0.0)
    vc = float(vote_count or 0.0)
    wr = 6.5
    if vc > 0 and va > 0:
        wr = (vc * va + 150.0 * 6.5) / (vc + 150.0)
        if wr < 5.3:
            ceiling = max(35.0, wr * 10.0)
            phi = min(phi, ceiling)

    # Amortecimento Bayesiano de Cold-Start para novos usuários (n_u < 8)
    is_preliminary = False
    if user_item_count < 8:
        is_preliminary = True
        k0 = 8.0
        prior_quality = (wr * 10.0) if (vc > 0 and va > 0) else 65.0
        w_user = user_item_count / (user_item_count + k0)
        phi = w_user * phi + (1.0 - w_user) * prior_quality

    final_score = round(max(0.0, min(100.0, phi)), 1)

    return {
        "match_score": final_score,
        "match_tags": rich_tags,
        "is_preliminary": is_preliminary,
        "best_persona_id": best_persona_id,
        "raw_potential": round(phi, 1),
    }

CURATED_KEYWORDS_MAP: Dict[str, str] = {
    "time travel": "⏳ Viagem no Tempo",
    "space": "🌌 Exploração Espacial",
    "space travel": "🌌 Exploração Espacial",
    "dystopia": "🏙️ Distopia Futurista",
    "post-apocalyptic": "🌋 Pós-Apocalíptico",
    "artificial intelligence": "🤖 Inteligência Artificial",
    "robot": "🤖 Robôs & Tecnologia",
    "cyberpunk": "🌆 Universo Cyberpunk",
    "superhero": "🦸 Super-Heróis",
    "heist": "💰 Golpes & Estratégia",
    "magic": "✨ Magia & Fantasia",
    "witch": "🧙 Bruxaria & Misticismo",
    "revenge": "⚔️ Trama de Vingança",
    "psychological thriller": "🧠 Suspense Psicológico",
    "psychological": "🧠 Suspense Psicológico",
    "plot twist": "⚡ Reviravoltas no Enredo",
    "twist ending": "⚡ Final Surpreendente",
    "investigation": "🔍 Investigação Policial",
    "detective": "🕵️ Caso de Detetive",
    "serial killer": "🔪 Caçada a Assassino",
    "murder": "🔍 Mistério & Crime",
    "survival": "🏕️ Luta pela Sobrevivência",
    "zombie": "🧟 Apocalipse Zumbi",
    "vampire": "🧛 Universo Sobrenatural",
    "alien": "👽 Invasão Extraterrestre",
    "haunted house": "👻 Casa Mal-Assombrada",
    "conspiracy": "🕵️ Conspiração & Segredos",
    "espionage": "🕵️ Espionagem & Intriga",
    "spy": "🕵️ Ação & Espionagem",
    "courtroom": "⚖️ Drama de Tribunal",
    "dark comedy": "🎭 Humor Ácido",
    "satire": "🎭 Sátira Social",
    "coming of age": "🌱 Juventude & Amadurecimento",
    "based on novel": "📚 Baseado em Livro",
    "based on true story": "📰 Baseado em Fatos Reais",
    "biography": "📜 História Real",
    "sports": "🏆 Superação no Esporte",
}

NICHE_LANGUAGE_MAP: Dict[str, str] = {
    "ko": "🇰🇷 Dorama em Alta",
    "ja": "🇯🇵 Produção Japonesa",
    "pt": "🇧🇷 Cinema Nacional",
    "fr": "🇫🇷 Cinema Francês",
    "es": "🇪🇸 Produção Hispânica",
    "it": "🇮🇹 Cinema Italiano",
    "de": "🇩🇪 Cinema Alemão",
}

def calculate_rewatch_boost(
    candidate_vector: Dict[str, float],
    cand_emb: Optional[List[float]],
    user_items: Optional[List[Any]]
) -> Tuple[float, Optional[str]]:
    """
    Calcula um bônus direto no Match Score (0.0 a 6.0 pontos) quando o candidato
    possui forte afinidade semântica ou de features (gêneros/elenco/direção/temas)
    com uma obra que o usuário reassistiu (ou reassistiu episódios).
    Retorna (score_boost, rewatch_tag).
    """
    if not user_items:
        return 0.0, None

    from app.recommendation.embeddings import calculate_cosine_similarity
    from app.recommendation.vector_builder import build_sparse_vector

    best_boost = 0.0
    best_tag = None

    for item in user_items:
        if isinstance(item, dict):
            m_title = item.get("title")
            m_emb = item.get("embedding")
            media_obj = item.get("media_obj")
            rewatch_cnt = item.get("rewatch_count", 0) or 0
            ep_rewatches = item.get("ep_rewatches", 0) or 0
        else:
            try:
                media = getattr(item, "media", None)
                if not media:
                    continue
                m_title = getattr(media, "title", None)
                m_emb = getattr(media, "embedding", None)
                media_obj = media
                rewatch_cnt = getattr(item, "rewatch_count", 0) or 0
                episodes_list = getattr(item, "__dict__", {}).get("episodes") or []
                ep_rewatches = sum((getattr(ep, "rewatch_count", 0) or 0) for ep in episodes_list)
            except Exception:
                continue

        if not m_title:
            continue

        if rewatch_cnt <= 0 and ep_rewatches <= 0:
            continue

        # 1. Similaridade Semântica (se houver embeddings)
        sem_sim = 0.0
        if cand_emb and m_emb:
            sem_sim = calculate_cosine_similarity(cand_emb, m_emb)

        # 2. Afinidade de Features Esparsas (gênero, elenco, direção, keywords)
        feat_sim = 0.0
        if candidate_vector and media_obj:
            try:
                item_vec = build_sparse_vector(media_obj)
                if item_vec:
                    feat_sim = cosine_similarity(item_vec, candidate_vector)
            except Exception:
                feat_sim = 0.0

        effective_sim = max(sem_sim, feat_sim * 0.90)
        if effective_sim < 0.45:
            continue

        if rewatch_cnt > 0:
            # Obra inteira reassistida: bônus base de 3.5 até 6.0 pontos proporcional à similaridade e nº de rewatches
            base_pts = 3.5 + min(1.0 * (rewatch_cnt - 1), 2.5)
            boost = round(base_pts * min(1.0, effective_sim / 0.65), 1)
            tag = (
                f'🔄 Porque você reassistiu "{m_title}" ({rewatch_cnt + 1}x)'
                if rewatch_cnt > 1
                else f'🔄 Porque você reassistiu "{m_title}"'
            )
        else:
            # Episódios reassistidos individualmente: bônus moderado (até 2.5 pontos)
            base_pts = min(2.5, 0.8 + 0.35 * ep_rewatches)
            boost = round(base_pts * min(1.0, effective_sim / 0.65), 1)
            tag = f'🔄 Porque você reassistiu episódios de "{m_title}"'

        if boost > best_boost:
            best_boost = boost
            best_tag = tag if effective_sim >= 0.55 else None

    return best_boost, best_tag


def build_rich_explanation_tags(
    candidate: Dict[str, Any],
    user_vector: Dict[str, float],
    user_items: Optional[List[Any]] = None,
    cand_emb: Optional[List[float]] = None,
    target_persona: Optional[Dict[str, Any]] = None,
    fallback_tags: Optional[List[str]] = None,
) -> List[str]:
    """
    Gera tags de explicabilidade ricas, humanas e contextuais para a recomendação:
    1. 'Porque você curtiu / favoritou / reassistiu [Obra X]' (conexão semântica com itens de alto peso)
    2. Talentos de destaque (Diretor ou Ator favorito)
    3. Associação com Persona do usuário ('Para sua faceta Explorador Sci-Fi')
    4. Microtemas / Vibe da obra (Viagem no tempo, Distopia, etc.)
    5. Idioma de nicho (Coreano, Japonês, Nacional - NUNCA 'Áudio Inglês')
    """
    from app.recommendation.embeddings import calculate_cosine_similarity

    rich_tags: List[str] = []

    # 1. Conexão com obra que o usuário assistiu e amou (Item-to-Item Similarity)
    if user_items and cand_emb:
        best_sim = -1.0
        best_item_title = None
        best_rating = None
        best_is_fav = False
        best_rewatch_cnt = 0
        best_ep_rewatches = 0

        for item in user_items:
            if isinstance(item, dict):
                m_title = item.get("title")
                m_emb = item.get("embedding")
                rating = item.get("rating") or 0.0
                status = item.get("status") or ""
                is_fav = item.get("is_favorite", False)
                rewatch_cnt = item.get("rewatch_count", 0) or 0
                ep_rewatches = item.get("ep_rewatches", 0) or 0
            else:
                try:
                    media = getattr(item, "media", None)
                    if not media or not media.embedding or not media.title:
                        continue
                    m_title = media.title
                    m_emb = media.embedding
                    rating = getattr(item, "rating", None) or 0.0
                    status = getattr(item, "status", "")
                    is_fav = getattr(item, "is_favorite", False)
                    rewatch_cnt = getattr(item, "rewatch_count", 0) or 0
                    episodes_list = getattr(item, "__dict__", {}).get("episodes") or []
                    ep_rewatches = sum((getattr(ep, "rewatch_count", 0) or 0) for ep in episodes_list)
                except Exception:
                    continue

            if not m_title or not m_emb:
                continue

            # Prioriza obras com boa avaliação, completadas, favoritadas ou reassistidas
            if rating < 3.5 and status != "completed" and not is_fav and rewatch_cnt <= 0 and ep_rewatches <= 0:
                continue

            sim = calculate_cosine_similarity(cand_emb, m_emb)
            fav_mult = 1.25 if is_fav else 1.0
            rewatch_mult = (
                (1.25 + min(0.10 * (rewatch_cnt - 1), 0.25))
                if rewatch_cnt > 0
                else (1.12 if ep_rewatches > 0 else 1.0)
            )
            sim_effective = sim * fav_mult * rewatch_mult
            if sim_effective > best_sim:
                best_sim = sim_effective
                best_item_title = m_title
                best_rating = rating
                best_is_fav = is_fav
                best_rewatch_cnt = rewatch_cnt
                best_ep_rewatches = ep_rewatches

        # Se houver uma obra com afinidade semântica expressiva (>= 0.65)
        if best_item_title and best_sim >= 0.65:
            if best_is_fav and best_rewatch_cnt > 0:
                rich_tags.append(f'❤️🔄 Porque você favoritou e reassistiu "{best_item_title}"')
            elif best_is_fav:
                rich_tags.append(f'❤️ Porque você favoritou "{best_item_title}"')
            elif best_rewatch_cnt > 0:
                rich_tags.append(f'🔄 Porque você reassistiu "{best_item_title}"')
            elif best_ep_rewatches >= 2:
                rich_tags.append(f'🔄 Porque você reassistiu episódios de "{best_item_title}"')
            elif best_rating and best_rating >= 4.5:
                rich_tags.append(f'🍿 Porque você deu {best_rating:.0f}★ em "{best_item_title}"')
            elif best_rating and best_rating >= 4.0:
                rich_tags.append(f'🍿 Porque você curtiu "{best_item_title}"')
            else:
                rich_tags.append(f'🍿 Na mesma vibe de "{best_item_title}"')

    # 2. Diretor ou Ator que o usuário acompanha com peso alto
    directors = candidate.get("directors") or candidate.get("crew") or []
    if not directors and isinstance(candidate.get("credits"), dict):
        directors = [c.get("name") for c in candidate["credits"].get("crew", []) if c.get("job") in ("Director", "Creator")]

    cast_members = candidate.get("cast") or []
    if not cast_members and isinstance(candidate.get("credits"), dict):
        cast_members = [c.get("name") for c in candidate["credits"].get("cast", [])]

    found_talent = False
    for d in directors[:4]:
        name = d.get("name") if isinstance(d, dict) else str(d)
        if name and user_vector.get(f"director_{name.lower()}", 0.0) >= 1.5:
            rich_tags.append(f"🎬 Dirigido por {name}")
            found_talent = True
            break

    if not found_talent:
        for c in cast_members[:6]:
            name = c.get("name") if isinstance(c, dict) else str(c)
            if name and user_vector.get(f"cast_{name.lower()}", 0.0) >= 1.5:
                rich_tags.append(f"🌟 Estrelado por {name}")
                break

    # 3. Associação à Persona / Faceta de Gosto
    if target_persona:
        from app.recommendation.cluster_manager import is_valid_persona_name
        p_name = (target_persona.get("name") or "").strip()
        p_emoji = target_persona.get("emoji", "✨")
        if p_name and is_valid_persona_name(p_name):
            rich_tags.append(f"{p_emoji} Para sua faceta {p_name}")

    # 4. Microtemas / Vibe Narrativa a partir de keywords do candidato
    keywords = candidate.get("keywords_list") or candidate.get("keywords") or []
    for kw in keywords[:10]:
        kw_str = (kw.get("name") if isinstance(kw, dict) else str(kw)).lower()
        if kw_str in CURATED_KEYWORDS_MAP and len(rich_tags) < 3:
            badge = CURATED_KEYWORDS_MAP[kw_str]
            if badge not in rich_tags:
                rich_tags.append(badge)

    # 5. Idioma de Nicho Marcante (NUNCA Inglês)
    orig_lang = (candidate.get("original_language") or "").lower()
    if orig_lang in NICHE_LANGUAGE_MAP and len(rich_tags) < 3:
        badge = NICHE_LANGUAGE_MAP[orig_lang]
        if badge not in rich_tags:
            rich_tags.append(badge)

    # 6. Fallback com gênero elegante caso falte contexto
    if len(rich_tags) < 2:
        genres = candidate.get("genres") or []
        genre_names = []
        for g in genres:
            g_name = (g.get("name") if isinstance(g, dict) else str(g)).title()
            if g_name and g_name not in genre_names:
                genre_names.append(g_name)

        if len(genre_names) >= 2:
            rich_tags.append(f"🎬 {genre_names[0]} & {genre_names[1]}")
        elif len(genre_names) == 1:
            rich_tags.append(f"🎬 {genre_names[0]}")

    # 7. Fallback com tags originais se ainda estiver vazio
    if not rich_tags and fallback_tags:
        for t in fallback_tags:
            if "Áudio" not in t and t not in rich_tags:
                rich_tags.append(t)

    return rich_tags[:3]

