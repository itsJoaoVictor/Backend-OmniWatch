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

def _calculate_match_score_detailed(
    user_vector: Dict[str, float],
    media_vector: Dict[str, float],
) -> Tuple[float, List[str], Dict[str, Any]]:
    """
    Versão interna detalhada de `calculate_match_score`.
    Mantém exatamente os mesmos cálculos e tags, retornando também o breakdown por pilar.
    """
    empty_detail: Dict[str, Any] = {
        "genres": {"score": 0.0, "max_score": 40.0, "percent": 0, "is_prior": True, "matched_items": [], "unmatched_items": []},
        "format": {"score": 0.0, "max_score": 5.0, "percent": 0, "is_prior": True, "media_type": None, "label": "Formato"},
        "talent": {"score": 0.0, "max_score": 25.0, "percent": 0, "is_prior": True, "matched_directors": [], "matched_cast": []},
        "themes": {"score": 0.0, "max_score": 20.0, "percent": 0, "is_prior": True, "matched_items": []},
        "language": {"score": 0.0, "max_score": 10.0, "percent": 0, "is_prior": True, "code": None, "label": None},
    }
    if not user_vector or not media_vector:
        return 0.0, [], empty_detail

    total_score = 0.0

    # 1. Componente de Gêneros (peso 40.0)
    media_genres = {k: v for k, v in media_vector.items() if k.startswith("genre_")}
    genre_score = 0.0
    genre_is_prior = False
    matched_genres: List[str] = []
    unmatched_genres: List[str] = []
    if media_genres:
        user_genres = {k: v for k, v in user_vector.items() if k.startswith("genre_")}
        max_u_genre = max(abs(v) for v in user_genres.values()) if user_genres else 1.0
        if max_u_genre == 0:
            max_u_genre = 1.0

        genre_ratios = []
        matched_genre_weights = []
        for k in media_genres:
            u_val = user_vector.get(k, 0.0)
            ratio = (u_val / max_u_genre) if max_u_genre > 0 else 0.0
            genre_ratios.append(ratio)
            g_label = k.replace("genre_", "").title()
            if u_val > 0:
                matched_genre_weights.append((g_label, u_val))
            else:
                unmatched_genres.append(g_label)

        matched_genre_weights.sort(key=lambda x: x[1], reverse=True)
        matched_genres = [name for name, _ in matched_genre_weights[:4]]
        unmatched_genres = unmatched_genres[:3]

        genre_ratios.sort(reverse=True)
        # Soft Top-r Genre Matching (Ideia 4): 0.50 para 1º, 0.30 para 2º, 0.20 para 3º
        # Usa piso suave (0.75) no denominador para evitar que obras catalogadas com apenas 1 gênero
        # ganhem 100% dos 40 pts automaticamente em detrimento de obras multi-gênero.
        pos_weights = [0.50, 0.30, 0.20]
        weighted_genre_ratio = 0.0
        weight_sum = 0.0
        for i, r in enumerate(genre_ratios[:3]):
            w = pos_weights[i]
            weighted_genre_ratio += r * w
            weight_sum += w
        if weight_sum > 0:
            effective_weight_sum = max(weight_sum, 0.75)
            genre_ratio = max(0.0, min(1.0, weighted_genre_ratio / effective_weight_sum))
            genre_score = genre_ratio * 40.0
            total_score += genre_score
        else:
            genre_is_prior = True
    else:
        genre_score = 0.35 * 40.0  # Shrinkage prior (14.0 pts)
        genre_is_prior = True
        total_score += genre_score

    # 2. Componente de Formato Proporcional (Filme vs Série - peso 5.0)
    u_movie = max(0.0, user_vector.get("type_movie", 0.0))
    u_tv = max(0.0, user_vector.get("type_tv", 0.0))
    total_type = u_movie + u_tv
    is_movie = any(k == "type_movie" for k in media_vector)
    is_tv = any(k == "type_tv" for k in media_vector)
    detected_media_type = "movie" if is_movie else ("tv" if is_tv else None)
    format_label = "Filmes" if is_movie else ("Séries" if is_tv else "Formato")
    if total_type > 0:
        if is_movie:
            type_ratio = u_movie / total_type
        elif is_tv:
            type_ratio = u_tv / total_type
        else:
            type_ratio = 0.5
        format_score = max(0.0, min(1.0, type_ratio)) * 5.0
        format_is_prior = not (is_movie or is_tv)
        total_score += format_score
    else:
        format_score = 0.5 * 5.0  # Neutro (2.5 pts)
        format_is_prior = True
        total_score += format_score

    # 3. Componente de Talento: Elenco e Direção (peso 25.0)
    media_people = {k: v for k, v in media_vector.items() if k.startswith("cast_") or k.startswith("director_")}
    matched_directors: List[str] = []
    matched_cast: List[str] = []
    if media_people:
        user_people = {k: v for k, v in user_vector.items() if k.startswith("cast_") or k.startswith("director_")}
        max_u_person = max(abs(v) for v in user_people.values()) if user_people else 1.0
        if max_u_person == 0:
            max_u_person = 1.0

        matched_people = [
            (k, user_vector[k])
            for k in media_people
            if k in user_vector and user_vector[k] > 0
        ]
        matched_people.sort(key=lambda x: x[1], reverse=True)
        matched_directors = [k.replace("director_", "").title() for k, _ in matched_people if k.startswith("director_")][:3]
        matched_cast = [k.replace("cast_", "").title() for k, _ in matched_people if k.startswith("cast_")][:4]

        matched_ratios = [
            max(0.0, min(1.0, val / max_u_person))
            for _, val in matched_people
        ]
        matched_sum = sum(matched_ratios)
        if matched_sum > 0:
            # Saturação suave exponencial: 1 - exp(-1.2 * matched_sum)
            sat = 1.0 - math.exp(-1.2 * matched_sum)
            talent_score = sat * 25.0
            talent_is_prior = False
            total_score += talent_score
        else:
            talent_score = 0.18 * 25.0  # Bayesian prior calibrado para zero matches (4.5 pts)
            talent_is_prior = True
            total_score += talent_score
    else:
        talent_score = 0.18 * 25.0  # Bayesian prior calibrado (4.5 pts)
        talent_is_prior = True
        total_score += talent_score

    # 4. Componente de Palavras-chave e Temas (peso 20.0)
    from app.recommendation.vector_builder import NOISE_KEYWORDS
    media_kw = {
        k: v for k, v in media_vector.items()
        if k.startswith("keyword_") and k.replace("keyword_", "").lower().strip() not in NOISE_KEYWORDS
    }
    matched_themes: List[str] = []
    if media_kw:
        user_kw = {
            k: v for k, v in user_vector.items()
            if k.startswith("keyword_") and k.replace("keyword_", "").lower().strip() not in NOISE_KEYWORDS
        }
        max_u_kw = max(abs(v) for v in user_kw.values()) if user_kw else 1.0
        if max_u_kw == 0:
            max_u_kw = 1.0

        matched_kw_entries = [
            (k, user_kw[k])
            for k in media_kw
            if k in user_kw and user_kw[k] > 0
        ]
        matched_kw_entries.sort(key=lambda x: x[1], reverse=True)
        seen_themes = set()
        for k, _ in matched_kw_entries:
            clean_kw = k.replace("keyword_", "").lower().strip()
            if clean_kw in CURATED_KEYWORDS_MAP:
                curated = CURATED_KEYWORDS_MAP[clean_kw]
                theme_label = curated.split(" ", 1)[1].strip() if " " in curated else curated
            elif clean_kw in BREAKDOWN_KEYWORD_PT_MAP:
                theme_label = BREAKDOWN_KEYWORD_PT_MAP[clean_kw]
            else:
                theme_label = clean_kw.title()
            if theme_label and theme_label not in seen_themes:
                seen_themes.add(theme_label)
                matched_themes.append(theme_label)
            if len(matched_themes) >= 5:
                break

        matched_kw_ratios = [
            max(0.0, min(1.0, val / max_u_kw))
            for _, val in matched_kw_entries
        ]
        matched_kw_sum = sum(matched_kw_ratios)
        if matched_kw_sum > 0:
            # Saturação suave exponencial: 1 - exp(-1.2 * matched_kw_sum)
            sat_kw = 1.0 - math.exp(-1.2 * matched_kw_sum)
            themes_score = sat_kw * 20.0
            themes_is_prior = False
            total_score += themes_score
        else:
            themes_score = 0.18 * 20.0  # Bayesian prior calibrado para zero matches (3.6 pts)
            themes_is_prior = True
            total_score += themes_score
    else:
        themes_score = 0.18 * 20.0  # Bayesian prior calibrado (3.6 pts)
        themes_is_prior = True
        total_score += themes_score

    # 5. Componente de Idioma Original (peso 10.0)
    media_langs = {k: v for k, v in media_vector.items() if k.startswith("lang_")}
    lang_code: Optional[str] = None
    if media_langs:
        lang_code = next(iter(media_langs)).replace("lang_", "").lower()
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
        lang_is_prior = False
        if lang_ratio == 0.0:
            lang_ratio = 0.7 if "lang_en" in media_langs else 0.35
            lang_is_prior = True
        lang_score = lang_ratio * 10.0
        total_score += lang_score
    else:
        lang_score = 0.7 * 10.0  # Prior neutro inglês (7.0 pts)
        lang_is_prior = True
        total_score += lang_score

    lang_label = LANGUAGE_DISPLAY_MAP.get(lang_code, lang_code.upper()) if lang_code else None
    score = round(max(0.0, min(100.0, total_score)), 1)

    # Identificar principais contribuintes para explicabilidade
    contributors = []
    for k in set(user_vector) & set(media_vector):
        if k.startswith("keyword_") and k.replace("keyword_", "").lower().strip() in NOISE_KEYWORDS:
            continue
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

    detail: Dict[str, Any] = {
        "genres": {
            "score": round(genre_score, 1),
            "max_score": 40.0,
            "percent": int(round(max(0.0, min(100.0, (genre_score / 40.0) * 100.0)))),
            "is_prior": genre_is_prior,
            "matched_items": matched_genres,
            "unmatched_items": unmatched_genres,
        },
        "format": {
            "score": round(format_score, 1),
            "max_score": 5.0,
            "percent": int(round(max(0.0, min(100.0, (format_score / 5.0) * 100.0)))),
            "is_prior": format_is_prior,
            "media_type": detected_media_type,
            "label": format_label,
        },
        "talent": {
            "score": round(talent_score, 1),
            "max_score": 25.0,
            "percent": int(round(max(0.0, min(100.0, (talent_score / 25.0) * 100.0)))),
            "is_prior": talent_is_prior,
            "matched_directors": matched_directors,
            "matched_cast": matched_cast,
        },
        "themes": {
            "score": round(themes_score, 1),
            "max_score": 20.0,
            "percent": int(round(max(0.0, min(100.0, (themes_score / 20.0) * 100.0)))),
            "is_prior": themes_is_prior,
            "matched_items": matched_themes,
        },
        "language": {
            "score": round(lang_score, 1),
            "max_score": 10.0,
            "percent": int(round(max(0.0, min(100.0, (lang_score / 10.0) * 100.0)))),
            "is_prior": lang_is_prior,
            "code": lang_code,
            "label": lang_label,
        },
    }

    return score, top_tags, detail


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
    score, top_tags, _ = _calculate_match_score_detailed(user_vector, media_vector)
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
    include_breakdown: bool = False,
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
    used_sparse_detail: Optional[Dict[str, Any]] = None
    used_sem_score: Optional[float] = None
    persona_scores: List[Dict[str, Any]] = []

    if target_persona:
        p_vec = target_persona.get("feature_vector") or user_vector
        p_emb = target_persona.get("centroid_embedding") or user_embedding
        matched_persona = target_persona
        best_persona_id = target_persona.get("id")

        aff_score, tags, used_sparse_detail = _calculate_match_score_detailed(p_vec, candidate_vector)
        if p_emb and candidate_embedding:
            sem_sim = calculate_cosine_similarity(p_emb, candidate_embedding)
            sem_score = sem_sim * 100.0
            used_sem_score = sem_score
            phi = 0.55 * sem_score + 0.30 * aff_score + 0.15 * math.sqrt(max(0.0, sem_score * aff_score))
            if sem_sim >= 0.60 and "✨ Sintonia Semântica" not in tags:
                tags.append("✨ Sintonia Semântica")
        else:
            phi = aff_score

        if include_breakdown:
            from app.recommendation.cluster_manager import is_valid_persona_name
            p_name = (target_persona.get("name") or "").strip()
            if p_name and is_valid_persona_name(p_name):
                persona_scores.append({
                    "id": str(best_persona_id or ""),
                    "name": p_name,
                    "emoji": target_persona.get("emoji") or "✨",
                    "score": round(max(0.0, min(100.0, phi)), 1),
                    "is_primary": True,
                })
    elif active_personas and len(active_personas) > 1:
        # Multi-Persona: calcula potencial para cada cluster ativo e agrega via LogSumExp ponderado (Sub-etapa 3.3 / Ideia 7)
        phi_items = []
        tau = 0.08  # Temperatura para LogSumExp suave na escala 0-100

        for p in active_personas:
            p_vec = p.get("feature_vector") or user_vector
            p_emb = p.get("centroid_embedding") or user_embedding
            p_aff, p_tags, p_detail = _calculate_match_score_detailed(p_vec, candidate_vector)
            p_sem_score: Optional[float] = None
            if p_emb and candidate_embedding:
                sem_sim = calculate_cosine_similarity(p_emb, candidate_embedding)
                p_sem_score = sem_sim * 100.0
                p_phi = 0.55 * p_sem_score + 0.30 * p_aff + 0.15 * math.sqrt(max(0.0, p_sem_score * p_aff))
                if sem_sim >= 0.60 and "✨ Sintonia Semântica" not in p_tags:
                    p_tags.append("✨ Sintonia Semântica")
            else:
                p_phi = p_aff

            pi_k = float(p.get("pi_k") or (1.0 / len(active_personas)))
            phi_items.append((p_phi, pi_k, p, p_tags, p_detail, p_sem_score))

        # Encontra o cluster de maior afinidade absoluta
        phi_items.sort(key=lambda x: x[0], reverse=True)
        best_p_phi, best_pi, best_persona, best_tags, used_sparse_detail, used_sem_score = phi_items[0]
        best_persona_id = best_persona.get("id")
        matched_persona = best_persona

        if include_breakdown:
            from app.recommendation.cluster_manager import is_valid_persona_name
            for p_phi_val, _, p_obj, _, _, _ in phi_items:
                p_name = (p_obj.get("name") or "").strip()
                if p_name and is_valid_persona_name(p_name):
                    persona_scores.append({
                        "id": str(p_obj.get("id") or ""),
                        "name": p_name,
                        "emoji": p_obj.get("emoji") or "✨",
                        "score": round(max(0.0, min(100.0, p_phi_val)), 1),
                        "is_primary": p_obj.get("id") == best_persona_id,
                    })
                if len(persona_scores) >= 5:
                    break

        # Agregação LogSumExp Ponderada por pi_k:
        # S_lse = M + (1/tau) * ln(sum_k (pi_k / sum_pi) * exp(tau * (phi_k - M)))
        sum_pi = sum(item[1] for item in phi_items) or 1.0
        M = best_p_phi
        lse_inner = sum((item[1] / sum_pi) * math.exp(tau * (item[0] - M)) for item in phi_items)
        lse_score = M + (1.0 / tau) * math.log(max(1e-9, lse_inner))

        # Preserva 100% da força do pico da persona principal (unificando a escala entre Explore e Personas)
        phi = max(lse_score, best_p_phi)
        tags = best_tags
    else:
        aff_score, tags, used_sparse_detail = _calculate_match_score_detailed(user_vector, candidate_vector)
        if user_embedding and candidate_embedding:
            sem_sim = calculate_cosine_similarity(user_embedding, candidate_embedding)
            sem_score = sem_sim * 100.0
            used_sem_score = sem_score
            phi = 0.55 * sem_score + 0.30 * aff_score + 0.15 * math.sqrt(max(0.0, sem_score * aff_score))
            if sem_sim >= 0.60 and "✨ Sintonia Semântica" not in tags:
                tags.append("✨ Sintonia Semântica")
        else:
            phi = aff_score

    base_score = round(max(0.0, min(100.0, phi)), 1)
    base_source_label = "Base do perfil geral"
    if matched_persona:
        from app.recommendation.cluster_manager import is_valid_persona_name
        p_name = (matched_persona.get("name") or "").strip()
        if p_name and is_valid_persona_name(p_name):
            base_source_label = f"Base da faceta {p_name}"

    boosts: List[Dict[str, Any]] = []
    restraints: List[Dict[str, Any]] = []

    # Delta Comportamental e Tags Explicativas
    rich_tags = tags
    if candidate:
        rich_tags = build_rich_explanation_tags(
            candidate=candidate,
            user_vector=user_vector,
            user_items=user_items or [],
            cand_emb=candidate_embedding,
            target_persona=matched_persona,
            fallback_tags=tags,
            candidate_vector=candidate_vector
        )
        fav_tag = next((t for t in rich_tags if t.startswith("❤️")), None)
        if fav_tag:
            phi_before = phi
            phi = min(100.0, phi + 4.0)
            delta = round(phi - phi_before, 1)
            if include_breakdown and delta > 0:
                boosts.append({
                    "key": "favorite_connection",
                    "label": "Conexão com obra favoritada",
                    "detail": fav_tag.replace("❤️", "").replace("🔄", "").strip(),
                    "delta": delta,
                })

        rewatch_pts, rewatch_tag = calculate_rewatch_boost(candidate_vector, candidate_embedding, user_items or [])
        if rewatch_pts > 0:
            phi_before = phi
            phi = min(100.0, phi + rewatch_pts)
            delta = round(phi - phi_before, 1)
            if rewatch_tag and not any("reassistiu" in t for t in rich_tags):
                rich_tags = ([rewatch_tag] + rich_tags)[:3]
            if include_breakdown and delta > 0:
                boosts.append({
                    "key": "rewatch_boost",
                    "label": "Afinidade com obra reassistida",
                    "detail": rewatch_tag.replace("🔄", "").strip() if rewatch_tag else "Estilo alinhado a títulos que você reassistiu.",
                    "delta": delta,
                })

    # Gate Anti-Bomba (IMDb Weighted Rating)
    va = float(vote_average or 0.0)
    vc = float(vote_count or 0.0)
    wr = 6.5
    if vc > 0 and va > 0:
        wr = (vc * va + 150.0 * 6.5) / (vc + 150.0)
        if wr < 5.3:
            ceiling = max(35.0, wr * 10.0)
            phi_before = phi
            phi = min(phi, ceiling)
            delta = round(phi - phi_before, 1)
            if include_breakdown and delta < 0:
                restraints.append({
                    "key": "anti_bomb_gate",
                    "label": "Curadoria de qualidade",
                    "detail": f"A avaliação média ponderada ({wr:.1f}/10) aplicou um teto de proteção.",
                    "delta": delta,
                })

    # 1. Floor de Tração e Relevância Cultural para Obras Lançadas
    # Evita que obras com 1 a 10 votos (ex: produções obscuras) ranqueiem no topo
    is_movie = (candidate and candidate.get("media_type") == "movie") or (candidate and "title" in candidate and "first_air_date" not in candidate)
    r_date = (candidate.get("release_date") or candidate.get("first_air_date")) if candidate else None
    is_past_release = bool(r_date and len(str(r_date)) >= 4 and str(r_date)[:4].isdigit())

    if is_past_release and vc > 0:
        min_vc_threshold = 40.0 if is_movie else 25.0
        if vc < min_vc_threshold:
            traction = math.sqrt(max(1.0, vc) / min_vc_threshold)
            phi_before = phi
            phi = phi * (0.70 + 0.30 * traction)
            delta = round(phi - phi_before, 1)
            if include_breakdown and delta < 0:
                restraints.append({
                    "key": "low_vote_traction",
                    "label": "Poucos votos no TMDB",
                    "detail": f"Esta obra possui apenas {int(vc)} avaliações registradas até o momento.",
                    "delta": delta,
                })

    # Penalidade para idioma estrangeiro de nicho sem qualquer histórico na biblioteca
    cand_lang = (candidate.get("original_language") or "").lower() if candidate else ""
    if cand_lang and cand_lang not in ["en", "pt", "ja"]:
        user_lang_aff = user_vector.get(f"lang_{cand_lang}", 0.0)
        if user_lang_aff <= 0.0 and vc < 150:
            phi_before = phi
            phi = phi * 0.80
            delta = round(phi - phi_before, 1)
            if include_breakdown and delta < 0:
                lang_name = LANGUAGE_DISPLAY_MAP.get(cand_lang, cand_lang.upper())
                restraints.append({
                    "key": "niche_language",
                    "label": "Idioma pouco explorado",
                    "detail": f"{lang_name} ainda não aparece no seu histórico e tem poucos votos.",
                    "delta": delta,
                })

    # 2. Penalidade de Gêneros Residuais / Quase Inexistentes na Biblioteca
    # Ex: Doramas de Romance melodramático para usuários cujo perfil tem < 3% de Romance
    cand_gids = (candidate.get("genre_ids") or []) if candidate else []
    is_primarily_romance = bool(cand_gids and cand_gids[0] in [10749, 10766])
    if is_primarily_romance and user_item_count >= 30:
        romance_weight = user_vector.get("genre_romance", 0.0)
        genre_vals = [v for k, v in user_vector.items() if k.startswith("genre_")]
        top_genres_avg = (sum(sorted(genre_vals, reverse=True)[:3]) / 3.0) if genre_vals else 5.0
        if romance_weight < 0.55 * top_genres_avg:
            phi_before = phi
            phi = phi * 0.75
            delta = round(phi - phi_before, 1)
            if include_breakdown and delta < 0:
                restraints.append({
                    "key": "residual_genre",
                    "label": "Gênero pouco frequente no perfil",
                    "detail": "O foco em Romance destoa da predominância do seu histórico.",
                    "delta": delta,
                })

    # 3. Atenuação de Anacronismo Temporal para Comédias/Sitcoms
    # Sitcoms dos anos 70/80 (ex: The Jeffersons 1975) não devem competir com sitcoms modernas (B99, The Office)
    if is_past_release:
        release_year = int(str(r_date)[:4])
        if release_year < 1995:
            is_comedy = (cand_gids and 35 in cand_gids) or any(k in ["genre_comédia", "genre_comedy"] for k in candidate_vector)
            if is_comedy:
                has_vintage_comedy = False
                if user_items:
                    for it in user_items:
                        m_obj = getattr(it, "media_obj", None) or getattr(it, "media", None)
                        if m_obj:
                            m_date = getattr(m_obj, "release_date", None) or getattr(m_obj, "first_air_date", None)
                            if m_date and len(str(m_date)) >= 4 and str(m_date)[:4].isdigit() and int(str(m_date)[:4]) < 1995:
                                g_names = [str(g.get("name", "")).lower() for g in (getattr(m_obj, "genres", []) or []) if isinstance(g, dict)]
                                if "comédia" in g_names or "comedy" in g_names:
                                    has_vintage_comedy = True
                                    break
                if not has_vintage_comedy:
                    phi_before = phi
                    phi = phi * 0.82
                    delta = round(phi - phi_before, 1)
                    if include_breakdown and delta < 0:
                        restraints.append({
                            "key": "vintage_comedy",
                            "label": "Comédia clássica fora do padrão",
                            "detail": "Seu histórico tem poucas comédias anteriores a 1995.",
                            "delta": delta,
                        })

    # Amortecimento Bayesiano de Cold-Start para novos usuários (n_u < 8)
    is_preliminary = False
    if user_item_count < 8:
        is_preliminary = True
        k0 = 8.0
        prior_quality = (wr * 10.0) if (vc > 0 and va > 0) else 65.0
        w_user = user_item_count / (user_item_count + k0)
        phi_before = phi
        phi = w_user * phi + (1.0 - w_user) * prior_quality
        delta = round(phi - phi_before, 1)
        if include_breakdown:
            if delta < 0:
                restraints.append({
                    "key": "cold_start",
                    "label": "Perfil em construção",
                    "detail": f"Com {user_item_count} título(s) avaliado(s), o match ainda é suavizado pela nota média geral.",
                    "delta": delta,
                })
            elif delta > 0:
                boosts.append({
                    "key": "cold_start_prior",
                    "label": "Suavização por qualidade geral",
                    "detail": f"Com {user_item_count} título(s) na lista, a avaliação média ajudou a compor o match preliminar.",
                    "delta": delta,
                })

    final_score = round(max(0.0, min(100.0, phi)), 1)
    net_adjustment = round(final_score - base_score, 1)

    result: Dict[str, Any] = {
        "match_score": final_score,
        "match_tags": rich_tags,
        "is_preliminary": is_preliminary,
        "best_persona_id": best_persona_id,
        "raw_potential": round(phi, 1),
    }

    if include_breakdown and used_sparse_detail is not None:
        anchors = (
            find_top_anchor_matches(
                candidate=candidate,
                user_items=user_items or [],
                cand_emb=candidate_embedding,
                candidate_vector=candidate_vector,
                limit=3,
            )
            if candidate
            else []
        )
        pillars = _build_breakdown_pillars(
            sparse_detail=used_sparse_detail,
            sem_score=used_sem_score,
            candidate=candidate,
        )

        # Lacunas estruturais honestas (para responder "por que não foi 95%?")
        if final_score < 92.0:
            talent_detail = used_sparse_detail["talent"]
            themes_detail = used_sparse_detail["themes"]
            genres_detail = used_sparse_detail["genres"]

            if talent_detail["is_prior"]:
                restraints.append({
                    "key": "new_talent",
                    "label": "Elenco e direção inéditos",
                    "detail": "Nenhum ator ou diretor principal desta obra aparece com destaque no seu histórico.",
                    "delta": None,
                })
            elif talent_detail["percent"] < 45:
                restraints.append({
                    "key": "low_talent",
                    "label": "Elenco com baixa recorrência",
                    "detail": f"Apenas conexões pontuais de elenco e direção no seu histórico ({talent_detail['percent']}%).",
                    "delta": None,
                })

            if themes_detail["is_prior"]:
                restraints.append({
                    "key": "new_themes",
                    "label": "Temas ainda novos no perfil",
                    "detail": "As palavras-chave temáticas desta obra ainda não têm histórico forte na sua lista.",
                    "delta": None,
                })
            elif themes_detail["percent"] < 45:
                restraints.append({
                    "key": "low_themes",
                    "label": "Temas com baixa recorrência",
                    "detail": f"Pouca sobreposição temática com seus títulos mais frequentes ({themes_detail['percent']}%).",
                    "delta": None,
                })

            if genres_detail["unmatched_items"] and genres_detail["percent"] < 75:
                unmatched_str = ", ".join(genres_detail["unmatched_items"][:2])
                restraints.append({
                    "key": "unmatched_genres",
                    "label": "Gêneros menos frequentes",
                    "detail": f"Gêneros como {unmatched_str} têm pouco peso no seu perfil atual.",
                    "delta": None,
                })
            elif not genres_detail["is_prior"] and genres_detail["percent"] < 55:
                restraints.append({
                    "key": "moderate_genres",
                    "label": "Gêneros secundários no perfil",
                    "detail": f"Esses gêneros ocupam uma parcela menor do seu histórico ({genres_detail['percent']}%).",
                    "delta": None,
                })

            if used_sem_score is not None and used_sem_score < 65.0:
                restraints.append({
                    "key": "semantic_gap",
                    "label": "Atmosfera parcialmente diferente",
                    "detail": "O tom narrativo se distancia um pouco das obras centrais do seu gosto.",
                    "delta": None,
                })

            if not restraints and final_score < 88.0 and used_sem_score is not None and used_sem_score < 82.0:
                restraints.append({
                    "key": "semantic_moderate",
                    "label": "Atmosfera abaixo do pico central",
                    "detail": f"Boa sintonia narrativa ({round(used_sem_score)}%), mas um pouco distante das suas obras mais centrais.",
                    "delta": None,
                })

        restraints = restraints[:4]
        tier_label, summary = _build_match_verdict(
            final_score=final_score,
            is_preliminary=is_preliminary,
            pillars=pillars,
            restraints=restraints,
            anchors=anchors,
        )

        result["breakdown"] = {
            "score": final_score,
            "base_score": base_score,
            "base_source_label": base_source_label,
            "net_adjustment": net_adjustment,
            "tier_label": tier_label,
            "summary": summary,
            "is_preliminary": is_preliminary,
            "user_item_count": user_item_count,
            "semantic_available": used_sem_score is not None,
            "pillars": pillars,
            "anchors": anchors,
            "personas": persona_scores,
            "boosts": boosts,
            "restraints": restraints,
        }

    return result

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

BREAKDOWN_KEYWORD_PT_MAP: Dict[str, str] = {
    "anime": "Anime",
    "based on manga": "Baseado em Mangá",
    "manga": "Mangá",
    "slice of life": "Cotidiano",
    "seinen": "Seinen",
    "shonen": "Shonen",
    "shounen": "Shonen",
    "shoujo": "Shoujo",
    "josei": "Josei",
    "isekai": "Isekai",
    "mecha": "Mecha",
    "rivalry": "Rivalidade",
    "cooking": "Gastronomia",
    "food": "Gastronomia",
    "chef": "Culinária",
    "restaurant": "Restaurante",
    "friendship": "Amizade",
    "high school": "Vida Escolar",
    "school": "Vida Escolar",
    "school life": "Vida Escolar",
    "martial arts": "Artes Marciais",
    "sword fight": "Luta de Espadas",
    "swordplay": "Luta de Espadas",
    "samurai": "Samurai",
    "ninja": "Ninja",
    "demon": "Demônios",
    "monster": "Monstros",
    "supernatural": "Sobrenatural",
    "supernatural power": "Superpoderes",
    "super power": "Superpoderes",
    "family": "Relações Familiares",
    "family relationships": "Relações Familiares",
    "romantic comedy": "Comédia Romântica",
    "romance": "Romance",
    "love triangle": "Triângulo Amoroso",
    "workplace": "Ambiente de Trabalho",
    "politics": "Política",
    "war": "Guerra",
    "military": "Militar",
    "historical": "Histórico",
    "crime": "Crime",
    "organized crime": "Crime Organizado",
    "gangster": "Máfia",
    "mafia": "Máfia",
    "police": "Policial",
    "prison": "Prisão",
    "mystery": "Mistério",
    "thriller": "Suspense",
    "horror": "Terror",
    "gore": "Violência Explícita",
    "slasher": "Slasher",
    "ghost": "Fantasmas",
    "apocalypse": "Apocalipse",
    "time loop": "Loop Temporal",
    "multiverse": "Multiverso",
    "parallel universe": "Universo Paralelo",
    "virtual reality": "Realidade Virtual",
    "video game": "Videogames",
    "music": "Música",
    "band": "Banda & Música",
    "art": "Arte",
    "painting": "Arte & Pintura",
    "redemption": "Redenção",
    "grief": "Luto & Superação",
    "mental health": "Saúde Mental",
    "adolescence": "Adolescência",
    "teenager": "Juventude",
    "underdog": "Superação",
    "tournament": "Torneio",
    "competition": "Competição",
    "based on light novel": "Baseado em Light Novel",
    "based on video game": "Baseado em Jogo",
    "based on comic book": "Baseado em HQ",
    "based on play or musical": "Baseado em Peça",
    "based on short story": "Baseado em Conto",
    "adult animation": "Animação Adulta",
    "sitcom": "Sitcom",
    "anthology": "Antologia",
    "miniseries": "Minissérie",
    "buddy cop": "Dupla Policial",
    "anti hero": "Anti-Herói",
    "villain": "Vilões",
    "chosen one": "O Escolhido",
    "found family": "Família Escolhida",
    "road trip": "Viagem na Estrada",
    "survival horror": "Terror de Sobrevivência",
    "dark fantasy": "Fantasia Sombria",
    "epic": "Épico",
    "period drama": "Drama de Época",
    "small town": "Cidade Pequena",
    "secret identity": "Identidade Secreta",
    "double life": "Vida Dupla",
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

LANGUAGE_DISPLAY_MAP: Dict[str, str] = {
    "ja": "Produção Japonesa",
    "ko": "Produção Coreana",
    "pt": "Produção em Português",
    "en": "Produção em Inglês",
    "es": "Produção Hispânica",
    "fr": "Cinema Francês",
    "it": "Cinema Italiano",
    "de": "Cinema Alemão",
    "zh": "Produção Chinesa",
    "cn": "Produção Chinesa",
    "hi": "Cinema Indiano",
}


def _extract_user_item_fields(item: Any) -> Optional[Dict[str, Any]]:
    """Extrai campos padronizados de um item da lista do usuário (dict ou ORM UserListItem)."""
    if isinstance(item, dict):
        m_title = item.get("title")
        if not m_title:
            return None
        return {
            "tmdb_id": item.get("tmdb_id") or item.get("id"),
            "media_type": item.get("media_type"),
            "title": m_title,
            "poster_path": item.get("poster_path"),
            "embedding": item.get("embedding"),
            "media_obj": item.get("media_obj"),
            "feature_vector": item.get("feature_vector"),
            "rating": item.get("rating") or 0.0,
            "status": item.get("status") or "",
            "is_favorite": bool(item.get("is_favorite", False)),
            "rewatch_count": item.get("rewatch_count", 0) or 0,
            "ep_rewatches": item.get("ep_rewatches", 0) or 0,
        }

    try:
        media = getattr(item, "media", None)
        if not media or not getattr(media, "title", None):
            return None
        episodes_list = getattr(item, "__dict__", {}).get("episodes") or []
        ep_rewatches = sum((getattr(ep, "rewatch_count", 0) or 0) for ep in episodes_list)
        return {
            "tmdb_id": getattr(media, "tmdb_id", None),
            "media_type": getattr(media, "media_type", None),
            "title": media.title,
            "poster_path": getattr(media, "poster_path", None),
            "embedding": getattr(media, "embedding", None),
            "media_obj": media,
            "feature_vector": None,
            "rating": getattr(item, "rating", None) or 0.0,
            "status": getattr(item, "status", "") or "",
            "is_favorite": bool(getattr(item, "is_favorite", False)),
            "rewatch_count": getattr(item, "rewatch_count", 0) or 0,
            "ep_rewatches": ep_rewatches,
        }
    except Exception:
        return None


def _get_candidate_canonical_genres(
    candidate: Dict[str, Any],
    candidate_vector: Optional[Dict[str, float]] = None,
) -> set:
    from app.recommendation.vector_builder import normalize_genre_to_canonical
    if candidate_vector:
        canon = {k.replace("genre_", "") for k in candidate_vector if k.startswith("genre_")}
        if canon:
            return canon
    canon = set()
    for g in list(candidate.get("genre_ids", [])) + list(candidate.get("genres", [])):
        for c_g in normalize_genre_to_canonical(g):
            canon.add(c_g)
    return canon


def _is_anchor_genre_coherent(m_obj: Any, cand_genres_canon: set, sim: float) -> bool:
    """Verifica coerência temática de gênero entre uma obra-âncora e o candidato."""
    if m_obj is None or not cand_genres_canon:
        return True
    from app.recommendation.vector_builder import normalize_genre_to_canonical
    item_genres_canon = set()
    for g in getattr(m_obj, "genres", []) or []:
        for c_g in normalize_genre_to_canonical(g):
            item_genres_canon.add(c_g)
    if not item_genres_canon:
        return True
    if not (cand_genres_canon & item_genres_canon) and sim < 0.72:
        return False
    if ("animação" in item_genres_canon) != ("animação" in cand_genres_canon) and sim < 0.72:
        return False
    return True


def _format_anchor_reason(
    is_fav: bool,
    rewatch_cnt: int,
    ep_rewatches: int,
    rating: float,
    status: str,
) -> str:
    if is_fav and rewatch_cnt > 0:
        return "Favorito & Reassistido"
    if is_fav:
        return "Favorito"
    if rewatch_cnt > 1:
        return f"Reassistido {rewatch_cnt + 1}x"
    if rewatch_cnt == 1:
        return "Reassistido"
    if ep_rewatches >= 2:
        return "Episódios reassistidos"
    if rating and rating > 0:
        return f"{rating:.1f}★".replace(".0★", "★")
    return "Concluído" if status == "completed" else "Na sua lista"


def find_top_anchor_matches(
    candidate: Dict[str, Any],
    user_items: Optional[List[Any]] = None,
    cand_emb: Optional[List[float]] = None,
    candidate_vector: Optional[Dict[str, float]] = None,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """
    Retorna até `limit` obras da biblioteca do usuário com maior conexão ao candidato
    (sem autorreferência ao próprio título). Usa embeddings quando disponíveis e
    fallback por similaridade esparsa quando `cand_emb` não existe localmente.
    """
    if not user_items:
        return []

    from app.recommendation.embeddings import calculate_cosine_similarity
    from app.recommendation.vector_builder import build_sparse_vector

    cand_tmdb_id = candidate.get("id") or candidate.get("tmdb_id")
    cand_type = candidate.get("media_type")
    cand_genres_canon = _get_candidate_canonical_genres(candidate, candidate_vector)

    scored_anchors: List[Tuple[float, Dict[str, Any]]] = []
    seen_keys = set()

    for raw_item in user_items:
        fields = _extract_user_item_fields(raw_item)
        if not fields:
            continue

        m_tmdb_id = fields["tmdb_id"]
        m_type = fields["media_type"]
        if m_tmdb_id is None or m_type not in ("movie", "tv"):
            continue
        if cand_tmdb_id is not None and int(m_tmdb_id) == int(cand_tmdb_id) and m_type == cand_type:
            continue

        item_key = (int(m_tmdb_id), m_type)
        if item_key in seen_keys:
            continue

        rating = float(fields["rating"] or 0.0)
        status = fields["status"]
        is_fav = fields["is_favorite"]
        rewatch_cnt = int(fields["rewatch_count"] or 0)
        ep_rewatches = int(fields["ep_rewatches"] or 0)

        if rating < 3.5 and status != "completed" and not is_fav and rewatch_cnt <= 0 and ep_rewatches <= 0:
            continue

        m_emb = fields["embedding"]
        m_obj = fields["media_obj"]
        sim: Optional[float] = None

        if cand_emb and m_emb:
            sim = calculate_cosine_similarity(cand_emb, m_emb)
            if sim < 0.60:
                continue
        elif not cand_emb and candidate_vector:
            item_vec = fields["feature_vector"]
            if item_vec is None and m_obj is not None:
                try:
                    item_vec = build_sparse_vector(m_obj)
                except Exception:
                    item_vec = None
            if not item_vec:
                continue
            sim = cosine_similarity(item_vec, candidate_vector)
            if sim < 0.45:
                continue
        else:
            continue

        if not _is_anchor_genre_coherent(m_obj, cand_genres_canon, sim):
            continue

        shared_genres: List[str] = []
        if m_obj is not None and cand_genres_canon:
            from app.recommendation.vector_builder import normalize_genre_to_canonical
            item_genres_canon = set()
            for g in getattr(m_obj, "genres", []) or []:
                for c_g in normalize_genre_to_canonical(g):
                    item_genres_canon.add(c_g)
            if item_genres_canon:
                shared_genres = [g.title() for g in sorted(cand_genres_canon & item_genres_canon)[:2]]

        seen_keys.add(item_key)
        fav_mult = 1.12 if is_fav else 1.0
        rewatch_mult = (
            (1.15 + min(0.08 * (rewatch_cnt - 1), 0.15))
            if rewatch_cnt > 0
            else (1.08 if ep_rewatches > 0 else 1.0)
        )
        sim_effective = sim * fav_mult * rewatch_mult

        scored_anchors.append((
            sim_effective,
            {
                "tmdb_id": int(m_tmdb_id),
                "media_type": m_type,
                "title": fields["title"],
                "poster_path": fields["poster_path"],
                "similarity": int(round(max(0.0, min(0.99, sim)) * 100)),
                "is_favorite": is_fav,
                "rating": round(rating, 1) if rating > 0 else None,
                "rewatch_count": rewatch_cnt,
                "ep_rewatches": ep_rewatches,
                "reason_label": _format_anchor_reason(is_fav, rewatch_cnt, ep_rewatches, rating, status),
                "shared_genres": shared_genres,
                "priority_note": None,
            },
        ))

    scored_anchors.sort(key=lambda x: x[0], reverse=True)
    top_anchors = [anchor for _, anchor in scored_anchors[:limit]]
    for i, anchor in enumerate(top_anchors):
        if anchor["is_favorite"] or anchor["rewatch_count"] > 0:
            if any(top_anchors[j]["similarity"] > anchor["similarity"] for j in range(i + 1, len(top_anchors))):
                anchor["priority_note"] = (
                    "Maior peso (favorito)" if anchor["is_favorite"] else "Maior peso (reassistido)"
                )
    return top_anchors


def _status_from_percent(percent: int, strong_threshold: int = 70) -> str:
    if percent >= strong_threshold:
        return "strong"
    if percent >= 40:
        return "moderate"
    return "low"


def _build_breakdown_pillars(
    sparse_detail: Dict[str, Any],
    sem_score: Optional[float],
    candidate: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Traduz os detalhes esparsos e semânticos em pilares legíveis para o Raio-X."""
    pillars: List[Dict[str, Any]] = []

    if sem_score is not None:
        sem_pct = int(round(max(0.0, min(100.0, sem_score))))
        if sem_pct >= 75:
            sem_desc = "Enredo e atmosfera muito próximos das obras que definem seu gosto."
        elif sem_pct >= 55:
            sem_desc = "Boa afinidade narrativa com o conjunto da sua biblioteca."
        else:
            sem_desc = "Proposta narrativa um pouco distinta do centro do seu perfil."
        pillars.append({
            "key": "semantic",
            "label": "Atmosfera & Trama",
            "percent": sem_pct,
            "status": _status_from_percent(sem_pct),
            "impact": "high",
            "impact_label": "Alto impacto",
            "description": sem_desc,
            "chips": [],
        })

    g_detail = sparse_detail["genres"]
    g_pct = g_detail["percent"]
    g_chips = g_detail["matched_items"][:4]
    if g_detail["is_prior"]:
        g_desc = "Gêneros não informados; aplicada referência neutra."
    elif g_chips:
        if g_pct >= 65:
            g_desc = f"Alinhamento forte com {', '.join(g_chips[:3])}."
        else:
            g_desc = f"Combina com {', '.join(g_chips[:2])}, mas inclui gêneros menos comuns para você."
    else:
        g_desc = "Esses gêneros ainda aparecem pouco na sua biblioteca."
    pillars.append({
        "key": "genres",
        "label": "Gêneros",
        "percent": g_pct,
        "status": _status_from_percent(g_pct),
        "impact": "high",
        "impact_label": "Alto impacto",
        "description": g_desc,
        "chips": g_chips,
    })

    th_detail = sparse_detail["themes"]
    th_pct = th_detail["percent"]
    th_chips = th_detail["matched_items"][:5]
    if th_detail["is_prior"]:
        th_status = "novelty"
        th_desc = "Temas ainda novos ou sem peso forte no seu histórico."
    else:
        th_status = _status_from_percent(th_pct)
        if th_pct < 50:
            th_desc = f"Alguns temas em comum com o seu histórico ({', '.join(th_chips[:3])})."
        else:
            th_desc = f"Reúne temas recorrentes no seu perfil ({', '.join(th_chips[:3])})."
    pillars.append({
        "key": "themes",
        "label": "Temas & Vibe",
        "percent": th_pct,
        "status": th_status,
        "impact": "medium",
        "impact_label": "Médio impacto",
        "description": th_desc,
        "chips": th_chips,
    })

    t_detail = sparse_detail["talent"]
    t_pct = t_detail["percent"]
    t_chips = (t_detail["matched_directors"] + t_detail["matched_cast"])[:5]
    if t_detail["is_prior"]:
        t_status = "novelty"
        t_desc = "Elenco e direção sem nomes recorrentes no seu histórico."
    else:
        t_status = _status_from_percent(t_pct)
        if t_pct < 50:
            t_desc = f"Participação pontual de nomes já vistos no seu histórico ({', '.join(t_chips[:2])})."
        elif t_detail["matched_directors"] and t_detail["matched_cast"]:
            t_desc = f"Direção e elenco que você acompanha ({', '.join(t_chips[:2])})."
        elif t_detail["matched_directors"]:
            t_desc = f"Direção conhecida do seu perfil ({t_detail['matched_directors'][0]})."
        else:
            t_desc = f"Nomes frequentes no seu histórico ({', '.join(t_detail['matched_cast'][:2])})."
    pillars.append({
        "key": "talent",
        "label": "Elenco & Direção",
        "percent": t_pct,
        "status": t_status,
        "impact": "medium",
        "impact_label": "Médio impacto",
        "description": t_desc,
        "chips": t_chips,
    })

    l_detail = sparse_detail["language"]
    l_pct = l_detail["percent"]
    lang_code = l_detail["code"] or ((candidate.get("original_language") or "").lower().strip() if candidate else "")
    lang_label = l_detail["label"] or (LANGUAGE_DISPLAY_MAP.get(lang_code, lang_code.upper()) if lang_code else None)
    if not l_detail["is_prior"] and lang_label:
        l_desc = f"Idioma original alinhado às suas preferências ({lang_label})."
    elif lang_code == "en":
        l_desc = "Produção em inglês (referência neutra do perfil)."
    elif lang_label:
        l_desc = f"{lang_label} ainda pouco explorada no seu histórico."
    else:
        l_desc = "Idioma original não informado."
    pillars.append({
        "key": "language",
        "label": "Origem & Idioma",
        "percent": l_pct,
        "status": _status_from_percent(l_pct),
        "impact": "low",
        "impact_label": "Baixo impacto",
        "description": l_desc,
        "chips": [lang_label] if lang_label else [],
    })

    f_detail = sparse_detail["format"]
    f_pct = f_detail["percent"]
    m_type = f_detail["media_type"] or (candidate.get("media_type") if candidate else None)
    f_label = "Filmes" if m_type == "movie" else ("Séries" if m_type == "tv" else f_detail["label"])
    if f_detail["is_prior"]:
        f_desc = "Distribuição equilibrada entre filmes e séries no seu perfil."
    else:
        f_desc = f"{f_label} representam {f_pct}% da sua afinidade de formato."
    pillars.append({
        "key": "format",
        "label": "Formato",
        "percent": f_pct,
        "status": _status_from_percent(f_pct, strong_threshold=60),
        "impact": "low",
        "impact_label": "Baixo impacto",
        "description": f_desc,
        "chips": [f_label],
    })

    return pillars


def _build_match_verdict(
    final_score: float,
    is_preliminary: bool,
    pillars: List[Dict[str, Any]],
    restraints: List[Dict[str, Any]],
    anchors: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """Gera o selo qualitativo e um resumo humano determinístico em pt-BR."""
    if is_preliminary:
        tier_label = "Match Preliminar"
    elif final_score >= 88.0:
        tier_label = "Sintonia Excepcional"
    elif final_score >= 75.0:
        tier_label = "Alta Compatibilidade"
    elif final_score >= 60.0:
        tier_label = "Boa Compatibilidade"
    elif final_score >= 45.0:
        tier_label = "Compatibilidade Moderada"
    else:
        tier_label = "Fora da sua Zona de Conforto"

    pillar_names = {
        "semantic": "atmosfera narrativa",
        "genres": "gêneros",
        "themes": "temas",
        "talent": "elenco e direção",
        "language": "origem cultural",
        "format": "formato",
    }
    importance_weights = {
        "semantic": 1.45,
        "genres": 1.30,
        "themes": 1.00,
        "talent": 0.95,
        "language": 0.55,
    }
    strong_pillars = [
        pillar_names[p["key"]]
        for p in sorted(
            pillars,
            key=lambda x: x["percent"] * importance_weights.get(x["key"], 0.5),
            reverse=True,
        )
        if p["key"] in importance_weights and p["status"] == "strong"
    ]

    if strong_pillars:
        lead = " e ".join(strong_pillars[:2])
        base = f"Forte sintonia em {lead}"
        if anchors:
            base += f', puxada pela semelhança com "{anchors[0]["title"]}"'
    elif anchors:
        base = f'Afinidade conectada a títulos como "{anchors[0]["title"]}"'
    else:
        base = "Leitura mista em relação às suas preferências atuais"

    if final_score < 90.0 and restraints:
        top_restraint = restraints[0]
        if top_restraint["key"] == "new_talent":
            tail = ", equilibrada por elenco e direção ainda inéditos para você."
        elif top_restraint["key"] == "low_talent":
            tail = ", equilibrada por pouca recorrência no elenco e direção."
        elif top_restraint["key"] == "new_themes":
            tail = ", equilibrada por temas pouco explorados no seu perfil."
        elif top_restraint["key"] == "low_themes":
            tail = ", equilibrada por baixa recorrência temática."
        elif top_restraint["key"] == "cold_start":
            tail = ", suavizada enquanto seu perfil ganha mais títulos."
        elif top_restraint["key"] == "anti_bomb_gate":
            tail = ", limitada pela avaliação média geral da obra."
        else:
            tail = f", moderada por {top_restraint['label'].lower()}."
        return tier_label, base + tail

    return tier_label, base + "."

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
    candidate_vector: Optional[Dict[str, float]] = None,
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
        cand_genres_canon = _get_candidate_canonical_genres(candidate, candidate_vector)

        best_sim = -1.0
        best_item_title = None
        best_rating = None
        best_is_fav = False
        best_rewatch_cnt = 0
        best_ep_rewatches = 0

        for item in user_items:
            fields = _extract_user_item_fields(item)
            if not fields:
                continue

            m_title = fields["title"]
            m_emb = fields["embedding"]
            m_obj = fields["media_obj"]
            rating = fields["rating"]
            status = fields["status"]
            is_fav = fields["is_favorite"]
            rewatch_cnt = fields["rewatch_count"]
            ep_rewatches = fields["ep_rewatches"]

            if not m_title or not m_emb:
                continue

            # Prioriza obras com boa avaliação, completadas, favoritadas ou reassistidas
            if rating < 3.5 and status != "completed" and not is_fav and rewatch_cnt <= 0 and ep_rewatches <= 0:
                continue

            sim = calculate_cosine_similarity(cand_emb, m_emb)
            if sim < 0.60:
                continue

            # Verifica coerência temática de gênero entre a âncora e o candidato
            if not _is_anchor_genre_coherent(m_obj, cand_genres_canon, sim):
                continue

            fav_mult = 1.12 if is_fav else 1.0
            rewatch_mult = (
                (1.15 + min(0.08 * (rewatch_cnt - 1), 0.15))
                if rewatch_cnt > 0
                else (1.08 if ep_rewatches > 0 else 1.0)
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

    # 3. Associação à Persona / Faceta de Gosto com Gate Angular Estrito (Sub-etapa 3.3 / Ideia 7)
    if target_persona:
        from app.recommendation.cluster_manager import is_valid_persona_name
        p_name = (target_persona.get("name") or "").strip()
        p_emoji = target_persona.get("emoji", "✨")
        if p_name and is_valid_persona_name(p_name):
            # Limiar angular mínimo (theta_k - 25º percentil do cluster)
            theta_k = float(target_persona.get("theta_k") or 0.55)
            p_emb = target_persona.get("centroid_embedding")

            allow_persona_tag = True
            if cand_emb and p_emb:
                angular_sim = calculate_cosine_similarity(cand_emb, p_emb)
                if angular_sim < theta_k:
                    allow_persona_tag = False  # Bloqueado pelo Gate Angular Estrito!
            elif not cand_emb:
                p_vec = target_persona.get("feature_vector") or {}
                if p_vec and candidate_vector:
                    cand_aff, _ = calculate_match_score(p_vec, candidate_vector)
                    if cand_aff < 55.0:
                        allow_persona_tag = False

            if allow_persona_tag:
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

