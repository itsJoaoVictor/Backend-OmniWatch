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
    Calcula o Match Score percentual (0.0 a 100.0) de forma modular calibrada
    e retorna (match_score, top_contributors).

    Pesos por categoria ativa na obra:
    - Gêneros compatíveis: até 45%
    - Tipo de mídia (Filme vs Série): até 10%
    - Elenco e Direção: até 25%
    - Palavras-chave e Temas: até 20%
    """
    if not user_vector or not media_vector:
        return 0.0, []

    total_score = 0.0
    active_weight_sum = 0.0

    # 1. Componente de Gêneros (peso 45)
    media_genres = {k: v for k, v in media_vector.items() if k.startswith("genre_")}
    if media_genres:
        active_weight_sum += 45.0
        user_genres = {k: v for k, v in user_vector.items() if k.startswith("genre_")}
        max_u_genre = max(abs(v) for v in user_genres.values()) if user_genres else 1.0
        if max_u_genre == 0:
            max_u_genre = 1.0

        g_match_sum = 0.0
        for k, v in media_genres.items():
            u_val = user_vector.get(k, 0.0)
            if u_val > 0:
                g_match_sum += (u_val / max_u_genre) * v
            elif u_val < 0:
                # Penaliza gêneros explicitamente rejeitados pelo usuário
                g_match_sum -= (abs(u_val) / max_u_genre) * v

        g_max_sum = sum(media_genres.values())
        if g_max_sum > 0:
            genre_ratio = max(0.0, g_match_sum / g_max_sum)
            total_score += genre_ratio * 45.0

    # 2. Componente de Tipo de Mídia (peso 10)
    media_types = {k: v for k, v in media_vector.items() if k.startswith("type_")}
    if media_types:
        active_weight_sum += 10.0
        t_match_sum = 0.0
        for k, v in media_types.items():
            u_val = user_vector.get(k, 0.0)
            if u_val > 0:
                # Usuário consome e aprecia este formato (filme ou série)
                t_match_sum += v
            elif u_val < 0:
                t_match_sum -= v

        t_max_sum = sum(media_types.values())
        if t_max_sum > 0:
            type_ratio = max(0.0, t_match_sum / t_max_sum)
            total_score += type_ratio * 10.0

    # 3. Componente de Elenco e Direção (peso 25)
    media_people = {k: v for k, v in media_vector.items() if k.startswith("cast_") or k.startswith("director_")}
    if media_people:
        active_weight_sum += 25.0
        user_people = {k: v for k, v in user_vector.items() if k.startswith("cast_") or k.startswith("director_")}
        max_u_person = max(abs(v) for v in user_people.values()) if user_people else 1.0
        if max_u_person == 0:
            max_u_person = 1.0

        matched_ratios = [
            max(0.0, user_vector[k] / max_u_person)
            for k in media_people
            if k in user_vector and user_vector[k] > 0
        ]
        # Se a obra tem apenas 1 pessoa, ela satura os 25 pontos. Se tiver mais, satura em 2 pessoas (12.5 cada).
        per_person_weight = 25.0 / min(2, len(media_people))
        people_score = min(25.0, sum(matched_ratios) * per_person_weight)
        total_score += people_score

    # 4. Componente de Palavras-chave e Temas (peso 20)
    media_kw = {k: v for k, v in media_vector.items() if k.startswith("keyword_")}
    if media_kw:
        active_weight_sum += 20.0
        user_kw = {k: v for k, v in user_vector.items() if k.startswith("keyword_")}
        max_u_kw = max(abs(v) for v in user_kw.values()) if user_kw else 1.0
        if max_u_kw == 0:
            max_u_kw = 1.0

        matched_kw_ratios = [
            max(0.0, user_vector[k] / max_u_kw)
            for k in media_kw
            if k in user_vector and user_vector[k] > 0
        ]
        # Se a obra tem apenas 1 keyword, satura os 20 pontos. Se tiver mais, satura em 2 keywords (10 cada).
        per_kw_weight = 20.0 / min(2, len(media_kw))
        kw_score = min(20.0, sum(matched_kw_ratios) * per_kw_weight)
        total_score += kw_score

    # Normalização em relação às categorias ativas na obra
    if active_weight_sum > 0:
        raw_percentage = (total_score / active_weight_sum) * 100.0
    else:
        raw_percentage = 0.0

    score = round(max(0.0, min(100.0, raw_percentage)), 1)

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
    1. 'Porque você curtiu [Obra X]' (conexão semântica com itens avaliados com nota alta)
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
        best_item = None

        for item in user_items:
            media = getattr(item, "media", None)
            if not media or not media.embedding or not media.title:
                continue

            # Prioriza obras com boa avaliação ou completadas
            rating = getattr(item, "rating", None) or 0.0
            status = getattr(item, "status", "")
            if rating < 3.5 and status != "completed":
                continue

            sim = calculate_cosine_similarity(cand_emb, media.embedding)
            if sim > best_sim:
                best_sim = sim
                best_item = item

        # Se houver uma obra com afinidade semântica expressiva (>= 0.65)
        if best_item and best_sim >= 0.65:
            m_title = best_item.media.title
            m_rating = getattr(best_item, "rating", None)
            if m_rating and m_rating >= 4.5:
                rich_tags.append(f'🍿 Porque você deu {m_rating:.0f}★ em "{m_title}"')
            elif m_rating and m_rating >= 4.0:
                rich_tags.append(f'🍿 Porque você curtiu "{m_title}"')
            else:
                rich_tags.append(f'🍿 Na mesma vibe de "{m_title}"')

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

