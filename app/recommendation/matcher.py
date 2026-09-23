import math
from typing import Dict, List, Tuple, Any

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
            top_tags.append(f"🧑‍🎤 {k.replace('cast_', '').title()}")
        elif k.startswith("director_"):
            top_tags.append(f"🎬 {k.replace('director_', '').title()}")
        elif k.startswith("type_"):
            pass  # omite tipo puro das tags visuais
        elif k.startswith("lang_"):
            lang = k.replace("lang_", "")
            lang_map = {
                "ja": "Japonês", 
                "ko": "Coreano", 
                "es": "Espanhol", 
                "fr": "Francês", 
                "it": "Italiano", 
                "de": "Alemão", 
                "pt": "Português", 
                "zh": "Chinês",
                "hi": "Indiano"
            }
            friendly_lang = lang_map.get(lang, lang.upper())
            top_tags.append(f"🗣️ Áudio {friendly_lang}")
        else:
            clean_kw = k.replace("keyword_", "").title()
            top_tags.append(f"🔥 {clean_kw}")

    return score, top_tags
