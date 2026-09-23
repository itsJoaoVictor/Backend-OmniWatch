from typing import List, Dict, Any
from app.media.models import Media

WEIGHTS = {
    "genre": 1.5,
    "director": 1.2,
    "cast": 1.0,
    "media_type": 0.8,
    "keyword": 0.9,      # Fase 1: keywords temáticas
    "language": 1.0,     # Fase 2: idioma original (feature emergente)
}

def build_sparse_vector(media: Media) -> Dict[str, float]:
    """
    Converts a Media object into a sparse dictionary vector,
    where keys are features (e.g., "genre_action") and values are weights.
    
    Features included:
    - type_movie / type_tv (media type)
    - genre_* (genres in pt-BR)
    - director_* (director names)
    - cast_* (top cast names)
    - keyword_* (thematic keywords from TMDB) — Fase 1
    - lang_* (original language code) — Fase 2
    """
    vector = {}

    # Media Type
    if media.media_type:
        vector[f"type_{media.media_type}"] = WEIGHTS["media_type"]

    # Genres (One-hot effectively, but scaled by weight)
    if media.genres:
        for genre in media.genres:
            g_name = genre.get("name", "").lower() if isinstance(genre, dict) else str(genre).lower()
            if g_name:
                vector[f"genre_{g_name}"] = WEIGHTS["genre"]

    # Directors
    if media.directors:
        for director in media.directors:
            if director:
                vector[f"director_{director.lower()}"] = WEIGHTS["director"]

    # Main Cast
    if media.main_cast:
        for actor in media.main_cast:
            if actor:
                vector[f"cast_{actor.lower()}"] = WEIGHTS["cast"]

    # --- Fase 1: Keywords temáticas ---
    # Keywords são adicionadas com peso 0.9 — ligeiramente abaixo de gêneros.
    # Isso permite que temas recorrentes (ex: "super-herói", "viagem no tempo")
    # influenciem as recomendações sem dominar o perfil.
    if hasattr(media, 'keywords') and media.keywords:
        for kw in media.keywords:
            if kw:
                kw_clean = str(kw).lower().strip()
                if kw_clean:
                    vector[f"keyword_{kw_clean}"] = WEIGHTS["keyword"]

    # --- Fase 2: Idioma Original Emergente ---
    # NÃO criamos um "subtype_anime" artificial — deixamos as features emergir.
    # lang_ja + genre_animação = padrão de anime que acumula naturalmente.
    # O idioma tem peso 1.0 — igual ao cast, mas menor que gênero (1.5).
    # Só adicionamos ao vetor se o idioma não for "en" (inglês é o padrão e não
    # representa preferência específica — quase tudo tem versão em inglês).
    if hasattr(media, 'original_language') and media.original_language:
        lang = media.original_language.lower().strip()
        if lang and lang != "en":
            # Ex: lang_ja, lang_ko, lang_pt, lang_es, lang_zh
            vector[f"lang_{lang}"] = WEIGHTS["language"]

    return vector

def merge_vectors(base_vector: Dict[str, float], new_vector: Dict[str, float], scale: float = 1.0) -> Dict[str, float]:
    """
    Merges two vectors, applying a scaling factor to the new vector.
    Used for incremental profile updates.
    """
    result = base_vector.copy()
    for k, v in new_vector.items():
        result[k] = result.get(k, 0.0) + (v * scale)
    return result
