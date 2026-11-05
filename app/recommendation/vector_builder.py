from typing import List, Dict, Any, Union
from app.media.models import Media

WEIGHTS = {
    "genre": 1.5,
    "director": 1.2,
    "cast": 1.0,
    "media_type": 0.8,
    "keyword": 0.9,      # Fase 1: keywords temáticas
    "language": 1.0,     # Fase 2: idioma original (feature emergente)
}

# Mapeamento canônico de IDs do TMDB para lista de gêneros em pt-BR
TMDB_GENRE_ID_TO_CANONICAL: Dict[int, List[str]] = {
    # Filmes e TV comuns
    28: ["ação"],
    12: ["aventura"],
    16: ["animação"],
    35: ["comédia"],
    80: ["crime"],
    99: ["documentário"],
    18: ["drama"],
    10751: ["família"],
    14: ["fantasia"],
    36: ["história"],
    27: ["terror"],
    10402: ["música"],
    9648: ["mistério"],
    10749: ["romance"],
    878: ["ficção científica"],
    10770: ["cinema tv"],
    53: ["thriller"],
    10752: ["guerra"],
    37: ["faroeste"],
    # Séries de TV (Compostos e Específicos)
    10759: ["ação", "aventura"],              # Action & Adventure
    10765: ["ficção científica", "fantasia"],    # Sci-Fi & Fantasy
    10768: ["guerra", "política"],            # War & Politics
    10762: ["infantil"],                      # Kids
    10763: ["notícias"],                      # News
    10764: ["reality"],                       # Reality
    10766: ["novela"],                        # Soap
    10767: ["talk show"],                     # Talk
}

# Mapeamento de rótulos textuais (inglês e compostos) para eixos unificados em pt-BR
CANONICAL_GENRE_NAME_MAP: Dict[str, List[str]] = {
    # Tags compostas de Séries do TMDB
    "action & adventure": ["ação", "aventura"],
    "action and adventure": ["ação", "aventura"],
    "sci-fi & fantasy": ["ficção científica", "fantasia"],
    "scifi & fantasy": ["ficção científica", "fantasia"],
    "sci-fi and fantasy": ["ficção científica", "fantasia"],
    "war & politics": ["guerra", "política"],
    "war and politics": ["guerra", "política"],
    # Tags em inglês do TMDB
    "science fiction": ["ficção científica"],
    "sci-fi": ["ficção científica"],
    "fantasy": ["fantasia"],
    "action": ["ação"],
    "adventure": ["aventura"],
    "animation": ["animação"],
    "comedy": ["comédia"],
    "crime": ["crime"],
    "documentary": ["documentário"],
    "drama": ["drama"],
    "family": ["família"],
    "history": ["história"],
    "horror": ["terror"],
    "music": ["música"],
    "mystery": ["mistério"],
    "romance": ["romance"],
    "thriller": ["thriller"],
    "suspense": ["thriller"],
    "war": ["guerra"],
    "western": ["faroeste"],
    "soap": ["novela"],
    "kids": ["infantil"],
    "news": ["notícias"],
    "reality": ["reality"],
    "talk": ["talk show"],
}

def normalize_genre_to_canonical(genre_input: Any) -> List[str]:
    """
    Decompõe e normaliza qualquer representação de gênero (ID inteiro, dicionário TMDB
    ou string em inglês/composta) para a lista canônica unificada de eixos em pt-BR.
    
    Exemplos:
      - 10765 -> ["ficção científica", "fantasia"]
      - "Sci-Fi & Fantasy" -> ["ficção científica", "fantasia"]
      - "Action & Adventure" -> ["ação", "aventura"]
      - "War & Politics" -> ["guerra", "política"]
      - "Soap" -> ["novela"]
      - "Kids" -> ["infantil"]
      - "Ficção científica" -> ["ficção científica"]
      - {"id": 10759, "name": "Action & Adventure"} -> ["ação", "aventura"]
    """
    if not genre_input:
        return []

    # 1. Se for inteiro (ID direto do TMDB)
    if isinstance(genre_input, int):
        return TMDB_GENRE_ID_TO_CANONICAL.get(genre_input, [])

    # 2. Se for dicionário {"id": ..., "name": ...}
    if isinstance(genre_input, dict):
        g_id = genre_input.get("id")
        if isinstance(g_id, int) and g_id in TMDB_GENRE_ID_TO_CANONICAL:
            return TMDB_GENRE_ID_TO_CANONICAL[g_id]
        raw_name = str(genre_input.get("name", "")).strip().lower()
    else:
        raw_name = str(genre_input).strip().lower()

    if not raw_name:
        return []

    # 3. Mapeamento de nome composto / em inglês
    if raw_name in CANONICAL_GENRE_NAME_MAP:
        return CANONICAL_GENRE_NAME_MAP[raw_name]

    # 4. Caso a string contenha '&' ou 'and' não mapeados exatamente
    if "action" in raw_name and "adventure" in raw_name:
        return ["ação", "aventura"]
    if ("sci-fi" in raw_name or "science fiction" in raw_name) and "fantasy" in raw_name:
        return ["ficção científica", "fantasia"]
    if "war" in raw_name and "politics" in raw_name:
        return ["guerra", "política"]

    # 5. Já está em português ou normalizado
    return [raw_name]

def build_sparse_vector(media: Media) -> Dict[str, float]:
    """
    Converts a Media object into a sparse dictionary vector,
    where keys are features (e.g., "genre_action") and values are weights.
    
    Features included:
    - type_movie / type_tv (media type)
    - genre_* (canonical genres in pt-BR decomposed cross-media)
    - director_* (director names)
    - cast_* (top cast names)
    - keyword_* (thematic keywords from TMDB) — Fase 1
    - lang_* (original language code) — Fase 2
    """
    vector = {}

    # Media Type
    if media.media_type:
        vector[f"type_{media.media_type}"] = WEIGHTS["media_type"]

    # Genres (Canonical Decomposition em pt-BR)
    if media.genres:
        for genre in media.genres:
            canonical_genres = normalize_genre_to_canonical(genre)
            for g_name in canonical_genres:
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
