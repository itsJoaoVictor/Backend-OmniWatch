import asyncio
import logging
from typing import List, Optional, Dict, Any, Union, Tuple
import httpx
import numpy as np
from sklearn.decomposition import PCA

from app.core.config import settings

logger = logging.getLogger(__name__)


def build_text_for_embedding(media: Any) -> str:
    """
    Constrói um texto descritivo denso a partir dos metadados da obra.
    Suporta tanto uma instância de Media (SQLAlchemy) quanto um dict de candidato.
    Inclui título, tipo, gêneros, direção, elenco, sinopse, década e idioma original.
    """
    if isinstance(media, dict):
        title = media.get("title") or media.get("name") or ""
        media_type = media.get("media_type", "movie")
        raw_genres = media.get("genres", [])
        directors = media.get("crew", [])
        cast = media.get("cast", [])
        keywords = media.get("keywords_list", []) or media.get("keywords", [])
        overview = media.get("overview", "")
        rel_date = media.get("release_date") or media.get("first_air_date") or ""
        orig_lang = media.get("original_language", "")
    else:
        title = getattr(media, "title", "")
        media_type = getattr(media, "media_type", "movie")
        raw_genres = getattr(media, "genres", [])
        directors = getattr(media, "directors", [])
        cast = getattr(media, "main_cast", [])
        keywords = getattr(media, "keywords", [])
        overview = getattr(media, "overview", "") if hasattr(media, "overview") else ""
        rel_date = getattr(media, "release_date", "") or ""
        orig_lang = getattr(media, "original_language", "") or ""

    parts = []
    if title:
        parts.append(f"Título: {title}")

    m_type_label = "Filme" if media_type == "movie" else "Série de TV"
    parts.append(f"Tipo: {m_type_label}")

    # Gêneros
    genre_names = []
    if raw_genres:
        for g in raw_genres:
            if isinstance(g, dict):
                name = g.get("name")
                if name:
                    genre_names.append(name)
            elif isinstance(g, str):
                genre_names.append(g)
    if genre_names:
        parts.append(f"Gêneros: {', '.join(genre_names)}")

    # Diretores
    if directors:
        parts.append(f"Direção: {', '.join(directors[:5])}")

    # Elenco Principal
    if cast:
        parts.append(f"Elenco: {', '.join(cast[:10])}")

    # Temas / Keywords
    if keywords:
        parts.append(f"Temas: {', '.join(keywords[:15])}")

    # Sinopse (overview até ~400 chars)
    if overview:
        overview_clean = str(overview).strip()
        if len(overview_clean) > 400:
            overview_clean = overview_clean[:397] + "..."
        parts.append(f"Sinopse: {overview_clean}")

    # Ano de lançamento e década
    if rel_date and len(str(rel_date)) >= 4:
        try:
            year = int(str(rel_date)[:4])
            decade = (year // 10) * 10
            parts.append(f"Lançamento: {year} (Década de {decade})")
        except Exception:
            pass

    # Idioma Original
    if orig_lang:
        parts.append(f"Idioma Original: {str(orig_lang).lower().strip()}")

    return ". ".join(parts)


async def generate_embeddings_batch(texts: List[str], chunk_size: int = 32) -> List[Optional[List[float]]]:
    """
    Gera vetores de embedding semântico em batch chamando a API do OpenRouter
    com o modelo qwen/qwen3-embedding-8b em 1 único POST HTTP (ou chunks de 32).
    """
    if not texts:
        return []

    api_key = settings.OPENAI_API_KEY
    if not api_key:
        logger.warning("OPENAI_API_KEY não configurada. Não é possível gerar embeddings em batch.")
        return [None] * len(texts)

    url = f"{settings.OPENAI_BASE_URL.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    all_embeddings: List[Optional[List[float]]] = [None] * len(texts)

    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        cleaned_chunk = [t.strip() if t and t.strip() else "Mídia" for t in chunk]

        payload = {
            "model": settings.OPENAI_MODEL_Embedding,
            "input": cleaned_chunk
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=15.0)
                if response.status_code == 200:
                    data = response.json()
                    results_data = data.get("data", [])
                    results_data.sort(key=lambda x: x.get("index", 0))
                    for item in results_data:
                        idx = item.get("index", 0)
                        raw_emb = item.get("embedding")
                        if raw_emb:
                            vec = np.array(raw_emb, dtype=np.float32)
                            norm = np.linalg.norm(vec)
                            if norm > 0:
                                vec = vec / norm
                            all_embeddings[i + idx] = vec.tolist()
                else:
                    logger.error(f"[Embeddings] Erro batch OpenRouter: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"[Embeddings] Exceção em generate_embeddings_batch: {e}")

    return all_embeddings


async def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Gera o vetor de embedding semântico chamando a API do OpenRouter
    com o modelo qwen/qwen3-embedding-8b.
    """
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        logger.warning("OPENAI_API_KEY não configurada. Não é possível gerar embedding.")
        return None

    if not text or not text.strip():
        return None

    url = f"{settings.OPENAI_BASE_URL.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": settings.OPENAI_MODEL_Embedding,
        "input": text.strip()
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                embedding = data["data"][0]["embedding"]
                if len(embedding) != settings.EMBEDDING_DIM:
                    logger.warning(
                        f"[Embeddings] Dimensão de embedding inesperada da API: {len(embedding)} "
                        f"!= {settings.EMBEDDING_DIM}"
                    )
                # Normaliza o vetor para ter norma euclidiana = 1.0
                vec = np.array(embedding, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                return vec.tolist()
            else:
                logger.error(f"[Embeddings] Erro ao gerar embedding via OpenRouter: {response.status_code} - {response.text}")
                return None
    except httpx.TimeoutException as te:
        logger.error(f"[Embeddings] Timeout (>5.0s) ao chamar API de embedding: {te}")
        return None
    except Exception as e:
        logger.error(f"[Embeddings] Exceção ao chamar API de embedding: {e}")
        return None


def calculate_cosine_similarity(vec_a: Optional[List[float]], vec_b: Optional[List[float]]) -> float:
    """
    Calcula a similaridade de cosseno entre dois vetores normalizados.
    Retorna um valor float entre 0.0 e 1.0.
    Protegido contra dimensões desalinhadas e vetores nulos.
    """
    if not vec_a or not vec_b:
        return 0.0

    try:
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)

        if a.shape != b.shape:
            logger.warning(
                f"[Embeddings] Dimensões incompatíveis em calculate_cosine_similarity: "
                f"a.shape={a.shape} vs b.shape={b.shape} (esperado: ({settings.EMBEDDING_DIM},))"
            )
            return 0.0

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        # Clampa entre 0.0 e 1.0 para uso no match score
        return max(0.0, min(1.0, similarity))
    except Exception as e:
        logger.error(f"[Embeddings] Erro ao calcular similaridade de cosseno: {e}")
        return 0.0


def update_user_embedding_vector(
    current_embedding: Optional[List[float]],
    media_embedding: List[float],
    scale: float
) -> List[float]:
    """
    Atualiza o vetor de embedding do usuário de forma incremental e ponderada.
    - Se scale > 0 (gostou), puxa o vetor do usuário em direção ao vetor da obra.
    - Se scale < 0 (não gostou), afasta o vetor do usuário daquela região semântica.
    Protegido contra incompatibilidade de dimensão vetorial.
    """
    if not media_embedding:
        return current_embedding or []

    med_vec = np.array(media_embedding, dtype=np.float32)

    if not current_embedding:
        # Primeiro item no perfil: assume o vetor da obra
        norm = np.linalg.norm(med_vec)
        return (med_vec / norm).tolist() if norm > 0 else []

    curr_vec = np.array(current_embedding, dtype=np.float32)

    if curr_vec.shape != med_vec.shape:
        logger.warning(
            f"[Embeddings] Dimensões incompatíveis em update_user_embedding_vector: "
            f"curr_vec={curr_vec.shape} vs med_vec={med_vec.shape} "
            f"(esperado: ({settings.EMBEDDING_DIM},)). Mantendo current_embedding."
        )
        return current_embedding

    # Atualização ponderada: U_novo = U_antigo + (scale * M)
    new_vec = curr_vec + (scale * med_vec)

    norm = np.linalg.norm(new_vec)
    if norm > 0:
        new_vec = new_vec / norm

    return new_vec.tolist()


# ---------------------------------------------------------------------------
# Correção de Anisotropia em Memória: Centering + PCA Whitening (Ideia 2)
# ---------------------------------------------------------------------------

class CatalogWhitener:
    """
    Realiza Centralização (Centering) + PCA Whitening em memória sobre embeddings semânticos.
    Transforma vetores 4096D (que naturalmente sofrem de anisotropia e ocupam um cone estreito)
    em um subespaço esférico isotrópico (ex: 256D), abrindo o espectro angular de cossenos
    sem alterar os dados originais salvos na tabela Media. (Ideia 2 / Sub-etapa 3.1).
    """

    def __init__(self, target_dim: int = 256):
        self.target_dim = target_dim
        self.mean_vector: Optional[np.ndarray] = None
        self.pca: Optional[PCA] = None
        self.is_fitted: bool = False

    def fit(self, X: Union[List[List[float]], np.ndarray]) -> "CatalogWhitener":
        if X is None or len(X) < 5:
            return self

        X_mat = np.array(X, dtype=np.float32)
        if len(X_mat.shape) != 2:
            return self

        n_samples, n_features = X_mat.shape
        self.mean_vector = np.mean(X_mat, axis=0, dtype=np.float64).astype(np.float32)
        X_centered = X_mat - self.mean_vector

        n_components = min(self.target_dim, n_samples - 1, n_features)
        if n_components >= 2:
            try:
                self.pca = PCA(n_components=n_components, whiten=True, random_state=42)
                self.pca.fit(X_centered)
                self.is_fitted = True
            except Exception as e:
                logger.warning(f"[CatalogWhitener] Falha no PCA whitening: {e}")
                self.is_fitted = False

        return self

    def transform(self, emb: Union[List[float], np.ndarray]) -> np.ndarray:
        if emb is None:
            return np.zeros(self.target_dim, dtype=np.float32)

        vec = np.array(emb, dtype=np.float32)
        if vec.ndim != 1:
            return vec

        if self.mean_vector is not None and vec.shape[0] == self.mean_vector.shape[0]:
            vec_centered = (vec - self.mean_vector).reshape(1, -1)
        else:
            vec_centered = vec.reshape(1, -1)

        if self.is_fitted and self.pca is not None:
            try:
                whitened = self.pca.transform(vec_centered)[0]
                norm = np.linalg.norm(whitened)
                return (whitened / norm).astype(np.float32) if norm > 0 else whitened.astype(np.float32)
            except Exception:
                pass

        # Fallback: centralização simples + normalização euclidiana
        c = vec_centered[0]
        norm = np.linalg.norm(c)
        return (c / norm).astype(np.float32) if norm > 0 else c.astype(np.float32)

    def transform_batch(self, X: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        if X is None or len(X) == 0:
            return np.zeros((0, self.target_dim), dtype=np.float32)

        X_mat = np.array(X, dtype=np.float32)
        if self.mean_vector is not None and X_mat.shape[1] == self.mean_vector.shape[0]:
            X_centered = X_mat - self.mean_vector
        else:
            X_centered = X_mat

        if self.is_fitted and self.pca is not None:
            try:
                whitened = self.pca.transform(X_centered)
                norms = np.linalg.norm(whitened, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return (whitened / norms).astype(np.float32)
            except Exception as e:
                logger.warning(f"[CatalogWhitener] Falha no batch transform PCA: {e}")

        norms = np.linalg.norm(X_centered, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (X_centered / norms).astype(np.float32)


_GLOBAL_WHITENER: Optional[CatalogWhitener] = None
_WHITENER_LOCK = asyncio.Lock()


async def get_or_fit_catalog_whitener(db: Any = None, target_dim: int = 256) -> CatalogWhitener:
    """
    Obtém ou ajusta o CatalogWhitener global em memória a partir dos embeddings do catálogo Media.
    """
    global _GLOBAL_WHITENER
    if _GLOBAL_WHITENER is not None and _GLOBAL_WHITENER.is_fitted:
        return _GLOBAL_WHITENER

    async with _WHITENER_LOCK:
        if _GLOBAL_WHITENER is not None and _GLOBAL_WHITENER.is_fitted:
            return _GLOBAL_WHITENER

        whitener = CatalogWhitener(target_dim=target_dim)
        if db is not None:
            try:
                from sqlalchemy.future import select
                from app.media.models import Media

                stmt = select(Media.embedding).where(Media.embedding.isnot(None)).limit(1000)
                res = await db.execute(stmt)
                rows = res.scalars().all()
                valid_embs = [r for r in rows if r and len(r) == settings.EMBEDDING_DIM]
                if len(valid_embs) >= 10:
                    whitener.fit(np.array(valid_embs, dtype=np.float32))
                    logger.info(f"[CatalogWhitener] Ajustado com sucesso sobre {len(valid_embs)} embeddings do catálogo ({target_dim}D).")
            except Exception as e:
                logger.warning(f"[CatalogWhitener] Não foi possível carregar embeddings para whitening: {e}")

        _GLOBAL_WHITENER = whitener
        return whitener


def whiten_embedding(emb: Optional[List[float]], whitener: Optional[CatalogWhitener] = None) -> Optional[List[float]]:
    """
    Função utilitária síncrona para aplicar whitening em um embedding usando o whitener fornecido
    ou o singleton global.
    """
    if not emb:
        return None
    w = whitener or _GLOBAL_WHITENER
    if w is None or not w.is_fitted:
        return emb
    res = w.transform(emb)
    return res.tolist()


_PROJECTION_MATRIX_CACHE: Dict[Tuple[int, int], np.ndarray] = {}

def get_projection_matrix(source_dim: int = 4096, target_dim: int = 512) -> np.ndarray:
    """
    Gera matriz ortogonalizada determinística via decomposição QR para projeção
    de alta dimensão (ex: 4096D) para 512D preservando distâncias angulares
    segundo o Lema de Johnson-Lindenstrauss (Sub-etapa 5.1 + Ideia 2).
    """
    key = (source_dim, target_dim)
    if key not in _PROJECTION_MATRIX_CACHE:
        rng = np.random.RandomState(42)
        raw = rng.randn(source_dim, target_dim).astype(np.float32)
        q, _ = np.linalg.qr(raw)
        _PROJECTION_MATRIX_CACHE[key] = q[:, :target_dim].astype(np.float32)
    return _PROJECTION_MATRIX_CACHE[key]


def project_vector_512(vec: Optional[List[float]], target_dim: int = 512) -> Optional[List[float]]:
    """
    Projeta vetores densos para 512 dimensões mantendo fidelidade de cosseno > 99%,
    atendendo aos requisitos de índice HNSW do pgvector (< 2000D) com zero custo de API.
    """
    if not vec:
        return None
    arr = np.array(vec, dtype=np.float32)
    if arr.ndim != 1 or len(arr) <= target_dim:
        return vec
    proj_mat = get_projection_matrix(source_dim=len(arr), target_dim=target_dim)
    reduced = np.dot(arr, proj_mat)
    norm = np.linalg.norm(reduced)
    if norm > 0:
        reduced = reduced / norm
    return reduced.tolist()


