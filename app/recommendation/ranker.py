import os
import math
import logging
import joblib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.users.models import User
from app.tracking.models import UserListItem
from app.media.models import Media
from app.recommendation.models import RecommendationInteractionEvent, UserDismissedRecommendation
from app.recommendation.matcher import calculate_match_score, compute_omnimatch_score
from app.recommendation.embeddings import calculate_cosine_similarity
from app.recommendation.orchestrator import build_candidate_vector

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "lgbm_ranker.pkl")
CALIBRATOR_PATH = os.path.join(os.path.dirname(__file__), "lgbm_calibrator.pkl")
ENABLE_LGBM_RANKER = False  # Desabilitado em favor da Equação Canônica Analítica Calibrada (OmniMatch v3) até conclusão da telemetria offline (Ideia 10)


def extract_ranker_features(
    user_vector: Dict[str, float],
    candidate: Dict[str, Any],
    user_embedding: Optional[List[float]] = None,
    cand_emb: Optional[List[float]] = None,
    base_omnimatch: Optional[float] = None,
    user_avg_year: float = 2015.0,
    user_top_languages: Optional[Set[str]] = None,
    user_streaming_providers: Optional[Set[str]] = None,
    user_items: Optional[List[Dict[str, Any]]] = None,
    is_leave_one_out: bool = False,
    item_weight: float = 1.0,
    active_personas: Optional[List[Dict[str, Any]]] = None,
) -> List[float]:
    """
    Extrai o vetor de 11 features numéricas para o LGBMRanker com protocolo Leave-One-Out estrito
    (Sub-etapa 6.2 + Ideia 10).
    """
    candidate_vector = build_candidate_vector(candidate)

    # 1. Ajuste Leave-One-Out para eliminar vazamento de dados durante o treino
    eff_user_vector = dict(user_vector or {})
    eff_user_embedding = user_embedding

    if is_leave_one_out and candidate_vector:
        # S_u^{(-i)} = S_u - w_i * m_i
        for tag, val in candidate_vector.items():
            if tag in eff_user_vector:
                new_v = eff_user_vector[tag] - (item_weight * float(val))
                if new_v > 0.001:
                    eff_user_vector[tag] = new_v
                else:
                    eff_user_vector.pop(tag, None)

        # Ajuste do centroide denso
        if user_embedding and cand_emb and user_items and len(user_items) > 1:
            try:
                n = float(len(user_items))
                u_arr = np.array(user_embedding, dtype=np.float32)
                c_arr = np.array(cand_emb, dtype=np.float32)
                adj = (n * u_arr - c_arr) / (n - 1.0)
                norm = np.linalg.norm(adj)
                if norm > 1e-6:
                    eff_user_embedding = (adj / norm).tolist()
                else:
                    eff_user_embedding = None
            except Exception:
                eff_user_embedding = None

    # Feature 1: OmniMatch v3 Score
    if base_omnimatch is not None and not is_leave_one_out:
        omni_score = float(base_omnimatch)
    else:
        try:
            omni_res = compute_omnimatch_score(
                user_vector=eff_user_vector,
                candidate_vector=candidate_vector,
                user_embedding=eff_user_embedding,
                candidate_embedding=cand_emb,
                active_personas=active_personas,
                vote_average=float(candidate.get("vote_average", 0.0) or 0.0),
                vote_count=int(candidate.get("vote_count", 0) or 0),
                user_item_count=len(user_items) if user_items else 10,
                candidate=candidate,
                user_items=user_items,
            )
            omni_score = float(omni_res.get("match_score", 50.0))
        except Exception:
            omni_score = 50.0

    # Feature 2: Similaridade Semântica (Embeddings)
    sem_sim = 0.0
    if eff_user_embedding and cand_emb:
        try:
            sem_sim = float(calculate_cosine_similarity(eff_user_embedding, cand_emb))
        except Exception:
            sem_sim = 0.0

    # Feature 3: Afinidade Esparsa
    sparse_score = 0.0
    if eff_user_vector and candidate_vector:
        try:
            aff, _ = calculate_match_score(eff_user_vector, candidate_vector)
            sparse_score = float(aff)
        except Exception:
            sparse_score = 0.0

    # Feature 4: Log da Popularidade
    pop = float(candidate.get("popularity", 0.0) or 0.0)
    candidate_popularity_log = float(math.log1p(max(0.0, pop)))

    # Feature 5: Nota Média (Vote Average)
    candidate_vote_average = float(candidate.get("vote_average", 0.0) or 0.0)

    # Feature 6: Log da Contagem de Votos
    vcount = int(candidate.get("vote_count", 0) or 0)
    candidate_vote_count_log = float(math.log1p(max(0, vcount)))

    # Feature 7: Media Type (1.0 filme, 0.0 série)
    is_movie = 1.0 if candidate.get("media_type") == "movie" else 0.0

    # Feature 8: Diferença de Ano de Lançamento
    date_str = str(candidate.get("release_date") or candidate.get("first_air_date") or "")
    cand_year = user_avg_year
    if len(date_str) >= 4 and date_str[:4].isdigit():
        try:
            cand_year = float(date_str[:4])
        except Exception:
            cand_year = user_avg_year
    release_year_diff = float(abs(cand_year - user_avg_year))

    # Feature 9: Correspondência de Idioma
    lang = str(candidate.get("original_language", "") or "").lower().strip()
    if user_top_languages and lang in user_top_languages:
        lang_match = 1.0
    elif eff_user_vector and f"lang_{lang}" in eff_user_vector:
        lang_match = 1.0
    else:
        lang_match = 0.0

    # Feature 10: Pertence a Franquia / Coleção
    has_collection = 1.0 if candidate.get("belongs_to_collection") else 0.0

    # Feature 11: Sobreposição de Provedores de Streaming BR
    cand_provs = candidate.get("watch_providers_br") or candidate.get("flatrate_providers_br") or []
    c_set = set()
    if isinstance(cand_provs, list):
        for p in cand_provs:
            if isinstance(p, dict):
                p_id = p.get("provider_id") or p.get("provider_name")
                if p_id:
                    c_set.add(str(p_id))
            elif isinstance(p, (str, int)):
                c_set.add(str(p))

    if user_streaming_providers and c_set:
        inter = len(c_set.intersection(user_streaming_providers))
        union = len(c_set.union(user_streaming_providers))
        streaming_br_overlap = float(inter / union) if union > 0 else 0.0
    else:
        streaming_br_overlap = 0.0

    return [
        omni_score,
        sem_sim,
        sparse_score,
        candidate_popularity_log,
        candidate_vote_average,
        candidate_vote_count_log,
        is_movie,
        release_year_diff,
        lang_match,
        has_collection,
        streaming_br_overlap,
    ]


class LightGBMRanker:
    _cached_model = None
    _cached_calibrator = None
    _cached_mtime = 0

    def __init__(self):
        self.model = None
        self.calibrator = None
        self._load_model()

    def _load_model(self):
        if not ENABLE_LGBM_RANKER:
            self.model = None
            self.calibrator = None
            return

        if not os.path.exists(MODEL_PATH) or not os.path.exists(CALIBRATOR_PATH):
            if LightGBMRanker._cached_mtime == 0:
                logger.info("ℹ️ [LightGBM] Arquivo de modelo ou calibrador não encontrado. Usando fallback analítico (OmniMatch v3).")
                LightGBMRanker._cached_mtime = -1
            self.model = None
            self.calibrator = None
            return

        current_mtime = max(os.path.getmtime(MODEL_PATH), os.path.getmtime(CALIBRATOR_PATH))
        if LightGBMRanker._cached_mtime == current_mtime:
            self.model = LightGBMRanker._cached_model
            self.calibrator = LightGBMRanker._cached_calibrator
            return

        try:
            self.model = joblib.load(MODEL_PATH)
            self.calibrator = joblib.load(CALIBRATOR_PATH)
            LightGBMRanker._cached_model = self.model
            LightGBMRanker._cached_calibrator = self.calibrator
            LightGBMRanker._cached_mtime = current_mtime
            logger.info("[LightGBM] Modelo LGBMRanker e Calibrador Isotonico carregados com sucesso!")
        except Exception as e:
            logger.error(f"[LightGBM] Falha ao carregar modelo LightGBM ou calibrador: {e}")
            self.model = None
            self.calibrator = None

    def is_ready(self) -> bool:
        if self.model is None or self.calibrator is None:
            self._load_model()
        return ENABLE_LGBM_RANKER and self.model is not None and self.calibrator is not None

    def extract_features(
        self,
        user: Any,
        candidate: Dict[str, Any],
        cand_emb=None,
        base_omnimatch: Optional[float] = None,
        user_items: Optional[List[Dict[str, Any]]] = None,
        user_avg_year: float = 2015.0,
        user_top_languages: Optional[Set[str]] = None,
        user_streaming_providers: Optional[Set[str]] = None,
        is_leave_one_out: bool = False,
        item_weight: float = 1.0,
        active_personas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[float]:
        user_vector = getattr(user, "feature_vector", None) if user else {}
        user_embedding = getattr(user, "embedding", None) if user else None

        return extract_ranker_features(
            user_vector=user_vector or {},
            candidate=candidate,
            user_embedding=user_embedding,
            cand_emb=cand_emb,
            base_omnimatch=base_omnimatch,
            user_avg_year=user_avg_year,
            user_top_languages=user_top_languages,
            user_streaming_providers=user_streaming_providers,
            user_items=user_items,
            is_leave_one_out=is_leave_one_out,
            item_weight=item_weight,
            active_personas=active_personas,
        )

    def predict_score(
        self,
        user: Any,
        candidate: Dict[str, Any],
        cand_emb=None,
        base_omnimatch: Optional[float] = None,
        user_items: Optional[List[Dict[str, Any]]] = None,
        user_avg_year: float = 2015.0,
        user_top_languages: Optional[Set[str]] = None,
        user_streaming_providers: Optional[Set[str]] = None,
        active_personas: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        Prediz score ordenado e calibrado [30.0, 98.0] via LGBMRanker + IsotonicRegression.
        Se indisponível, fallback transparente para OmniMatch v3.
        """
        if not self.is_ready():
            if base_omnimatch is not None:
                return float(base_omnimatch)
            user_vector = getattr(user, "feature_vector", None) if user else {}
            candidate_vector = build_candidate_vector(candidate)
            res = compute_omnimatch_score(
                user_vector=user_vector,
                candidate_vector=candidate_vector,
                user_embedding=getattr(user, "embedding", None) if user else None,
                candidate_embedding=cand_emb,
                active_personas=active_personas,
                vote_average=float(candidate.get("vote_average", 0.0) or 0.0),
                vote_count=int(candidate.get("vote_count", 0) or 0),
                user_item_count=len(user_items) if user_items else 10,
                candidate=candidate,
                user_items=user_items,
            )
            return float(res.get("match_score", 50.0))

        try:
            features = self.extract_features(
                user=user,
                candidate=candidate,
                cand_emb=cand_emb,
                base_omnimatch=base_omnimatch,
                user_items=user_items,
                user_avg_year=user_avg_year,
                user_top_languages=user_top_languages,
                user_streaming_providers=user_streaming_providers,
                is_leave_one_out=False,
                active_personas=active_personas,
            )
            raw = float(self.model.predict(np.array([features], dtype=np.float32))[0])
            calibrated = float(self.calibrator.predict(np.array([raw]))[0])
            return round(float(np.clip(calibrated, 30.0, 98.0)), 1)
        except Exception as e:
            logger.warning(f"[LightGBM] Erro na inferencia do ranker ({e}); fallback para OmniMatch v3.")
            if base_omnimatch is not None:
                return float(base_omnimatch)
            user_vector = getattr(user, "feature_vector", None) if user else {}
            candidate_vector = build_candidate_vector(candidate)
            res = compute_omnimatch_score(
                user_vector=user_vector,
                candidate_vector=candidate_vector,
                user_embedding=getattr(user, "embedding", None) if user else None,
                candidate_embedding=cand_emb,
                active_personas=active_personas,
                vote_average=float(candidate.get("vote_average", 0.0) or 0.0),
                vote_count=int(candidate.get("vote_count", 0) or 0),
                user_item_count=len(user_items) if user_items else 10,
                candidate=candidate,
                user_items=user_items,
            )
            return float(res.get("match_score", 50.0))


def optimize_ranker_hyperparameters(X: np.ndarray, y: np.ndarray, groups: List[int], n_trials: int = 15) -> Dict[str, Any]:
    """
    Otimiza hiperparâmetros do LGBMRanker com Optuna buscando maximizar correlação/NDCG (Ideia 10).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [5, 10],
                "num_leaves": trial.suggest_int("num_leaves", 7, 31),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 150),
                "min_child_samples": trial.suggest_int("min_child_samples", 1, max(2, min(10, len(X) // len(groups)))),
                "random_state": 42,
                "verbose": -1,
            }
            import lightgbm as lgb
            m = lgb.LGBMRanker(**params)
            m.fit(X, y, group=groups)
            preds = m.predict(X)
            from scipy.stats import spearmanr
            corr, _ = spearmanr(preds, y)
            return float(corr) if not math.isnan(corr) else 0.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        logger.info(f"[Optuna] Melhores hiperparametros encontrados: {study.best_params}")
        return study.best_params
    except Exception as e:
        logger.warning(f"[Optuna] Tuning não concluído ({e}); utilizando defaults estáticos.")
        return {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 100, "min_child_samples": 2}


async def train_ranker_model(db: AsyncSession, tune_hyperparameters: bool = False) -> bool:
    """
    Treina o LGBMRanker com protocolo Leave-One-Out estrito, rótulos graduados e calibração isotônica.
    """
    import lightgbm as lgb
    import random
    from sklearn.isotonic import IsotonicRegression

    logger.info("Iniciando treinamento do LGBMRanker Leak-Free com rótulos graduados...")

    # 1. Carregar todos os usuários
    users_result = await db.execute(select(User))
    users = users_result.scalars().all()
    if not users:
        logger.info("[LightGBM] Nenhum usuário encontrado para treino.")
        return False

    # 2. Pré-carregar pool de mídias para amostragem de negativos
    media_result = await db.execute(select(Media))
    all_media = media_result.scalars().all()
    media_by_id = {m.id: m for m in all_media}

    X_train: List[List[float]] = []
    y_train: List[int] = []
    query_groups: List[int] = []

    for user in users:
        if not user.feature_vector:
            continue

        # 2.1 Buscar itens da lista do usuário
        list_result = await db.execute(select(UserListItem).where(UserListItem.user_id == user.id))
        user_items = list_result.scalars().all()

        # 2.2 Buscar eventos de telemetria
        telemetry_result = await db.execute(
            select(RecommendationInteractionEvent).where(RecommendationInteractionEvent.user_id == user.id)
        )
        telemetry_events = telemetry_result.scalars().all()

        # 2.3 Buscar recomendações dispensadas
        dismissed_result = await db.execute(
            select(UserDismissedRecommendation).where(UserDismissedRecommendation.user_id == user.id)
        )
        dismissed_items = dismissed_result.scalars().all()

        if not user_items and not telemetry_events and not dismissed_items:
            continue

        # Métricas agregadas do usuário
        years = []
        user_top_languages: Set[str] = set()
        user_streaming_providers: Set[str] = set()
        watched_media_ids: Set[Any] = set()
        user_item_snapshots = []

        for it in user_items:
            watched_media_ids.add(it.media_id)
            m = it.media or media_by_id.get(it.media_id)
            if not m:
                continue
            d_str = str(m.release_date or "")
            if len(d_str) >= 4 and d_str[:4].isdigit():
                try:
                    years.append(float(d_str[:4]))
                except Exception:
                    pass
            if m.original_language:
                user_top_languages.add(str(m.original_language).lower().strip())
            provs = getattr(m, "watch_providers_br", None) or []
            if isinstance(provs, list):
                for p in provs:
                    if isinstance(p, dict):
                        pid = p.get("provider_id") or p.get("provider_name")
                        if pid:
                            user_streaming_providers.add(str(pid))
                    elif isinstance(p, (str, int)):
                        user_streaming_providers.add(str(p))

            user_item_snapshots.append({
                "id": m.tmdb_id,
                "media_type": m.media_type,
                "genres": m.genres or [],
                "vote_average": float(m.vote_average or 0.0),
                "rating": it.rating,
                "status": it.status,
            })

        user_avg_year = float(np.mean(years)) if years else 2015.0

        user_X: List[List[float]] = []
        user_y: List[int] = []
        seen_sample_keys = set()

        # 2.4 Rótulos a partir de UserListItem (com Leave-One-Out estrito)
        for it in user_items:
            m = it.media or media_by_id.get(it.media_id)
            if not m:
                continue

            sample_key = (m.media_type, m.tmdb_id)
            if sample_key in seen_sample_keys:
                continue

            rating_val = float(it.rating) if it.rating is not None else None
            # Determina rótulo graduado
            # 4: Favorito / Nota >= 4.5
            # 3: Nota 3.5 a 4.0 / Concluído
            # 2: Nota 2.5 a 3.0 / Watchlist
            # 0: Dropped / Nota <= 2.0
            if it.is_favorite or (rating_val is not None and (rating_val >= 4.5 or (rating_val > 5.0 and rating_val >= 9.0))):
                label = 4
                weight = 1.0
            elif (rating_val is not None and ((3.5 <= rating_val < 4.5) or (rating_val > 5.0 and 7.0 <= rating_val < 9.0))) or (it.status == "completed" and rating_val is None):
                label = 3
                weight = 0.8
            elif (rating_val is not None and ((2.5 <= rating_val < 3.5) or (rating_val > 5.0 and 5.0 <= rating_val < 7.0))) or (it.status in ("watching", "plan_to_watch")):
                label = 2
                weight = 0.5
            elif it.status == "dropped" or (rating_val is not None and (rating_val <= 2.0 or (rating_val > 5.0 and rating_val <= 4.0))):
                label = 0
                weight = 0.2
            else:
                label = 2
                weight = 0.5

            candidate_dict = {
                "id": m.tmdb_id,
                "media_type": m.media_type,
                "genres": m.genres or [],
                "genre_ids": [g.get("id") if isinstance(g, dict) else g for g in (m.genres or [])],
                "original_language": m.original_language or "",
                "popularity": float(getattr(m, "popularity", 0.0) or 0.0),
                "vote_average": float(getattr(m, "vote_average", 0.0) or 0.0),
                "vote_count": int(getattr(m, "vote_count", 0) or 0),
                "release_date": m.release_date,
                "belongs_to_collection": getattr(m, "belongs_to_collection", None),
                "watch_providers_br": getattr(m, "watch_providers_br", None),
            }

            feat = extract_ranker_features(
                user_vector=user.feature_vector or {},
                candidate=candidate_dict,
                user_embedding=user.embedding,
                cand_emb=m.embedding,
                user_avg_year=user_avg_year,
                user_top_languages=user_top_languages,
                user_streaming_providers=user_streaming_providers,
                user_items=user_item_snapshots,
                is_leave_one_out=True,
                item_weight=weight,
            )
            user_X.append(feat)
            user_y.append(label)
            seen_sample_keys.add(sample_key)

        # 2.5 Rótulos a partir de Telemetria
        for ev in telemetry_events:
            sample_key = (ev.media_type, ev.tmdb_id)
            if sample_key in seen_sample_keys:
                continue

            cand_m = next((m for m in all_media if m.tmdb_id == ev.tmdb_id and m.media_type == ev.media_type), None)
            cand_dict = {
                "id": ev.tmdb_id,
                "media_type": ev.media_type,
                "genres": cand_m.genres if cand_m else [],
                "genre_ids": [g.get("id") if isinstance(g, dict) else g for g in (cand_m.genres or [])] if cand_m else [],
                "original_language": cand_m.original_language if cand_m else "",
                "popularity": float(getattr(cand_m, "popularity", 0.0) or 0.0) if cand_m else 0.0,
                "vote_average": float(getattr(cand_m, "vote_average", 0.0) or 0.0) if cand_m else 0.0,
                "vote_count": int(getattr(cand_m, "vote_count", 0) or 0) if cand_m else 0,
                "release_date": cand_m.release_date if cand_m else None,
                "belongs_to_collection": getattr(cand_m, "belongs_to_collection", None) if cand_m else None,
                "watch_providers_br": getattr(cand_m, "watch_providers_br", None) if cand_m else None,
            }
            cand_emb = cand_m.embedding if cand_m else None

            ev_type = (ev.event_type or "").lower()
            if ev_type in ("click", "add_to_list"):
                label = 1
            elif ev_type == "dismiss":
                label = 0
            else:
                continue

            feat = extract_ranker_features(
                user_vector=user.feature_vector or {},
                candidate=cand_dict,
                user_embedding=user.embedding,
                cand_emb=cand_emb,
                user_avg_year=user_avg_year,
                user_top_languages=user_top_languages,
                user_streaming_providers=user_streaming_providers,
                user_items=user_item_snapshots,
                is_leave_one_out=False,
            )
            user_X.append(feat)
            user_y.append(label)
            seen_sample_keys.add(sample_key)

        # 2.6 Rótulos a partir de UserDismissedRecommendation (Hard Negative: y=0)
        for dis in dismissed_items:
            sample_key = (dis.media_type, dis.tmdb_id)
            if sample_key in seen_sample_keys:
                continue
            cand_m = next((m for m in all_media if m.tmdb_id == dis.tmdb_id and m.media_type == dis.media_type), None)
            cand_dict = {
                "id": dis.tmdb_id,
                "media_type": dis.media_type,
                "genres": cand_m.genres if cand_m else [],
                "genre_ids": [g.get("id") if isinstance(g, dict) else g for g in (cand_m.genres or [])] if cand_m else [],
                "original_language": cand_m.original_language if cand_m else "",
                "popularity": float(getattr(cand_m, "popularity", 0.0) or 0.0) if cand_m else 0.0,
                "vote_average": float(getattr(cand_m, "vote_average", 0.0) or 0.0) if cand_m else 0.0,
                "vote_count": int(getattr(cand_m, "vote_count", 0) or 0) if cand_m else 0,
                "release_date": cand_m.release_date if cand_m else None,
                "belongs_to_collection": getattr(cand_m, "belongs_to_collection", None) if cand_m else None,
                "watch_providers_br": getattr(cand_m, "watch_providers_br", None) if cand_m else None,
            }
            cand_emb = dis.embedding or (cand_m.embedding if cand_m else None)

            feat = extract_ranker_features(
                user_vector=user.feature_vector or {},
                candidate=cand_dict,
                user_embedding=user.embedding,
                cand_emb=cand_emb,
                user_avg_year=user_avg_year,
                user_top_languages=user_top_languages,
                user_streaming_providers=user_streaming_providers,
                user_items=user_item_snapshots,
                is_leave_one_out=False,
            )
            user_X.append(feat)
            user_y.append(0)
            seen_sample_keys.add(sample_key)

        # 2.7 Negativos Sintéticos (Amostragem de obras não consumidas)
        unwatched = [m for m in all_media if m.id not in watched_media_ids and (m.media_type, m.tmdb_id) not in seen_sample_keys]
        if unwatched:
            neg_samples = random.sample(unwatched, min(3, len(unwatched)))
            for neg_m in neg_samples:
                neg_dict = {
                    "id": neg_m.tmdb_id,
                    "media_type": neg_m.media_type,
                    "genres": neg_m.genres or [],
                    "genre_ids": [g.get("id") if isinstance(g, dict) else g for g in (neg_m.genres or [])],
                    "original_language": neg_m.original_language or "",
                    "popularity": float(getattr(neg_m, "popularity", 0.0) or 0.0),
                    "vote_average": float(getattr(neg_m, "vote_average", 0.0) or 0.0),
                    "vote_count": int(getattr(neg_m, "vote_count", 0) or 0),
                    "release_date": neg_m.release_date,
                    "belongs_to_collection": getattr(neg_m, "belongs_to_collection", None),
                    "watch_providers_br": getattr(neg_m, "watch_providers_br", None),
                }
                feat = extract_ranker_features(
                    user_vector=user.feature_vector or {},
                    candidate=neg_dict,
                    user_embedding=user.embedding,
                    cand_emb=neg_m.embedding,
                    user_avg_year=user_avg_year,
                    user_top_languages=user_top_languages,
                    user_streaming_providers=user_streaming_providers,
                    user_items=user_item_snapshots,
                    is_leave_one_out=False,
                )
                user_X.append(feat)
                user_y.append(0)

        # Verifica se o grupo do usuário tem pelo menos 2 amostras com variação de rótulo
        if len(user_y) >= 2 and len(set(user_y)) > 1:
            X_train.extend(user_X)
            y_train.extend(user_y)
            query_groups.append(len(user_y))

    if not query_groups or len(X_train) < 6:
        logger.info(f"[LightGBM] Dados insuficientes para treino ({len(X_train)} amostras, {len(query_groups)} grupos). Mantendo fallback.")
        return False

    logger.info(f"[LightGBM] Treinando com {len(X_train)} amostras distribuídas em {len(query_groups)} grupos de usuários...")

    X_arr = np.array(X_train, dtype=np.float32)
    y_arr = np.array(y_train, dtype=np.int32)

    params = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": max(1, min(2, len(X_train) // len(query_groups))),
    }

    if tune_hyperparameters:
        opt_params = optimize_ranker_hyperparameters(X_arr, y_arr, query_groups)
        params.update(opt_params)

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[5, 10],
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        importance_type="gain",
        random_state=42,
        verbose=-1,
    )
    model.fit(X_arr, y_arr, group=query_groups)

    # 3. Calibração Isotônica suave [30.0, 98.0]
    label_to_score = {4: 95.0, 3: 85.0, 2: 72.0, 1: 55.0, 0: 35.0}
    target_scores = np.array([label_to_score.get(int(y), 50.0) for y in y_arr], dtype=np.float32)
    raw_preds = model.predict(X_arr)

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=30.0, y_max=98.0)
    calibrator.fit(raw_preds, target_scores)

    # 4. Salvar modelo e calibrador
    joblib.dump(model, MODEL_PATH)
    joblib.dump(calibrator, CALIBRATOR_PATH)

    LightGBMRanker._cached_model = model
    LightGBMRanker._cached_calibrator = calibrator
    LightGBMRanker._cached_mtime = max(os.path.getmtime(MODEL_PATH), os.path.getmtime(CALIBRATOR_PATH))

    logger.info(f"✅ [LightGBM] Modelo LGBMRanker e Calibrador Isotonico treinados e salvos com sucesso em {MODEL_PATH}")
    return True


async def run_ranker_training_loop(interval_hours: int = 24):
    """
    Loop que roda indefinidamente em background (gerenciado pelo FastAPI Lifespan).
    Treina o modelo a cada X horas.
    """
    import asyncio
    from app.core.database import AsyncSessionLocal

    await asyncio.sleep(180)

    while True:
        if not ENABLE_LGBM_RANKER:
            logger.info("[LightGBM] Treinamento do ranker pausado via flag ENABLE_LGBM_RANKER.")
            await asyncio.sleep(interval_hours * 3600)
            continue

        try:
            logger.info("Iniciando treinamento automático agendado do LGBMRanker...")
            async with AsyncSessionLocal() as db:
                await train_ranker_model(db)
        except Exception as e:
            logger.warning(f"Erro no treinamento automático do ranker: {e}")

        await asyncio.sleep(interval_hours * 3600)
