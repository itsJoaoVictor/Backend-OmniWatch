import os
import joblib
import numpy as np
from typing import Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.users.models import User
from app.tracking.models import UserListItem
from app.media.models import Media
from app.recommendation.matcher import calculate_match_score
from app.recommendation.embeddings import calculate_cosine_similarity
from app.recommendation.orchestrator import build_candidate_vector

MODEL_PATH = os.path.join(os.path.dirname(__file__), "lgbm_ranker.pkl")

class LightGBMRanker:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
            except Exception as e:
                print(f"Failed to load LightGBM model: {e}")

    def is_ready(self) -> bool:
        return self.model is not None

    def extract_features(self, user: User, candidate: Dict[str, Any], cand_emb=None) -> List[float]:
        """
        Extrai vetor numérico de features para o modelo LightGBM.
        """
        user_vector = user.feature_vector or {}
        candidate_vector = build_candidate_vector(candidate)
        
        # Feature 1: Afinidade de tags (Sparse Score)
        affinity_score, _ = calculate_match_score(user_vector, candidate_vector)
        
        # Feature 2: Similaridade Semântica (Embeddings)
        semantic_sim = 0.0
        if user.embedding and cand_emb:
            semantic_sim = calculate_cosine_similarity(user.embedding, cand_emb)
            
        # Feature 3: Popularidade
        popularity = float(candidate.get("popularity", 0.0))
        
        # Feature 4: Nota média (Vote Average)
        vote_average = float(candidate.get("vote_average", 0.0))
        
        # Feature 5: Media Type (1 = Movie, 0 = TV)
        is_movie = 1.0 if candidate.get("media_type") == "movie" else 0.0
        
        return [
            float(affinity_score),
            float(semantic_sim),
            float(popularity),
            float(vote_average),
            float(is_movie)
        ]

    def predict_score(self, user: User, candidate: Dict[str, Any], cand_emb=None) -> float:
        if not self.is_ready():
            user_vector = user.feature_vector or {}
            candidate_vector = build_candidate_vector(candidate)
            aff_score, _ = calculate_match_score(user_vector, candidate_vector)
            if user.embedding and cand_emb:
                sem_sim = calculate_cosine_similarity(user.embedding, cand_emb)
                return round(0.65 * (sem_sim * 100) + 0.35 * aff_score, 1)
            return aff_score
            
        features = self.extract_features(user, candidate, cand_emb)
        try:
            prob = self.model.predict_proba(np.array([features], dtype=np.float64))[0][1]
            return round(prob * 100.0, 1)
        except Exception:
            # Fallback seguro
            user_vector = user.feature_vector or {}
            candidate_vector = build_candidate_vector(candidate)
            aff_score, _ = calculate_match_score(user_vector, candidate_vector)
            if user.embedding and cand_emb:
                sem_sim = calculate_cosine_similarity(user.embedding, cand_emb)
                return round(0.65 * (sem_sim * 100) + 0.35 * aff_score, 1)
            return aff_score

async def train_ranker_model(db: AsyncSession):
    """
    Treina o modelo LightGBM varrendo o banco de dados.
    """
    import lightgbm as lgb
    import random
    
    print("Iniciando treinamento do LightGBM...")
    
    # 1. Buscar todos os usuários com histórico
    users_result = await db.execute(select(User))
    users = users_result.scalars().all()
    
    X_train = []
    y_train = []
    
    # Pre-carregar midias para amostragem negativa
    media_result = await db.execute(select(Media))
    all_media = media_result.scalars().all()
    if not all_media:
        print("Nenhuma mídia no banco para treinamento.")
        return
        
    ranker = LightGBMRanker()
    
    for user in users:
        if not user.feature_vector:
            continue
            
        # Buscar lista do usuário
        list_result = await db.execute(select(UserListItem).where(UserListItem.user_id == user.id))
        user_items = list_result.scalars().all()
        
        if not user_items:
            continue
            
        watched_media_ids = {item.media_id for item in user_items}
        
        for item in user_items:
            if not item.media:
                continue
                
            # Converter Media para o formato esperado pelo candidate
            candidate_dict = {
                "media_type": item.media.media_type,
                "genre_ids": [g.get("id") if isinstance(g, dict) else g for g in (item.media.genres or [])],
                "genres": item.media.genres,
                "original_language": item.media.original_language,
                "popularity": 50.0, # Placeholder médio se não tiver no DB
                "vote_average": 7.0,
            }
            
            # Positivo
            features_pos = ranker.extract_features(user, candidate_dict, cand_emb=item.media.embedding)
            X_train.append(features_pos)
            y_train.append(1)
            
            # Gerar 2 Negativos aleatórios (obras que o usuário NÃO assistiu)
            unwatched = [m for m in all_media if m.id not in watched_media_ids]
            if unwatched:
                neg_samples = random.sample(unwatched, min(2, len(unwatched)))
                for neg_m in neg_samples:
                    neg_dict = {
                        "media_type": neg_m.media_type,
                        "genre_ids": [g.get("id") if isinstance(g, dict) else g for g in (neg_m.genres or [])],
                        "genres": neg_m.genres,
                        "original_language": neg_m.original_language,
                        "popularity": 20.0,
                        "vote_average": 5.0,
                    }
                    features_neg = ranker.extract_features(user, neg_dict, cand_emb=neg_m.embedding)
                    X_train.append(features_neg)
                    y_train.append(0)

    if len(X_train) < 10:
        print("Dados insuficientes para treinar o modelo. Assista a mais algumas obras primeiro!")
        return
        
    print(f"Treinando com {len(X_train)} amostras (Positivos/Negativos)...")
    
    # 2. Treinar o modelo
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(np.array(X_train), np.array(y_train))
    
    # 3. Salvar
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo LightGBM salvo com sucesso em {MODEL_PATH}")

async def run_ranker_training_loop(interval_hours: int = 24):
    """
    Loop que roda indefinidamente em background (gerenciado pelo FastAPI Lifespan).
    Treina o modelo a cada X horas.
    """
    import asyncio
    from app.core.database import AsyncSessionLocal
    
    # Aguarda 5 minutos antes do primeiro treino para o servidor subir em paz
    await asyncio.sleep(5 * 60)
    
    while True:
        try:
            print("Iniciando treinamento automático agendado do LightGBM...")
            async with AsyncSessionLocal() as db:
                await train_ranker_model(db)
        except Exception as e:
            print(f"Erro no treinamento automático do ranker: {e}")
            
        await asyncio.sleep(interval_hours * 3600)
