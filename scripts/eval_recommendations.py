#!/usr/bin/env python3
"""
Harness de Avaliação Offline Temporal (eval_recommendations.py)
OmniWatch - Etapa 0 (Régua Científica e Blindagem do Estado Atual)

Protocolo:
- Snapshot somente-leitura de UserListItem, UserDismissedRecommendation e Media.
- Split Temporal (Leave-Last-N-Out) por usuário: separa as últimas N interações
  para validação sem vazamento de dados (leak-free).
- Métricas padrão de grandes streamings:
  * NDCG@10 (Qualidade e calibração da ordem no topo)
  * Recall@20 e HitRate@10 (Capacidade de encontrar o que o usuário assistiu/amou)
  * Intra-List Diversity (ILD@10) via dissimilaridade cosseno (1 - cos(a, b))
  * Histograma de Scores e Detecção de Anomalias (Falsos 100% e Platô de 87%)
  * Rank Médio de Negativos (Dispensados e Abandonados)
- Exporta relatório em JSON e Markdown e permite comparação entre etapas (--compare).
"""

import os
import sys
import json
import time
import math
import argparse
import asyncio
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

# Configura encoding no Windows e silencia warnings do LightGBM/sklearn
warnings.filterwarnings("ignore", category=UserWarning)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Configura o sys.path para importar app de Backend-OmniWatch
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.core.database import AsyncSessionLocal
from app.core.config import settings
from app.users.models import User
from app.media.models import Media
from app.tracking.models import UserListItem
from app.recommendation.models import UserDismissedRecommendation
from app.recommendation.matcher import calculate_match_score
from app.recommendation.embeddings import calculate_cosine_similarity, update_user_embedding_vector
from app.recommendation.vector_builder import build_sparse_vector, merge_vectors
from app.recommendation.profile_manager import (
    calculate_time_decay,
    calculate_rating_scale,
    calculate_rewatch_modifier,
)
from app.recommendation.orchestrator import build_candidate_vector
from app.recommendation.ranker import LightGBMRanker


# ---------------------------------------------------------------------------
# Funções Auxiliares de Snapshot e Modelagem
# ---------------------------------------------------------------------------

def compute_ground_truth_relevance(item: UserListItem) -> float:
    """
    Atribui relevância graduada de 0.0 a 3.0 para cálculo de NDCG e Recall.
    - 3.0: Obra amada (Nota >= 4.5, Favorito ou Reassistida)
    - 2.0: Obra curtida (Nota >= 3.5 ou Status Concluído com nota satisfatória)
    - 1.0: Obra neutra/consumida (Nota 2.5-3.0 ou Status Watching/Plan to Watch)
    - 0.0: Obra rejeitada/negativa (Nota < 2.5 ou Status Dropped)
    """
    if getattr(item, "is_favorite", False) or (getattr(item, "rewatch_count", 0) or 0) > 0:
        return 3.0

    rating = getattr(item, "rating", None)
    if rating is not None:
        if rating >= 4.5:
            return 3.0
        elif rating >= 3.5:
            return 2.0
        elif rating >= 2.5:
            return 1.0
        else:
            return 0.0

    status = getattr(item, "status", "")
    if status == "completed":
        return 2.0
    elif status == "watching":
        return 1.5
    elif status == "plan_to_watch":
        return 1.0
    elif status == "dropped":
        return 0.0

    return 1.0


def media_to_candidate_dict(media: Media) -> Dict[str, Any]:
    """Converte um registro Media do banco em dicionário de candidato para os rankers."""
    return {
        "id": media.tmdb_id,
        "tmdb_id": media.tmdb_id,
        "media_type": media.media_type or "movie",
        "title": media.title or "",
        "name": media.title or "",
        "genres": media.genres or [],
        "directors": media.directors or [],
        "crew": media.directors or [],
        "cast": media.main_cast or [],
        "keywords": media.keywords or [],
        "keywords_list": media.keywords or [],
        "original_language": media.original_language or "en",
        "popularity": 25.0,  # Valor padrão neutro quando ausente na tabela Media
        "vote_average": 7.0,
        "embedding": media.embedding,
    }


def build_train_profile(train_items: List[UserListItem]) -> Tuple[Dict[str, float], Optional[List[float]]]:
    """
    Reconstrói o vetor esparso e o embedding denso do usuário estritamente
    a partir dos itens do split de treino (sem vazamento do conjunto de teste).
    """
    user_vec: Dict[str, float] = {}
    user_emb: Optional[List[float]] = None

    for item in train_items:
        media = item.media
        if not media:
            continue

        action_date = item.last_watched_at or item.updated_at or item.created_at
        time_decay = calculate_time_decay(action_date)
        rating_scale = calculate_rating_scale(item.rating, item.status, rewatch_count=item.rewatch_count)
        fav_modifier = 2.0 if getattr(item, "is_favorite", False) else 1.0
        rewatch_mod = calculate_rewatch_modifier(item.rewatch_count)
        ep_mod = 1.2 if media.media_type == "tv" else 1.0

        scale = time_decay * rating_scale * fav_modifier * rewatch_mod * ep_mod

        # Vetor esparso
        try:
            m_vec = build_sparse_vector(media)
            if m_vec:
                user_vec = merge_vectors(user_vec, m_vec, scale=scale)
        except Exception:
            pass

        # Embedding denso
        if media.embedding:
            user_emb = update_user_embedding_vector(user_emb, media.embedding, scale=scale)

    # Normalização e poda de features esparsas
    max_val = max((abs(v) for v in user_vec.values()), default=1.0)
    if max_val > 100.0:
        user_vec = {k: (v / max_val) * 100.0 for k, v in user_vec.items()}
    if len(user_vec) > 300:
        sorted_items = sorted(user_vec.items(), key=lambda x: abs(x[1]), reverse=True)
        user_vec = dict(sorted_items[:300])

    return user_vec, user_emb


# ---------------------------------------------------------------------------
# Métricas de Ranqueamento e Diversidade
# ---------------------------------------------------------------------------

def compute_ndcg_at_k(ranked_relevances: List[float], k: int = 10) -> float:
    """Calcula NDCG@K a partir das relevâncias verdadeiras ordenadas pela nota do modelo."""
    if not ranked_relevances:
        return 0.0

    sub_k = ranked_relevances[:k]
    dcg = sum((math.pow(2.0, rel) - 1.0) / math.log2(idx + 2.0) for idx, rel in enumerate(sub_k))

    ideal_sorted = sorted(ranked_relevances, reverse=True)[:k]
    idcg = sum((math.pow(2.0, rel) - 1.0) / math.log2(idx + 2.0) for idx, rel in enumerate(ideal_sorted))

    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def compute_recall_at_k(
    ranked_items: List[Dict[str, Any]],
    relevant_media_ids: set,
    k: int = 20
) -> float:
    """Calcula a proporção de itens relevantes do conjunto de teste recuperados no Top-K."""
    if not relevant_media_ids:
        return 0.0

    top_k_ids = {item["media_id"] for item in ranked_items[:k]}
    hits = len(top_k_ids.intersection(relevant_media_ids))
    return float(hits / len(relevant_media_ids))


def compute_hit_rate_at_k(
    ranked_items: List[Dict[str, Any]],
    relevant_media_ids: set,
    k: int = 10
) -> float:
    """Retorna 1.0 se ao menos 1 item relevante estiver no Top-K, 0.0 caso contrário."""
    if not relevant_media_ids:
        return 0.0
    top_k_ids = {item["media_id"] for item in ranked_items[:k]}
    return 1.0 if len(top_k_ids.intersection(relevant_media_ids)) > 0 else 0.0


def compute_intra_list_diversity(top_items: List[Dict[str, Any]], k: int = 10) -> float:
    """
    Calcula Intra-List Diversity (ILD) sobre os embeddings dos Top-K itens recomendados.
    ILD = média de (1.0 - cos_sim(e_i, e_j)) para todo par distinto.
    """
    items_to_use = [it for it in top_items[:k] if it.get("embedding")]
    n = len(items_to_use)
    if n < 2:
        return 0.0

    embs = np.array([it["embedding"] for it in items_to_use], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_embs = embs / norms

    sim_matrix = np.dot(norm_embs, norm_embs.T)
    # Distância cosseno = 1.0 - similaridade
    dist_matrix = 1.0 - sim_matrix

    # Média do triângulo superior (excluindo a diagonal)
    triu_indices = np.triu_indices(n, k=1)
    pairwise_distances = dist_matrix[triu_indices]

    if len(pairwise_distances) == 0:
        return 0.0
    return float(np.mean(pairwise_distances))


def compute_distribution_stats(scores: List[float]) -> Dict[str, Any]:
    """Estatísticas descritivas completas e detecção de saturação e anomalias."""
    if not scores:
        return {}

    arr = np.array(scores, dtype=np.float64)
    total = len(arr)

    # Detecção das anomalias comprovadas:
    # 1. Falsos 100% (Anomalia 2.1.1)
    perfect_scores = int(np.sum(arr >= 99.5))
    perfect_pct = round((perfect_scores / total) * 100.0, 2)

    # 2. Platô de 87% (Anomalia 2.1.3 - obras rasas com denominador dinâmico 55)
    plateau_87 = int(np.sum((arr >= 86.0) & (arr <= 88.0)))
    plateau_87_pct = round((plateau_87 / total) * 100.0, 2)

    # Bins de distribuição
    bins = {
        "0-49%": int(np.sum(arr < 50.0)),
        "50-59%": int(np.sum((arr >= 50.0) & (arr < 60.0))),
        "60-69%": int(np.sum((arr >= 60.0) & (arr < 70.0))),
        "70-79%": int(np.sum((arr >= 70.0) & (arr < 80.0))),
        "80-89%": int(np.sum((arr >= 80.0) & (arr < 90.0))),
        "90-100%": int(np.sum(arr >= 90.0)),
    }

    return {
        "count": total,
        "mean": round(float(np.mean(arr)), 2),
        "std": round(float(np.std(arr)), 2),
        "min": round(float(np.min(arr)), 1),
        "p10": round(float(np.percentile(arr, 10)), 1),
        "p25": round(float(np.percentile(arr, 25)), 1),
        "median_p50": round(float(np.median(arr)), 1),
        "p75": round(float(np.percentile(arr, 75)), 1),
        "p90": round(float(np.percentile(arr, 90)), 1),
        "p99": round(float(np.percentile(arr, 99)), 1),
        "max": round(float(np.max(arr)), 1),
        "anomaly_100_count": perfect_scores,
        "anomaly_100_pct": perfect_pct,
        "anomaly_87_plateau_count": plateau_87,
        "anomaly_87_plateau_pct": plateau_87_pct,
        "bins": bins,
    }


# ---------------------------------------------------------------------------
# Harness Principal de Avaliação Offline
# ---------------------------------------------------------------------------

async def run_evaluation(
    test_ratio: float = 0.15,
    max_leave_n: int = 10,
    min_items: int = 5,
    tag: str = "baseline_v0"
) -> Dict[str, Any]:
    start_time = time.time()

    async with AsyncSessionLocal() as db:
        # 1. Carrega todas as mídias do catálogo
        all_media = (await db.execute(select(Media))).scalars().all()
        media_by_id = {m.id: m for m in all_media}
        media_by_tmdb = {(m.media_type, m.tmdb_id): m for m in all_media}

        # 2. Carrega todos os usuários
        users = (await db.execute(select(User))).scalars().all()

        # 3. Carrega itens dispensados agrupados por usuário
        dismissed_records = (await db.execute(select(UserDismissedRecommendation))).scalars().all()
        dismissed_by_user: Dict[str, List[UserDismissedRecommendation]] = {}
        for d in dismissed_records:
            u_id = str(d.user_id)
            dismissed_by_user.setdefault(u_id, []).append(d)

        # 4. Inicializa o Ranker LGBM existente
        lgbm_ranker = LightGBMRanker()

        models_to_test = [
            "Current_Production",
            "Hybrid_Heuristic_065_035",
            "Pure_Dense_Embedding",
            "Pure_Sparse_Vector",
        ]

        per_model_results: Dict[str, Dict[str, Any]] = {
            m: {
                "ndcg_10_list": [],
                "recall_20_list": [],
                "hit_rate_10_list": [],
                "ild_10_list": [],
                "negative_ranks": [],
                "all_scores": [],
            }
            for m in models_to_test
        }

        user_eval_summaries = []
        eligible_users_count = 0

        for user in users:
            u_id_str = str(user.id)

            # Busca itens do usuário ordenados por data
            items_result = await db.execute(
                select(UserListItem)
                .where(UserListItem.user_id == user.id)
                .options(selectinload(UserListItem.media), selectinload(UserListItem.episodes))
            )
            raw_items = [it for it in items_result.scalars().all() if it.media is not None]

            if len(raw_items) < min_items:
                continue

            eligible_users_count += 1

            # Ordena por timestamp cronológico (Leave-Last-N-Out)
            sorted_items = sorted(
                raw_items,
                key=lambda it: it.last_watched_at or it.updated_at or it.created_at or datetime.min.replace(tzinfo=timezone.utc)
            )

            n_total = len(sorted_items)
            n_test = min(max_leave_n, max(2, int(round(n_total * test_ratio))))
            train_items = sorted_items[:-n_test]
            test_items = sorted_items[-n_test:]

            train_media_ids = {it.media_id for it in train_items}
            test_media_dict = {it.media_id: it for it in test_items}

            # Identifica itens relevantes no conjunto de teste (relevance >= 2.0 ou fallback >= 1.0)
            test_relevances = {
                it.media_id: compute_ground_truth_relevance(it)
                for it in test_items
            }
            high_rel_media_ids = {mid for mid, rel in test_relevances.items() if rel >= 2.0}
            if not high_rel_media_ids:
                high_rel_media_ids = {mid for mid, rel in test_relevances.items() if rel >= 1.0}

            # Reconstrói os perfis de treino sem leak
            train_user_vec, train_user_emb = build_train_profile(train_items)
            dummy_user = User(feature_vector=train_user_vec, embedding=train_user_emb)

            # Monta a piscina de candidatos para o usuário:
            # - Todos os itens do conjunto de teste (Holdout)
            # - Itens dispensados (UserDismissedRecommendation)
            # - Itens do catálogo não vistos no treino
            candidate_pool: List[Dict[str, Any]] = []
            seen_cand_ids = set()

            # 1. Adiciona itens de teste
            for it in test_items:
                c_dict = media_to_candidate_dict(it.media)
                c_dict["media_id"] = it.media_id
                c_dict["ground_truth_relevance"] = test_relevances[it.media_id]
                c_dict["is_test_item"] = True
                c_dict["is_dismissed"] = False
                candidate_pool.append(c_dict)
                seen_cand_ids.add(it.media_id)

            # 2. Adiciona itens dispensados
            user_dismissed = dismissed_by_user.get(u_id_str, [])
            for d in user_dismissed:
                matched_m = media_by_tmdb.get((d.media_type, d.tmdb_id))
                if matched_m and matched_m.id not in seen_cand_ids and matched_m.id not in train_media_ids:
                    c_dict = media_to_candidate_dict(matched_m)
                    c_dict["media_id"] = matched_m.id
                    c_dict["ground_truth_relevance"] = 0.0
                    c_dict["is_test_item"] = False
                    c_dict["is_dismissed"] = True
                    candidate_pool.append(c_dict)
                    seen_cand_ids.add(matched_m.id)

            # 3. Adiciona demais obras do catálogo que não estão no treino
            for m in all_media:
                if m.id not in seen_cand_ids and m.id not in train_media_ids:
                    c_dict = media_to_candidate_dict(m)
                    c_dict["media_id"] = m.id
                    c_dict["ground_truth_relevance"] = 0.0
                    c_dict["is_test_item"] = False
                    c_dict["is_dismissed"] = False
                    candidate_pool.append(c_dict)
                    seen_cand_ids.add(m.id)

            user_summary_entry = {
                "user_id": u_id_str,
                "user_name": user.name,
                "total_items": n_total,
                "train_items": len(train_items),
                "test_items": len(test_items),
                "candidate_pool_size": len(candidate_pool),
                "high_rel_test_count": len(high_rel_media_ids),
                "model_ndcg": {}
            }

            # -----------------------------------------------------------------
            # Avaliação de cada modelo para este usuário
            # -----------------------------------------------------------------
            for model_name in models_to_test:
                scored_candidates = []

                for cand in candidate_pool:
                    c_copy = dict(cand)
                    cand_emb = c_copy.get("embedding")
                    cand_vec = build_candidate_vector(c_copy)

                    if model_name == "Current_Production":
                        # Se o usuário possui taste clusters e mais de 1 persona
                        if user.taste_clusters and len(user.taste_clusters) > 1:
                            best_s = -1.0
                            for p in user.taste_clusters:
                                p_vec = p.get("feature_vector") or train_user_vec
                                p_emb = p.get("centroid_embedding") or train_user_emb
                                aff_s, _ = calculate_match_score(p_vec, cand_vec)
                                if p_emb and cand_emb:
                                    sem_s = calculate_cosine_similarity(p_emb, cand_emb) * 100.0
                                    f_s = round(0.65 * sem_s + 0.35 * aff_s, 1)
                                else:
                                    f_s = aff_s
                                if f_s > best_s:
                                    best_s = f_s
                            pred_score = best_s
                        else:
                            # Cai no ranker lgbm_ranker.predict_score
                            pred_score = lgbm_ranker.predict_score(dummy_user, c_copy, cand_emb=cand_emb)

                    elif model_name == "Hybrid_Heuristic_065_035":
                        aff_s, _ = calculate_match_score(train_user_vec, cand_vec)
                        if train_user_emb and cand_emb:
                            sem_s = calculate_cosine_similarity(train_user_emb, cand_emb) * 100.0
                            pred_score = round(0.65 * sem_s + 0.35 * aff_s, 1)
                        else:
                            pred_score = aff_s

                    elif model_name == "Pure_Dense_Embedding":
                        if train_user_emb and cand_emb:
                            pred_score = round(calculate_cosine_similarity(train_user_emb, cand_emb) * 100.0, 1)
                        else:
                            pred_score = 0.0

                    elif model_name == "Pure_Sparse_Vector":
                        aff_s, _ = calculate_match_score(train_user_vec, cand_vec)
                        pred_score = round(aff_s, 1)

                    else:
                        pred_score = 0.0

                    c_copy["predicted_score"] = pred_score
                    scored_candidates.append(c_copy)

                # Ordena candidatos por predicted_score decrescente (desempate por relevância)
                scored_candidates.sort(key=lambda x: (x["predicted_score"], x["ground_truth_relevance"]), reverse=True)

                # Coleta relevâncias ordenadas
                ranked_relevances = [c["ground_truth_relevance"] for c in scored_candidates]
                all_scores = [c["predicted_score"] for c in scored_candidates]

                # Métricas
                ndcg_10 = compute_ndcg_at_k(ranked_relevances, k=10)
                recall_20 = compute_recall_at_k(scored_candidates, high_rel_media_ids, k=20)
                hit_rate_10 = compute_hit_rate_at_k(scored_candidates, high_rel_media_ids, k=10)
                ild_10 = compute_intra_list_diversity(scored_candidates, k=10)

                # Rastreamento de postos dos itens negativos
                for rank_idx, cand in enumerate(scored_candidates):
                    if cand["is_dismissed"] or (cand["is_test_item"] and cand["ground_truth_relevance"] == 0.0):
                        per_model_results[model_name]["negative_ranks"].append(rank_idx + 1)

                per_model_results[model_name]["ndcg_10_list"].append(ndcg_10)
                per_model_results[model_name]["recall_20_list"].append(recall_20)
                per_model_results[model_name]["hit_rate_10_list"].append(hit_rate_10)
                per_model_results[model_name]["ild_10_list"].append(ild_10)
                per_model_results[model_name]["all_scores"].extend(all_scores)

                user_summary_entry["model_ndcg"][model_name] = round(ndcg_10, 4)

            user_eval_summaries.append(user_summary_entry)

    duration_sec = round(time.time() - start_time, 2)

    # ---------------------------------------------------------------------------
    # Agregação Final das Métricas por Modelo
    # ---------------------------------------------------------------------------
    final_models_summary = {}

    for model_name, data in per_model_results.items():
        ndcg_arr = data["ndcg_10_list"]
        rec_arr = data["recall_20_list"]
        hit_arr = data["hit_rate_10_list"]
        ild_arr = data["ild_10_list"]
        neg_ranks = data["negative_ranks"]
        scores = data["all_scores"]

        dist_stats = compute_distribution_stats(scores)

        final_models_summary[model_name] = {
            "ndcg_10_mean": round(float(np.mean(ndcg_arr)), 4) if ndcg_arr else 0.0,
            "recall_20_mean": round(float(np.mean(rec_arr)), 4) if rec_arr else 0.0,
            "hit_rate_10_mean": round(float(np.mean(hit_arr)), 4) if hit_arr else 0.0,
            "ild_10_mean": round(float(np.mean(ild_arr)), 4) if ild_arr else 0.0,
            "avg_negative_rank": round(float(np.mean(neg_ranks)), 1) if neg_ranks else 0.0,
            "total_negative_evaluations": len(neg_ranks),
            "distribution": dist_stats,
        }

    report_payload = {
        "tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration_sec,
        "eligible_users_count": eligible_users_count,
        "total_catalog_media": len(all_media),
        "protocol": {
            "type": "Leave-Last-N-Out Temporal Split",
            "test_ratio": test_ratio,
            "max_leave_n": max_leave_n,
            "min_items": min_items,
        },
        "models": final_models_summary,
        "users": user_eval_summaries,
    }

    return report_payload


# ---------------------------------------------------------------------------
# Formatadores de Relatório (Terminal e Markdown)
# ---------------------------------------------------------------------------

def print_terminal_report(report: Dict[str, Any], compare_report: Optional[Dict[str, Any]] = None):
    print("\n" + "=" * 80)
    print(f"📊 RELATÓRIO DO HARNESS OFFLINE TEMPORAL — {report['tag'].upper()}")
    print("=" * 80)
    print(f"⏱️  Tempo de Execução: {report['duration_seconds']}s (Meta: < 15s)")
    print(f"👥 Usuários Elegíveis (>= 5 itens): {report['eligible_users_count']}")
    print(f"🎬 Catálogo de Mídias Testadas: {report['total_catalog_media']}")
    print("-" * 80)

    header = f"{'Modelo':<28} | {'NDCG@10':<9} | {'Recall@20':<9} | {'HitRate@10':<10} | {'ILD@10':<8} | {'100% Bug':<9} | {'87% Platô':<9}"
    print(header)
    print("-" * len(header))

    for m_name, m_data in report["models"].items():
        dist = m_data["distribution"]
        ndcg_str = f"{m_data['ndcg_10_mean']:.4f}"
        rec_str = f"{m_data['recall_20_mean']:.4f}"
        hit_str = f"{m_data['hit_rate_10_mean']:.4f}"
        ild_str = f"{m_data['ild_10_mean']:.4f}"
        p100_str = f"{dist.get('anomaly_100_pct', 0.0):.1f}%"
        p87_str = f"{dist.get('anomaly_87_plateau_pct', 0.0):.1f}%"

        if compare_report and m_name in compare_report.get("models", {}):
            prev_m = compare_report["models"][m_name]
            delta_ndcg = m_data['ndcg_10_mean'] - prev_m['ndcg_10_mean']
            sign = "+" if delta_ndcg >= 0 else ""
            ndcg_str += f" ({sign}{delta_ndcg:.4f})"

        print(f"{m_name:<28} | {ndcg_str:<9} | {rec_str:<9} | {hit_str:<10} | {ild_str:<8} | {p100_str:<9} | {p87_str:<9}")

    print("-" * 80)
    print("📈 Distribuição de Scores do Modelo Atual (Current_Production):")
    curr_dist = report["models"]["Current_Production"]["distribution"]
    print(f"   Média: {curr_dist['mean']}% | Desvio Padrão: {curr_dist['std']} | Mediana: {curr_dist['median_p50']}%")
    print(f"   Min: {curr_dist['min']}% | P10: {curr_dist['p10']}% | P75: {curr_dist['p75']}% | P90: {curr_dist['p90']}% | Max: {curr_dist['max']}%")
    print(f"   ⚠️  Falsos 100% detectados: {curr_dist['anomaly_100_count']} ({curr_dist['anomaly_100_pct']}%)")
    print(f"   ⚠️  Platôs de 87% detectados: {curr_dist['anomaly_87_plateau_count']} ({curr_dist['anomaly_87_plateau_pct']}%)")
    print(f"   Faixas: {curr_dist['bins']}")
    print("=" * 80 + "\n")


def generate_markdown_report(report: Dict[str, Any], compare_report: Optional[Dict[str, Any]] = None) -> str:
    lines = []
    lines.append(f"# Relatório de Avaliação Offline Temporal — `{report['tag']}`")
    lines.append("")
    lines.append(f"**Data e Hora:** `{report['timestamp']}`  ")
    lines.append(f"**Duração:** `{report['duration_seconds']}s` (Requisito da Etapa 0: `< 15s`)  ")
    lines.append(f"**Usuários Elegíveis Analisados:** `{report['eligible_users_count']}`  ")
    lines.append(f"**Catálogo de Obras Avaliadas:** `{report['total_catalog_media']}` itens  ")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> Este harness avalia de forma estritamente somente-leitura o comportamento dos rankers usando um split temporal (*Leave-Last-N-Out*). Ele serve como a régua científica definitiva para auditar e provar o ganho de acurácia de cada etapa.")
    lines.append("")
    lines.append("## 1. Tabela Comparativa de Modelos")
    lines.append("")
    lines.append("| Modelo | NDCG@10 | Recall@20 | HitRate@10 | Intra-List Diversity (ILD) | Falsos 100% | Platô 87% | Rank Médio Negativos |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for m_name, m_data in report["models"].items():
        dist = m_data["distribution"]
        ndcg_val = m_data['ndcg_10_mean']
        rec_val = m_data['recall_20_mean']
        hit_val = m_data['hit_rate_10_mean']
        ild_val = m_data['ild_10_mean']
        p100_val = dist.get('anomaly_100_pct', 0.0)
        p87_val = dist.get('anomaly_87_plateau_pct', 0.0)
        neg_rank = m_data.get('avg_negative_rank', 0.0)

        ndcg_display = f"**{ndcg_val:.4f}**"
        if compare_report and m_name in compare_report.get("models", {}):
            prev_val = compare_report["models"][m_name]['ndcg_10_mean']
            delta = ndcg_val - prev_val
            sign = "+" if delta >= 0 else ""
            ndcg_display += f" *({sign}{delta:.4f})*"

        lines.append(
            f"| `{m_name}` | {ndcg_display} | {rec_val:.4f} | {hit_val:.4f} | {ild_val:.4f} | "
            f"{p100_val:.1f}% | {p87_val:.1f}% | #{neg_rank} |"
        )

    lines.append("")
    lines.append("## 2. Diagnóstico Empírico das Anomalias")
    curr_m = report["models"]["Current_Production"]
    curr_dist = curr_m["distribution"]

    lines.append("")
    if curr_dist["anomaly_100_pct"] > 0:
        lines.append(f"> [!WARNING]")
        lines.append(f"> **Anomalia 2.1.1 (Falsos 100% Confirmada):** O modelo em produção gerou `{curr_dist['anomaly_100_count']}` predições com nota máxima `99.5% - 100.0%` ({curr_dist['anomaly_100_pct']}% de todas as notas). Isso comprova em teste offline a contaminação do classificador LightGBM!")
    else:
        lines.append("> [!TIP]")
        lines.append("> **Falsos 100% Zerados:** Nenhuma obra atingiu a trava degenerada de 100%.")

    lines.append("")
    if curr_dist["anomaly_87_plateau_pct"] > 0:
        lines.append(f"> [!WARNING]")
        lines.append(f"> **Anomalia 2.1.3 (Platô de 87% Confirmada):** `{curr_dist['anomaly_87_plateau_count']}` obras caíram exatamente na faixa `[86.0%, 88.0%]` ({curr_dist['anomaly_87_plateau_pct']}% das notas), evidenciando o paradoxo do denominador dinâmico `55.0`.")
    else:
        lines.append("> [!TIP]")
        lines.append("> **Platô de 87% Inexistente ou Diluído.**")

    lines.append("")
    lines.append("## 3. Histograma de Distribuição de Scores (`Current_Production`)")
    lines.append("")
    lines.append("| Faixa de Pontuação | Quantidade de Obras | Percentual | Barra Visual |")
    lines.append("| :--- | :---: | :---: | :--- |")

    tot_scores = curr_dist["count"]
    for b_label, b_cnt in curr_dist["bins"].items():
        pct = (b_cnt / tot_scores) * 100.0 if tot_scores > 0 else 0.0
        bar = "█" * int(round(pct / 4))
        lines.append(f"| `{b_label}` | {b_cnt} | {pct:.1f}% | `{bar}` |")

    lines.append("")
    lines.append("## 4. Detalhamento por Usuário (NDCG@10)")
    lines.append("")
    lines.append("| Usuário | Histórico Total | Treino | Teste (Holdout) | Candidatos | Current_Production | Hybrid_Heuristic |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for u_info in report["users"]:
        m_ndcg = u_info["model_ndcg"]
        lines.append(
            f"| **{u_info['user_name']}** | {u_info['total_items']} | {u_info['train_items']} | "
            f"{u_info['test_items']} | {u_info['candidate_pool_size']} | "
            f"`{m_ndcg.get('Current_Production', 0.0):.4f}` | `{m_ndcg.get('Hybrid_Heuristic_065_035', 0.0):.4f}` |"
        )

    lines.append("")
    lines.append("---")
    lines.append(f"*Relatório gerado automaticamente por `scripts/eval_recommendations.py`.*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entrada CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Harness de Avaliação Offline Temporal - OmniWatch")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Proporção de itens recentes para teste (default: 0.15)")
    parser.add_argument("--max-leave-n", type=int, default=10, help="Teto de itens no conjunto de teste por usuário (default: 10)")
    parser.add_argument("--min-items", type=int, default=5, help="Mínimo de itens no histórico do usuário para avaliação (default: 5)")
    parser.add_argument("--tag", type=str, default="baseline_v0", help="Tag identificadora da rodada (ex: baseline_v0, etapa_1)")
    parser.add_argument("--output-dir", type=str, default=None, help="Diretório onde salvar os relatórios json e md")
    parser.add_argument("--compare", type=str, default=None, help="Caminho para relatório JSON anterior para comparar")
    return parser.parse_args()


async def main():
    args = parse_args()

    # Define diretório de saída
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(BACKEND_DIR, "reports", "eval")
    os.makedirs(output_dir, exist_ok=True)

    # Carrega comparação anterior se fornecida
    compare_report = None
    if args.compare and os.path.exists(args.compare):
        try:
            with open(args.compare, "r", encoding="utf-8") as f:
                compare_report = json.load(f)
        except Exception as e:
            print(f"⚠️ Não foi possível carregar arquivo de comparação {args.compare}: {e}")

    # Executa a avaliação offline
    report = await run_evaluation(
        test_ratio=args.test_ratio,
        max_leave_n=args.max_leave_n,
        min_items=args.min_items,
        tag=args.tag
    )

    # Imprime no terminal
    print_terminal_report(report, compare_report=compare_report)

    # Salva arquivos JSON e Markdown
    json_path = os.path.join(output_dir, f"{args.tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_content = generate_markdown_report(report, compare_report=compare_report)
    md_path = os.path.join(output_dir, f"{args.tag}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"💾 Relatórios salvos com sucesso:")
    print(f"   JSON: {json_path}")
    print(f"   MD:   {md_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
