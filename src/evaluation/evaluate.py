import numpy as np
import pandas as pd

from src.data.preprocess import load_behaviors
from src.evaluation.metrics import (
    reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    intra_list_diversity
)
from src.ranking.diversity import mmr_rerank
from src.models.cold_start import build_popularity


behaviors = load_behaviors(
    "data/raw/valid/behaviors.tsv"
)

# quick test first
behaviors = behaviors.head(5000)

embeddings = np.load(
    "data/processed/article_embeddings.npy"
)

article_ids = pd.read_csv(
    "data/processed/article_ids.csv"
)

id_to_index = {
    news_id: idx
    for idx, news_id in enumerate(
        article_ids["news_id"]
    )
}

# popularity is built only from TRAIN data
popularity = build_popularity()


def evaluate_row(row):
    history = row["history"]
    impressions = row["impressions"]

    candidate_ids = [
        news_id
        for news_id, label in impressions
        if news_id in id_to_index
    ]

    if len(candidate_ids) == 0:
        return None

    label_map = {
        news_id: label
        for news_id, label in impressions
    }

    candidate_indices = [
        id_to_index[news_id]
        for news_id in candidate_ids
    ]

    candidate_embeddings = embeddings[
        candidate_indices
    ]

    history_indices = [
        id_to_index[news_id]
        for news_id in history
        if news_id in id_to_index
    ]

    # ==================================
    # COLD START
    # ==================================

    if len(history_indices) == 0:

        relevance_scores = np.array(
            [
                popularity.get(news_id, 0)
                for news_id in candidate_ids
            ],
            dtype=float
        )

        ranked_indices = np.argsort(
            relevance_scores
        )[::-1]

        ranked_labels = [
            label_map[candidate_ids[idx]]
            for idx in ranked_indices
        ]

        ild10 = intra_list_diversity(
            candidate_embeddings,
            ranked_indices,
            10
        )

        return {
            "mrr": reciprocal_rank(
                ranked_labels
            ),
            "ndcg5": ndcg_at_k(
                ranked_labels,
                5
            ),
            "ndcg10": ndcg_at_k(
                ranked_labels,
                10
            ),
            "recall5": recall_at_k(
                ranked_labels,
                5
            ),
            "recall10": recall_at_k(
                ranked_labels,
                10
            ),
            "ild10": ild10,
            "cold_start": True
        }

    # ==================================
    # PERSONALIZED RANKING
    # ==================================

    history_embeddings = embeddings[
        history_indices
    ]

    similarity_matrix = (
        candidate_embeddings
        @ history_embeddings.T
    )

    # strongest 3 historical matches
    top_k_hist = min(
        3,
        similarity_matrix.shape[1]
    )

    top_scores = np.sort(
        similarity_matrix,
        axis=1
    )[:, -top_k_hist:]

    relevance_scores = (
        top_scores.mean(axis=1)
    )

    # base semantic ranking
    base_ranked_indices = np.argsort(
        relevance_scores
    )[::-1]

    # shortlist top 20
    top_n = min(
        20,
        len(candidate_ids)
    )

    shortlist = (
        base_ranked_indices[:top_n]
    )

    subset_embeddings = (
        candidate_embeddings[shortlist]
    )

    subset_scores = (
        relevance_scores[shortlist]
    )

    # MMR reranking
    mmr_order = mmr_rerank(
        subset_embeddings,
        subset_scores,
        top_k=top_n,
        lambda_weight=0.9
    )

    ranked_indices = [
        shortlist[idx]
        for idx in mmr_order
    ]

    selected_set = set(
        ranked_indices
    )

    remaining_indices = [
        idx
        for idx in base_ranked_indices
        if idx not in selected_set
    ]

    ranked_indices.extend(
        remaining_indices
    )

    ranked_labels = [
        label_map[candidate_ids[idx]]
        for idx in ranked_indices
    ]

    ild10 = intra_list_diversity(
        candidate_embeddings,
        ranked_indices,
        10
    )

    return {
        "mrr": reciprocal_rank(
            ranked_labels
        ),
        "ndcg5": ndcg_at_k(
            ranked_labels,
            5
        ),
        "ndcg10": ndcg_at_k(
            ranked_labels,
            10
        ),
        "recall5": recall_at_k(
            ranked_labels,
            5
        ),
        "recall10": recall_at_k(
            ranked_labels,
            10
        ),
        "ild10": ild10,
        "cold_start": False
    }


results = []

for _, row in behaviors.iterrows():

    result = evaluate_row(row)

    if result is not None:
        results.append(result)

    if (
        len(results) > 0
        and len(results) % 1000 == 0
    ):
        print(
            "Processed:",
            len(results)
        )


results = pd.DataFrame(
    results
)

print(
    "\nEvaluated rows:",
    len(results)
)


# ==================================
# OVERALL RESULTS
# ==================================

print("\nOVERALL")

print(
    "MRR:",
    results["mrr"].mean()
)

print(
    "NDCG@5:",
    results["ndcg5"].mean()
)

print(
    "NDCG@10:",
    results["ndcg10"].mean()
)

print(
    "Recall@5:",
    results["recall5"].mean()
)

print(
    "Recall@10:",
    results["recall10"].mean()
)

print(
    "ILD@10:",
    results["ild10"].mean()
)


# ==================================
# COLD START RESULTS
# ==================================

cold_results = results[
    results["cold_start"] == True
]

print("\nCOLD START")

print(
    "Rows:",
    len(cold_results)
)

if len(cold_results) > 0:

    print(
        "MRR:",
        cold_results["mrr"].mean()
    )

    print(
        "NDCG@5:",
        cold_results["ndcg5"].mean()
    )

    print(
        "NDCG@10:",
        cold_results["ndcg10"].mean()
    )

    print(
        "Recall@5:",
        cold_results["recall5"].mean()
    )

    print(
        "Recall@10:",
        cold_results["recall10"].mean()
    )

    print(
        "ILD@10:",
        cold_results["ild10"].mean()
    )


# ==================================
# PERSONALIZED RESULTS
# ==================================

personal_results = results[
    results["cold_start"] == False
]

print("\nPERSONALIZED")

print(
    "Rows:",
    len(personal_results)
)

if len(personal_results) > 0:

    print(
        "MRR:",
        personal_results["mrr"].mean()
    )

    print(
        "NDCG@5:",
        personal_results["ndcg5"].mean()
    )

    print(
        "NDCG@10:",
        personal_results["ndcg10"].mean()
    )

    print(
        "Recall@5:",
        personal_results["recall5"].mean()
    )

    print(
        "Recall@10:",
        personal_results["recall10"].mean()
    )

    print(
        "ILD@10:",
        personal_results["ild10"].mean()
    )