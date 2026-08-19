import faiss
import numpy as np
import pandas as pd

from src.data.preprocess import load_behaviors, load_news
from src.ranking.diversity import mmr_rerank
from src.models.cold_start import build_popularity


INDEX_PATH = "data/processed/article_index.faiss"
EMBEDDING_PATH = "data/processed/article_embeddings.npy"
ARTICLE_IDS_PATH = "data/processed/article_ids.csv"


index = faiss.read_index(INDEX_PATH)

embeddings = np.load(
    EMBEDDING_PATH
).astype("float32")

article_ids = pd.read_csv(
    ARTICLE_IDS_PATH
)

train_news = load_news(
    "data/raw/train/news.tsv"
)

valid_news = load_news(
    "data/raw/valid/news.tsv"
)

news = pd.concat(
    [train_news, valid_news],
    ignore_index=True
)

news = news.drop_duplicates(
    subset=["news_id"]
).reset_index(drop=True)

behaviors = load_behaviors(
    "data/raw/valid/behaviors.tsv"
)

popularity = build_popularity()

id_to_index = {
    news_id: idx
    for idx, news_id in enumerate(
        article_ids["news_id"]
    )
}

news_lookup = news.set_index(
    "news_id"
)


def retrieve_candidates(
    history,
    top_k=100
):
    history_indices = [
        id_to_index[news_id]
        for news_id in history
        if news_id in id_to_index
    ]

    if len(history_indices) == 0:
        return []

    history_embeddings = embeddings[
        history_indices
    ]

    user_vector = history_embeddings.mean(
        axis=0
    )

    norm = np.linalg.norm(
        user_vector
    )

    if norm > 0:
        user_vector = (
            user_vector / norm
        )

    user_vector = user_vector.astype(
        "float32"
    )

    search_k = min(
        top_k + len(history),
        index.ntotal
    )

    scores, indices = index.search(
        user_vector.reshape(1, -1),
        search_k
    )

    history_set = set(history)

    candidates = []

    for idx in indices[0]:

        if idx == -1:
            continue

        news_id = article_ids.iloc[
            idx
        ]["news_id"]

        if news_id in history_set:
            continue

        candidates.append(news_id)

        if len(candidates) == top_k:
            break

    return candidates


def get_cold_start_candidates(
    top_k=100
):
    popular_articles = popularity.most_common(
        top_k
    )

    return [
        news_id
        for news_id, clicks in popular_articles
        if news_id in id_to_index
    ]


def rank_candidates(
    history,
    candidate_ids
):
    history_indices = [
        id_to_index[news_id]
        for news_id in history
        if news_id in id_to_index
    ]

    candidate_ids = [
        news_id
        for news_id in candidate_ids
        if news_id in id_to_index
    ]

    if (
        len(history_indices) == 0
        or len(candidate_ids) == 0
    ):
        return None

    history_embeddings = embeddings[
        history_indices
    ]

    candidate_indices = [
        id_to_index[news_id]
        for news_id in candidate_ids
    ]

    candidate_embeddings = embeddings[
        candidate_indices
    ]

    similarity_matrix = (
        candidate_embeddings
        @ history_embeddings.T
    )

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

    return (
        candidate_ids,
        candidate_embeddings,
        relevance_scores
    )


def cold_start_recommend(
    final_k=10,
    shortlist_k=20
):
    candidate_ids = get_cold_start_candidates(
        top_k=100
    )

    candidate_indices = [
        id_to_index[news_id]
        for news_id in candidate_ids
    ]

    candidate_embeddings = embeddings[
        candidate_indices
    ]

    relevance_scores = np.array(
        [
            popularity.get(news_id, 0)
            for news_id in candidate_ids
        ],
        dtype="float32"
    )

    base_order = np.argsort(
        relevance_scores
    )[::-1]

    shortlist_k = min(
        shortlist_k,
        len(candidate_ids)
    )

    shortlist = base_order[
        :shortlist_k
    ]

    subset_embeddings = (
        candidate_embeddings[
            shortlist
        ]
    )

    subset_scores = (
        relevance_scores[
            shortlist
        ]
    )

    mmr_order = mmr_rerank(
        subset_embeddings,
        subset_scores,
        top_k=min(
            final_k,
            shortlist_k
        ),
        lambda_weight=0.9
    )

    final_indices = [
        shortlist[idx]
        for idx in mmr_order
    ]

    recommendations = []

    for idx in final_indices:

        news_id = candidate_ids[idx]

        if news_id not in news_lookup.index:
            continue

        row = news_lookup.loc[
            news_id
        ]

        recommendations.append(
            {
                "news_id": news_id,
                "score": float(
                    relevance_scores[idx]
                ),
                "category": row[
                    "category"
                ],
                "title": row[
                    "title"
                ],
                "mode": "cold_start"
            }
        )

    return recommendations


def recommend(
    history,
    retrieval_k=100,
    shortlist_k=20,
    final_k=10
):
    # ----------------------------------
    # COLD START
    # ----------------------------------

    if history is None or len(history) == 0:

        return cold_start_recommend(
            final_k=final_k,
            shortlist_k=shortlist_k
        )

    # ----------------------------------
    # PERSONALIZED RETRIEVAL
    # ----------------------------------

    candidate_ids = retrieve_candidates(
        history,
        top_k=retrieval_k
    )

    if len(candidate_ids) == 0:

        return cold_start_recommend(
            final_k=final_k,
            shortlist_k=shortlist_k
        )

    ranked_data = rank_candidates(
        history,
        candidate_ids
    )

    if ranked_data is None:

        return cold_start_recommend(
            final_k=final_k,
            shortlist_k=shortlist_k
        )

    (
        candidate_ids,
        candidate_embeddings,
        relevance_scores
    ) = ranked_data

    base_order = np.argsort(
        relevance_scores
    )[::-1]

    shortlist_k = min(
        shortlist_k,
        len(candidate_ids)
    )

    shortlist = base_order[
        :shortlist_k
    ]

    subset_embeddings = (
        candidate_embeddings[
            shortlist
        ]
    )

    subset_scores = (
        relevance_scores[
            shortlist
        ]
    )

    mmr_order = mmr_rerank(
        subset_embeddings,
        subset_scores,
        top_k=min(
            final_k,
            shortlist_k
        ),
        lambda_weight=0.9
    )

    final_indices = [
        shortlist[idx]
        for idx in mmr_order
    ]

    recommendations = []

    for idx in final_indices:

        news_id = candidate_ids[
            idx
        ]

        if news_id not in news_lookup.index:
            continue

        row = news_lookup.loc[
            news_id
        ]

        recommendations.append(
            {
                "news_id": news_id,
                "score": float(
                    relevance_scores[idx]
                ),
                "category": row[
                    "category"
                ],
                "title": row[
                    "title"
                ],
                "mode": "personalized"
            }
        )

    return recommendations


if __name__ == "__main__":

    row = behaviors.iloc[0]

    print(
        "User:",
        row["user_id"]
    )

    print(
        "History size:",
        len(row["history"])
    )

    recommendations = recommend(
        row["history"]
    )

    print(
        "\nPERSONALIZED RECOMMENDATIONS\n"
    )

    for rank, item in enumerate(
        recommendations,
        start=1
    ):
        print(
            rank,
            "|",
            item["mode"],
            "|",
            item["news_id"],
            "|",
            round(
                item["score"],
                4
            ),
            "|",
            item["category"],
            "|",
            item["title"]
        )

    print(
        "\nCOLD START TEST\n"
    )

    cold_recommendations = recommend(
        []
    )

    for rank, item in enumerate(
        cold_recommendations,
        start=1
    ):
        print(
            rank,
            "|",
            item["mode"],
            "|",
            item["news_id"],
            "|",
            round(
                item["score"],
                4
            ),
            "|",
            item["category"],
            "|",
            item["title"]
        )