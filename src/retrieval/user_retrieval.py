import faiss
import numpy as np
import pandas as pd

from src.data.preprocess import load_behaviors


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

behaviors = load_behaviors(
    "data/raw/valid/behaviors.tsv"
)

id_to_index = {
    news_id: idx
    for idx, news_id in enumerate(
        article_ids["news_id"]
    )
}


def build_user_vector(history):

    history_indices = [
        id_to_index[news_id]
        for news_id in history
        if news_id in id_to_index
    ]

    if len(history_indices) == 0:
        return None

    history_embeddings = embeddings[
        history_indices
    ]

    user_vector = history_embeddings.mean(
        axis=0
    )

    # normalize because FAISS IndexFlatIP
    # works like cosine similarity
    # when vectors are normalized
    norm = np.linalg.norm(user_vector)

    if norm > 0:
        user_vector = (
            user_vector / norm
        )

    return user_vector.astype("float32")


def retrieve_for_user(
    history,
    top_k=100
):

    user_vector = build_user_vector(
        history
    )

    if user_vector is None:
        return []

    # ask FAISS for extra articles
    # because some retrieved articles
    # may already exist in user history
    search_k = min(
        top_k + len(history),
        index.ntotal
    )

    scores, indices = index.search(
        user_vector.reshape(1, -1),
        search_k
    )

    history_set = set(history)

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        # safety check
        if idx == -1:
            continue

        news_id = article_ids.iloc[
            idx
        ]["news_id"]

        # don't recommend articles
        # user already clicked
        if news_id in history_set:
            continue

        results.append(
            {
                "news_id": news_id,
                "score": float(score)
            }
        )

        if len(results) == top_k:
            break

    return results


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

    results = retrieve_for_user(
        row["history"],
        top_k=10
    )

    print(
        "\nRetrieved unseen articles:"
    )

    for result in results:

        print(
            result["news_id"],
            "|",
            round(
                result["score"],
                4
            )
        )