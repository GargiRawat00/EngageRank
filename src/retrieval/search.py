import faiss
import numpy as np
import pandas as pd


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


def search_similar_articles(
    query_vector,
    top_k=10
):
    query_vector = np.array(
        query_vector,
        dtype="float32"
    ).reshape(1, -1)

    scores, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):
        news_id = article_ids.iloc[idx]["news_id"]

        results.append(
            {
                "news_id": news_id,
                "score": float(score)
            }
        )

    return results


if __name__ == "__main__":

    # testing using first article vector
    query_vector = embeddings[0]

    results = search_similar_articles(
        query_vector,
        top_k=5
    )

    for result in results:
        print(
            result["news_id"],
            "|",
            result["score"]
        )