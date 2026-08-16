from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from src.data.preprocess import load_news


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_PATH = Path(
    "data/processed/article_embeddings.npy"
)

ARTICLE_IDS_PATH = Path(
    "data/processed/article_ids.csv"
)


model = SentenceTransformer(
    MODEL_NAME,
    device="cuda"
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

texts = news["text"].tolist()


if EMBEDDING_PATH.exists() and ARTICLE_IDS_PATH.exists():

    print("Loading saved embeddings...")

    embeddings = np.load(
        EMBEDDING_PATH
    )

    article_ids = pd.read_csv(
        ARTICLE_IDS_PATH
    )

else:

    print("Generating article embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    np.save(
        EMBEDDING_PATH,
        embeddings
    )

    news["news_id"].to_csv(
        ARTICLE_IDS_PATH,
        index=False
    )

    article_ids = news[["news_id"]]


print(
    "Embedding shape:",
    embeddings.shape
)


query_index = 0

query_embedding = embeddings[
    query_index
]


similarities = cos_sim(
    query_embedding,
    embeddings
)[0]


top_indices = similarities.argsort(
    descending=True
)[1:6]


print("\nQUERY ARTICLE:")

print(
    news.iloc[
        query_index
    ]["title"]
)


print(
    "\nMOST SIMILAR ARTICLES:"
)


for idx in top_indices:

    idx = int(idx)

    print(
        round(
            similarities[idx].item(),
            4
        ),
        "|",
        news.iloc[idx]["category"],
        "|",
        news.iloc[idx]["title"]
    )
print(embeddings.shape)