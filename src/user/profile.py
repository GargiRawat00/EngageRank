import numpy as np
import pandas as pd
from src.data.preprocess import load_behaviors, load_news
from sentence_transformers.util import cos_sim
news = load_news(
    "data/raw/train/news.tsv"
)

behaviors = load_behaviors(
    "data/raw/train/behaviors.tsv"
)

embeddings = np.load(
    "data/processed/article_embeddings.npy"
)

article_ids = pd.read_csv(
    "data/processed/article_ids.csv"
)
id_to_index = {
    news_id: idx
    for idx, news_id in enumerate(article_ids["news_id"])
}
first_user = behaviors.iloc[0]

history = first_user["history"]
history_indices = [
    id_to_index[news_id]
    for news_id in history
    if news_id in id_to_index
]
history_embeddings = embeddings[history_indices]
user_vector = history_embeddings.mean(axis=0)
current_impressions = first_user["impressions"]
candidate_ids = [
    news_id
    for news_id, label in current_impressions
    if news_id in id_to_index
]
candidate_indices = [
    id_to_index[news_id]
    for news_id in candidate_ids
]
candidate_embeddings = embeddings[candidate_indices]
scores = cos_sim(
    user_vector,
    candidate_embeddings
)[0]
ranked_indices = scores.argsort(
    descending=True
)

for idx in ranked_indices[:10]:
    idx = int(idx)

    news_id = candidate_ids[idx]

    row = news[
        news["news_id"] == news_id
    ].iloc[0]
    label_map = {
        news_id: label
        for news_id, label in current_impressions
    }
    print(
        round(scores[idx].item(), 4),
        "|",
        news_id,
        "|",
        row["category"],
        "|",
        row["title"]
    )
print("History embeddings shape:", history_embeddings.shape)
print("User vector shape:", user_vector.shape)