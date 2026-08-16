from collections import Counter

from src.data.preprocess import load_behaviors


def build_popularity():
    behaviors = load_behaviors(
        "data/raw/train/behaviors.tsv"
    )

    clicks = Counter()

    for impressions in behaviors["impressions"]:
        for news_id, label in impressions:
            if label == 1:
                clicks[news_id] += 1

    return clicks