from collections import Counter
from src.data.preprocess import load_behaviors
from src.data.preprocess import load_news

def build_popularity_scores(behaviors):
    clicks = Counter()

    for impressions in behaviors["impressions"]:
        for news_id, label in impressions:
            if label == 1:
                clicks[news_id] += 1

    return clicks


if __name__ == "__main__":
    behaviors = load_behaviors(
        "data/raw/train/behaviors.tsv"
    )
    news = load_news(
    "data/raw/train/news.tsv"
    )
    scores = build_popularity_scores(behaviors)

    print(scores.most_common(10))
    top_articles = scores.most_common(10)

    for news_id, clicks in top_articles:
        row = news[news["news_id"] == news_id].iloc[0]

        print(
        news_id,
        "|",
        clicks,
        "|",
        row["category"],
        "|",
        row["title"]
    )