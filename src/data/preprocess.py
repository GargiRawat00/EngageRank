import pandas as pd


NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities"
]


BEHAVIOR_COLUMNS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions"
]


def load_news(path):
    df = pd.read_csv(
        path,
        sep="\t",
        names=NEWS_COLUMNS
    )

    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")

    df["text"] = (
        df["title"] + " " + df["abstract"]
    ).str.strip()

    return df


def parse_impressions(x):
    return [
        (news_id, int(label))
        for news_id, label in (
            item.split("-")
            for item in x.split()
        )
    ]


def load_behaviors(path):
    df = pd.read_csv(
        path,
        sep="\t",
        names=BEHAVIOR_COLUMNS
    )

    df["history"] = df["history"].fillna("")

    df["history"] = df["history"].apply(
        lambda x: x.split() if x else []
    )

    df["impressions"] = df["impressions"].apply(
        parse_impressions
    )

    return df


if __name__ == "__main__":

    train_news = load_news(
        "data/raw/train/news.tsv"
    )

    train_behaviors = load_behaviors(
        "data/raw/train/behaviors.tsv"
    )

    print("News shape:")
    print(train_news.shape)

    print("\nBehaviors shape:")
    print(train_behaviors.shape)

    print("\nFirst news row:")
    print(
        train_news[
            [
                "news_id",
                "category",
                "subcategory",
                "title",
                "text"
            ]
        ].iloc[0]
    )

    print("\nFirst behavior row:")
    print(
        train_behaviors[
            [
                "user_id",
                "history",
                "impressions"
            ]
        ].iloc[0]
    )