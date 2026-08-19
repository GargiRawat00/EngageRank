from src.ranking.recommender import recommend, news_lookup


def build_briefing_context(
    history,
    top_k=5
):
    recommendations = recommend(
        history,
        final_k=top_k
    )

    context = []

    for item in recommendations:

        news_id = item["news_id"]

        if news_id not in news_lookup.index:
            continue

        row = news_lookup.loc[news_id]

        context.append(
            {
                "news_id": news_id,
                "title": row["title"],
                "abstract": row["abstract"],
                "category": row["category"]
            }
        )

    return context


if __name__ == "__main__":

    history = [
        "N55189",
        "N42782",
        "N34694",
        "N45794",
        "N18445"
    ]

    context = build_briefing_context(
        history,
        top_k=5
    )

    for i, article in enumerate(
        context,
        start=1
    ):
        print("\nARTICLE", i)
        print("ID:", article["news_id"])
        print("Category:", article["category"])
        print("Title:", article["title"])
        print("Abstract:", article["abstract"])