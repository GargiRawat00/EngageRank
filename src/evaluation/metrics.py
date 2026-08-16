import math


def reciprocal_rank(labels):
    for rank, label in enumerate(labels, start=1):
        if label == 1:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(labels, k):
    dcg = sum(
        label / math.log2(rank + 1)
        for rank, label in enumerate(labels[:k], start=1)
    )

    ideal = sorted(labels, reverse=True)[:k]

    idcg = sum(
        label / math.log2(rank + 1)
        for rank, label in enumerate(ideal, start=1)
    )

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(labels, k):
    total_relevant = sum(labels)

    if total_relevant == 0:
        return 0.0

    return sum(labels[:k]) / total_relevant


def intra_list_diversity(
    candidate_embeddings,
    ranked_indices,
    k
):
    top_indices = ranked_indices[:k]

    if len(top_indices) < 2:
        return 0.0

    total_distance = 0.0
    pairs = 0

    for i in range(len(top_indices)):
        for j in range(i + 1, len(top_indices)):

            similarity = (
                candidate_embeddings[top_indices[i]]
                @ candidate_embeddings[top_indices[j]]
            )

            distance = 1.0 - similarity

            total_distance += distance
            pairs += 1

    return total_distance / pairs