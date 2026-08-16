def mmr_rerank(
    candidate_embeddings,
    relevance_scores,
    top_k=10,
    lambda_weight=0.9
):

    selected = []
    remaining = list(
        range(len(relevance_scores))
    )

    while remaining and len(selected) < top_k:

        best_idx = None
        best_score = -1e9

        for idx in remaining:

            relevance = relevance_scores[idx]

            if len(selected) == 0:
                diversity_penalty = 0.0

            else:
                similarities = (
                    candidate_embeddings[idx]
                    @ candidate_embeddings[selected].T
                )

                diversity_penalty = (
                    similarities.max()
                )

            score = (
                lambda_weight * relevance
                - (1 - lambda_weight)
                * diversity_penalty
            )

            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected