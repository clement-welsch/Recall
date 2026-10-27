def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> dict[str, float]:
    """Calculate per-query retrieval metrics at cutoff k.

    IDs represent source documents, ordered from most to least relevant in
    ``retrieved_ids``. Duplicate retrieved IDs count only at their first rank.
    ``relevant_ids`` must contain at least one source, and ``k`` must be a
    positive integer. Precision uses k as its denominator, including when
    fewer than k sources were retrieved.

    Return ``precision_at_k``, ``recall_at_k``, and
    ``reciprocal_rank_at_k``. An empty retrieval returns zero for every metric.
    See ``notebooks/2026-10-20-retrieval-evaluation.ipynb`` for the implementation
    exercise.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"Cutoff k must be a positive integer, got {k}.")

    if relevant_ids is None or len(relevant_ids) == 0:
        raise ValueError("relevant_ids must contain at least one source.")

    # Remove duplicate source IDs while preserving ranking order.
    unique_ids = []
    seen_ids = set()

    for source_id in retrieved_ids:
        if source_id not in seen_ids:
            unique_ids.append(source_id)
            seen_ids.add(source_id)

    # Keep at most the first k unique source IDs.
    top_k_ids = unique_ids[:k]

    # Count IDs in the selected results that belong to relevant_ids.
    hits = 0

    for source_id in top_k_ids:
        if source_id in relevant_ids:
            hits += 1

    # Calculate Precision@k.
    precision_at_k = hits / k

    # Calculate Recall@k.
    recall_at_k = hits / len(relevant_ids)

    # Calculate Reciprocal Rank@k.
    rank = 0

    for cur_rank, source_id in enumerate(top_k_ids, start=1):
        if source_id in relevant_ids:
            rank = cur_rank
            break

    reciprocal_rank_at_k = 1.0 / rank if rank > 0 else 0.0

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "reciprocal_rank_at_k": reciprocal_rank_at_k,
    }