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
    raise NotImplementedError(
        "Implement the retrieval metrics described in the guided TP."
    )
