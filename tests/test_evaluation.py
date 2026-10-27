import pytest

from devassistant.evaluation import evaluate_retrieval


def test_evaluate_retrieval_returns_expected_metrics() -> None:
    """Calculate all metrics correctly for a standard example."""
    retrieved_ids = ["noise.md", "python.md", "rag.md"]
    relevant_ids = {"python.md", "rag.md"}

    result = evaluate_retrieval(retrieved_ids, relevant_ids, k=2)

    assert result == {
        "precision_at_k": 0.5,
        "recall_at_k": 0.5,
        "reciprocal_rank_at_k": 0.5,
    }


def test_evaluate_retrieval_returns_exact_metric_keys() -> None:
    """Return exactly the three expected metrics."""
    result = evaluate_retrieval(["python.md"], {"python.md"}, k=1)

    assert set(result.keys()) == {
        "precision_at_k",
        "recall_at_k",
        "reciprocal_rank_at_k",
    }
    assert all(isinstance(value, float) for value in result.values())


def test_evaluate_retrieval_returns_zero_for_empty_results() -> None:
    """Return zero for every metric when no documents are retrieved."""
    result = evaluate_retrieval([], {"python.md"}, k=5)

    assert result == {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "reciprocal_rank_at_k": 0.0,
    }


def test_evaluate_retrieval_deduplicates_before_applying_k() -> None:
    """Keep first occurrences and apply k to unique source IDs."""
    retrieved_ids = ["noise.md", "noise.md", "python.md"]
    relevant_ids = {"python.md"}

    result = evaluate_retrieval(retrieved_ids, relevant_ids, k=2)

    assert result["precision_at_k"] == 0.5
    assert result["recall_at_k"] == 1.0
    assert result["reciprocal_rank_at_k"] == 0.5


def test_evaluate_retrieval_preserves_first_occurrence_order() -> None:
    """Calculate the reciprocal rank using the original ranking order."""
    retrieved_ids = ["python.md", "noise.md", "python.md", "rag.md"]
    relevant_ids = {"rag.md"}

    result = evaluate_retrieval(retrieved_ids, relevant_ids, k=3)

    assert result["precision_at_k"] == pytest.approx(1 / 3)
    assert result["recall_at_k"] == 1.0
    assert result["reciprocal_rank_at_k"] == 1 / 3


def test_evaluate_retrieval_uses_k_as_precision_denominator() -> None:
    """Use k as the denominator even when fewer than k results exist."""
    result = evaluate_retrieval(["python.md"], {"python.md"}, k=5)

    assert result["precision_at_k"] == 0.2
    assert result["recall_at_k"] == 1.0
    assert result["reciprocal_rank_at_k"] == 1.0


def test_evaluate_retrieval_returns_zero_reciprocal_rank_when_no_hit() -> None:
    """Return zero reciprocal rank when no relevant source is retrieved."""
    result = evaluate_retrieval(
        ["noise.md", "other.md"],
        {"python.md"},
        k=2,
    )

    assert result["precision_at_k"] == 0.0
    assert result["recall_at_k"] == 0.0
    assert result["reciprocal_rank_at_k"] == 0.0


@pytest.mark.parametrize("k", [0, -1, 2.5, True, False, "5"])
def test_evaluate_retrieval_rejects_invalid_k(k: object) -> None:
    """Reject cutoffs that are not positive integers."""
    with pytest.raises(ValueError, match="Cutoff k"):
        evaluate_retrieval(["python.md"], {"python.md"}, k=k)


def test_evaluate_retrieval_rejects_empty_relevant_ids() -> None:
    """Reject an empty set of relevant source IDs."""
    with pytest.raises(ValueError, match="relevant_ids"):
        evaluate_retrieval(["python.md"], set(), k=5)
