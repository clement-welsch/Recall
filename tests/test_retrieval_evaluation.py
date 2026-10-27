import pytest

from devassistant.evaluation import evaluate_retrieval


def test_evaluate_retrieval_perfect_ranking():
    result = evaluate_retrieval(
        retrieved_ids=["python.md", "rag.md"],
        relevant_ids={"python.md", "rag.md"},
        k=2,
    )

    assert result == {
        "precision_at_k": 1.0,
        "recall_at_k": 1.0,
        "reciprocal_rank_at_k": 1.0,
    }


def test_evaluate_retrieval_partial_hits():
    result = evaluate_retrieval(
        retrieved_ids=["noise.md", "python.md", "rag.md"],
        relevant_ids={"python.md", "rag.md"},
        k=2,
    )

    assert result == {
        "precision_at_k": 0.5,
        "recall_at_k": 0.5,
        "reciprocal_rank_at_k": 0.5,
    }


def test_evaluate_retrieval_ignores_results_after_k():
    result = evaluate_retrieval(
        retrieved_ids=["noise.md", "python.md", "rag.md"],
        relevant_ids={"python.md", "rag.md", "docker.md"},
        k=2,
    )

    assert result == {
        "precision_at_k": 0.5,
        "recall_at_k": pytest.approx(1 / 3),
        "reciprocal_rank_at_k": 0.5,
    }


def test_evaluate_retrieval_deduplicates_sources_at_first_rank():
    result = evaluate_retrieval(
        retrieved_ids=["python.md", "python.md", "rag.md"],
        relevant_ids={"rag.md"},
        k=2,
    )

    assert result == {
        "precision_at_k": 0.5,
        "recall_at_k": 1.0,
        "reciprocal_rank_at_k": 0.5,
    }


def test_evaluate_retrieval_precision_uses_k_when_fewer_results_are_returned():
    result = evaluate_retrieval(
        retrieved_ids=["python.md"],
        relevant_ids={"python.md"},
        k=3,
    )

    assert result == {
        "precision_at_k": pytest.approx(1 / 3),
        "recall_at_k": 1.0,
        "reciprocal_rank_at_k": 1.0,
    }


def test_evaluate_retrieval_returns_zero_metrics_for_empty_results():
    result = evaluate_retrieval(
        retrieved_ids=[],
        relevant_ids={"python.md"},
        k=3,
    )

    assert result == {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "reciprocal_rank_at_k": 0.0,
    }


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_evaluate_retrieval_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="k"):
        evaluate_retrieval(["python.md"], {"python.md"}, k=k)


def test_evaluate_retrieval_rejects_empty_relevant_ids():
    with pytest.raises(ValueError, match="relevant_ids"):
        evaluate_retrieval(["python.md"], set(), k=1)
