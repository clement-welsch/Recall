import pytest

from devassistant.search import aggregate_document_scores, search


def test_search(monkeypatch):
    documents = [
        {
            "content": "Python is a programming language.",
            "source": "python.md",
        },
        {
            "content": "RAG retrieves relevant information.",
            "source": "rag.md",
        },
    ]

    def fake_embed(texts):
        if texts == [
            "Python is a programming language.",
            "RAG retrieves relevant information.",
        ]:
            return [
                [0, 1],
                [1, 0],
            ]

        return [[1, 0]]

    monkeypatch.setattr(
        "devassistant.search.embed",
        fake_embed,
    )

    results = search(
        documents,
        "What is RAG?",
    )

    assert results[0] == {
        "chunk_score": 1.0,
        "source_score": 1.0,
        "chunk": {
            "content": "RAG retrieves relevant information.",
            "source": "rag.md",
        },
    }


def test_search_top_k(monkeypatch):
    documents = [
        {
            "content": "Python is a programming language.",
            "source": "python.md",
        },
        {
            "content": "RAG retrieves relevant information.",
            "source": "rag.md",
        },
        {
            "content": "C++ is used for game development.",
            "source": "cpp.md",
        },
    ]

    def fake_embed(texts):
        if texts == [
            "Python is a programming language.",
            "RAG retrieves relevant information.",
            "C++ is used for game development.",
        ]:
            return [
                [0, 1],
                [1, 0],
                [0, 0.5],
            ]

        return [[1, 0]]

    monkeypatch.setattr(
        "devassistant.search.embed",
        fake_embed,
    )

    results = search(
        documents,
        "What is RAG?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["chunk"]["source"] == "rag.md"


def test_search_keeps_document_metadata(monkeypatch):
    documents = [
        {
            "content": "Python is a programming language.",
            "source": "python.md",
        },
        {
            "content": "RAG retrieves relevant information.",
            "source": "rag.md",
        },
    ]

    def fake_embed(texts):
        if texts == [
            "Python is a programming language.",
            "RAG retrieves relevant information.",
        ]:
            return [
                [0, 1],
                [1, 0],
            ]

        return [[1, 0]]

    monkeypatch.setattr(
        "devassistant.search.embed",
        fake_embed,
    )

    results = search(
        documents,
        "What is RAG?",
    )

    assert results[0]["chunk"] == {
        "content": "RAG retrieves relevant information.",
        "source": "rag.md",
    }


def test_search_filters_results_below_score_threshold(monkeypatch):
    documents = [
        {
            "content": "Relevant document",
            "source": "relevant.md",
        },
        {
            "content": "Very relevant document",
            "source": "very_relevant.md",
        },
        {
            "content": "Irrelevant document",
            "source": "irrelevant.md",
        },
    ]

    def fake_embed(texts):
        if texts == ["What is relevant?"]:
            return [[1.0, 0.0]]

        return [
            [1.0, 0.0],
            [0.9, 0.0],
            [0.4, 0.0],
        ]

    def fake_similarity(a, b):
        return a[0]

    monkeypatch.setattr(
        "devassistant.search.embed",
        fake_embed,
    )

    monkeypatch.setattr(
        "devassistant.search.cosine_similarity",
        fake_similarity,
    )

    results = search(
        documents,
        question="What is relevant?",
        top_k=3,
        score_threshold=0.7,
    )

    assert [result["chunk_score"] for result in results] == [1.0, 0.9]
    assert [result["source_score"] for result in results] == [1.0, 0.9]


def test_aggregate_document_scores_uses_max_chunk_score():
    results = [
        (
            0.6,
            {
                "content": "First RAG chunk.",
                "source": "rag.md",
            },
        ),
        (
            0.9,
            {
                "content": "Second RAG chunk.",
                "source": "rag.md",
            },
        ),
        (
            0.8,
            {
                "content": "Python chunk.",
                "source": "python.md",
            },
        ),
    ]

    scores = aggregate_document_scores(results)

    assert scores == [
        (0.9, "rag.md"),
        (0.8, "python.md"),
    ]


def test_search_top_k_limits_documents(monkeypatch):
    documents = [
        {
            "content": "RAG chunk one.",
            "source": "rag.md",
        },
        {
            "content": "RAG chunk two.",
            "source": "rag.md",
        },
        {
            "content": "Python chunk.",
            "source": "python.md",
        },
        {
            "content": "C++ chunk.",
            "source": "cpp.md",
        },
    ]

    def fake_embed(texts):
        if len(texts) == 1:
            return [[1.0, 0.0]]

        return [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.6, 0.8],
            [0.0, 1.0],
        ]

    monkeypatch.setattr(
        "devassistant.search.embed",
        fake_embed,
    )

    results = search(
        documents,
        "What is relevant?",
        top_k=2,
    )

    sources = {result["chunk"]["source"] for result in results}

    assert sources == {
        "rag.md",
        "python.md",
    }

    assert len(results) == 3


def test_search_rejects_negative_top_k():
    with pytest.raises(ValueError, match="top_k"):
        search([], "What is relevant?", top_k=-1)


def test_search_returns_empty_list_when_documents_are_empty(monkeypatch):
    def fail_if_called(*args, **kwargs):
        pytest.fail("Embeddings should not be computed for empty documents")

    monkeypatch.setattr("devassistant.search.embed", fail_if_called)

    assert search([], "What is relevant?") == []


def test_search_returns_empty_list_when_top_k_is_zero(monkeypatch):
    def fail_if_called(*args, **kwargs):
        pytest.fail("Embeddings should not be computed when top_k is zero")

    monkeypatch.setattr("devassistant.search.embed", fail_if_called)

    documents = [
        {
            "content": "Example document",
            "source": "example.md",
        }
    ]

    assert search(documents, "What is relevant?", top_k=0) == []


@pytest.mark.parametrize(
    "score_threshold",
    [
        -1.01,
        1.01,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_search_rejects_invalid_score_threshold(score_threshold):
    with pytest.raises(ValueError, match="score_threshold"):
        search([], "What is relevant?", score_threshold=score_threshold)


@pytest.mark.parametrize("score_threshold", [-1.0, 1.0])
def test_search_accepts_score_threshold_boundaries(score_threshold):
    assert (
        search(
            [],
            "What is relevant?",
            score_threshold=score_threshold,
        )
        == []
    )


def test_aggregate_document_scores_breaks_ties_by_source():
    results = [
        (0.8, {"source": "zebra.md"}),
        (0.9, {"source": "guide.md"}),
        (0.8, {"source": "architecture.md"}),
    ]

    assert aggregate_document_scores(results) == [
        (0.9, "guide.md"),
        (0.8, "architecture.md"),
        (0.8, "zebra.md"),
    ]


def test_aggregate_document_scores_counts_each_source_once():
    results = [
        (0.7, {"source": "guide.md", "content": "Chunk 1"}),
        (0.9, {"source": "guide.md", "content": "Chunk 2"}),
        (0.8, {"source": "installation.md", "content": "Chunk 1"}),
        (0.6, {"source": "guide.md", "content": "Chunk 3"}),
    ]

    assert aggregate_document_scores(results) == [
        (0.9, "guide.md"),
        (0.8, "installation.md"),
    ]


def test_search_preserves_individual_chunk_scores_for_selected_sources(monkeypatch):
    documents = [
        {
            "content": "First RAG chunk.",
            "source": "rag.md",
        },
        {
            "content": "Second RAG chunk.",
            "source": "rag.md",
        },
        {
            "content": "Python chunk.",
            "source": "python.md",
        },
    ]

    def fake_embed(texts):
        if len(texts) == 1:
            return [[1.0, 0.0]]

        return [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.6, 0.8],
        ]

    monkeypatch.setattr(
        "devassistant.search.embed",
        fake_embed,
    )

    results = search(
        documents,
        "What is relevant?",
        top_k=1,
    )

    assert results == [
        {
            "chunk_score": 1.0,
            "source_score": 1.0,
            "chunk": documents[0],
        },
        {
            "chunk_score": 0.8,
            "source_score": 1.0,
            "chunk": documents[1],
        },
    ]
