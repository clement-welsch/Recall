import pytest

from devassistant.rag import build_context, get_answer


@pytest.fixture
def documents(tmp_path):
    python_file = tmp_path / "python.md"
    python_file.write_text(
        "Python is a high-level programming language."
    )

    rag_file = tmp_path / "rag.md"
    rag_file.write_text(
        "RAG combines document retrieval with language model generation."
    )

    return tmp_path


@pytest.fixture
def fake_search():
    def _fake_search(
        documents,
        question,
        top_k=3,
        score_threshold=0.0,
    ):
        return [
            {
                "chunk_score": 1.0,
                "source_score": 1.0,
                "chunk": {
                    "content": (
                        "RAG combines document retrieval "
                        "with language model generation."
                    ),
                    "source": "rag.md",
                },
            },
            {
                "chunk_score": 0.7,
                "source_score": 0.7,
                "chunk": {
                    "content": (
                        "Python is a high-level programming language."
                    ),
                    "source": "python.md",
                },
            },
        ]

    return _fake_search


@pytest.fixture
def fake_ask():
    captured = {}

    def _fake_ask(prompt):
        captured["prompt"] = prompt
        return "Here is the generated answer."

    return _fake_ask, captured


def test_get_answer(monkeypatch, documents, fake_ask):
    def fake_search(
        documents,
        question,
        top_k=3,
        score_threshold=0.0,
    ):
        return [
            {
                "chunk_score": 1.0,
                "source_score": 1.0,
                "chunk": {
                    "content": (
                        "RAG combines document retrieval "
                        "with language model generation."
                    ),
                    "source": "rag.md",
                },
            },
            {
                "chunk_score": 0.7,
                "source_score": 0.7,
                "chunk": {
                    "content": (
                        "Python is a high-level programming language."
                    ),
                    "source": "python.md",
                },
            },
        ]

    ask, captured = fake_ask

    monkeypatch.setattr("devassistant.rag.search", fake_search)
    monkeypatch.setattr("devassistant.rag.ask", ask)

    answer = get_answer(
        directory=documents,
        question="What is RAG?",
    )

    assert answer == {
        "answer": "Here is the generated answer.",
        "sources": ["rag.md", "python.md"],
        "answer_mode": "local",
    }

    assert captured["prompt"] == (
        "Context:\n"
        "Source: rag.md\n"
        "RAG combines document retrieval with language model generation.\n"
        "Source: python.md\n"
        "Python is a high-level programming language.\n"
        "Answer the question using the provided context. "
        "If the context does not contain the answer, "
        "say that the information was not found in the local documentation.\n"
        "Question:\n"
        "What is RAG?"
    )


def test_get_answer_empty_directory(monkeypatch, tmp_path, fake_ask):
    ask, captured = fake_ask

    monkeypatch.setattr(
        "devassistant.rag.search",
        lambda documents, question, top_k=3, score_threshold=0.0: [],
    )
    monkeypatch.setattr("devassistant.rag.ask", ask)

    answer = get_answer(
        directory=tmp_path,
        question="What is RAG?",
    )

    assert answer == {
        "answer": "Here is the generated answer.",
        "sources": [],
        "answer_mode": "model",
    }

    assert captured["prompt"] == (
        "You are a helpful assistant.\n"
        "Answer the question using your general knowledge.\n"
        "No relevant local documentation was retrieved. "
        "Do not claim that your answer is based on local documents.\n"
        "If you are uncertain, state that clearly.\n"
        "Question:\n"
        "What is RAG?"
    )


def test_get_answer_with_chunks(monkeypatch, tmp_path):
    document = tmp_path / "document.md"
    document.write_text(
        "Python is a programming language. "
        "RAG retrieves relevant document chunks. "
        "C++ is commonly used for game development."
    )

    captured = {}

    def fake_search(
        documents,
        question,
        top_k=3,
        score_threshold=0.0,
    ):
        captured["documents"] = documents

        return [
            {
                "chunk_score": 1.0,
                "source_score": 1.0,
                "chunk": documents[1],
            },
        ]

    def fake_ask(prompt):
        captured["prompt"] = prompt
        return "RAG retrieves relevant information."

    monkeypatch.setattr("devassistant.rag.search", fake_search)
    monkeypatch.setattr("devassistant.rag.ask", fake_ask)

    answer = get_answer(
        directory=tmp_path,
        question="What does RAG retrieve?",
        chunk_size=10,
        overlap=2,
    )

    assert answer == {
        "answer": "RAG retrieves relevant information.",
        "sources": ["document.md"],
        "answer_mode": "local",
    }

    assert len(captured["documents"]) > 1

    assert captured["prompt"] == (
        "Context:\n"
        f"Source: {captured['documents'][1]['source']}\n"
        f"{captured['documents'][1]['content']}\n"
        "Answer the question using the provided context. "
        "If the context does not contain the answer, "
        "say that the information was not found in the local documentation.\n"
        "Question:\n"
        "What does RAG retrieve?"
    )


def test_get_answer_with_top_k(monkeypatch, tmp_path):
    document = tmp_path / "document.md"
    document.write_text(
        "Python is a programming language. "
        "RAG retrieves relevant document chunks. "
        "C++ is commonly used for game development."
    )

    captured = {}

    def fake_search(
        documents,
        question,
        top_k=3,
        score_threshold=0.0,
    ):
        captured["top_k"] = top_k
        return []

    def fake_ask(prompt):
        return "Answer"

    monkeypatch.setattr("devassistant.rag.search", fake_search)
    monkeypatch.setattr("devassistant.rag.ask", fake_ask)

    answer = get_answer(
        directory=tmp_path,
        question="What does RAG retrieve?",
        chunk_size=10,
        overlap=2,
        top_k=5,
    )

    assert answer == {
        "answer": "Answer",
        "sources": [],
        "answer_mode": "model",
    }

    assert captured["top_k"] == 5


def test_get_answer_includes_sources(monkeypatch, tmp_path):
    document = tmp_path / "rag.md"
    document.write_text("RAG retrieves relevant information.")

    def fake_search(
        documents,
        question,
        top_k=3,
        score_threshold=0.0,
    ):
        return [
            {
                "chunk_score": 1.0,
                "source_score": 1.0,
                "chunk": {
                    "content": "RAG retrieves relevant information.",
                    "source": "rag.md",
                },
            },
        ]

    def fake_ask(prompt):
        return "Answer"

    monkeypatch.setattr("devassistant.rag.search", fake_search)
    monkeypatch.setattr("devassistant.rag.ask", fake_ask)

    answer = get_answer(
        directory=tmp_path,
        question="What does RAG retrieve?",
    )

    assert answer == {
        "answer": "Answer",
        "sources": ["rag.md"],
        "answer_mode": "local",
    }


def test_get_answer_returns_sources(monkeypatch, tmp_path):
    document = tmp_path / "rag.md"
    document.write_text("RAG retrieves relevant information.")

    def fake_search(
        documents,
        question,
        top_k=3,
        score_threshold=0.0,
    ):
        return [
            {
                "chunk_score": 1.0,
                "source_score": 1.0,
                "chunk": {
                    "content": "RAG retrieves relevant information.",
                    "source": "rag.md",
                },
            },
        ]

    def fake_ask(prompt):
        return "RAG retrieves relevant information."

    monkeypatch.setattr("devassistant.rag.search", fake_search)
    monkeypatch.setattr("devassistant.rag.ask", fake_ask)

    result = get_answer(
        directory=tmp_path,
        question="What does RAG retrieve?",
    )

    assert result == {
        "answer": "RAG retrieves relevant information.",
        "sources": ["rag.md"],
        "answer_mode": "local",
    }


def test_get_answer_returns_unique_sources(monkeypatch, tmp_path):
    document = tmp_path / "rag.md"
    document.write_text("RAG retrieves relevant information.")

    def fake_search(
        documents,
        question,
        top_k=3,
        score_threshold=0.0,
    ):
        return [
            {
                "chunk_score": 1.0,
                "source_score": 1.0,
                "chunk": {
                    "content": "First relevant chunk.",
                    "source": "rag.md",
                },
            },
            {
                "chunk_score": 0.9,
                "source_score": 1.0,
                "chunk": {
                    "content": "Second relevant chunk.",
                    "source": "rag.md",
                },
            },
            {
                "chunk_score": 0.8,
                "source_score": 0.8,
                "chunk": {
                    "content": "Another relevant chunk.",
                    "source": "python.md",
                },
            },
        ]

    def fake_ask(prompt):
        return "Here is the generated answer."

    monkeypatch.setattr("devassistant.rag.search", fake_search)
    monkeypatch.setattr("devassistant.rag.ask", fake_ask)

    result = get_answer(
        directory=tmp_path,
        question="What does RAG retrieve?",
    )

    assert result == {
        "answer": "Here is the generated answer.",
        "sources": ["rag.md", "python.md"],
        "answer_mode": "local",
    }


def test_build_context():
    documents = [
        {
            "chunk_score": 1.0,
            "source_score": 1.0,
            "chunk": {
                "content": "First relevant chunk.",
                "source": "rag.md",
            },
        },
        {
            "chunk_score": 0.8,
            "source_score": 0.8,
            "chunk": {
                "content": "Second relevant chunk.",
                "source": "python.md",
            },
        },
    ]

    context = build_context(documents)

    assert context == (
        "Source: rag.md\n"
        "First relevant chunk.\n"
        "Source: python.md\n"
        "Second relevant chunk."
    )


def test_get_answer_passes_score_threshold(monkeypatch, tmp_path):
    document = tmp_path / "rag.md"
    document.write_text("RAG retrieves relevant information.")

    captured = {}

    def fake_search(
        documents,
        question,
        top_k=3,
        score_threshold=0.0,
    ):
        captured["score_threshold"] = score_threshold

        return [
            {
                "chunk_score": 1.0,
                "source_score": 1.0,
                "chunk": {
                    "content": "RAG retrieves relevant information.",
                    "source": "rag.md",
                },
            },
        ]

    def fake_ask(prompt):
        return "Here is the generated answer."

    monkeypatch.setattr("devassistant.rag.search", fake_search)
    monkeypatch.setattr("devassistant.rag.ask", fake_ask)

    get_answer(
        directory=tmp_path,
        question="What does RAG retrieve?",
        score_threshold=0.7,
    )

    assert captured["score_threshold"] == 0.7


def test_get_answer_excludes_documents_below_score_threshold(
    monkeypatch,
    tmp_path,
):
    document = tmp_path / "rag.md"
    document.write_text("RAG retrieves relevant information.")

    captured = {}

    def fake_search(
        documents,
        question,
        top_k=3,
        score_threshold=0.0,
    ):
        assert score_threshold == 0.7

        return [
            {
                "chunk_score": 0.9,
                "source_score": 0.9,
                "chunk": {
                    "content": "Relevant information.",
                    "source": "relevant.md",
                },
            },
        ]

    def fake_ask(prompt):
        captured["prompt"] = prompt
        return "Here is the generated answer."

    monkeypatch.setattr("devassistant.rag.search", fake_search)
    monkeypatch.setattr("devassistant.rag.ask", fake_ask)

    result = get_answer(
        directory=tmp_path,
        question="What is relevant?",
        score_threshold=0.7,
    )

    assert result["answer"] == "Here is the generated answer."
    assert result["sources"] == ["relevant.md"]
    assert result["answer_mode"] == "local"
    assert "Relevant information." in captured["prompt"]
    assert "Irrelevant information." not in captured["prompt"]