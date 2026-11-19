from devassistant.rag import get_answer


def test_rag_integration(tmp_path):
    python_file = tmp_path / "python.md"
    python_file.write_text("Python is a high-level programming language.")

    rag_file = tmp_path / "rag.md"
    rag_file.write_text(
        "RAG combines document retrieval with language model generation."
    )

    question = "What is RAG?"

    answer = get_answer(
        directory=tmp_path,
        question=question,
    )

    assert isinstance(answer, dict)
    assert answer["answer"]
    assert answer["sources"]
