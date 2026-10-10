import pytest

from devassistant.prompt import build_prompt


def test_build_prompt():
    prompt = build_prompt(
        "RAG retrieves relevant information.",
        "What does RAG retrieve?",
    )

    assert prompt == (
        "Context:\n"
        "RAG retrieves relevant information.\n"
        "Answer the question using the provided context. "
        "If the context does not contain the answer, "
        "say that the information was not found in the local documentation.\n"
        "Question:\n"
        "What does RAG retrieve?"
    )


def test_build_prompt_model_mode():
    prompt = build_prompt(
        "",
        "What is RAG?",
        answer_mode="model",
    )

    assert prompt == (
        "You are a helpful assistant.\n"
        "Answer the question using your general knowledge.\n"
        "No relevant local documentation was retrieved. "
        "Do not claim that your answer is based on local documents.\n"
        "If you are uncertain, state that clearly.\n"
        "Question:\n"
        "What is RAG?"
    )


def test_build_prompt_invalid_mode():
    with pytest.raises(ValueError, match="answer_mode"):
        build_prompt("", "What is RAG?", answer_mode="invalid")