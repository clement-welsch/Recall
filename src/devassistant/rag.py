from devassistant.document_loader import load_documents
from devassistant.search import search
from devassistant.lmstudio_client import ask
from devassistant.prompt import build_prompt


def build_context(documents):
    contexts = []

    for result in documents:
        chunk = result["chunk"]
        source = chunk["source"]

        contexts.append(
            f"Source: {source}\n"
            f"{chunk['content']}"
        )

    return "\n".join(contexts)


def get_answer(
    directory,
    question,
    chunk_size=60,
    overlap=12,
    top_k=3,
    score_threshold=0.0,
):
    documents = load_documents(
        directory,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    best_docs = search(
        documents,
        question,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    sources = []

    for result in best_docs:
        source = result["chunk"]["source"]

        if source not in sources:
            sources.append(source)

    context = build_context(best_docs)
    answer_mode = "local" if best_docs else "model"

    prompt = build_prompt(
        context,
        question,
        answer_mode=answer_mode,
    )

    response = ask(prompt)

    return {
        "answer": response,
        "sources": sources,
        "answer_mode": answer_mode,
    }