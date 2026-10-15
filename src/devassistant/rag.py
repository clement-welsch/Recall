from math import isfinite

from devassistant.document_loader import load_documents
from devassistant.models import KnowledgeDomain
from devassistant.search import search
from devassistant.lmstudio_client import ask
from devassistant.prompt import build_prompt


def build_context(documents):
    contexts = []

    for result in documents:
        chunk = result["chunk"]
        source = chunk["source"]
        domain = f"Domain: {result['domain']}\n" if "domain" in result else ""

        contexts.append(
            f"{domain}Source: {source}\n"
            f"{chunk['content']}"
        )

    return "\n".join(contexts)


def retrieve_documents(
    directory,
    question,
    chunk_size=60,
    overlap=12,
    top_k=3,
    score_threshold=0.0,
):
    """Load and rank relevant chunks from one document directory.

    Return search results containing chunk and score metadata without building
    a prompt or calling the language model.
    """
    documents = load_documents(
        directory,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return search(
        documents,
        question,
        top_k=top_k,
        score_threshold=score_threshold,
    )


def get_answer(
    directory,
    question,
    chunk_size=60,
    overlap=12,
    top_k=3,
    score_threshold=0.0,
):
    best_docs = retrieve_documents(
        directory,
        question,
        chunk_size=chunk_size,
        overlap=overlap,
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


def get_answer_for_domains(
    domains,
    question,
    chunk_size=60,
    overlap=12,
    top_k=3,
    score_threshold=0.0,
):
    """Generate one answer from retrieved chunks in configured domains.

    Domains share the retrieval implementation. The response identifies each
    source by domain and filename. Scores are compared across domains, so the
    same embedding model and retrieval settings must be used for all domains.
    """
    if top_k < 0:
        raise ValueError("top_k must be greater than or equal to 0")
    if not isfinite(score_threshold) or not -1.0 <= score_threshold <= 1.0:
        raise ValueError(
            "score_threshold must be a finite value between -1.0 and 1.0"
        )

    domain_results = []
    domain_names = {}
    domain_order = {}
    seen_domain_ids = set()

    for domain in domains:
        if not isinstance(domain, KnowledgeDomain):
            raise TypeError("domains must contain KnowledgeDomain instances")
        if domain.id in seen_domain_ids:
            raise ValueError("domains must not contain duplicate domain IDs")
        seen_domain_ids.add(domain.id)
        domain_names[domain.id] = domain.name
        domain_order[domain.id] = len(domain_order)

        results = retrieve_documents(
            domain.directory,
            question,
            chunk_size=chunk_size,
            overlap=overlap,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        for result in results:
            domain_results.append(
                {
                    **result,
                    "domain": domain.name,
                    "domain_id": domain.id,
                }
            )

    source_scores = {}
    for result in domain_results:
        source = result["chunk"]["source"]
        source_key = (result["domain_id"], source)
        source_scores[source_key] = max(
            source_scores.get(source_key, float("-inf")),
            result["source_score"],
        )

    ranked_sources = sorted(
        source_scores,
        key=lambda key: (
            -source_scores[key],
            domain_names[key[0]].casefold(),
            key[1],
            domain_order[key[0]],
        ),
    )
    selected_sources = set(ranked_sources[:top_k])
    best_docs = [
        result
        for result in domain_results
        if (result["domain_id"], result["chunk"]["source"])
        in selected_sources
    ]
    best_docs.sort(
        key=lambda result: (
            -result["chunk_score"],
            result["domain"].casefold(),
            result["chunk"]["source"],
            domain_order[result["domain_id"]],
        )
    )

    sources = []
    seen_sources = set()
    for result in best_docs:
        source_key = (result["domain_id"], result["chunk"]["source"])
        if source_key not in seen_sources:
            sources.append(
                {
                    "domain": result["domain"],
                    "source": result["chunk"]["source"],
                }
            )
            seen_sources.add(source_key)

    context = build_context(best_docs)
    answer_mode = "local" if best_docs else "model"
    prompt = build_prompt(context, question, answer_mode=answer_mode)
    response = ask(prompt)

    return {
        "answer": response,
        "sources": sources,
        "answer_mode": answer_mode,
    }
