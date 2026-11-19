from math import isfinite

from devassistant.embeddings import embed
from devassistant.similarity import cosine_similarity


def search(
    documents,
    question,
    top_k=3,
    score_threshold=0.0,
):
    """Search for relevant chunks and return their individual and source scores."""
    if top_k < 0:
        raise ValueError("top_k must be greater than or equal to 0")

    if not isfinite(score_threshold) or not -1.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be a finite value between -1.0 and 1.0")

    if top_k == 0 or not documents:
        return []

    contents = [document["content"] for document in documents]

    vectors_docs = embed(contents)
    vectors_question = embed([question])
    question_vector = vectors_question[0]

    scored_chunks = []

    for vector, document in zip(vectors_docs, documents):
        chunk_score = cosine_similarity(vector, question_vector)

        if chunk_score >= score_threshold:
            scored_chunks.append((chunk_score, document))

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    document_scores = aggregate_document_scores(scored_chunks)

    source_scores = {source: score for score, source in document_scores}

    selected_sources = {source for _, source in document_scores[:top_k]}

    return [
        {
            "chunk_score": chunk_score,
            "source_score": source_scores[chunk["source"]],
            "chunk": chunk,
        }
        for chunk_score, chunk in scored_chunks
        if chunk["source"] in selected_sources
    ]


def aggregate_document_scores(results):
    """Return each source's maximum chunk score, sorted by score then source."""
    document_scores = {}

    for score, document in results:
        source = document["source"]

        if source not in document_scores:
            document_scores[source] = score
        else:
            document_scores[source] = max(
                document_scores[source],
                score,
            )

    return sorted(
        [(score, source) for source, score in document_scores.items()],
        key=lambda item: (-item[0], item[1]),
    )
