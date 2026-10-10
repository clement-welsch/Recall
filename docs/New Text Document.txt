# Recall — System Architecture

## 1. Overview

Recall is a local AI-powered knowledge retrieval assistant designed to help developers retrieve information from their technical documentation.

It uses Retrieval-Augmented Generation (RAG) to find relevant content in local Markdown documents and provide that content as context to a language model running through LM Studio.

The architecture separates document loading, chunk retrieval, similarity scoring, context construction, and answer generation.

## 2. RAG Pipeline Architecture

![RAG Pipeline Architecture](../img/rag-pipeline-overview.png)

The RAG pipeline follows these main steps:

1. **Document loading** — Recall loads documents from the specified directory.
2. **Chunking** — Documents are split into smaller text segments with configurable chunk size and overlap.
3. **Embedding generation** — The embedding model converts document chunks and the user's question into vectors.
4. **Similarity search** — Recall computes cosine similarity between the question vector and document chunk vectors.
5. **Source ranking** — Chunks are filtered by the configured score threshold. Sources are ranked using the maximum score among their retained chunks.
6. **Context construction** — Recall assembles the selected chunks into a context for the language model.
7. **Answer generation** — The context and question are sent to the generation model through LM Studio.
8. **Response** — Recall returns the generated answer and the names of the source documents.

The current implementation uses an in-memory embedding cache to avoid redundant embedding requests for repeated text.

## 3. Adaptive Retrieval Fallback

![Adaptive Retrieval Fallback Flow](../img/adaptive-retrieval-fallback.png)

An adaptive retrieval fallback is a proposed extension to the current RAG pipeline.

The intended behavior is to prioritize local documentation when it contains relevant information and fall back to the model's general knowledge when the retrieved context is insufficient.

The planned decision flow is:

1. Search the local documentation.
2. Evaluate the retrieved results.
3. If the context is sufficient, generate an answer grounded in the local documents.
4. If the context is insufficient, generate a response using the model's general knowledge.
5. Clearly indicate whether the answer is based on local documentation, general model knowledge, or both.

**Implementation status:** this fallback strategy is not yet implemented. The current `get_answer()` function always builds a prompt and calls the generation model, even when the search returns no chunks. The model therefore receives an empty or incomplete context in that case.

A future implementation should distinguish the absence of search results from the presence of results that are not sufficiently relevant. Cosine similarity scores are not calibrated probabilities of relevance, so any relevance threshold must be evaluated rather than treated as a guarantee.

## 4. RAG Data Flow

![RAG Data Flow Diagram](../img/rag-data-flow.png)

The main data flow starts with a user question and a directory containing local documents.

### Inputs

* **Document directory** — the location of the documents to search.
* **Question** — the user's query.
* **Retrieval parameters** — chunk size, overlap, number of sources to select, and similarity score threshold.

### Processing

The document loader reads the available documents and produces chunks. Recall embeds the chunks and question, calculates cosine similarity scores, filters the results, and selects the highest-ranked sources.

The selected chunks are assembled into a textual context. The prompt builder combines this context with the question, and the generation client sends the prompt to LM Studio.

### Output

The current `get_answer()` function returns:

```python
{
    "answer": response,
    "sources": sources,
}
```

The `sources` field contains unique source names associated with the retrieved chunks.

Although the search layer exposes `chunk_score` and `source_score`, these scores are not currently included in the final RAG response.

## 5. Main Components

| Component            | Responsibility                                                                      |
| -------------------- | ----------------------------------------------------------------------------------- |
| `document_loader.py` | Loads documents from the specified directory.                                       |
| `chunking.py`        | Splits document content into chunks.                                                |
| `embeddings.py`      | Generates embeddings and maintains an in-memory LRU cache.                          |
| `similarity.py`      | Computes cosine similarity.                                                         |
| `search.py`          | Scores chunks, filters by threshold, aggregates source scores, and selects sources. |
| `prompt.py`          | Builds the prompt from the context and question.                                    |
| `lmstudio_client.py` | Communicates with LM Studio for answer generation.                                  |
| `rag.py`             | Coordinates retrieval, context construction, and answer generation.                 |
| `config.py`          | Provides application configuration.                                                 |

## 6. Search Scoring

Recall uses cosine similarity to compare the question embedding with each document chunk embedding.

The `score_threshold` parameter filters chunks before source ranking. Its default value is `0.0`.

For each source, Recall uses the maximum score among its retained chunks as the source score.

The `top_k` parameter determines the number of sources selected, not the total number of chunks returned. Multiple chunks from a selected source may be included in the final context.

These scores measure embedding similarity. They should not be interpreted as calibrated confidence values or proof that a chunk contains a complete and correct answer.

## 7. Embedding Cache

Recall maintains an in-memory Least Recently Used (LRU) cache for embeddings.

The cache has a maximum capacity of 1,024 entries. Cache keys include the embedding model and input text.

Repeated texts can reuse cached embeddings, and missing unique texts are sent to the embedding endpoint in batches. The cache is not persistent and is cleared when the process terminates.

This reduces redundant embedding API calls but does not replace a persistent document index.

## 8. Design Principles

The architecture follows these principles:

* **Local-first processing** — document retrieval and model inference are designed to use local resources.
* **Separation of concerns** — loading, chunking, embedding, search, prompting, and generation are handled by separate modules.
* **Configurable retrieval** — chunking and search behavior can be adjusted through parameters.
* **Transparent sourcing** — the current response includes the names of documents used to build the context.
* **Testability** — individual components can be tested independently, including with mocked model interactions.
* **Explicit limitations** — retrieval scores are treated as similarity measures, not guarantees of factual correctness.

## 9. Current Limitations and Future Work

The current implementation has several limitations:

* No persistent embedding index.
* Basic word-count-based chunking.
* No implemented adaptive fallback strategy.
* No comprehensive retrieval evaluation dataset.
* No calibrated relevance threshold.
* No source or chunk scores in the final RAG response.
* Limited evaluation of answers when retrieved context is insufficient.

Potential future improvements include:

1. Implementing adaptive retrieval fallback.
2. Evaluating retrieval quality with a representative question dataset.
3. Improving chunking and context selection.
4. Exposing source metadata and scores through the RAG response.
5. Adding persistent indexing if performance measurements justify it.
6. Evaluating answer grounding and behavior when no relevant context is available.

---

*This document describes the current architecture and explicitly identifies proposed behavior that has not yet been implemented.*
