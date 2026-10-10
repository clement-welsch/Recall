# Recall — Project Log

## 1. Project Overview

Recall is a local AI-powered knowledge retrieval assistant designed to help developers retrieve information from their technical documentation.

The project uses Retrieval-Augmented Generation (RAG) to retrieve relevant Markdown content and provide context to a local language model.

### Current technology stack

* **Language:** Python 3.12+
* **LLM runtime:** LM Studio
* **Generation model:** Gemma 4 E4B
* **Embedding model:** Nomic Embed Text v1.5
* **API client:** OpenAI-compatible Python client
* **Numerical operations:** NumPy
* **Testing:** pytest
* **Version control:** Git and GitHub

The application is designed to run locally, using LM Studio for embedding generation and answer generation.

---

## 2. Development History

### September 24, 2026 — Project Initialization

* Created the Recall repository.
* Established the initial project structure and development workflow.

### September 28, 2026 — Python Package and Embeddings

* Created the Python package structure.
* Integrated the LM Studio embedding endpoint.
* Implemented the initial embedding functionality.

### September 29, 2026 — Similarity Search and RAG

* Implemented cosine similarity calculations.
* Added tests for similarity calculations and embeddings.
* Implemented the initial RAG pipeline.
* Added document chunking.
* Introduced configurable retrieval parameters.
* Refactored prompt construction.

### October 7, 2026 — Source Tracking and Context Construction

* Added source tracking to retrieved document chunks.
* Implemented context construction for the RAG pipeline.
* Updated the answer-generation workflow to return the generated answer and the list of source documents.
* Added tests covering source tracking and context construction.

### October 7, 2026 — Search Threshold and Scoring Validation

* Introduced a configurable `score_threshold` parameter for similarity search.
* Defined cosine similarity as the basis for retrieval scoring.
* Added validation for search parameters.
* Stabilized score ordering and retrieval behavior.
* Added tests for parameter validation and scoring edge cases.

The default threshold is `0.0`. It is a cosine similarity threshold, not a calibrated probability of relevance.

PR #18: `Validate search parameters and stabilize scoring`.

### October 10, 2026 — Search Score Contract

Updated the search result contract to expose individual chunk scores and aggregated source scores.

Each search result contains:

* `chunk_score`: cosine similarity between the chunk embedding and the question embedding.
* `source_score`: the maximum score among the retained chunks belonging to that source.
* `chunk`: the original chunk and its associated metadata.

Source scores are aggregated using the maximum chunk score, rather than an average.

Sources are ordered by descending score. When scores are equal, source names provide a deterministic secondary ordering.

The `top_k` parameter selects the number of sources, not the number of individual chunks. All retained chunks belonging to the selected sources are returned.

The score threshold is applied to individual chunks before source aggregation.

PR #19: `Expose chunk and source scores in search results`.

### October 10, 2026 — Embedding Cache

Implemented an in-memory Least Recently Used (LRU) cache for embedding results.

The cache:

* Has a maximum capacity of 1,024 entries.
* Uses the embedding model and input text as its cache key.
* Reuses embeddings for previously processed texts.
* Deduplicates repeated texts within a single call.
* Sends missing unique texts to the embedding endpoint in batches.
* Preserves the original input order and duplicate entries in the returned results.
* Evicts the least recently used entries when the cache reaches capacity.
* Provides `clear_embedding_cache()` to clear the cache.

The cache reduces redundant embedding API calls for identical inputs during the current process lifetime.

It does not provide persistent storage of embeddings across application restarts.

PR #20: `Cache document embeddings with LRU eviction`.

---

## 3. Current Architecture

| Module               | Responsibility                                                                         |
| -------------------- | -------------------------------------------------------------------------------------- |
| `config.py`          | Application configuration                                                              |
| `lmstudio_client.py` | Communication with LM Studio and answer generation                                     |
| `embeddings.py`      | Embedding generation and in-memory LRU caching                                         |
| `similarity.py`      | Cosine similarity calculations                                                         |
| `search.py`          | Chunk scoring, threshold filtering, source score aggregation, and source selection     |
| `chunking.py`        | Splitting document content into chunks                                                 |
| `document_loader.py` | Loading documents for retrieval                                                        |
| `prompt.py`          | Constructing prompts for answer generation                                             |
| `rag.py`             | Orchestrating document loading, retrieval, context construction, and answer generation |

---

## 4. Current Search and RAG Contracts

### 4.1 Search

The `search()` function accepts:

* `documents`
* `question`
* `top_k`, defaulting to `3`
* `score_threshold`, defaulting to `0.0`

Validation rules:

* `top_k` must be greater than or equal to zero.
* `score_threshold` must be finite and between `-1.0` and `1.0`.
* If `top_k` is zero or the document list is empty, the function returns an empty list.

For non-empty searches, the function computes embeddings, scores chunks, applies the threshold, aggregates scores by source, selects the best sources, and returns their retained chunks.

Each result has the following structure:

```python
{
    "chunk_score": float,
    "source_score": float,
    "chunk": dict,
}
```

The exact numerical values depend on the embedding model and the cosine similarity calculation.

### 4.2 Source Score Aggregation

For each source, `aggregate_document_scores()` retains the maximum chunk score.

Sources are sorted by:

1. Descending score.
2. Ascending source name when scores are equal.

The source score represents the strongest matching chunk from that source. It is not an average score or a calibrated confidence measure.

### 4.3 RAG Answer Generation

The `get_answer()` function:

1. Loads documents from the specified directory.
2. Splits documents into chunks using the configured chunk size and overlap.
3. Retrieves relevant chunks using `search()`.
4. Collects unique source names.
5. Builds the context from the retained chunks.
6. Constructs the prompt.
7. Sends the prompt to LM Studio.
8. Returns the generated answer and source names.

The current return contract is:

```python
{
    "answer": response,
    "sources": sources,
}
```

**Important distinction:** chunk and source scores are available from `search()`, but `get_answer()` does not currently expose these scores in its returned dictionary.

The source list contains unique source names, not detailed source metadata or score information.

---

## 5. Current Configuration and Dependencies

The development environment was recorded as:

* Python `3.12.10`
* OpenAI-compatible Python client `3.19.2`
* NumPy `2.5.3`
* pytest `9.1.1`

These versions reflect the environment recorded during development and are not a guarantee that every environment uses the same versions.

The project uses LM Studio through its OpenAI-compatible local API.

The embedding and generation models are configured separately.

**Known dependency-management limitation:** runtime dependencies were not yet fully declared in `pyproject.toml` in the last documented environment review. This should be checked before considering dependency management complete.

---

## 6. Testing Status

The project has tests covering:

* Embedding generation.
* Embedding cache reuse.
* Repeated input deduplication.
* Batched embedding requests.
* LRU eviction.
* Similarity calculations.
* Search parameter validation.
* Search scoring and ordering.
* Source score aggregation.
* RAG behavior.
* Source tracking and context construction.

The user confirmed that the test suite passed on October 10, 2026, after the recent search scoring and embedding cache changes.

This confirms that the current test suite passes; it does not establish complete coverage of all possible inputs or production workloads.

---

## 7. Known Limitations

The following limitations remain relevant to the current implementation.

### Retrieval and indexing

* No persistent embedding index has been implemented.
* The in-memory embedding cache is lost when the process terminates.
* The cache avoids repeated API calls for identical inputs, but does not replace a persistent document index.
* The chunking strategy is basic and based on word counts.
* The documented default chunk size is 60 words, with an overlap of 12 words.
* Retrieval quality has not yet been established through a comprehensive evaluation dataset.
* The cosine similarity threshold is configurable but is not calibrated as a probability of relevance.

### Source scoring

* A source score is defined as the maximum retained chunk score.
* This score does not measure the overall quality or completeness of a source.
* Search results expose scores, but the RAG response currently returns only the answer and unique source names.

### Generation and grounding

* Prompt constraints remain relatively limited.
* The system does not yet have a comprehensive evaluation of factual grounding or unsupported answers.
* Behavior when no relevant context is found requires further evaluation.

### Input formats

* Document ingestion is currently focused on Markdown.
* Support for additional document formats has not been established as implemented.

### Error handling and configuration

* Error handling and configuration robustness may require further improvement.
* Runtime dependency declarations should be verified and completed in `pyproject.toml`.

---

## 8. Potential Next Steps

The following items are candidates for future work, not completed features.

1. Improve prompt instructions to constrain answers to retrieved evidence.
2. Evaluate behavior when no relevant context is found.
3. Create a retrieval evaluation dataset with representative developer questions.
4. Measure retrieval quality and tune the chunking strategy.
5. Implement a persistent embedding index.
6. Measure embedding latency, search latency, and memory consumption.
7. Improve document chunking beyond fixed word counts.
8. Consider ingestion of additional document formats.
9. Complete runtime dependency declarations in `pyproject.toml`.
10. Improve error handling and configuration validation.
11. Evaluate whether exposing source and chunk scores through the RAG response would be useful to downstream consumers.

Future work should be prioritized based on measurable improvements to retrieval quality, reliability, usability, or performance.

---

## 9. Project Log Maintenance Policy

This file records the development history and the state of the implementation.

When updating this log:

* Add a dated entry for significant completed milestones.
* Record the relevant commit or pull request when known.
* Distinguish implemented features from proposed work.
* Record test results only when they have actually been observed.
* Update the architecture and known limitations when implementation changes affect them.
* Avoid treating a successful test run as proof of complete correctness.
* Keep the README focused on how to use the current project.
* Keep this file focused on development history, implementation status, and outstanding work.
