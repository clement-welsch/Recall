
# Agent Instructions — Recall

## Project Overview

Recall is a local, Python-based knowledge retrieval assistant for developers.

It combines semantic search, document retrieval, and Retrieval-Augmented Generation (RAG) using local models served through LM Studio's OpenAI-compatible API.

The repository is named `Recall`. Its Python package is named `devassistant`.

## Project Architecture

Main source directory:

`src/devassistant/`

Core modules:

- `config.py`: LM Studio and model configuration.
- `document_loader.py`: Markdown document ingestion.
- `chunking.py`: Word-based text chunking with overlap.
- `embeddings.py`: Embedding generation and LRU caching.
- `similarity.py`: Cosine similarity calculations.
- `search.py`: Semantic retrieval and ranking.
- `prompt.py`: Prompt construction.
- `lmstudio_client.py`: Local model API integration.
- `rag.py`: RAG orchestration and answer generation.
- `evaluation.py`: Retrieval evaluation exercises.

Additional directories:

- `tests/`: Automated tests.
- `notebooks/`: Guided development exercises.
- `docs/architecture/`: Architecture documentation.

Read `README.md` and the relevant architecture documentation before making structural changes.

## Technology Stack

- Python 3.12 or later.
- pytest for automated testing.
- LM Studio for local inference.
- OpenAI-compatible local API.
- Nomic Embed Text v1.5 for embeddings.
- Gemma 4 E4B for answer generation.

Default local API endpoint:

`http://localhost:1234/v1`

Do not assume LM Studio or its models are available in CI.

## Development Principles

1. Prefer simple, modular Python implementations.
2. Preserve separation of concerns between modules.
3. Avoid unnecessary dependencies.
4. Preserve existing public APIs unless a change is explicitly required.
5. Maintain backward compatibility where practical.
6. Use descriptive names and type hints for new code.
7. Document non-obvious algorithms and design decisions.
8. Do not introduce network services or cloud inference without explicit approval.
9. Avoid unrelated refactoring.
10. Keep changes focused and independently testable.

## RAG-Specific Requirements

Preserve the following behavior:

- Markdown document ingestion.
- Configurable chunk size and overlap.
- Cosine similarity ranking.
- Configurable similarity thresholds.
- Top-k source selection.
- Source provenance.
- Embedding cache behavior.
- Adaptive retrieval fallback.
- Multi-domain retrieval.

Important API contracts:

`get_answer()` supports single-domain retrieval.

`get_answer_for_domains()` supports retrieval across explicit `KnowledgeDomain` collections.

The answer mode is:

- `local`: At least one retrieval result exists.
- `model`: No retrieval result exists.

Both modes use the configured local LLM.

Do not interpret `local` as proof that retrieved context is sufficient or factually correct.

Do not silently change the meaning of `top_k`. It currently limits selected source files, not necessarily individual chunks.

## Embedding Requirements

The configured embedding model produces 768-dimensional vectors.

The in-memory LRU cache:

- Has a maximum capacity of 1,024 entries.
- Uses model identifier and input text as its key.
- Deduplicates repeated input texts.
- Batches missing embedding requests.
- Preserves input order.
- Evicts least recently used entries.

Preserve these properties when modifying embedding code.

Do not introduce persistent embedding storage without documenting cache invalidation and model compatibility.

## Testing Requirements

Run the automated test suite:

```bash
python -m pytest -v
```

Run focused tests during development:

```bash
python -m pytest -v tests/test_search.py
python -m pytest -v tests/test_rag.py
```

Before declaring a task complete:

1. Run relevant tests.
2. Run the complete test suite when feasible.
3. Report passed, failed, and skipped tests.
4. Identify tests that could not be executed.
5. Do not claim success without execution evidence.

Prefer mocks and deterministic fixtures for unit tests.

Tests requiring a running LM Studio server must be identified separately.

Never require live local model inference for ordinary unit tests.

## Educational Development Rules

Recall is also a learning project.

The `notebooks/` directory contains guided exercises designed to help the developer progressively implement new capabilities.

When working on an educational exercise:

1. Explain the objective and expected behavior.
2. Identify the relevant modules.
3. Provide progressive hints before complete solutions.
4. Encourage the developer to implement the solution.
5. Use acceptance tests to validate progress.
6. Review submitted implementations.
7. Explain failures and suggest focused corrections.

Do not automatically complete an educational exercise unless explicitly requested.

Do not replace an unfinished exercise with a complete implementation merely to make acceptance tests pass.

Tests intentionally raising `NotImplementedError` may represent planned exercises rather than regressions.

Distinguish completed functionality from work in progress.

## Documentation

Update documentation when changes affect:

- Public APIs.
- Configuration.
- Model requirements.
- Retrieval behavior.
- Architecture.
- Installation.
- Test commands.

Keep `README.md` consistent with the implementation.

Use the existing architecture documentation:

`docs/architecture/system-architecture.md`

Name new educational notebooks using:

`YYYY-MM-DD-<topic>.ipynb`

## Security and Privacy

Recall is designed for local processing.

- Never commit credentials or API secrets.
- Never commit private course documents without authorization.
- Do not upload user documents to external services.
- Avoid logging document contents unnecessarily.
- Keep local configuration and generated artifacts out of commits where appropriate.

The default LM Studio API key is a local placeholder, not a production secret.

## Git Workflow

- Prefer small, focused changes.
- Use descriptive branch names.
- Write meaningful commit messages.
- Include tests for behavioral changes.
- Update documentation when necessary.
- Do not merge pull requests without explicit authorization.
- Do not modify unrelated files.

## Completion Report

When completing a development task, report:

- Summary of changes.
- Files modified.
- Tests executed and their results.
- Known limitations.
- Remaining work.

Never claim that a feature is implemented, tested, or merged unless that action has been verified.
