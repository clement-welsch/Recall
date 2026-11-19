
# Contributing to Recall

Thank you for your interest in contributing to Recall.

Recall is a local Retrieval-Augmented Generation (RAG) assistant built with Python and LM Studio.

The project focuses on modular architecture, reliable semantic retrieval, local inference, and progressive learning through practical development exercises.

## 1. Development Environment

### Requirements

- Python 3.12 or later.
- Git.
- LM Studio for local inference and integration testing.
- A compatible chat model.
- A compatible embedding model.

The currently documented models are:

- Chat: `gemma-4-e4b`
- Embeddings: `text-embedding-nomic-embed-text-v1.5`

### Clone the Repository

```bash
git clone https://github.com/clement-welsch/Recall.git
cd Recall
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Install pytest if necessary:

```bash
python -m pip install pytest
```

Verify the environment:

```bash
python --version
python -m pip check
```

## 2. Project Structure

```text
Recall/
├── docs/
│   └── architecture/
├── notebooks/
├── src/
│   └── devassistant/
├── tests/
├── pyproject.toml
├── requirements.txt
├── README.md
└── PROJECT_LOG.md
```

The Python package is `devassistant`.

The repository and application are named Recall.

## 3. Development Workflow

The recommended contribution workflow is:

1. Start from an up-to-date development branch.
2. Create a dedicated feature or fix branch.
3. Implement a focused change.
4. Add or update relevant tests.
5. Run the appropriate checks.
6. Update documentation when necessary.
7. Open a pull request for review.

Example:

```bash
git checkout develop
git pull origin develop
git checkout -b feat/retrieval-evaluation
```

Suggested branch prefixes:

| Prefix | Purpose |
|--------|---------|
| `feat/` | New functionality |
| `fix/` | Bug fixes |
| `docs/` | Documentation |
| `test/` | Tests |
| `refactor/` | Internal restructuring |
| `chore/` | Maintenance |

These prefixes are recommended conventions, not enforced repository rules.

## 4. Coding Guidelines

Follow these principles:

- Write clear, maintainable Python.
- Keep modules focused on a single responsibility.
- Prefer explicit behavior over unnecessary abstractions.
- Use descriptive function and variable names.
- Add type hints to new public interfaces.
- Keep functions reasonably small.
- Avoid unnecessary third-party dependencies.
- Preserve existing interfaces when possible.
- Document important design decisions.

Before introducing a new dependency, explain why the standard library or existing dependencies are insufficient.

## 5. Testing

Recall uses pytest.

Run the complete test suite:

```bash
python -m pytest -v
```

Run a specific test module:

```bash
python -m pytest -v tests/test_search.py
```

Run the RAG integration test separately:

```bash
python -m pytest -v tests/test_rag_integration.py
```

### Unit Tests

Unit tests should:

- Be deterministic.
- Avoid unnecessary external dependencies.
- Mock LM Studio API calls when appropriate.
- Cover expected behavior and edge cases.
- Verify error handling.
- Avoid relying on private local documents.

### Integration Tests

Some integration tests require LM Studio.

Start the local API server at:

`http://localhost:1234/v1`

Verify that the required models are available and loaded.

Integration tests must clearly document their external requirements.

A successful unit test suite does not guarantee retrieval quality or factual accuracy.

## 6. RAG Development Guidelines

When modifying the retrieval pipeline, consider:

- Chunk size and overlap.
- Embedding model compatibility.
- Similarity calculations.
- Retrieval ranking.
- Similarity thresholds.
- Top-k behavior.
- Source attribution.
- Answer mode selection.
- Multi-domain retrieval.

The current retrieval pipeline uses cosine similarity.

Similarity scores are ranking signals, not calibrated probabilities.

Changes to retrieval algorithms should include tests and, where possible, measurable quality comparisons.

Avoid claiming retrieval improvements based solely on anecdotal examples.

## 7. Educational Exercises

Recall includes guided development exercises in the `notebooks/` directory.

These exercises are designed for progressive implementation and learning.

A typical exercise includes:

1. A development objective.
2. Acceptance criteria.
3. Relevant technical context.
4. Progressive hints.
5. Automated acceptance tests.
6. Implementation work.
7. Validation and review.

Use the notebook naming convention:

`YYYY-MM-DD-<topic>.ipynb`

An unfinished exercise may contain tests that intentionally fail until the expected implementation is completed.

Do not remove or bypass acceptance tests simply to obtain a passing test suite.

When contributing to an exercise, distinguish between:

- Implemented functionality.
- Planned functionality.
- Tests awaiting implementation.

## 8. Documentation

Update the documentation whenever a contribution changes:

- Installation instructions.
- Public APIs.
- Configuration.
- Supported models.
- Retrieval behavior.
- Architecture.
- Test procedures.

Relevant files include:

- `README.md`
- `PROJECT_LOG.md`
- `docs/architecture/system-architecture.md`

Keep examples executable and consistent with the current implementation.

## 9. Commit Messages

Use concise, descriptive commit messages.

Suggested format:

```text
<type>: <short description>
```

Examples:

```text
feat: add retrieval evaluation metrics
fix: preserve source provenance across domains
test: cover embedding cache eviction
docs: document adaptive fallback behavior
refactor: simplify semantic search ranking
```

Suggested commit types:

- `feat`
- `fix`
- `docs`
- `test`
- `refactor`
- `chore`

## 10. Pull Requests

Each pull request should include:

### Description

Explain what changed and why.

### Scope

Identify the affected modules.

### Validation

List the tests executed and their results.

### Documentation

Mention any documentation updates.

### Limitations

Identify unfinished work, known issues, or environmental constraints.

Keep pull requests focused. Avoid combining unrelated changes.

## 11. Privacy and Security

Recall is designed to process documentation locally.

Contributors must:

- Avoid committing credentials.
- Avoid committing private course materials.
- Avoid exposing sensitive document contents in logs.
- Avoid introducing external document transmission without explicit approval.
- Keep generated files and local environment artifacts out of commits when appropriate.

Any change that introduces external services must clearly document its privacy implications.

## 12. Reporting Issues

When reporting a bug, provide:

- A clear description.
- Steps to reproduce.
- Expected behavior.
- Actual behavior.
- Python version.
- Relevant model configuration.
- Error messages or logs.
- A minimal reproducible example when possible.

Do not include credentials or private documents in issue reports.

## 13. Project Priorities

Current development priorities include:

- Retrieval-quality evaluation.
- Improved technical-document chunking.
- Persistent embedding storage.
- Performance benchmarking.
- Better context relevance assessment.
- Multi-domain retrieval evaluation.

These are development directions, not guarantees that the features are already implemented.

Refer to the README and project issues for the latest status.
