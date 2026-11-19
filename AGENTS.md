# Agent Instructions — Recall

## 1. Project Overview

Recall is a local Python knowledge retrieval assistant for developers, combining semantic search and Retrieval-Augmented Generation (RAG).

- Repository: `Recall`
- Python package: `devassistant`
- Source: `src/devassistant/`
- Tests: `tests/`
- Educational exercises: `notebooks/`
- Architecture: `docs/architecture/`

Recall is both a functional application and an educational project. Preserve existing behavior and support progressive learning.

## 2. Architecture

| Module | Responsibility |
|---|---|
| `config.py` | Local models and configuration |
| `document_loader.py` | Markdown ingestion |
| `chunking.py` | Text chunking and overlap |
| `embeddings.py` | Embeddings and LRU cache |
| `similarity.py` | Cosine similarity |
| `search.py` | Retrieval and ranking |
| `prompt.py` | Prompt construction |
| `lmstudio_client.py` | Local model API |
| `rag.py` | RAG orchestration |
| `evaluation.py` | Retrieval evaluation |

Before architectural changes, consult `README.md` and relevant documentation in `docs/architecture/`.

Avoid unnecessary abstractions, dependencies and module reorganizations.

## 3. Technology Stack

- Python 3.12+, subject to dependency compatibility
- pytest — automated tests
- Ruff — linting and formatting
- pre-commit — Git quality hooks
- LM Studio — local inference
- Nomic Embed Text v1.5 — 768-dimensional embeddings
- Gemma 4 E4B — answer generation

Default local API: `http://localhost:1234/v1`

Do not assume LM Studio is running, especially in CI. Do not assume a specific operating system, Python executable or virtual environment.

## 4. Development Rules

1. Inspect relevant code and tests before modifying anything.
2. Prefer minimal, modular and independently testable changes.
3. Preserve public APIs and backward compatibility unless explicitly required.
4. Use descriptive names, type hints and clear error handling.
5. Fix root causes instead of suppressing failures.
6. Avoid unrelated refactoring and repository-wide formatting.
7. Keep behavior deterministic and portable across Windows and Linux.
8. Document non-obvious algorithms and architectural decisions.
9. Never introduce cloud inference or external services without authorization.
10. Respect existing architecture and project conventions.

## 5. RAG Contracts

Preserve:

- Markdown ingestion and configurable chunking.
- Cosine similarity and configurable thresholds.
- Top-k source selection and source provenance.
- Adaptive retrieval fallback.
- Single-domain and multi-domain retrieval.
- Existing embedding cache behavior.

### Public APIs

- `get_answer()` — single-domain retrieval.
- `get_answer_for_domains()` — retrieval across explicit `KnowledgeDomain` collections.

Answer modes:

- `local`: at least one retrieval result exists.
- `model`: no retrieval result exists.

Both modes use the configured local LLM.

**Important:**
- `local` does not guarantee sufficient or accurate retrieved context.
- `top_k` limits selected source files, not necessarily individual chunks.
- Do not silently change these semantics.

## 6. Embedding Cache Contracts

The configured embedding model produces 768-dimensional vectors.

The in-memory LRU cache must:

- Hold at most 1,024 entries.
- Use `(model_identifier, input_text)` as its key.
- Deduplicate repeated inputs.
- Batch missing embedding requests.
- Preserve input order.
- Evict least recently used entries.

Do not introduce persistent embedding storage without documenting invalidation and model compatibility.

## 7. Code Quality

Ruff is the canonical Python linter and formatter.

Configuration source of truth: `pyproject.toml`.

Required checks:

```bash
python -m ruff check .
python -m ruff format --check .
```

For targeted corrections:

```bash
python -m ruff check --fix path/to/file.py
python -m ruff format path/to/file.py
```

Rules:

- Respect the existing Ruff configuration.
- Avoid blanket `noqa` suppressions.
- Do not weaken lint rules to hide violations.
- Do not change runtime behavior solely to satisfy linting.
- Do not format unrelated files.
- Review exception contracts before changing exception types.
- Do not introduce competing formatters without approval.

## 8. Pre-commit

Configuration: `.pre-commit-config.yaml`

Expected hooks include `ruff-check` and `ruff-format`.

Install when needed:

```bash
python -m pip install pre-commit
python -m pre_commit install
```

Validate:

```bash
pre-commit run --all-files
```

Use the executable from the appropriate tooling environment if necessary.

Rules:

- Never bypass hooks using `--no-verify` without authorization.
- Review files modified by hooks before staging.
- Rerun hooks after automatic corrections.
- Report environment failures instead of disabling hooks.
- A successful local hook run does not guarantee CI success.

## 9. Testing Strategy

### Default: deterministic tests

```bash
python -m pytest -m "not lmstudio"
```

This is the default for agents and CI. It must work without LM Studio.

### Live integration tests

```bash
python -m pytest
```

Run the complete suite only when LM Studio and required models are available.

### Focused tests

```bash
python -m pytest -v tests/test_search.py
python -m pytest -v tests/test_rag.py
```

Testing requirements:

- Prefer deterministic fixtures and mocked model calls.
- Mark live inference tests with `@pytest.mark.lmstudio`.
- Add regression tests for confirmed bugs.
- Preserve existing coverage.
- Do not alter assertions to conceal defects.
- Do not depend on unspecified filesystem ordering or timing.
- Keep tests portable across Windows and Linux.
- Do not automatically install or start LM Studio.

Report passed, failed, skipped, deselected and unexecuted tests accurately.

Never claim tests passed without execution evidence.

## 10. CI — GitHub Actions

Expected checks:

- Ruff linting.
- Ruff formatting.
- pytest excluding `lmstudio` tests.

CI must remain independent of local inference.

Prefer fast, deterministic workflows and dependency-compatible Python versions.

Do not disable failing tests to obtain green checks.

Investigate platform-specific failures.

**A PR is not ready to merge while required CI checks are failing or pending.**

## 11. Educational Workflow

Recall supports guided learning through `notebooks/`.

For educational exercises:

1. Explain the objective and acceptance criteria.
2. Identify relevant modules.
3. Provide progressive hints before complete solutions.
4. Let the developer implement the exercise.
5. Review submitted code and test results.
6. Explain failures and suggest focused corrections.

Do not complete educational exercises automatically unless explicitly requested.

Tests intentionally raising `NotImplementedError` may represent unfinished exercises rather than regressions.

Distinguish verified functionality from planned work.

## 12. Documentation

Update documentation when changing:

- Public APIs or configuration.
- Model requirements or retrieval behavior.
- Architecture or installation.
- Testing or development tooling.

Keep `README.md` synchronized with implementation.

Architecture reference: `docs/architecture/system-architecture.md`

Educational notebook naming: `YYYY-MM-DD-<topic>.ipynb`

Update `CONTRIBUTING.md` when development workflows change.

## 13. Security and Privacy

Recall is designed for local processing.

- Never commit credentials or private course documents.
- Never upload user documents to external services without authorization.
- Never transmit RAG documents to cloud models without authorization.
- Avoid logging sensitive document contents.
- Exclude virtual environments, caches, logs and generated artifacts from commits.
- Treat the default LM Studio API key as a local placeholder, not a production secret.

## 14. Git Workflow

- Use focused branches, commits and pull requests.
- Inspect `git status` and review diffs before staging.
- Preserve unrelated uncommitted work.
- Stage only intended files.
- Do not commit, push or merge without authorization.
- Never bypass branch protection or required CI checks.

### Validation checklist

Before completing a development task, run the applicable checks:

```bash
git status --short
git diff --check
python -m ruff check .
python -m ruff format --check .
python -m pytest -m "not lmstudio"
pre-commit run --all-files
```

Before committing, also inspect staged changes:

```bash
git diff --cached --check
```

If a check cannot run, document why. Do not report it as passed.

## 15. Definition of Done

A development task is complete when:

- [ ] Requested behavior is implemented.
- [ ] Existing architecture and API contracts are preserved.
- [ ] Relevant tests are added or updated.
- [ ] Applicable tests and quality checks pass.
- [ ] Documentation is updated when necessary.
- [ ] No unrelated changes are included.
- [ ] Git diff is reviewed.
- [ ] Known limitations are documented.
- [ ] CI status is reported when applicable.

Distinguish local completion from CI validation.

### Completion report

Summarize:

1. Changes and affected files.
2. Tests and quality checks executed.
3. CI status, if applicable.
4. Limitations and remaining work.

Never claim unverified implementation, testing, commit, push or merge operations.

## 16. Conversation Scope Management

Keep ChatGPT and Codex conversations focused on coherent functional areas rather than individual GitHub issues.

### Functional areas

Recall's primary functional areas include:

- Document ingestion and chunking.
- Embeddings and caching.
- Semantic search and ranking.
- RAG orchestration and fallback.
- LM Studio integration.
- Retrieval evaluation.
- Tests, tooling and CI.
- Architecture and documentation.

A conversation may cover several related issues within one functional area.

### When to recommend a new conversation

Recommend a transition when:

1. The user switches to a different functional area.
2. The main objective changes substantially.
3. Multiple unrelated topics accumulate.
4. Earlier requirements become difficult to track reliably.
5. A major development phase ends and a distinct phase begins.

Do not recommend a transition merely because an issue changes, a conversation grows longer or a minor correction is requested.

Do not claim access to exact context-window usage unless explicitly available.

### Transition procedure

When a boundary is appropriate:

1. Complete the current response.
2. Explain briefly why a new conversation is recommended.
3. Provide a concise handoff summary containing:
   - Functional area and objective.
   - Verified completed work.
   - Relevant decisions and technical constraints.
   - Issues, PRs and branch, if known.
   - Outstanding work and known problems.
4. Provide a ready-to-copy prompt for the next conversation.

Conversation boundaries are recommendations, not mandatory interruptions.

Never abandon active work solely to suggest a transition.