# Agent Instructions — Recall

## 1. Project Overview

Recall is a local, Python-based knowledge retrieval assistant for developers.

It combines semantic search, document retrieval, and Retrieval-Augmented Generation (RAG) using local models served through LM Studio's OpenAI-compatible API.

- Repository: `Recall`
- Python package: `devassistant`
- Main source directory: `src/devassistant/`

Recall is both a functional RAG application and an educational development project.

Agents must preserve existing functionality while supporting progressive learning and maintainable development.

## 2. Project Architecture

Core modules:

| Module | Responsibility |
|---|---|
| `config.py` | LM Studio and model configuration |
| `document_loader.py` | Markdown document ingestion |
| `chunking.py` | Word-based text chunking with overlap |
| `embeddings.py` | Embedding generation and LRU caching |
| `similarity.py` | Cosine similarity calculations |
| `search.py` | Semantic retrieval and ranking |
| `prompt.py` | Prompt construction |
| `lmstudio_client.py` | Local model API integration |
| `rag.py` | RAG orchestration and answer generation |
| `evaluation.py` | Retrieval evaluation exercises |

Additional directories:

- `tests/`: Automated tests.
- `notebooks/`: Guided development exercises.
- `docs/architecture/`: Architecture documentation.

Before structural changes, read `README.md` and the relevant documentation in `docs/architecture/`.

Do not reorganize modules or introduce architectural abstractions without a demonstrated need.

## 3. Technology Stack

- Python 3.12 or later, subject to dependency compatibility.
- pytest for automated testing.
- Ruff for Python linting and formatting.
- pre-commit for Git quality hooks.
- LM Studio for local inference.
- OpenAI-compatible local API.
- Nomic Embed Text v1.5 for embeddings.
- Gemma 4 E4B for answer generation.

Default local API endpoint:

`http://localhost:1234/v1`

Do not assume LM Studio or its models are available in CI.

Do not assume the development environment uses a specific operating system, Python executable, or virtual environment path.

## 4. Development Principles

1. Prefer simple, modular Python implementations.
2. Preserve separation of concerns between modules.
3. Avoid unnecessary dependencies.
4. Preserve public APIs unless a change is explicitly required.
5. Maintain backward compatibility where practical.
6. Use descriptive names and type hints for new code.
7. Document non-obvious algorithms and design decisions.
8. Do not introduce network services or cloud inference without explicit approval.
9. Avoid unrelated refactoring or formatting changes.
10. Keep changes focused and independently testable.
11. Prefer deterministic behavior across operating systems.
12. Investigate root causes rather than suppressing errors.

Before implementing a change, inspect the relevant existing code and tests.

Prefer the smallest change that satisfies the requirements.

## 5. Code Quality — Ruff

Ruff is the canonical Python linter and formatter for Recall.

The repository's `pyproject.toml` is the source of truth for Ruff configuration.

Do not introduce competing formatting or linting tools without explicit approval.

### Required checks

```bash
python -m ruff check .
python -m ruff format --check .
```

Both commands must pass before a task is considered complete.

### Automatic fixes

For targeted corrections:

```bash
python -m ruff check --fix path/to/file.py
python -m ruff format path/to/file.py
```

For repository-wide maintenance tasks:

```bash
python -m ruff check --fix .
python -m ruff format .
```

Repository-wide automatic fixes must not be used casually during unrelated feature development.

### Rules

- Respect the existing Ruff configuration.
- Keep imports organized.
- Use consistent formatting.
- Do not add blanket `noqa` suppressions.
- Document justified rule exceptions.
- Do not modify unrelated files solely to satisfy formatting.
- Do not change exception types without reviewing their callers and tests.
- Do not change runtime behavior merely to satisfy a lint recommendation.
- Do not weaken Ruff configuration to hide existing violations.

Prefer fixing the underlying issue over disabling a rule.

## 6. Git Hooks — pre-commit

Recall uses pre-commit to enforce code quality before commits.

Configuration:

`.pre-commit-config.yaml`

The configured hooks include:

- `ruff-check`: Python linting and import validation.
- `ruff-format`: Python formatting.

### Installation

```bash
python -m pip install pre-commit
python -m pre_commit install
```

If pre-commit is installed in a separate tooling environment, invoke its executable from that environment.

Do not assume pre-commit is installed in Recall's application virtual environment.

### Validation

```bash
pre-commit run --all-files
```

Use the appropriate executable path if pre-commit is not available on `PATH`.

### Rules

- Never bypass hooks with `--no-verify` without explicit authorization.
- If a hook modifies files, review the changes before staging them.
- Rerun the hooks after automatic corrections.
- Do not commit unrelated files.
- Do not commit virtual environments or generated local artifacts.
- Do not treat a successful local hook execution as proof that CI will pass.

If hooks fail because of environment or dependency problems, report the issue instead of disabling them.

## 7. RAG-Specific Requirements

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

### API contracts

`get_answer()` supports single-domain retrieval.

`get_answer_for_domains()` supports retrieval across explicit `KnowledgeDomain` collections.

Answer modes:

- `local`: At least one retrieval result exists.
- `model`: No retrieval result exists.

Both modes use the configured local LLM.

Do not interpret `local` as proof that retrieved context is sufficient or factually correct.

Do not silently change the meaning of `top_k`. It currently limits selected source files, not necessarily individual chunks.

## 8. Embedding Requirements

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

## 9. Testing Strategy

Recall distinguishes deterministic tests from tests requiring live LM Studio inference.

### Fast tests — no LM Studio

```bash
python -m pytest -m "not lmstudio"
```

This is the default test command for automated agents and CI.

It must not require an active LM Studio server.

### Complete test suite

```bash
python -m pytest
```

Run the complete suite when LM Studio and the required models are available.

Do not attempt to start or install LM Studio automatically without authorization.

### Focused tests

```bash
python -m pytest -v tests/test_search.py
python -m pytest -v tests/test_rag.py
```

Run relevant focused tests during development before executing broader checks.

### Testing requirements

1. Use deterministic fixtures for unit tests.
2. Mock external model calls when testing application logic.
3. Mark tests requiring live LM Studio with `@pytest.mark.lmstudio`.
4. Do not silently skip tests without reporting them.
5. Preserve existing test coverage when changing behavior.
6. Add regression tests for confirmed bugs.
7. Do not modify assertions merely to hide implementation defects.
8. Verify exception contracts when changing exception types.
9. Avoid dependence on filesystem iteration order.
10. Ensure tests are portable across Windows and Linux.

Tests must not rely on unspecified file ordering, operating-system-specific paths, or timing assumptions.

When ordering is part of an API contract, enforce it explicitly in the implementation and test it.

### Test reporting

Report:

- Tests passed.
- Tests failed.
- Tests skipped or deselected.
- Tests not executed.
- Environment limitations.

Never claim that tests passed without execution evidence.

## 10. CI — GitHub Actions

GitHub Actions provides independent validation of pull requests.

Local Ruff and pre-commit checks do not replace CI.

### Expected CI checks

- Ruff linting.
- Ruff formatting.
- pytest without LM Studio.

CI should not require local model inference.

### CI principles

- Prefer fast, deterministic workflows.
- Use dependency-compatible Python versions.
- Avoid unnecessary operating-system matrices.
- Keep tests independent of local machine configuration.
- Do not disable failing tests merely to obtain a green CI result.
- Investigate differences between Windows and Linux.
- Keep CI configuration aligned with `pyproject.toml` and `.pre-commit-config.yaml`.

A task is not considered CI-validated until the relevant GitHub Actions checks have completed successfully.

Do not claim a PR is ready to merge when required checks are failing or pending.

## 11. Educational Development Rules

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

## 12. Documentation

Update documentation when changes affect:

- Public APIs.
- Configuration.
- Model requirements.
- Retrieval behavior.
- Architecture.
- Installation.
- Test commands.
- Development tooling.

Keep `README.md` consistent with the implementation.

Use the existing architecture documentation:

`docs/architecture/system-architecture.md`

Name new educational notebooks using:

`YYYY-MM-DD-<topic>.ipynb`

Document changes to the development workflow in `CONTRIBUTING.md` when applicable.

## 13. Security and Privacy

Recall is designed for local processing.

- Never commit credentials or API secrets.
- Never commit private course documents without authorization.
- Do not upload user documents to external services.
- Avoid logging document contents unnecessarily.
- Keep local configuration and generated artifacts out of commits.
- Do not commit `.venv/`, `.venv-tools/`, caches, or local execution logs.
- Do not transmit local RAG documents to cloud models without explicit authorization.

The default LM Studio API key is a local placeholder, not a production secret.

## 14. Git Workflow

- Prefer small, focused changes.
- Use descriptive branch names.
- Write meaningful commit messages.
- Include tests for behavioral changes.
- Update documentation when necessary.
- Do not merge pull requests without explicit authorization.
- Do not modify unrelated files.
- Inspect the working tree before staging changes.
- Never overwrite unrelated uncommitted work.

### Before committing

1. Inspect `git status`.
2. Review the diff.
3. Run relevant tests.
4. Run Ruff checks.
5. Run pre-commit hooks.
6. Stage only intended files.
7. Verify staged changes.

Recommended commands:

```bash
git status --short
git diff --check
python -m ruff check .
python -m ruff format --check .
python -m pytest -m "not lmstudio"
pre-commit run --all-files
git diff --cached --check
```

Do not automatically commit or push unless explicitly requested or authorized by the task.

Do not bypass branch protection or required CI checks.

## 15. Definition of Done

A development task is complete when:

- [ ] The requested behavior is implemented.
- [ ] The implementation respects existing architecture.
- [ ] Relevant tests have been added or updated.
- [ ] Relevant tests pass.
- [ ] Ruff lint passes.
- [ ] Ruff format passes.
- [ ] pre-commit hooks pass when available.
- [ ] Documentation is updated when necessary.
- [ ] No unrelated files have been modified.
- [ ] The Git diff has been reviewed.
- [ ] Known limitations are documented.
- [ ] CI status is reported when applicable.

A task may be locally complete while CI validation is pending.

Clearly distinguish these states.

Do not declare a PR ready for merge while required CI checks are failing.

## 16. Completion Report

When completing a development task, report:

1. Summary of changes.
2. Files modified.
3. Tests executed and results.
4. Ruff and pre-commit results.
5. CI status, if applicable.
6. Known limitations.
7. Remaining work.

Never claim that a feature is implemented, tested, committed, pushed, or merged unless that action has been verified.

Keep completion reports concise and evidence-based.