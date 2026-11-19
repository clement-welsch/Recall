import pytest

from devassistant.embeddings import clear_embedding_cache, embed

texts = [
    "Le Nutri-Score évalue la qualité nutritionnelle des aliments.",
    "Le produit contient beaucoup de protéines.",
    "Les pommes sont des fruits.",
]


def test_embedding_count():
    vectors = embed(texts)
    assert len(vectors) == len(texts)


def test_embedding_dimension():
    vectors = embed(texts)

    for vector in vectors:
        assert len(vector) == 768


def test_embedding_type():
    vectors = embed(texts)

    for vector in vectors:
        assert isinstance(vector, list)


def test_embedding_values():
    vectors = embed(texts)
    for vector in vectors:
        for value in vector:
            assert isinstance(value, float)


def test_embedding_empty_input():
    with pytest.raises(ValueError):
        embed([])


def test_embedding_single_text():
    vectors = embed(["Le Nutri-Score évalue la qualité nutritionnelle."])

    assert len(vectors) == 1


###

from types import SimpleNamespace


@pytest.fixture
def mock_lmstudio(monkeypatch):
    monkeypatch.setattr(
        "devassistant.embeddings.client.embeddings.create",
        fake_create,
    )


def fake_create(model, input):
    if not input:
        raise ValueError("No embedding data received")

    return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 768) for _ in input])


def test_embedding_count_without_lmstudio(mock_lmstudio):
    vectors = embed(texts)
    assert len(vectors) == len(texts)


def test_embedding_dimension_without_lmstudio(mock_lmstudio):
    vectors = embed(texts)

    for vector in vectors:
        assert len(vector) == 768


def test_embedding_type_without_lmstudio(mock_lmstudio):
    vectors = embed(texts)
    for vector in vectors:
        assert isinstance(vector, list)


def test_embedding_values_without_lmstudio(mock_lmstudio):
    vectors = embed(texts)
    for vector in vectors:
        for value in vector:
            assert isinstance(value, float)


def test_embedding_empty_input_without_lmstudio(mock_lmstudio):
    with pytest.raises(ValueError):
        embed([])


def test_embedding_single_text_without_lmstudio(mock_lmstudio):
    vectors = embed(["Le Nutri-Score évalue la qualité nutritionnelle."])
    assert len(vectors) == 1


def test_embedding_cache_reuses_identical_texts(monkeypatch):
    clear_embedding_cache()
    calls = []

    def fake_create(model, input):
        calls.append(input)

        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(len(text))] * 768) for text in input]
        )

    monkeypatch.setattr(
        "devassistant.embeddings.client.embeddings.create",
        fake_create,
    )

    text_3 = [
        "Le Nutri-Score évalue la qualité nutritionnelle des aliments.",
        "Le produit contient beaucoup de protéines.",
        "Le Nutri-Score évalue la qualité nutritionnelle des aliments.",
    ]
    vectors = embed(text_3)

    assert len(vectors) == 3
    assert vectors[0] == vectors[2]
    assert sum(len(call) for call in calls) == 2


def test_embedding_cache_reuses_previous_results(monkeypatch):
    clear_embedding_cache()
    calls = []

    def fake_create(model, input):
        calls.append(input)

        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(len(text))] * 768) for text in input]
        )

    monkeypatch.setattr(
        "devassistant.embeddings.client.embeddings.create",
        fake_create,
    )

    first = embed(["Python est un langage."])
    second = embed(["Python est un langage."])

    assert first == second
    assert len(calls) == 1


def test_embedding_cache_batches_missing_texts(monkeypatch):
    clear_embedding_cache()
    calls = []

    def fake_create(model, input):
        calls.append(input)

        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(len(text))] * 768) for text in input]
        )

    monkeypatch.setattr(
        "devassistant.embeddings.client.embeddings.create",
        fake_create,
    )

    embed(["A", "B"])
    embed(["B", "C"])

    assert calls == [["A", "B"], ["C"]]


def test_embedding_cache_evicts_least_recently_used(monkeypatch):
    clear_embedding_cache()
    monkeypatch.setattr(
        "devassistant.embeddings.EMBEDDING_CACHE_MAXSIZE",
        2,
    )

    calls = []

    def fake_create(model, input):
        calls.append(input)

        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(len(text))] * 768) for text in input]
        )

    monkeypatch.setattr(
        "devassistant.embeddings.client.embeddings.create",
        fake_create,
    )

    embed(["A", "B"])
    embed(["A", "C"])
    embed(["B"])

    assert calls == [["A", "B"], ["C"], ["B"]]
