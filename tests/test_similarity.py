import numpy as np
import pytest

from devassistant.similarity import cosine_similarity


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_similarity_different_vectors():
    assert cosine_similarity([1, 1], [1, 0]) == pytest.approx(1 / np.sqrt(2))
