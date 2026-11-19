import pytest

from devassistant.lmstudio_client import ask


@pytest.mark.lmstudio
def test_ask():
    response = ask("Explique en une phrase ce qu'est le Nutri-Score.")

    assert isinstance(response, str)
    assert response
