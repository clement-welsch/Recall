from pathlib import Path
from uuid import UUID

import pytest

from devassistant.models import KnowledgeDomain


def test_knowledge_domain_creation():
    domain = KnowledgeDomain(
        name="Python",
        directory="docs/python",
        description="Python programming",
    )

    assert domain.name == "Python"
    assert domain.directory == Path("docs/python")
    assert domain.description == "Python programming"
    assert isinstance(domain.id, UUID)


def test_knowledge_domain_default_description():
    domain = KnowledgeDomain(name="Python", directory="docs/python")

    assert domain.description == ""


def test_knowledge_domains_receive_unique_ids():
    domain1 = KnowledgeDomain(name="Python", directory="docs/python")
    domain2 = KnowledgeDomain(name="Docker", directory="docs/docker")

    assert domain1.id != domain2.id


@pytest.mark.parametrize("invalid_name", ["", "   "])
def test_knowledge_domain_rejects_empty_name(invalid_name):
    with pytest.raises(ValueError, match="Name must be a non-empty string"):
        KnowledgeDomain(name=invalid_name, directory="docs")


@pytest.mark.parametrize("invalid_name", [None, 123, [], {}])
def test_knowledge_domain_rejects_non_string_name(invalid_name):
    with pytest.raises(ValueError, match="Name must be a non-empty string"):
        KnowledgeDomain(name=invalid_name, directory="docs")


@pytest.mark.parametrize("invalid_directory", ["", "   ", None, 123])
def test_knowledge_domain_rejects_invalid_directory(invalid_directory):
    with pytest.raises(ValueError, match="Directory must be a non-empty path"):
        KnowledgeDomain(name="Python", directory=invalid_directory)


def test_knowledge_domain_rejects_non_string_description():
    with pytest.raises(ValueError, match="Description must be a string"):
        KnowledgeDomain(
            name="Python",
            directory="docs/python",
            description=None,
        )


def test_knowledge_domain_rejects_non_uuid_id():
    with pytest.raises(ValueError, match="ID must be a UUID"):
        KnowledgeDomain(
            name="Python",
            directory="docs/python",
            id="python",
        )
