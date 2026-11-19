import pytest

from devassistant.chunking import chunk_text


def test_chunk_text():
    text = "one two three four five six"

    chunks = chunk_text(
        text,
        size=4,
        overlap=1,
    )

    assert chunks == [
        "one two three four",
        "four five six",
    ]


def test_chunk_text_without_overlap():
    text = "one two three four five six"

    chunks = chunk_text(
        text,
        size=3,
        overlap=0,
    )

    assert chunks == [
        "one two three",
        "four five six",
    ]


def test_chunk_text_smaller_than_size():
    text = "one two three"

    chunks = chunk_text(
        text,
        size=5,
        overlap=1,
    )

    assert chunks == [
        "one two three",
    ]


def test_chunk_text_empty():
    chunks = chunk_text(
        "",
        size=5,
        overlap=1,
    )

    assert chunks == []


def test_chunk_text_invalid_size():
    with pytest.raises(ValueError):
        chunk_text(
            "one two three",
            size=0,
            overlap=0,
        )


def test_chunk_text_negative_size():
    with pytest.raises(ValueError):
        chunk_text(
            "one two three",
            size=-1,
            overlap=0,
        )


def test_chunk_text_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text(
            "one two three",
            size=5,
            overlap=5,
        )


def test_chunk_text_negative_overlap():
    with pytest.raises(ValueError):
        chunk_text(
            "one two three",
            size=5,
            overlap=-1,
        )
