from devassistant.document_loader import load_documents


def test_load_documents(tmp_path):
    python_file = tmp_path / "python.md"
    python_file.write_text("Python content")

    rag_file = tmp_path / "rag.md"
    rag_file.write_text("RAG content")

    documents = load_documents(tmp_path)

    assert documents == [
        {
            "content": "Python content",
            "source": "python.md",
        },
        {
            "content": "RAG content",
            "source": "rag.md",
        },
    ]


def test_load_documents_empty_directory(tmp_path):
    documents = load_documents(tmp_path)

    assert documents == []


def test_load_documents_ignores_non_markdown_files(tmp_path):
    markdown_file = tmp_path / "python.md"
    markdown_file.write_text("Python content")

    text_file = tmp_path / "notes.txt"
    text_file.write_text("This file should be ignored")

    documents = load_documents(tmp_path)

    assert documents == [
        {
            "content": "Python content",
            "source": "python.md",
        }
    ]


def test_load_documents_empty_markdown_file(tmp_path):
    empty_file = tmp_path / "empty.md"
    empty_file.write_text("")

    documents = load_documents(tmp_path)

    assert documents == []


def test_load_documents_with_chunking(tmp_path):
    document = tmp_path / "document.md"
    document.write_text("one two three four five six")

    documents = load_documents(
        tmp_path,
        chunk_size=4,
        overlap=1,
    )

    assert documents == [
        {
            "content": "one two three four",
            "source": "document.md",
        },
        {
            "content": "four five six",
            "source": "document.md",
        },
    ]


def test_load_documents_default_chunking(tmp_path):
    document = tmp_path / "document.md"
    document.write_text("one two three")

    documents = load_documents(tmp_path)

    assert documents == [
        {
            "content": "one two three",
            "source": "document.md",
        }
    ]


def test_load_documents_keeps_source(tmp_path):
    document = tmp_path / "rag.md"
    document.write_text("RAG retrieves relevant information.")

    documents = load_documents(tmp_path)

    assert documents == [
        {
            "content": "RAG retrieves relevant information.",
            "source": "rag.md",
        }
    ]
