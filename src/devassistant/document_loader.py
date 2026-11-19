from devassistant.chunking import chunk_text


def load_documents(directory, chunk_size=60, overlap=12):
    documents = []

    for file in sorted(directory.glob("*.md")):
        text = file.read_text()

        chunks = chunk_text(
            text,
            size=chunk_size,
            overlap=overlap,
        )

        for chunk in chunks:
            documents.append(
                {
                    "content": chunk,
                    "source": file.name,
                }
            )

    return documents
