def chunk_text(text, size=60, overlap=12):
    """Split text into fixed-size word chunks with overlapping content."""

    if size <= 0:
        raise ValueError("The size must be greater than 0")

    if overlap < 0 or overlap >= size:
        raise ValueError("The overlap must be between 0 and size - 1")

    words = text.split()

    step = size - overlap

    chunks = []
    start = 0

    while start < len(words):
        chunks.append(" ".join(words[start : start + size]))

        if start + size >= len(words):
            break

        start += step

    return chunks
