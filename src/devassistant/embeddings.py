from collections import OrderedDict

from openai import OpenAI

from .config import (
    LMSTUDIO_API_KEY,
    LMSTUDIO_BASE_URL,
    LMSTUDIO_EMBEDDING_MODEL,
)

client = OpenAI(
    base_url=LMSTUDIO_BASE_URL,
    api_key=LMSTUDIO_API_KEY,
)

EMBEDDING_CACHE_MAXSIZE = 1024
_embedding_cache = OrderedDict()


def embed(texts):
    if not texts:
        raise ValueError("No embedding data received")

    model = LMSTUDIO_EMBEDDING_MODEL
    unique_texts = list(dict.fromkeys(texts))
    vectors_by_text = {}

    missing_texts = []

    for text in unique_texts:
        key = (model, text)

        if key in _embedding_cache:
            _embedding_cache.move_to_end(key)
            vectors_by_text[text] = _embedding_cache[key]
        else:
            missing_texts.append(text)

    if missing_texts:
        response = client.embeddings.create(
            model=model,
            input=missing_texts,
        )

        for text, embedding in zip(missing_texts, response.data, strict=False):
            vectors_by_text[text] = embedding.embedding

            key = (model, text)
            _embedding_cache[key] = embedding.embedding
            _embedding_cache.move_to_end(key)

            while len(_embedding_cache) > EMBEDDING_CACHE_MAXSIZE:
                _embedding_cache.popitem(last=False)

    return [vectors_by_text[text] for text in texts]


def clear_embedding_cache():
    _embedding_cache.clear()
