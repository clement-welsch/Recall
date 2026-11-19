from openai import OpenAI

from .config import (
    LMSTUDIO_API_KEY,
    LMSTUDIO_BASE_URL,
    LMSTUDIO_MODEL,
)

client = OpenAI(
    base_url=LMSTUDIO_BASE_URL,
    api_key=LMSTUDIO_API_KEY,
)


def ask(prompt: str) -> str:
    response = client.chat.completions.create(
        model=LMSTUDIO_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content
