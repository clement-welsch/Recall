def build_prompt(context, question, answer_mode="local"):
    if answer_mode == "model":
        return (
            "You are a helpful assistant.\n"
            "Answer the question using your general knowledge.\n"
            "No relevant local documentation was retrieved. "
            "Do not claim that your answer is based on local documents.\n"
            "If you are uncertain, state that clearly.\n"
            "Question:\n"
            f"{question}"
        )

    if answer_mode != "local":
        raise ValueError(
            "answer_mode must be either 'local' or 'model'"
        )

    return (
        "Context:\n"
        f"{context}\n"
        "Answer the question using the provided context. "
        "If the context does not contain the answer, "
        "say that the information was not found in the local documentation.\n"
        "Question:\n"
        f"{question}"
    )