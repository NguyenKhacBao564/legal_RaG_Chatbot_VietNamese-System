import os

from brain import openai_chat_complete

ENABLE_LLM_SUMMARY = os.environ.get("ENABLE_LLM_SUMMARY", "false").lower() == "true"


def summarize_text(text):
    if not ENABLE_LLM_SUMMARY:
        return text

    # prepare template for prompt
    template = """You are a very good assistant that summarizes text.

    Always keep important key points in the summary.

    ==================
    {text}
    ==================

    Write a summary of the content in Vietnamese.
    """

    prompt = template.format(text=text)

    messages = [
        {
            "role": "system",
            "content": "You summarize text concisely and preserve important details.",
        },
        {"role": "user", "content": prompt},
    ]
    return openai_chat_complete(messages, temperature=0, max_tokens=512)
