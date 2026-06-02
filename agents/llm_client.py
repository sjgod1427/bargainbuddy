"""
Thin wrapper around Groq chat completions that automatically falls back to
OpenAI gpt-4o-mini when Groq returns a 429 rate-limit error.

All agents call groq_chat() instead of Groq().chat.completions.create().
The response object is identical in shape (OpenAI-spec) so no caller changes
are needed beyond the import.
"""
import logging
import os

OPENAI_FALLBACK_MODEL = "gpt-4o-mini"


def _is_rate_limit(exc: Exception) -> bool:
    cls = type(exc).__name__
    msg = str(exc)
    return (
        "RateLimitError" in cls
        or "429" in msg
        or "rate_limit_exceeded" in msg
        or "Rate limit" in msg
    )


def groq_chat(model: str, messages: list, **kwargs):
    """
    Call Groq. On 429 rate-limit, retry with OpenAI gpt-4o-mini if
    OPENAI_API_KEY is set; otherwise re-raise the original error.

    Returns the full response object (choices[0].message.content etc.).
    """
    from groq import Groq
    try:
        return Groq().chat.completions.create(model=model, messages=messages, **kwargs)
    except Exception as exc:
        if not _is_rate_limit(exc):
            raise
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            logging.warning("Groq rate limit hit but OPENAI_API_KEY not set — re-raising")
            raise
        logging.warning(
            f"Groq rate limit — falling back to OpenAI {OPENAI_FALLBACK_MODEL}"
        )
        from openai import OpenAI
        return OpenAI(api_key=openai_key).chat.completions.create(
            model=OPENAI_FALLBACK_MODEL, messages=messages, **kwargs
        )
