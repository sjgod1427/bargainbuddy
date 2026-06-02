"""
Central LLM call wrapper for BargainBuddy.

Priority order on every call:
  1. Groq (server key — free, fast)
  2. On 429: user-provided key for this session thread (OpenAI / Anthropic / Gemini)
  3. On 429 with no user key: server OPENAI_API_KEY env var
  4. Re-raise if nothing is available

User keys are stored in threading.local() so they are isolated per request
thread and never bleed between concurrent sessions.
"""
import logging
import threading

_local = threading.local()

# Models used when falling back to each provider
_PROVIDER_MODELS = {
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini":    "gemini/gemini-1.5-flash",
}

# kwargs that some providers don't support — strip them on non-Groq calls
_GROQ_ONLY_KWARGS = {"seed", "response_format"}


def set_user_keys(provider: str, api_key: str) -> None:
    """Store the user's provider + key for the current thread."""
    _local.provider = provider.lower().strip()
    _local.api_key = api_key.strip()


def clear_user_keys() -> None:
    _local.provider = ""
    _local.api_key = ""


def _is_rate_limit(exc: Exception) -> bool:
    cls = type(exc).__name__
    msg = str(exc)
    return (
        "RateLimitError" in cls
        or "429" in msg
        or "rate_limit_exceeded" in msg
        or "Rate limit" in msg
    )


def _fallback_call(messages: list, **kwargs):
    """
    Try fallback providers in order:
      1. User's session key (thread-local)
      2. Server OPENAI_API_KEY env var
    Uses LiteLLM so OpenAI / Anthropic / Gemini all work through one interface.
    """
    from litellm import completion

    # Strip kwargs that non-Groq providers don't support
    clean = {k: v for k, v in kwargs.items() if k not in _GROQ_ONLY_KWARGS}

    # User session key only — server keys are never used as fallback
    user_provider = getattr(_local, "provider", "")
    user_key = getattr(_local, "api_key", "")
    if user_provider and user_key and user_provider in _PROVIDER_MODELS:
        model = _PROVIDER_MODELS[user_provider]
        logging.warning(f"Groq rate limit — using user's {user_provider} key ({model})")
        return completion(model=model, messages=messages, api_key=user_key, **clean)

    return None  # no user key provided


def groq_chat(model: str, messages: list, **kwargs):
    """
    Call Groq. On 429 rate-limit, try user session key then server OpenAI key.
    Returns a response object with .choices[0].message.content.
    """
    from groq import Groq
    try:
        return Groq().chat.completions.create(model=model, messages=messages, **kwargs)
    except Exception as exc:
        if not _is_rate_limit(exc):
            raise
        result = _fallback_call(messages, **kwargs)
        if result is not None:
            return result
        logging.warning("Groq rate limit — no fallback key available, re-raising")
        raise
