"""LLM layer: a thin, safe wrapper around the Groq chat API (reads .env directly)."""

from pathlib import Path

from dotenv import dotenv_values
from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

import prompts

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_CHARS = 24_000  # keeps a single request within the free-tier limits

# Read settings straight from the .env file next to this script
# (not from the system environment variables).
ENV_PATH = Path(__file__).resolve().parent / ".env"


class SummarizerError(Exception):
    """Raised with a friendly, user-facing message."""


def _config() -> dict:
    return dotenv_values(ENV_PATH)


def get_setting(name: str) -> str:
    """Read one value from the .env file ('' if missing)."""
    if not ENV_PATH.exists():
        return ""
    return (_config().get(name) or "").strip()


def has_api_key() -> bool:
    key = get_setting("GROQ_API_KEY")
    return bool(key) and not key.startswith(("your_", "paste_"))


def _get_client() -> Groq:
    if not ENV_PATH.exists():
        raise SummarizerError(
            "No .env file found. Copy .env.example to .env and add your GROQ_API_KEY."
        )
    api_key = (_config().get("GROQ_API_KEY") or "").strip()
    if not api_key or api_key.startswith(("your_", "paste_")):
        raise SummarizerError(
            "No API key found. Add your GROQ_API_KEY to the .env file "
            "(get one free at https://console.groq.com/keys)."
        )
    return Groq(api_key=api_key)


def chat(messages: list[dict], temperature: float = 0.3) -> str:
    """Send a list of chat messages to Groq and return the reply text."""
    client = _get_client()
    model = _config().get("GROQ_MODEL") or DEFAULT_MODEL
    try:
        response = client.chat.completions.create(
            model=model, temperature=temperature, messages=messages
        )
        reply = (response.choices[0].message.content or "").strip()
    except AuthenticationError:
        raise SummarizerError("Your API key was rejected. Check GROQ_API_KEY in .env.")
    except RateLimitError:
        raise SummarizerError("Free-tier rate limit reached. Wait a moment and retry.")
    except APIConnectionError:
        raise SummarizerError("Could not reach Groq. Check your internet connection.")
    except APIStatusError as exc:
        raise SummarizerError(f"Groq returned an error (HTTP {exc.status_code}).")
    except Exception as exc:  # last-resort safety net: never crash the CLI
        raise SummarizerError(f"Unexpected error: {exc}")
    if not reply:
        raise SummarizerError("The model returned an empty reply. Please try again.")
    return reply


def ask_llm(user_prompt: str, temperature: float = 0.3) -> str:
    """Convenience: one DocuMind system prompt + one user prompt."""
    return chat(
        [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature,
    )


def summarize(text: str, length: str = "medium") -> str:
    """Week 1 function: summarise `text` as bullet points."""
    if not text.strip():
        raise SummarizerError("The file is empty, so there is nothing to summarise.")
    if length not in prompts.LENGTH_INSTRUCTIONS:
        raise SummarizerError("Length must be one of: short, medium, long.")
    return ask_llm(prompts.summarize(text[:MAX_CHARS], length))
