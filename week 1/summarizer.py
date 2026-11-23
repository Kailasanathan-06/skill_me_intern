"""Core summarisation logic: a thin, safe wrapper around the Groq chat API."""

from pathlib import Path

from dotenv import dotenv_values
from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

# Read settings straight from the .env file next to this script
# (not from the system environment variables).
ENV_PATH = Path(__file__).resolve().parent / ".env"


def _config() -> dict:
    return dotenv_values(ENV_PATH)

DEFAULT_MODEL = "openai/gpt-oss-20b"
MAX_CHARS = 24_000  # keeps a single request within the free-tier limits

LENGTH_INSTRUCTIONS = {
    "short": "Summarise the text below in exactly 3 short bullet points.",
    "medium": "Summarise the text below in 3 clear bullet points.",
    "long": "Summarise the text below in 6 detailed bullet points, "
    "covering the main ideas and important supporting details.",
}

SYSTEM_PROMPT = (
    "You are DocuMind, a careful assistant that summarises documents. "
    "Use only the information in the text. Do not invent facts. "
    "Reply with bullet points only, each starting with '- '."
)


class SummarizerError(Exception):
    """Raised with a friendly, user-facing message."""


def _get_client() -> Groq:
    if not ENV_PATH.exists():
        raise SummarizerError(
            "No .env file found. Copy .env.example to .env and add your GROQ_API_KEY."
        )
    api_key = (_config().get("GROQ_API_KEY") or "").strip()
    if not api_key or api_key.startswith(("your_", "paste_")):
        raise SummarizerError(
            "No API key found. Copy .env.example to .env and add your "
            "GROQ_API_KEY (get one free at https://console.groq.com/keys)."
        )
    return Groq(api_key=api_key)


def summarize(text: str, length: str = "medium") -> str:
    """Send `text` to Groq and return a bullet-point summary."""
    if not text.strip():
        raise SummarizerError("The file is empty, so there is nothing to summarise.")
    if length not in LENGTH_INSTRUCTIONS:
        raise SummarizerError("Length must be one of: short, medium, long.")

    client = _get_client()
    model = _config().get("GROQ_MODEL") or DEFAULT_MODEL

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"{LENGTH_INSTRUCTIONS[length]}\n\n---\n{text[:MAX_CHARS]}",
                },
            ],
        )
        return response.choices[0].message.content.strip()
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
