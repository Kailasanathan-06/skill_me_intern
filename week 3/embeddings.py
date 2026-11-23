"""Embeddings. Groq has no embeddings endpoint, so DocuMind embeds locally.

Backends (set EMBED_BACKEND in .env):
  fastembed  - small local model (BAAI/bge-small-en-v1.5), best quality (default)
  hash       - tiny offline keyword-hashing fallback, no download needed
"""

import hashlib
import re
from pathlib import Path

import numpy as np
from dotenv import dotenv_values

from summarizer import SummarizerError

ENV_PATH = Path(__file__).resolve().parent / ".env"
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
HASH_DIM = 2048
_STOP = set(
    "a an the and or of to in on for with is are was were be by as at it this that these "
    "those from can how what when why which who does do did".split()
)
_model = None


class EmbeddingError(SummarizerError):
    """Friendly, user-facing embedding problem."""


def _backend_name() -> str:
    name = (dotenv_values(ENV_PATH).get("EMBED_BACKEND") or "fastembed").strip().lower()
    if name not in {"fastembed", "hash"}:
        raise EmbeddingError("EMBED_BACKEND in .env must be 'fastembed' or 'hash'.")
    return name


def backend_id() -> str:
    """Identifies the backend + model, so cached vectors are never mixed up."""
    return f"fastembed:{FASTEMBED_MODEL}" if _backend_name() == "fastembed" else f"hash:{HASH_DIM}"


def _fastembed():
    global _model
    if _model is None:
        try:
            from fastembed import TextEmbedding

            _model = TextEmbedding(model_name=FASTEMBED_MODEL)
        except ImportError:
            raise EmbeddingError("fastembed is not installed. Run: pip install -r requirements.txt")
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load the embedding model ({exc}). The first run needs internet "
                "to download it (~130 MB). For an offline fallback, set EMBED_BACKEND=hash in .env."
            )
    return _model


def _hash_vector(text: str) -> np.ndarray:
    vec = np.zeros(HASH_DIM, dtype=np.float32)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in _STOP:
            continue
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        vec[int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % HASH_DIM] += 1.0
    return np.log1p(vec)


def _normalise(vec) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def embed_many(texts: list[str]) -> list[np.ndarray]:
    """Embed a batch of passages."""
    if not texts:
        return []
    if _backend_name() == "hash":
        return [_normalise(_hash_vector(t)) for t in texts]
    return [_normalise(v) for v in _fastembed().embed(texts)]


def embed(text: str, is_query: bool = False) -> np.ndarray:
    """Embed one piece of text (set is_query=True for search questions)."""
    if _backend_name() == "fastembed" and is_query:
        model = _fastembed()
        if hasattr(model, "query_embed"):
            return _normalise(next(iter(model.query_embed([text]))))
    return embed_many([text])[0]
