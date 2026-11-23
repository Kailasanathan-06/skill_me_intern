"""Retrieval: embed chunks (with per-file caching) and find the closest ones."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import documents
import embeddings
from summarizer import SummarizerError

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "embeddings"


@dataclass
class Hit:
    text: str
    source: str
    page: int | None
    score: float

    @property
    def label(self) -> str:
        return f"{self.source} (p.{self.page})" if self.page else self.source


def cosine_similarity(a, b) -> float:
    """cos(theta) = (a . b) / (|a| |b|)"""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


class DocumentIndex:
    """All loaded documents as a plain list of chunks with their vectors."""

    def __init__(self):
        self.chunks: list[dict] = []
        self.docs: dict[str, dict] = {}

    # ------------------------------------------------------------- loading
    def add_document(self, path) -> dict:
        path = Path(path).expanduser()
        if not path.exists():
            raise SummarizerError(f"File not found: {path}")
        if path.suffix.lower() not in documents.SUPPORTED:
            raise SummarizerError("Unsupported file type. Use .pdf, .txt or .md files.")

        digest = documents.file_hash(path)
        meta = {
            "filename": path.name,
            "sha256": digest,
            "backend": embeddings.backend_id(),
            "size": documents.CHUNK_SIZE,
            "overlap": documents.CHUNK_OVERLAP,
        }
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
        cache_file = CACHE_DIR / f"{safe}-{digest[:12]}.json"

        loaded, cached = self._read_cache(cache_file, meta), True
        if loaded is None:
            cached = False
            pages = documents.extract_pages(path)
            chunks = documents.build_chunks(pages, path.name)
            vectors = embeddings.embed_many([c["text"] for c in chunks])
            for chunk, vec in zip(chunks, vectors):
                chunk["vector"] = vec
            loaded = {"pages": len(pages), "chunks": chunks}
            self._write_cache(cache_file, meta, loaded)

        # re-adding a file replaces its old chunks
        self.chunks = [c for c in self.chunks if c["source"] != path.name]
        self.chunks.extend(loaded["chunks"])
        info = {"name": path.name, "pages": loaded["pages"], "chunks": len(loaded["chunks"]), "cached": cached}
        self.docs[path.name] = info
        return info

    @staticmethod
    def _read_cache(cache_file: Path, meta: dict):
        if not cache_file.exists():
            return None
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            if payload["meta"] != meta:
                return None
            chunks = [
                {
                    "text": c["text"],
                    "source": meta["filename"],
                    "page": c["page"],
                    "vector": np.asarray(c["vector"], dtype=np.float32),
                }
                for c in payload["chunks"]
            ]
            return {"pages": payload["pages"], "chunks": chunks}
        except (ValueError, KeyError, TypeError, OSError):
            return None  # corrupt or outdated cache: just re-embed

    @staticmethod
    def _write_cache(cache_file: Path, meta: dict, loaded: dict) -> None:
        payload = {
            "meta": meta,
            "pages": loaded["pages"],
            "chunks": [
                {"text": c["text"], "page": c["page"], "vector": [round(float(x), 6) for x in c["vector"]]}
                for c in loaded["chunks"]
            ],
        }
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass  # caching is an optimisation, never a failure

    def remove_document(self, name: str) -> None:
        self.chunks = [c for c in self.chunks if c["source"] != name]
        self.docs.pop(name, None)

    def clear(self) -> None:
        self.chunks, self.docs = [], {}

    # ----------------------------------------------------------- searching
    def retrieve(self, question: str, top_k: int = 3) -> list[Hit]:
        """Return the `top_k` chunks most similar to the question, across all documents."""
        if not self.chunks:
            raise SummarizerError("No documents loaded yet. Use /add <file.pdf> first.")
        query = embeddings.embed(question, is_query=True)
        scored = [(cosine_similarity(query, c["vector"]), c) for c in self.chunks]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [Hit(c["text"], c["source"], c["page"], score) for score, c in scored[:top_k]]
