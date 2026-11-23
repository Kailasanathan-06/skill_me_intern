"""Retrieval-augmented answering with guardrails and a confidence indicator."""

import re
from dataclasses import dataclass, field

import embeddings
import prompts
import summarizer
from retrieval import DocumentIndex, Hit
from summarizer import SummarizerError

RERANK_POOL = 10

# Similarity cut-offs per embedding backend (heuristics; tune with /retrieve and evaluate.py).
#   floor  - below this the question is treated as not covered (no LLM call)
#   medium / high - confidence bands for the best retrieved chunk
THRESHOLDS = {
    "fastembed": {"floor": 0.35, "medium": 0.55, "high": 0.70},
    "hash": {"floor": 0.15, "medium": 0.20, "high": 0.33},
}


@dataclass
class Answer:
    text: str
    found: bool
    used: list[Hit]  # sources the answer relied on
    retrieved: list[Hit] = field(default_factory=list)  # every chunk that was considered
    top_score: float = 0.0
    confidence: str = "low"  # high | medium | low
    guarded: bool = False  # True when the similarity floor blocked the LLM call


def min_similarity() -> float:
    """Guardrail floor. Override with MIN_SIMILARITY in .env."""
    custom = summarizer.get_setting("MIN_SIMILARITY")
    try:
        return float(custom)
    except ValueError:
        return THRESHOLDS[embeddings.backend_name()]["floor"]


def confidence_level(score: float) -> str:
    bands = THRESHOLDS[embeddings.backend_name()]
    if score >= bands["high"]:
        return "high"
    return "medium" if score >= bands["medium"] else "low"


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] source: {h.label}\n{h.text}" for i, h in enumerate(hits, 1))


def rerank(question: str, hits: list[Hit], keep: int = 3) -> list[Hit]:
    """Ask the LLM to pick the `keep` best of the retrieved chunks."""
    if len(hits) <= keep:
        return hits
    reply = summarizer.ask_llm(prompts.rerank_prompt(question, [h.text for h in hits], keep), temperature=0)
    order: list[int] = []
    for number in re.findall(r"\d+", reply):
        i = int(number) - 1
        if 0 <= i < len(hits) and i not in order:
            order.append(i)
    chosen = [hits[i] for i in order[:keep]]
    for hit in hits:  # top up from the cosine ranking if the model returned too few
        if len(chosen) >= keep:
            break
        if not any(hit is c for c in chosen):
            chosen.append(hit)
    return chosen


def answer_question(question: str, index: DocumentIndex, top_k: int = 3, use_rerank: bool = False) -> Answer:
    question = question.strip()
    if not question:
        raise SummarizerError("Please type a question.")
    candidates = index.retrieve(question, RERANK_POOL if use_rerank else top_k)
    top_score = candidates[0].score

    # Guardrail 1 (code): nothing in the documents is even close -> don't ask the LLM at all.
    if top_score < min_similarity():
        return Answer(prompts.NOT_FOUND_TEXT, False, [], candidates[:top_k], top_score, "low", guarded=True)

    hits = rerank(question, candidates, top_k) if use_rerank else candidates

    # Guardrail 2 (prompt): answer only from the excerpts, otherwise say "not found".
    text = summarizer.ask_llm(prompts.rag_prompt(question, format_context(hits)), temperature=0.1)
    found = "not found in the documents" not in re.sub(r"[^a-z ]", "", text.lower())

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
    used = [hits[i - 1] for i in cited] if cited else hits
    return Answer(
        text=text,
        found=found,
        used=used if found else [],
        retrieved=hits,
        top_score=top_score,
        confidence=confidence_level(top_score) if found else "low",
    )
