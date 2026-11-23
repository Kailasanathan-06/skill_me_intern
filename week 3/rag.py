"""Retrieval-augmented answering: retrieve -> (re-rank) -> grounded answer."""

import re
from dataclasses import dataclass

import prompts
import summarizer
from retrieval import DocumentIndex, Hit
from summarizer import SummarizerError

RERANK_POOL = 10


@dataclass
class Answer:
    text: str
    found: bool
    used: list[Hit]  # the sources the answer actually relied on


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
        raise SummarizerError("Please type a question after /ask.")
    candidates = index.retrieve(question, RERANK_POOL if use_rerank else top_k)
    hits = rerank(question, candidates, top_k) if use_rerank else candidates

    text = summarizer.ask_llm(prompts.rag_prompt(question, format_context(hits)), temperature=0.1)
    found = "not found in the documents" not in re.sub(r"[^a-z ]", "", text.lower())

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
    used = [hits[i - 1] for i in cited] if cited else hits
    return Answer(text=text, found=found, used=used if found else [])
