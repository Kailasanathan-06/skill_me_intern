"""Document pipeline: split -> summarise each section -> combine -> follow-ups."""

import re
import time

import prompts
import summarizer
from summarizer import MAX_CHARS, SummarizerError


# ---------------------------------------------------------------- splitting
def _chunk(text: str, max_chars: int) -> list[str]:
    """Break oversize text into pieces of at most `max_chars` at sentence ends."""
    if len(text) <= max_chars:
        return [text]
    pieces, buf = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        while len(sentence) > max_chars:  # no sentence boundary: hard cut
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if buf and len(buf) + len(sentence) + 1 > max_chars:
            pieces.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        pieces.append(buf)
    return pieces


def split_sections(text: str, min_chars: int = 120, max_chars: int = 4000) -> list[str]:
    """Split a document on Markdown headings, else on blank-line paragraphs.

    Tiny pieces are merged into their neighbours and oversize pieces are cut
    into fixed-size chunks, so every section is a sensible size for the model.
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    parts = [p.strip() for p in re.split(r"(?m)^(?=#{1,6}\s)", text) if p.strip()]
    if len(parts) < 2:  # no headings: fall back to paragraphs
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    merged, buf = [], ""
    for part in parts:
        buf = f"{buf}\n\n{part}".strip() if buf else part
        if len(buf) >= min_chars:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] += "\n\n" + buf
        else:
            merged.append(buf)

    sections = []
    for section in merged:
        sections.extend(_chunk(section, max_chars))
    return sections


# ------------------------------------------------------------ summarisation
def summarize_document(sections, length="medium", progress=None, delay=0.5):
    """Two-step pipeline. Returns (section_summaries, final_summary).

    Step 1 summarises every section on its own; step 2 combines those
    summaries into one coherent summary with a second LLM call.
    """
    if not sections:
        raise SummarizerError("The document is empty, so there is nothing to summarise.")

    section_summaries = []
    for i, section in enumerate(sections, 1):
        if progress:
            progress(f"Summarising section {i}/{len(sections)}...")
        section_summaries.append(summarizer.summarize(section, "short"))
        time.sleep(delay)  # respect free-tier rate limits

    if len(section_summaries) == 1 and length == "short":
        return section_summaries, section_summaries[0]

    if progress:
        progress("Combining section summaries...")
    final = summarizer.ask_llm(prompts.combine_summaries(section_summaries, length))
    return section_summaries, final


def get_key_points(summary: str) -> str:
    """Extract 5-8 key points using the combined summary as input."""
    return summarizer.ask_llm(prompts.key_points(summary))


def get_quiz(text: str) -> str:
    """Generate 3-5 revision questions (with answers) from the document."""
    return summarizer.ask_llm(prompts.generate_questions(text[:MAX_CHARS]), temperature=0.6)


def compare_documents(name_a, summary_a, name_b, summary_b) -> str:
    return summarizer.ask_llm(prompts.compare(name_a, summary_a, name_b, summary_b))


# ------------------------------------------------------------- conversation
class Conversation:
    """Small chat history so the user can ask follow-up questions."""

    def __init__(self, summary: str, section_summaries: list[str], max_turns: int = 6):
        self.max_turns = max_turns
        self.history: list[dict] = []
        self._system = (
            prompts.SYSTEM_PROMPT + "\n\n" + prompts.followup_context(summary, section_summaries)
        )

    def ask(self, question: str) -> str:
        messages = (
            [{"role": "system", "content": self._system}]
            + self.history
            + [{"role": "user", "content": question}]
        )
        reply = summarizer.chat(messages)
        self.history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": reply},
        ]
        self.history = self.history[-self.max_turns * 2 :]  # keep history small
        return reply
