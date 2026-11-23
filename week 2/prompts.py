"""Prompt templates for DocuMind. Each function returns the user-message text."""

SYSTEM_PROMPT = (
    "You are DocuMind, a careful document assistant for students. "
    "Use only the information in the text you are given and never invent facts. "
    "Write clearly and concisely."
)

LENGTH_INSTRUCTIONS = {
    "short": "Summarise the text below in exactly 3 short bullet points.",
    "medium": "Summarise the text below in 3 clear bullet points.",
    "long": "Summarise the text below in 6 detailed bullet points, covering the "
    "main ideas and important supporting details.",
}

COMBINE_FORMATS = {
    "short": "a one-sentence overview followed by 3 bullet points",
    "medium": "a two-sentence overview followed by 4-5 bullet points",
    "long": "a short overview paragraph followed by 6-8 bullet points "
    "covering every major section",
}


def summarize(text: str, length: str = "medium") -> str:
    """Template 1: summarise a piece of text as bullet points."""
    return (
        f"{LENGTH_INSTRUCTIONS[length]}\n"
        "Reply with bullet points only, each starting with '- '.\n\n"
        f"---\n{text}"
    )


def key_points(text: str) -> str:
    """Template 2: extract 5-8 key points (fed with the combined summary)."""
    return (
        "Extract 5 to 8 key points from the document summary below.\n"
        "Rules: bullet points only, each starting with '- ', one sentence each, "
        "no duplicates, and only facts present in the text.\n\n"
        f"---\n{text}"
    )


def generate_questions(text: str) -> str:
    """Template 3: revision questions with answers."""
    return (
        "Create 3 to 5 revision questions, each with a short answer, based ONLY "
        "on the document below. Mix factual and understanding questions.\n"
        "Use exactly this format, with a blank line between every line:\n\n"
        "**Q1. <question>**\n\n"
        "Answer: <answer>\n\n"
        f"---\n{text}"
    )


def combine_summaries(summaries: list[str], length: str = "medium") -> str:
    """Second step of the pipeline: merge per-section summaries into one."""
    numbered = "\n\n".join(f"Section {i}:\n{s}" for i, s in enumerate(summaries, 1))
    return (
        "Below are summaries of consecutive sections of ONE document. Combine "
        "them into a single coherent summary that reads as a whole, not as "
        "disconnected fragments. Merge overlapping points, keep the document's "
        "order and use smooth transitions.\n"
        f"Format: {COMBINE_FORMATS[length]}. Use '- ' for bullets.\n\n"
        f"{numbered}"
    )


def followup_context(summary: str, section_summaries: list[str]) -> str:
    """Context block placed in the system message for follow-up questions."""
    sections = "\n".join(f"- Section {i}: {s}" for i, s in enumerate(section_summaries, 1))
    return (
        "The user is studying a document. Answer their follow-up questions using "
        "only the material below. If the answer is not covered, say so honestly.\n\n"
        f"DOCUMENT SUMMARY:\n{summary}\n\nSECTION NOTES:\n{sections[:12000]}"
    )


def compare(name_a: str, summary_a: str, name_b: str, summary_b: str) -> str:
    """Stretch goal: compare two documents."""
    return (
        "Compare the two documents below. Give: one line on what they share, then "
        "3-5 bullet points on their key differences (topic, focus, conclusions). "
        "Use only the summaries provided.\n\n"
        f"DOCUMENT A ({name_a}):\n{summary_a}\n\n"
        f"DOCUMENT B ({name_b}):\n{summary_b}"
    )
