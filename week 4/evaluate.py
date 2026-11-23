"""Score DocuMind on a fixed question set.

    python evaluate.py                    # full run (uses your Groq key)
    python evaluate.py --retrieval-only   # no LLM: checks the right chunks are found
"""

import argparse
import json
import sys
import time
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import rag
from retrieval import DocumentIndex
from summarizer import SummarizerError, has_api_key

ROOT = Path(__file__).resolve().parent
TEST_DOCS = ["samples/renewable_energy.pdf", "samples/machine_learning_basics.pdf", "samples/study_skills.pdf"]
QUESTIONS_FILE = ROOT / "eval_questions.json"
console = Console()


def contains_any(text: str, keywords: list[str]) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def evaluate_one(item: dict, index: DocumentIndex, retrieval_only: bool, top_k: int) -> tuple[str, str]:
    """Return (PASS | FAIL | SKIP, detail)."""
    question, keywords = item["question"], item["expected"]
    unanswerable = item["type"] == "unanswerable"

    if retrieval_only:
        hits = index.retrieve(question, top_k)
        best = hits[0].score
        if unanswerable:
            blocked = best < rag.min_similarity()
            return ("PASS", f"guardrail blocked (best {best:.2f})") if blocked else ("SKIP", f"needs the LLM (best {best:.2f})")
        ok = contains_any(" ".join(h.text for h in hits), keywords)
        return ("PASS" if ok else "FAIL"), f"best {best:.2f} · {hits[0].source}"

    answer = rag.answer_question(question, index, top_k)
    if unanswerable:
        how = "guardrail" if answer.guarded else "model"
        return ("PASS" if not answer.found else "FAIL"), f"said not found ({how})" if not answer.found else "answered from nowhere!"
    ok = answer.found and contains_any(answer.text, keywords)
    return ("PASS" if ok else "FAIL"), f"{answer.confidence} confidence · best {answer.top_score:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DocuMind answers.")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM and only check retrieval")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if not args.retrieval_only and not has_api_key():
        console.print(Panel("No Groq key found. Add GROQ_API_KEY to .env, or run with --retrieval-only.",
                            title="Oops", border_style="red"))
        return 2

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    index = DocumentIndex()
    try:
        with console.status("Indexing test documents..."):
            for doc in TEST_DOCS:
                index.add_document(ROOT / doc)
    except SummarizerError as err:
        console.print(Panel(str(err), title="Oops", border_style="red"))
        return 2

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Question")
    table.add_column("Result", justify="center")
    table.add_column("Detail", style="dim")
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    colors = {"PASS": "green", "FAIL": "red", "SKIP": "yellow"}

    for number, item in enumerate(questions, 1):
        try:
            result, detail = evaluate_one(item, index, args.retrieval_only, args.top_k)
        except SummarizerError as err:
            result, detail = "FAIL", str(err)
        counts[result] += 1
        table.add_row(str(number), item["question"], f"[bold {colors[result]}]{result}[/]", detail)
        if not args.retrieval_only:
            time.sleep(1)  # respect free-tier rate limits

    console.print(table)
    checked = counts["PASS"] + counts["FAIL"]
    mode = "retrieval-only" if args.retrieval_only else "full"
    color = "green" if counts["FAIL"] == 0 else "red"
    console.print(Panel(
        f"[bold {color}]{counts['PASS']}/{checked} checks passed[/]  ·  {counts['FAIL']} failed  ·  "
        f"{counts['SKIP']} skipped  ·  mode: {mode}",
        box=box.ROUNDED, border_style=color))
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
