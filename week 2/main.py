"""DocuMind - Week 2: summarise, key points, quiz and follow-ups (Groq)."""

import shlex
from dataclasses import dataclass, field
from pathlib import Path

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

import pipeline
from pipeline import Conversation
from summarizer import SummarizerError

console = Console()
ACCENT = "cyan"
EXTENSIONS = {".txt", ".md", ".markdown"}
COMMANDS = {
    "quit", "exit", "q", "help", "length", "load", "summary", "sections",
    "key_points", "keypoints", "quiz_me", "quiz", "ask", "compare",
}


@dataclass
class Document:
    name: str
    text: str
    sections: list = field(default_factory=list)
    section_summaries: list = field(default_factory=list)
    summary: str = ""


# ----------------------------------------------------------------------- UI
def banner() -> None:
    title = Text("DocuMind", style=f"bold {ACCENT}")
    title.append("  ·  Week 2 · Structured Summaries", style="dim")
    console.print()
    console.print(Panel(title, box=box.ROUNDED, border_style=ACCENT, padding=(1, 4)))


def help_table(length: str) -> None:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("Command", style=f"bold {ACCENT}")
    table.add_column("What it does")
    rows = [
        ("<file.txt|.md>", "Load a document and summarise it (or: load <path>)"),
        ("summary", "Show the combined summary again"),
        ("sections", "Inspect each section's own summary"),
        ("key_points", "5-8 key points from the document"),
        ("quiz_me", "3-5 revision questions with answers"),
        ("ask <question>", "Follow-up question about the document"),
        ("compare <a> <b>", "Key differences between two documents"),
        ("length short|medium|long", f"Summary length (now: [bold]{length}[/bold])"),
        ("help / quit", "Show this menu / exit"),
    ]
    for row in rows:
        table.add_row(*row)
    console.print(table)


def panel(content, title: str, color: str = "green") -> None:
    console.print(
        Panel(
            Markdown(content) if isinstance(content, str) else content,
            title=f"[bold]{title}[/bold]",
            title_align="left",
            border_style=color,
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def sections_table(sections: list[str]) -> None:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Section preview")
    table.add_column("Chars", justify="right", style="dim")
    for i, sec in enumerate(sections, 1):
        preview = " ".join(sec.split())[:70]
        table.add_row(str(i), escape(preview) + ("…" if len(sec) > 70 else ""), f"{len(sec):,}")
    console.print(table)


# ------------------------------------------------------------------ helpers
def clean_path(raw: str) -> Path:
    return Path(raw.strip().strip("'\"")).expanduser()


def read_file(raw: str) -> tuple[str, str]:
    path = clean_path(raw)
    if not path.exists():
        raise SummarizerError(f"File not found: {path}")
    if path.suffix.lower() not in EXTENSIONS:
        raise SummarizerError("Please provide a .txt or .md file.")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SummarizerError(f"Could not read the file: {exc.strerror or exc}")
    if not text.strip():
        raise SummarizerError("The file is empty, so there is nothing to summarise.")
    return path.name, text


def process(raw: str, length: str, status) -> Document:
    """Read, split and summarise a file (shared by load and compare)."""
    name, text = read_file(raw)
    doc = Document(name=name, text=text, sections=pipeline.split_sections(text))
    doc.section_summaries, doc.summary = pipeline.summarize_document(
        doc.sections, length, progress=lambda msg: status.update(f"[{ACCENT}]{name}: {msg}")
    )
    return doc


# ----------------------------------------------------------------- commands
def cmd_load(raw: str, length: str) -> tuple[Document, Conversation]:
    name, text = read_file(raw)
    sections = pipeline.split_sections(text)
    console.print(f"\n[bold]{escape(name)}[/bold] → [green]{len(sections)} section(s)[/green]")
    sections_table(sections)
    with console.status(f"[{ACCENT}]Starting...", spinner="dots") as status:
        doc = process(raw, length, status)
    panel(doc.summary, f"Summary · {length}")
    return doc, Conversation(doc.summary, doc.section_summaries)


def cmd_compare(arg: str, length: str) -> None:
    try:
        paths = [p.strip("'\"") for p in shlex.split(arg, posix=False)]
    except ValueError:
        paths = []
    if len(paths) != 2:
        raise SummarizerError('Usage: compare <file1> <file2>  (quote paths that contain spaces)')
    with console.status(f"[{ACCENT}]Starting...", spinner="dots") as status:
        a = process(paths[0], length, status)
        b = process(paths[1], length, status)
        status.update(f"[{ACCENT}]Comparing documents...")
        result = pipeline.compare_documents(a.name, a.summary, b.name, b.summary)
    panel(result, f"Compare · {a.name} vs {b.name}", "magenta")


def looks_like_path(raw: str) -> bool:
    p = clean_path(raw)
    return p.suffix.lower() in EXTENSIONS or p.exists()


# --------------------------------------------------------------------- main
def main() -> None:
    banner()
    length = "medium"
    doc: Document | None = None
    convo: Conversation | None = None
    help_table(length)

    while True:
        label = f"[bold {ACCENT}]{escape(doc.name)}[/bold {ACCENT}] ›" if doc else f"[bold {ACCENT}]›[/bold {ACCENT}]"
        try:
            raw = Prompt.ask(f"\n{label} Command").strip()
        except (KeyboardInterrupt, EOFError):
            raw = "quit"
        if not raw:
            continue

        parts = raw.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in {"quit", "exit", "q"}:
            console.print("\n[dim]Goodbye! 👋[/dim]\n")
            break

        try:
            if cmd == "help":
                help_table(length)
            elif cmd == "length":
                if arg.lower() in {"short", "medium", "long"}:
                    length = arg.lower()
                    console.print(f"[green]✓ Summary length set to {length}.[/green]")
                    if doc:
                        console.print("[dim]Applies to the next document you load.[/dim]")
                else:
                    console.print("[red]✗ Use: length short | medium | long[/red]")
            elif cmd == "compare":
                cmd_compare(arg, length)
            elif cmd not in COMMANDS:  # a bare file path
                if not looks_like_path(raw):
                    raise SummarizerError("Unknown command. Type 'help' to see what I can do.")
                doc, convo = cmd_load(raw, length)
            elif cmd == "load":
                if not arg:
                    raise SummarizerError("Usage: load <path to .txt or .md file>")
                doc, convo = cmd_load(arg, length)
            else:
                if doc is None or convo is None:
                    raise SummarizerError("Load a document first, e.g. samples/climate_notes.md")
                if cmd == "summary":
                    panel(doc.summary, f"Summary · {doc.name}")
                elif cmd == "sections":
                    for i, s in enumerate(doc.section_summaries, 1):
                        panel(s, f"Section {i}/{len(doc.section_summaries)}", "blue")
                elif cmd in {"key_points", "keypoints"}:
                    with console.status(f"[{ACCENT}]Extracting key points...", spinner="dots"):
                        result = pipeline.get_key_points(doc.summary)
                    panel(result, "Key points")
                elif cmd in {"quiz_me", "quiz"}:
                    with console.status(f"[{ACCENT}]Writing questions...", spinner="dots"):
                        result = pipeline.get_quiz(doc.text)
                    panel(result, "Quiz me", "yellow")
                elif cmd == "ask":
                    if not arg:
                        raise SummarizerError("Usage: ask <your follow-up question>")
                    with console.status(f"[{ACCENT}]Thinking...", spinner="dots"):
                        result = convo.ask(arg)
                    panel(result, "Answer", "green")
        except SummarizerError as err:
            console.print(Panel(str(err), title="Oops", border_style="red", box=box.ROUNDED))


if __name__ == "__main__":
    main()
