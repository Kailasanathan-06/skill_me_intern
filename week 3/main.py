"""DocuMind - Week 3: summaries, quizzes and retrieval-based Q&A over your files."""

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

import documents
import pipeline
import rag
from pipeline import Conversation
from retrieval import DocumentIndex
from summarizer import SummarizerError

console = Console()
ACCENT = "cyan"
EXTENSIONS = documents.SUPPORTED
SUMMARY_COMMANDS = {"load", "summary", "sections", "key_points", "keypoints", "quiz_me", "quiz", "ask", "compare"}


@dataclass
class Document:
    name: str
    text: str
    sections: list = field(default_factory=list)
    section_summaries: list = field(default_factory=list)
    summary: str = ""


@dataclass
class State:
    length: str = "medium"
    doc: Document | None = None
    convo: Conversation | None = None
    index: DocumentIndex = field(default_factory=DocumentIndex)
    rerank: bool = False
    top_k: int = 3


# ----------------------------------------------------------------------- UI
def banner() -> None:
    title = Text("DocuMind", style=f"bold {ACCENT}")
    title.append("  ·  Week 3 · Retrieval over your PDFs", style="dim")
    console.print()
    console.print(Panel(title, box=box.ROUNDED, border_style=ACCENT, padding=(1, 4)))


def _menu(title: str, rows: list[tuple]) -> None:
    table = Table(title=title, title_justify="left", title_style="bold", box=box.SIMPLE_HEAD,
                  show_edge=False, pad_edge=False)
    table.add_column("Command", style=f"bold {ACCENT}")
    table.add_column("What it does")
    for row in rows:
        table.add_row(*row)
    console.print(table)


def help_menu(state: State) -> None:
    _menu("Ask your documents", [
        ("/add <file|folder>", "Index PDFs / .txt / .md files (try: /add samples)"),
        ("/ask <question>", "Grounded answer with sources"),
        ("/retrieve <question>", "Show the top matching chunks (no LLM)"),
        ("/docs", "List indexed documents and chunk counts"),
        ("/rerank on|off", f"LLM re-ranking of top 10 chunks (now: [bold]{'on' if state.rerank else 'off'}[/bold])"),
        ("/clear", "Remove all documents from the index"),
    ])
    _menu("Summarise one document", [
        ("<file> or load <file>", "Load and summarise a .pdf / .txt / .md"),
        ("summary · sections", "Combined summary · per-section summaries"),
        ("key_points · quiz_me", "5-8 key points · 3-5 quiz questions"),
        ("ask <question>", "Follow-up about the loaded document"),
        ("compare <a> <b>", "Key differences between two files"),
        ("length short|medium|long", f"Summary length (now: [bold]{state.length}[/bold])"),
        ("help · quit", "Show this menu · exit"),
    ])


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


def docs_table(index: DocumentIndex) -> None:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("Document", style="bold")
    table.add_column("Pages", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Embeddings", style="dim")
    for info in index.docs.values():
        table.add_row(escape(info["name"]), str(info["pages"]), str(info["chunks"]),
                      "from cache" if info["cached"] else "newly embedded")
    console.print(table)


def sources_table(hits, title="Sources") -> None:
    table = Table(title=title, title_justify="left", title_style="bold dim", box=box.SIMPLE_HEAD,
                  show_edge=False, pad_edge=False)
    table.add_column("Document", style=f"bold {ACCENT}")
    table.add_column("Page", justify="right")
    table.add_column("Similarity", justify="right")
    for h in hits:
        table.add_row(escape(h.source), str(h.page or "–"), f"{h.score:.2f}")
    console.print(table)


# ------------------------------------------------------------------ helpers
def clean_path(raw: str) -> Path:
    return Path(raw.strip().strip("'\"")).expanduser()


def split_args(arg: str) -> list[str]:
    try:
        return [p.strip("'\"") for p in shlex.split(arg, posix=False)]
    except ValueError:
        return [arg]


def read_file(raw: str) -> tuple[str, str]:
    path = clean_path(raw)
    if not path.exists():
        raise SummarizerError(f"File not found: {path}")
    if path.suffix.lower() not in EXTENSIONS:
        raise SummarizerError("Please provide a .pdf, .txt or .md file.")
    if path.suffix.lower() == ".pdf":
        return path.name, documents.extract_text(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SummarizerError(f"Could not read the file: {exc.strerror or exc}")
    if not text.strip():
        raise SummarizerError("The file is empty, so there is nothing to summarise.")
    return path.name, text


def process(raw: str, length: str, status) -> Document:
    name, text = read_file(raw)
    doc = Document(name=name, text=text, sections=pipeline.split_sections(text))
    doc.section_summaries, doc.summary = pipeline.summarize_document(
        doc.sections, length, progress=lambda msg: status.update(f"[{ACCENT}]{name}: {msg}")
    )
    return doc


def looks_like_path(raw: str) -> bool:
    p = clean_path(raw)
    return p.suffix.lower() in EXTENSIONS or p.exists()


# --------------------------------------------------------- Week 2 commands
def cmd_load(raw: str, state: State) -> None:
    name, text = read_file(raw)
    sections = pipeline.split_sections(text)
    console.print(f"\n[bold]{escape(name)}[/bold] → [green]{len(sections)} section(s)[/green]")
    sections_table(sections)
    with console.status(f"[{ACCENT}]Starting...", spinner="dots") as status:
        doc = process(raw, state.length, status)
    panel(doc.summary, f"Summary · {state.length}")
    state.doc, state.convo = doc, Conversation(doc.summary, doc.section_summaries)


def cmd_compare(arg: str, state: State) -> None:
    paths = split_args(arg)
    if len(paths) != 2:
        raise SummarizerError("Usage: compare <file1> <file2>  (quote paths that contain spaces)")
    with console.status(f"[{ACCENT}]Starting...", spinner="dots") as status:
        a = process(paths[0], state.length, status)
        b = process(paths[1], state.length, status)
        status.update(f"[{ACCENT}]Comparing documents...")
        result = pipeline.compare_documents(a.name, a.summary, b.name, b.summary)
    panel(result, f"Compare · {a.name} vs {b.name}", "magenta")


def handle_summary_command(cmd: str, arg: str, state: State) -> None:
    if cmd == "load":
        if not arg:
            raise SummarizerError("Usage: load <path to .pdf, .txt or .md file>")
        return cmd_load(arg, state)
    if cmd == "compare":
        return cmd_compare(arg, state)
    if state.doc is None or state.convo is None:
        raise SummarizerError("Load a document first, e.g. samples/climate_notes.md")
    doc = state.doc
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
            result = state.convo.ask(arg)
        panel(result, "Answer")


# --------------------------------------------------------- Week 3 commands
def cmd_add(arg: str, state: State) -> None:
    if not arg:
        raise SummarizerError("Usage: /add <file.pdf | folder> [more files]")
    paths: list[Path] = []
    for token in split_args(arg):
        p = clean_path(token)
        if p.is_dir():
            paths += sorted(f for f in p.iterdir() if f.suffix.lower() in documents.SUPPORTED)
        else:
            paths.append(p)
    if not paths:
        raise SummarizerError("No supported files found (.pdf, .txt, .md).")

    added = 0
    with console.status(f"[{ACCENT}]Starting...", spinner="dots") as status:
        for p in paths:
            status.update(f"[{ACCENT}]Indexing {p.name} (first run downloads the embedding model)...")
            try:
                state.index.add_document(p)
                added += 1
            except SummarizerError as err:
                console.print(f"[red]✗ {escape(p.name)}: {escape(str(err))}[/red]")
    if added:
        docs_table(state.index)
        console.print(f"[green]✓ {len(state.index.docs)} document(s) · {len(state.index.chunks)} chunks ready.[/green]")


def cmd_ask(arg: str, state: State) -> None:
    if not arg:
        raise SummarizerError("Usage: /ask <your question>")
    with console.status(f"[{ACCENT}]Searching documents...", spinner="dots") as status:
        if state.rerank:
            status.update(f"[{ACCENT}]Searching and re-ranking...")
        result = rag.answer_question(arg, state.index, state.top_k, state.rerank)
    panel(result.text, "Answer" if result.found else "Not found", "green" if result.found else "yellow")
    if result.found:
        sources_table(result.used)
    else:
        console.print("[dim]Nothing in the loaded documents answers this question.[/dim]")


def cmd_retrieve(arg: str, state: State) -> None:
    if not arg:
        raise SummarizerError("Usage: /retrieve <question>")
    with console.status(f"[{ACCENT}]Searching...", spinner="dots"):
        hits = state.index.retrieve(arg, state.top_k)
    for i, h in enumerate(hits, 1):
        panel(escape(h.text), f"#{i} · {h.label} · similarity {h.score:.2f}", "blue")


def handle_slash(cmd: str, arg: str, state: State) -> None:
    if cmd == "add":
        cmd_add(arg, state)
    elif cmd == "ask":
        cmd_ask(arg, state)
    elif cmd in {"retrieve", "search"}:
        cmd_retrieve(arg, state)
    elif cmd == "docs":
        if not state.index.docs:
            raise SummarizerError("No documents loaded yet. Use /add <file.pdf> first.")
        docs_table(state.index)
    elif cmd == "rerank":
        if arg.lower() not in {"on", "off"}:
            raise SummarizerError("Use: /rerank on  or  /rerank off")
        state.rerank = arg.lower() == "on"
        console.print(f"[green]✓ Re-ranking {'on' if state.rerank else 'off'}.[/green]")
    elif cmd == "clear":
        state.index.clear()
        console.print("[green]✓ Index cleared (cached embeddings are kept on disk).[/green]")
    elif cmd == "help":
        help_menu(state)
    else:
        raise SummarizerError("Unknown command. Type 'help' to see what I can do.")


# --------------------------------------------------------------------- main
def main() -> None:
    banner()
    state = State()
    help_menu(state)

    while True:
        label = f"[bold {ACCENT}]{escape(state.doc.name)}[/bold {ACCENT}] ›" if state.doc else f"[bold {ACCENT}]›[/bold {ACCENT}]"
        try:
            raw = Prompt.ask(f"\n{label} Command").strip()
        except (KeyboardInterrupt, EOFError):
            raw = "quit"
        if not raw:
            continue

        parts = raw.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in {"quit", "exit", "q", "/quit", "/exit"}:
            console.print("\n[dim]Goodbye! 👋[/dim]\n")
            break

        try:
            if cmd.startswith("/"):
                handle_slash(cmd[1:], arg, state)
            elif cmd == "help":
                help_menu(state)
            elif cmd == "length":
                if arg.lower() in {"short", "medium", "long"}:
                    state.length = arg.lower()
                    console.print(f"[green]✓ Summary length set to {state.length}.[/green]")
                else:
                    console.print("[red]✗ Use: length short | medium | long[/red]")
            elif cmd in SUMMARY_COMMANDS:
                handle_summary_command(cmd, arg, state)
            elif looks_like_path(raw):
                cmd_load(raw, state)
            else:
                raise SummarizerError("Unknown command. Type 'help' to see what I can do.")
        except SummarizerError as err:
            console.print(Panel(escape(str(err)), title="Oops", border_style="red", box=box.ROUNDED))


if __name__ == "__main__":
    main()
