"""DocuMind - Week 1: a clean terminal tool that summarises .txt files with Groq."""

import time
from pathlib import Path

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from summarizer import MAX_CHARS, SummarizerError, summarize

console = Console()
ACCENT = "cyan"


def banner() -> None:
    title = Text("DocuMind", style=f"bold {ACCENT}")
    title.append("  ·  Text Summariser", style="dim")
    console.print()
    console.print(Panel(title, box=box.ROUNDED, border_style=ACCENT, padding=(1, 4)))


def help_table(length: str) -> None:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("Input", style=f"bold {ACCENT}")
    table.add_column("What it does")
    table.add_row("<path to .txt>", "Summarise that file")
    table.add_row("length short|medium|long", f"Change summary length (now: [bold]{length}[/bold])")
    table.add_row("help", "Show this menu")
    table.add_row("quit", "Exit DocuMind")
    console.print(table)


def read_file(raw: str) -> str:
    path = Path(raw.strip().strip("'\"")).expanduser()
    if not path.exists():
        raise SummarizerError(f"File not found: {path}")
    if path.suffix.lower() != ".txt":
        raise SummarizerError("Please provide a .txt file.")
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SummarizerError(f"Could not read the file: {exc.strerror or exc}")


def run_summary(raw_path: str, length: str) -> None:
    text = read_file(raw_path)
    if len(text) > MAX_CHARS:
        console.print(
            f"[yellow]Long file: only the first {MAX_CHARS:,} characters will be summarised.[/yellow]"
        )
    with console.status(f"[{ACCENT}]Summarising ({length})...", spinner="dots"):
        summary = summarize(text, length)
    console.print(
        Panel(
            Markdown(summary),
            title=f"[bold]Summary · {length}[/bold]",
            title_align="left",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def main() -> None:
    banner()
    length = "medium"
    help_table(length)

    while True:
        try:
            raw = Prompt.ask(f"\n[bold {ACCENT}]›[/bold {ACCENT}] File path or command").strip()
        except (KeyboardInterrupt, EOFError):
            raw = "quit"

        if not raw:
            continue
        cmd = raw.lower()

        if cmd in {"quit", "exit", "q"}:
            console.print("\n[dim]Goodbye! 👋[/dim]\n")
            break
        if cmd == "help":
            help_table(length)
            continue
        if cmd.startswith("length"):
            choice = cmd.replace("length", "", 1).strip()
            if choice in {"short", "medium", "long"}:
                length = choice
                console.print(f"[green]✓ Summary length set to {length}.[/green]")
            else:
                console.print("[red]✗ Use: length short | medium | long[/red]")
            continue

        try:
            run_summary(raw, length)
        except SummarizerError as err:
            console.print(Panel(str(err), title="Oops", border_style="red", box=box.ROUNDED))
        time.sleep(1)  # gentle pause to respect free-tier rate limits


if __name__ == "__main__":
    main()
