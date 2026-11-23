# DocuMind · Week 2 — Prompt Templates & Structured Summaries

Builds on Week 1. DocuMind now splits a document into sections, summarises each
one, combines them into a single coherent summary, and lets you extract key
points, quiz yourself and ask follow-up questions. Powered by the **Groq API**
(free tier) with a clean [`rich`](https://github.com/Textualize/rich) terminal UI.

## What's new in Week 2
- `prompts.py` — system prompt + templates: `summarize`, `key_points`, `generate_questions` (plus `combine_summaries`, `followup_context`, `compare`)
- `split_sections(text)` — splits on Markdown headings, else blank-line paragraphs; merges tiny pieces and chunks oversize ones
- `summarize_document(sections)` — summarises each section, then a **second LLM call** combines them into one summary
- `key_points` and `quiz_me` commands
- `Conversation` class — small chat history for follow-up questions with the summary as context
- Stretch goal: `compare` two documents

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
```
Edit `.env` and set `GROQ_API_KEY` (free key: <https://console.groq.com/keys>).
The app reads `.env` directly; it never uses system environment variables.

```bash
python main.py
```

## Commands
| Command | Action |
|---|---|
| `samples/climate_notes.md` or `load <path>` | Load a `.txt`/`.md` file, show its sections and summary |
| `summary` | Show the combined summary again |
| `sections` | Inspect each section's own summary |
| `key_points` | 5-8 key points (from the combined summary) |
| `quiz_me` | 3-5 revision questions with answers |
| `ask <question>` | Follow-up question (keeps a short chat history) |
| `compare <a> <b>` | Key differences between two documents |
| `length short\|medium\|long` | Summary length for the next load |
| `help` / `quit` | Menu / exit |

## Project structure
```
documind/
├── main.py            # CLI / terminal UI
├── pipeline.py        # split_sections, summarize_document, Conversation
├── prompts.py         # system prompt + prompt templates
├── summarizer.py      # Groq wrapper, .env loading, friendly errors
├── samples/           # climate_notes.md (4 sections), article.txt, notes.txt
├── requirements.txt
├── .env.example       # template (safe to commit)
└── .gitignore         # keeps .env out of Git
```

## Pushing to GitHub
```bash
git add .
git status               # confirm .env is NOT listed
git commit -m "Week 2: prompt templates, section pipeline, key points, quiz, follow-ups"
```

## Notes
- Each section is limited to ~4,000 characters; longer ones are split automatically.
- Free tiers are rate-limited, so the pipeline pauses briefly between calls.
