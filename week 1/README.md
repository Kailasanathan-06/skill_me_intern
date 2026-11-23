# DocuMind · Week 1 — Your First LLM API Call

A clean terminal tool that summarises any `.txt` file in bullet points using the
**Groq API** (free tier). Part of the SkillMe GenAI internship project
*DocuMind: Document Q&A Tool with Retrieval*.

## Features
- `summarize(text)` function wrapping the Groq chat completion API
- API key kept in a `.env` file (git-ignored) and read directly from the file with `python-dotenv`
- Friendly error messages for a bad key, missing file, empty file, rate limit or no internet
- Polished terminal UI built with [`rich`](https://github.com/Textualize/rich)
- Loops until you type `quit`
- Stretch goal: `length short|medium|long` changes the summary style

## Setup

1. **Get a free key** at <https://console.groq.com/keys>.
2. **Install dependencies** (Python 3.9+):
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Add your key**:
   ```bash
   cp .env.example .env             # Windows: copy .env.example .env
   ```
   Then edit `.env` and replace `your_groq_api_key_here` with your real key.
4. **Run**:
   ```bash
   python main.py
   ```

## Usage

| Input | Action |
|---|---|
| `samples/article.txt` | Summarise a file |
| `length short` / `medium` / `long` | Change summary length |
| `help` | Show the menu |
| `quit` | Exit |

## Project structure
```
documind/
├── main.py            # CLI / terminal UI
├── summarizer.py      # summarize(text) + error handling
├── samples/           # test text files
├── requirements.txt
├── .env.example       # template (safe to commit)
└── .gitignore         # keeps .env out of Git
```

## Pushing to GitHub safely
```bash
git init
git add .
git status               # confirm .env is NOT listed
git commit -m "Week 1: DocuMind summariser CLI"
```
Never commit `.env`. If a key is ever exposed, revoke it in the Groq console.

## Notes
Free tiers are rate-limited; the CLI pauses 1 second between requests.
Files longer than 24,000 characters are truncated (chunking arrives in Week 2).
