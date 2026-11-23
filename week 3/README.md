# DocuMind · Week 3 — Retrieval Over Your Own PDFs

Ask questions about your own PDFs and text files and get answers grounded in
their content, with the source document and page shown for every answer.
Builds on Weeks 1-2 (summaries, key points, quiz, follow-ups). LLM: **Groq**.
Terminal UI: [`rich`](https://github.com/Textualize/rich).

## How retrieval works (simple RAG)
1. `extract_text(pdf_path)` reads a PDF with `pypdf` (page by page)
2. `chunk_text(text, size=500, overlap=50)` cuts overlapping chunks, tagged with source file and page
3. `embed(text)` turns each chunk into a vector; vectors are cached per file
4. `retrieve(question, top_k=3)` ranks every chunk with `cosine_similarity`
5. The top chunks go into a RAG prompt that says "answer only from this context,
   otherwise reply *Not found in the documents*"
6. `/ask` prints the answer plus the sources it used

> **Embeddings:** Groq does not offer an embeddings API, so DocuMind embeds
> locally with `fastembed` (model `BAAI/bge-small-en-v1.5`, free, no key). The
> first run downloads about 130 MB. For a no-download fallback set
> `EMBED_BACKEND=hash` in `.env` (works, but matches keywords more than meaning).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Open `.env`, set `GROQ_API_KEY` (free key: <https://console.groq.com/keys>).
The app reads `.env` directly, never system environment variables.

```bash
python main.py
```

## Try it
```
/add samples
/ask How long is a Pomodoro work session?          # answer in one PDF
/ask How can machine learning help renewable grids? # spans two PDFs
/ask What is the capital of Australia?              # not covered -> "Not found"
/retrieve wind turbine capacity                     # inspect the top chunks
```

## Commands
| Command | Action |
|---|---|
| `/add <file\|folder>` | Index `.pdf`, `.txt`, `.md` files (cached by filename + content hash) |
| `/ask <question>` | Grounded answer with source attribution |
| `/retrieve <question>` | Show the top matching chunks and scores (no LLM) |
| `/docs` · `/clear` | List indexed documents · empty the index |
| `/rerank on\|off` | Stretch goal: LLM picks the best 3 of the top 10 chunks |
| `<file>`, `summary`, `sections`, `key_points`, `quiz_me`, `ask`, `compare`, `length` | Week 2 commands (now also accept PDFs) |

## Caching
Embeddings are stored in `.cache/embeddings/` as `<name>-<hash>.json`. A file is
re-embedded only if its content, the embedding backend or the chunk settings change.

## Project structure
```
documind/
├── main.py         # CLI / terminal UI
├── documents.py    # extract_text, chunk_text (pypdf)
├── embeddings.py   # embed(), local backends
├── retrieval.py    # cosine_similarity, DocumentIndex.retrieve, caching
├── rag.py          # answer_question, rerank
├── pipeline.py     # Week 2: split, summarise, Conversation
├── prompts.py      # all prompt templates
├── summarizer.py   # Groq wrapper, .env loading, friendly errors
├── samples/        # 3 test PDFs + text files
├── requirements.txt · .env · .gitignore
```

## Known limitations
- Scanned or image-only PDFs have no extractable text (OCR is not included)
- Retrieval quality depends on the embedding model; 500-character chunks can split long ideas
- Answers are only as good as the retrieved chunks, so use `/retrieve` to check them

## Pushing to GitHub
```bash
git add .
git status      # confirm .env and .cache/ are NOT listed
git commit -m "Week 3: PDF retrieval with embeddings and cited answers"
```
