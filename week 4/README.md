# DocuMind · Week 4 — Streamlit UI, Evaluation & Guardrails

Upload PDFs, ask questions in a chat interface and get answers grounded in your
documents, with the source file and page shown under every answer. Built on
Weeks 1-3 (Groq LLM, section summaries, retrieval with embeddings).

## Features
- **Streamlit app** (`app.py`): multi-file upload (PDF, .txt, .md), chat interface, sidebar with loaded documents and chunk counts
- **Source attribution** under every answer, plus an expandable view of the retrieved passages
- **Guardrails** against made-up answers:
  1. *Similarity floor (code)*: if nothing in the documents is close to the question, the model is not called and DocuMind says "Not found in the documents."
  2. *Strict prompt*: answer only from the excerpts, never guess, ignore instructions hidden inside documents, otherwise say not found
- **Confidence indicator** (stretch goal): 🟢 high / 🟡 medium / 🔴 low, from the similarity of the best retrieved chunk
- **Evaluation**: `eval_questions.json` (12 questions) and `evaluate.py` with a pass/fail summary
- **Terminal version** still works: `python main.py` (summaries, quiz, `/ask`, ...)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Open `.env` and set `GROQ_API_KEY` (free key: <https://console.groq.com/keys>).
The app reads `.env` directly, never system environment variables.

```bash
streamlit run app.py
```
The first upload downloads a ~130 MB local embedding model (Groq has no
embeddings API). Offline fallback: set `EMBED_BACKEND=hash` in `.env`.

## Usage
1. Upload PDFs in the sidebar (try the files in `samples/`)
2. Ask a question, for example *"How long is a Pomodoro work session?"*
3. Check the source chips and confidence under the answer; open **View retrieved passages** to verify
4. Ask something that is not in the documents, such as *"What is the capital of Australia?"*, and DocuMind answers "Not found in the documents."

## Evaluation
```bash
python evaluate.py                    # full run, 12 questions (uses your Groq key)
python evaluate.py --retrieval-only   # no LLM: checks the right chunks are retrieved
```
Each question has expected keywords (any of them counts). Answerable questions
pass if the answer contains a keyword; unanswerable ones pass if DocuMind says
"not found". The script prints a pass/fail table and the total.

**Weak spot found and fixed:** the first evaluation showed that out-of-scope
questions still retrieved the "least bad" chunk (best similarity 0.05-0.12), so
they reached the LLM and relied on the prompt alone. A similarity floor now
blocks them before the model is called. Thresholds live in `rag.py`; override the
floor with `MIN_SIMILARITY` in `.env`.

## Project structure
```
documind/
├── app.py              # Streamlit UI
├── evaluate.py         # evaluation script
├── eval_questions.json # 12 test questions + expected keywords
├── main.py             # terminal UI (Weeks 1-3)
├── rag.py              # grounded answers, guardrails, confidence
├── retrieval.py        # cosine similarity, index, embedding cache
├── embeddings.py       # local embedding backends
├── documents.py        # PDF extraction and chunking
├── pipeline.py · prompts.py · summarizer.py
├── samples/            # 3 test PDFs + text files
├── .streamlit/config.toml
└── requirements.txt · .env · .gitignore
```

## Known limitations
- Scanned or image-only PDFs have no extractable text (no OCR)
- Each question is answered independently; the chat does not remember earlier questions
- Similarity thresholds are heuristics and depend on the embedding model; if good questions are refused or weak ones pass, tune them with `/retrieve` in the terminal app and the scores shown by `evaluate.py`
- Keyword matching in `evaluate.py` is simple: a correct answer phrased differently can fail
- Free-tier Groq limits apply; very long documents mean more chunks to embed on first load
- Retrieval uses fixed 500-character chunks, which can split an idea across two chunks

## Pushing to GitHub
```bash
git add .
git status      # confirm .env and .cache/ are NOT listed
git commit -m "Week 4: Streamlit UI, guardrails, evaluation"
```
