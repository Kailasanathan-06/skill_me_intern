"""DocuMind - Week 4: a Streamlit app for asking questions about your own documents.

Run with:  streamlit run app.py
"""

import hashlib
import html
import tempfile
from pathlib import Path

import streamlit as st

import rag
from retrieval import DocumentIndex
from summarizer import SummarizerError, has_api_key

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="centered", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 780px; }
      #MainMenu, footer { visibility: hidden; }
      .dm-hero h1 { margin: 0; font-size: 2.1rem; letter-spacing: -0.02em; }
      .dm-hero p { margin: .2rem 0 1.2rem 0; opacity: .7; }
      .dm-card { border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 1.1rem 1.3rem; margin: .6rem 0; }
      .dm-card ol { margin: .4rem 0 0 0; padding-left: 1.2rem; line-height: 1.8; }
      .dm-doc { display: flex; justify-content: space-between; gap: .6rem; padding: .45rem .7rem; margin-bottom: .35rem;
                border: 1px solid rgba(128,128,128,.25); border-radius: 10px; font-size: .88rem; }
      .dm-doc .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .dm-doc .count { opacity: .6; white-space: nowrap; }
      .dm-chip { display: inline-block; padding: .12rem .6rem; margin: .15rem .3rem .15rem 0; border-radius: 999px; font-size: .78rem;
                 background: rgba(13,148,136,.12); border: 1px solid rgba(13,148,136,.35); }
      .dm-conf-high { color: #16a34a; } .dm-conf-medium { color: #d97706; } .dm-conf-low { color: #dc2626; }
      .dm-meta { font-size: .8rem; margin-top: .3rem; opacity: .9; }
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = ["Summarise the key idea of each document", "What are the main numbers or dates mentioned?", "What is not covered here?"]
CONF_LABEL = {"high": "🟢 High confidence", "medium": "🟡 Medium confidence", "low": "🔴 Low confidence"}


# ------------------------------------------------------------------- state
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("index", DocumentIndex())  # chunks + vectors live here for the whole session
    ss.setdefault("messages", [])
    ss.setdefault("digests", {})  # filename -> content hash of processed uploads
    ss.setdefault("upload_errors", {})  # filename -> (content hash, message)
    ss.setdefault("tmp_dir", tempfile.mkdtemp(prefix="documind_"))


def sync_uploads(files) -> None:
    """Run extract -> chunk -> embed for new uploads; drop documents that were removed."""
    ss, index = st.session_state, st.session_state.index
    current = {Path(f.name).name for f in files}
    for name in list(index.docs):
        if name not in current:
            index.remove_document(name)
            ss.digests.pop(name, None)
    for name in list(ss.upload_errors):
        if name not in current:
            ss.upload_errors.pop(name)

    pending = []
    for f in files:
        name, data = Path(f.name).name, f.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        if ss.digests.get(name) == digest or ss.upload_errors.get(name, (None,))[0] == digest:
            continue  # already handled; Streamlit reruns the script on every interaction
        pending.append((name, data, digest))
    if not pending:
        return

    with st.spinner("Reading and embedding documents (the first run downloads the embedding model)..."):
        for name, data, digest in pending:
            path = Path(ss.tmp_dir) / name
            path.write_bytes(data)
            try:
                index.add_document(path)
                ss.digests[name] = digest
                ss.upload_errors.pop(name, None)
            except SummarizerError as err:
                ss.upload_errors[name] = (digest, str(err))


# ----------------------------------------------------------------- sidebar
def sidebar() -> tuple[int, bool]:
    with st.sidebar:
        st.markdown("## 📄 DocuMind")
        st.caption("Upload documents, then ask questions about them.")
        files = st.file_uploader(
            "Upload documents", type=["pdf", "txt", "md"], accept_multiple_files=True, label_visibility="collapsed"
        )
        sync_uploads(files or [])

        st.markdown("##### Loaded documents")
        docs = st.session_state.index.docs
        if docs:
            for info in docs.values():
                st.markdown(
                    f"<div class='dm-doc'><span class='name'>📄 {html.escape(info['name'])}</span>"
                    f"<span class='count'>{info['chunks']} chunks</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Nothing loaded yet.")
        for name, (_, message) in st.session_state.upload_errors.items():
            st.error(f"**{name}**: {message}")

        with st.expander("Settings"):
            top_k = st.slider("Passages used per answer", 1, 6, 3)
            use_rerank = st.toggle("Re-rank passages with the LLM", value=False, help="Picks the best of the top 10 passages.")

        if not has_api_key():
            st.warning("Add your `GROQ_API_KEY` to the `.env` file, then restart the app.")
        if st.button("Clear chat", use_container_width=True, disabled=not st.session_state.messages):
            st.session_state.messages = []
            st.rerun()
    return top_k, use_rerank


# ------------------------------------------------------------------- chat
def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📄"):
        if msg.get("error"):
            st.error(msg["content"])
            return
        if msg.get("found") is False:
            st.warning(msg["content"])
            st.caption("Nothing in the loaded documents answers this question.")
            return
        st.markdown(msg["content"])
        if msg["role"] != "assistant" or not msg.get("sources"):
            return

        chips = "".join(
            f"<span class='dm-chip'>📄 {html.escape(s['label'])}</span>" for s in msg["sources"]
        )
        conf = msg["confidence"]
        st.markdown(
            f"{chips}<div class='dm-meta'><span class='dm-conf-{conf}'>{CONF_LABEL[conf]}</span>"
            f" · best match {msg['top_score']:.2f}</div>",
            unsafe_allow_html=True,
        )
        with st.expander(f"View retrieved passages ({len(msg['sources'])})"):
            for s in msg["sources"]:
                st.markdown(f"**{s['label']}** · similarity {s['score']:.2f}")
                st.text(s["text"])


def answer_to_message(result: rag.Answer) -> dict:
    return {
        "role": "assistant",
        "content": result.text,
        "found": result.found,
        "confidence": result.confidence,
        "top_score": result.top_score,
        "sources": [{"label": h.label, "text": h.text, "score": h.score} for h in result.used],
    }


def welcome() -> None:
    st.markdown(
        """
        <div class="dm-card"><b>How it works</b>
        <ol><li>Upload one or more PDFs (or .txt / .md) in the sidebar</li>
        <li>Ask a question in the box below</li>
        <li>Check the sources shown under every answer</li></ol></div>
        <div class="dm-card"><b>Good to know</b><br>
        Answers come <i>only</i> from your documents. If the answer isn't there, DocuMind says so instead of guessing.</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    init_state()
    top_k, use_rerank = sidebar()

    st.markdown(
        "<div class='dm-hero'><h1>DocuMind</h1><p>Ask questions about your documents and get answers you can verify.</p></div>",
        unsafe_allow_html=True,
    )

    index = st.session_state.index
    if not index.docs and not st.session_state.messages:
        welcome()
    for msg in st.session_state.messages:
        render_message(msg)

    prompt = st.chat_input(
        "Ask a question about your documents..." if index.docs else "Upload a document to start",
        disabled=not index.docs,
    )
    if not prompt:
        return

    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.spinner("Searching your documents..."):
        try:
            reply = answer_to_message(rag.answer_question(prompt, index, top_k, use_rerank))
        except SummarizerError as err:
            reply = {"role": "assistant", "content": str(err), "error": True}
    st.session_state.messages.append(reply)
    render_message(reply)


main()
