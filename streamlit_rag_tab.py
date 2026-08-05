#!/usr/bin/env python3
"""
streamlit_rag_tab.py  —  the "Ask" tab for DAVA Insights (UI only).
================================================================================
This file is presentation glue. It calls:
  - dava_rag.Retriever            (semantic retrieval — always)
  - dava_rag.ExtractiveAnswerer   (no-LLM answer)      OR
  - ollama_generator.OllamaGenerator (local generation) — chosen by the user
and reuses elan_viz.render_elan_html to show the answer's references as a small,
audio-synced timeline you can click to hear the exact moment.

No generation or retrieval logic lives here — swap the backend without touching
this tab.

Drop-in:
    from streamlit_rag_tab import render_rag_tab
    tabs = st.tabs([..., "💬 Ask"])
    with tabs[-1]:
        render_rag_tab(default_transcript_path="output/session.json",
                       default_audio_path="output/session.wav")

Standalone:  streamlit run streamlit_rag_tab.py
"""

import base64
import hashlib
import mimetypes
import os
import tempfile

import streamlit as st
from streamlit.components.v1 import html as components_html

import dava_rag
import elan_viz                       # reuse the timeline for reference playback

try:
    import ollama_generator
    _HAVE_OLLAMA = True
except Exception:
    _HAVE_OLLAMA = False

_LANG_UI = {"Auto (match question)": "auto", "German": "de",
            "English": "en", "French": "fr"}


def _read_source(uploaded, default_path):
    """Return (bytes, signature) for the transcript from an upload or a path."""
    if uploaded is not None:
        b = uploaded.getvalue()
        return b, hashlib.md5(b).hexdigest()
    if default_path and os.path.exists(default_path):
        b = open(default_path, "rb").read()
        return b, hashlib.md5(b).hexdigest()
    return None, None


def _get_retriever(sig, transcript_bytes, embed_preset, window, overlap):
    """Build + index the retriever once per (source, settings); cache in session."""
    key = f"{sig}:{embed_preset}:{window}:{overlap}"
    cached = st.session_state.get("_rag_cache")
    if cached and cached["key"] == key:
        return cached["retriever"], cached["duration"]

    preset = dava_rag.EMBED_PRESETS.get(embed_preset, dava_rag.EMBED_PRESETS["mpnet"])
    try:
        embedder = dava_rag.SentenceTransformerEmbedder(
            model=preset["model"], query_prefix=preset["query_prefix"],
            passage_prefix=preset["passage_prefix"])
        # touch the model now so failures surface here, not mid-search
        embedder.encode(["_"], is_query=True)
    except Exception as e:
        st.warning(f"Couldn't load the embedding model ({e}). Falling back to a "
                   f"non-semantic hashing embedder — install sentence-transformers "
                   f"for real semantic search.")
        embedder = dava_rag.HashingEmbedder()

    retriever = dava_rag.Retriever(embedder=embedder, window_secs=window, overlap_secs=overlap)
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.write(transcript_bytes)
    tmp.close()
    try:
        with st.spinner("Indexing the transcript…"):
            n = retriever.index(tmp.name)
    finally:
        os.unlink(tmp.name)

    duration = max((p.end for p in retriever._passages.values()), default=0.0)
    st.session_state["_rag_cache"] = {"key": key, "retriever": retriever,
                                      "duration": duration, "n": n}
    return retriever, duration


def _audio_src(uploaded_audio, audio_url, default_audio_path):
    if uploaded_audio is not None:
        mime = mimetypes.guess_type(uploaded_audio.name)[0] or "audio/mpeg"
        return f"data:{mime};base64,{base64.b64encode(uploaded_audio.getvalue()).decode()}"
    if audio_url:
        return audio_url.strip()
    if default_audio_path and os.path.exists(default_audio_path):
        return elan_viz.embed_audio_datauri(default_audio_path)
    return None


def _references_timeline(passages, cited, duration, audio_src, height_px=200):
    """Render retrieved passages as one ELAN tier — cited ones highlighted — so the
    user can click to hear each referenced moment."""
    anns = []
    for i, p in enumerate(passages):
        anns.append({
            "id": None, "ref": None,
            "start": int(p.start * 1000), "end": int(p.end * 1000),
            "value": (p.text[:80] + "…") if len(p.text) > 80 else p.text,
            "color": "#f2c94c" if i in cited else "#3f6fa0",
        })
    data = {"media": None, "duration_ms": int(max(duration * 1000, 1)),
            "tiers": [{"tier_id": "Answer references", "participant": None,
                       "type": "reference", "parent": None, "color_mode": "solid",
                       "annotations": anns}]}
    html = elan_viz.render_elan_html(data, audio_src=audio_src,
                                     title="Answer references", height_px=height_px)
    components_html(html, height=height_px + 40, scrolling=False)


def render_rag_tab(default_transcript_path: str = None,
                   default_audio_path: str = None) -> None:
    st.subheader("Ask the document")
    st.caption("Ask a question in German, English or French. Retrieval is semantic "
               "and cross-lingual — a French question can surface a German passage — "
               "and every answer links back to the exact moments in the audio.")

    c1, c2 = st.columns(2)
    with c1:
        transcript = st.file_uploader("Transcript JSON (WhisperX segments)",
                                      type=["json"], key="rag_json")
    with c2:
        audio = st.file_uploader("Audio (optional, to play references)",
                                 type=["wav", "mp3", "m4a", "flac", "ogg"], key="rag_audio")
        audio_url = st.text_input("…or audio URL", value="", key="rag_audio_url")

    q = st.text_input("Your question", key="rag_q",
                      placeholder="e.g. Was bedeutet bedürfnisorientierte Erziehung?")

    o1, o2, o3 = st.columns([2, 2, 1])
    with o1:
        lang_ui = st.selectbox("Answer language", list(_LANG_UI.keys()), index=0, key="rag_lang")
    with o2:
        modes = ["Extractive (no LLM)"] + (["Ollama (local)"] if _HAVE_OLLAMA else [])
        mode = st.selectbox("Answer mode", modes, index=0, key="rag_mode")
    with o3:
        top_k = st.slider("Passages", 3, 10, 5, key="rag_k")

    with st.expander("Advanced"):
        a1, a2, a3 = st.columns(3)
        with a1:
            embed_preset = st.selectbox("Embedding model",
                                        list(dava_rag.EMBED_PRESETS.keys()), index=0,
                                        key="rag_embed",
                                        help="mpnet = your default; e5/bge-m3 = stronger "
                                             "cross-lingual (larger download).")
        with a2:
            window = st.number_input("Window (s)", 10.0, 120.0, 45.0, 5.0, key="rag_win")
        with a3:
            overlap = st.number_input("Overlap (s)", 0.0, 60.0, 10.0, 5.0, key="rag_ovl")
        model_name = st.text_input(
            "Ollama model", value=getattr(ollama_generator, "DEFAULT_MODEL", "qwen2.5:7b")
            if _HAVE_OLLAMA else "qwen2.5:7b", key="rag_model",
            help="Must be pulled: `ollama pull qwen2.5:7b`.")

    if not st.button("Ask", type="primary", key="rag_ask"):
        return

    src_bytes, sig = _read_source(transcript, default_transcript_path)
    if src_bytes is None:
        st.info("Upload a transcript JSON (or set default_transcript_path).")
        return
    if not q.strip():
        st.info("Type a question.")
        return

    retriever, duration = _get_retriever(sig, src_bytes, embed_preset, window, overlap)
    passages = retriever.search(q, k=top_k)
    if not passages:
        st.warning("Nothing retrieved — is the transcript empty?")
        return

    target_lang = _LANG_UI[lang_ui]
    if mode == "Ollama (local)":
        gen = ollama_generator.OllamaGenerator(model=model_name)
        ok, msg = gen.available()
        if not ok:
            st.error(msg + "  (Falling back to extractive answer.)")
            res = dava_rag.ExtractiveAnswerer().generate(q, passages, target_lang)
        else:
            with st.spinner(f"Generating with {model_name}…"):
                res = gen.generate(q, passages, target_lang)
    else:
        res = dava_rag.ExtractiveAnswerer().generate(q, passages, target_lang)

    st.markdown("### Answer")
    if res.provider == "extractive":
        st.caption("Extractive mode: the top matching passages, verbatim "
                   "(no synthesis, no translation).")
    st.write(res.answer)
    st.caption(f"answer language: {dava_rag.LANG_NAMES.get(res.target_lang, res.target_lang)} · "
               f"backend: {res.provider} · {len(passages)} passages retrieved")

    src = _audio_src(audio, audio_url, default_audio_path)
    st.markdown("### References")
    _references_timeline(passages, res.cited, duration, src)
    for i, p in enumerate(passages):
        badge = "✓ cited" if i in res.cited else ""
        with st.expander(f"[{p.start_fmt} – {p.end_fmt}]  {p.speaker}  ·  "
                         f"score {p.score:.3f}   {badge}"):
            st.write(p.text)


def _demo() -> None:
    st.set_page_config(page_title="DAVA Insights — Ask", layout="wide")
    st.title("DAVA Insights")
    render_rag_tab()


if __name__ == "__main__":
    _demo()
