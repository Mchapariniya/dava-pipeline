#!/usr/bin/env python3
"""
topic_modeling.py — discover topics within a WhisperX transcript.

Unlike ``qwen_topic_detector.py`` (which assigns ONE fixed-taxonomy label to the
whole file), this module performs genuine *topic modelling*: it discovers the
latent themes inside a single episode and assigns each segment to one of them.

Outputs
-------
1. ``<name>_topics.csv`` — one row per topic, columns compatible with BERTopic's
   ``get_topic_info()`` **and** with ``json_to_eaf.py``'s topic loader:
   ``Topic, Count, Name, Top_Words, Representative_Docs``.
2. Per-segment fields written onto the JSON: ``topic_id`` and ``topic_label``
   (so ELAN export and the dashboard get an exact, non-fuzzy mapping).
3. ``<name>_topics.json`` — the full topic table as JSON (handy for the dashboard).

Methods (choose with ``--method``)
----------------------------------
* ``lda``      — Latent Dirichlet Allocation (scikit-learn). Offline default.
* ``nmf``      — Non-negative Matrix Factorisation (scikit-learn). Offline.
* ``bertopic`` — BERTopic (embeddings + UMAP + HDBSCAN). Best quality; needs
  ``bertopic``/``sentence-transformers`` + model download.
* ``qwen``     — LLM topic modelling: a Qwen model *discovers* the themes (giving
  human-readable labels + keywords), then each chunk is assigned to one. Needs
  ``transformers``/``torch`` + a Qwen model (default ``Qwen/Qwen3-0.6B``). Reuses
  the same model family as ``qwen_topic_detector.py``.

Document unit (``--unit``)
--------------------------
Segments are short single utterances, which is sparse for topic modelling, so by
default consecutive segments are grouped into sliding *chunks*; each original
segment then inherits its chunk's topic.
* ``chunk``   — group ``--chunk_size`` consecutive segments (default). 
* ``segment`` — one segment == one document.
* ``speaker`` — concatenate everything a speaker says (topic-per-speaker).

Public entry point: :func:`run_topic_modeling`.
"""
from __future__ import annotations

import os
import re
import sys
import csv
import json
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

try:
    from . import common as C
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis import common as C


# --------------------------------------------------------------------------- #
# Document preparation
# --------------------------------------------------------------------------- #

def _prepare_documents(segments: List[dict], unit: str, chunk_size: int
                       ) -> Tuple[List[str], List[List[int]]]:
    """Return (documents, doc->[segment indices]) for the chosen unit."""
    sents = C.sentence_texts(segments)  # [(idx, text), ...] non-empty only

    if unit == "segment":
        docs = [t for _, t in sents]
        mapping = [[i] for i, _ in sents]
        return docs, mapping

    if unit == "speaker":
        by_spk: "defaultdict[str, List[int]]" = defaultdict(list)
        for i, _ in sents:
            by_spk[segments[i].get("speaker") or "UNKNOWN"].append(i)
        docs, mapping = [], []
        for _spk, idxs in by_spk.items():
            docs.append(" ".join((segments[i].get("text") or "") for i in idxs))
            mapping.append(idxs)
        return docs, mapping

    # default: sliding chunk of consecutive segments
    docs, mapping = [], []
    for start in range(0, len(sents), chunk_size):
        window = sents[start:start + chunk_size]
        if not window:
            continue
        docs.append(" ".join(t for _, t in window))
        mapping.append([i for i, _ in window])
    return docs, mapping


def _topic_name(top_words: List[str], k: int = 4) -> str:
    """BERTopic-style compact name: 'word1_word2_word3'."""
    return "_".join(top_words[:k]) if top_words else "misc"


# --------------------------------------------------------------------------- #
# scikit-learn backends (LDA / NMF) — fully offline
# --------------------------------------------------------------------------- #

def _sklearn_topics(docs: List[str], method: str, n_topics: int, lang: str,
                    top_words: int = 10, seed: int = 42):
    """Fit LDA or NMF and return (topics_info, doc_topic_ids, doc_topic_probs)."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    import numpy as np

    stop = sorted(C.get_stopwords(lang))
    # guard n_topics against tiny corpora
    n_topics = max(2, min(n_topics, max(2, len(docs) // 2)))

    if method == "nmf":
        vec = TfidfVectorizer(stop_words=stop, min_df=2, max_df=0.95,
                              token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
    else:  # lda
        vec = CountVectorizer(stop_words=stop, min_df=2, max_df=0.95,
                              token_pattern=r"(?u)\b[^\W\d_]{3,}\b")

    try:
        X = vec.fit_transform(docs)
    except ValueError:
        # vocabulary empty (very short corpus) -> relax min_df
        vec.set_params(min_df=1)
        X = vec.fit_transform(docs)

    vocab = np.array(vec.get_feature_names_out())
    if X.shape[1] == 0:
        raise RuntimeError("Empty vocabulary after vectorising; corpus too small.")

    if method == "nmf":
        model = NMF(n_components=n_topics, random_state=seed, init="nndsvda",
                    max_iter=400)
    else:
        model = LatentDirichletAllocation(n_components=n_topics,
                                          random_state=seed, learning_method="batch",
                                          max_iter=25)

    doc_topic = model.fit_transform(X)          # (n_docs, n_topics)
    comps = model.components_                    # (n_topics, vocab)

    topics_info = []
    for t in range(n_topics):
        order = comps[t].argsort()[::-1][:top_words]
        words = [vocab[j] for j in order]
        topics_info.append({"topic_id": t, "top_words": words,
                            "name": _topic_name(words)})

    doc_topic_ids = doc_topic.argmax(axis=1).tolist()
    doc_topic_probs = doc_topic.max(axis=1)
    # normalise probs to [0,1] per row (NMF weights aren't probabilities)
    row_sums = doc_topic.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    doc_topic_probs = (doc_topic.max(axis=1) / row_sums).tolist()
    return topics_info, doc_topic_ids, doc_topic_probs


# --------------------------------------------------------------------------- #
# BERTopic backend — production
# --------------------------------------------------------------------------- #

def _bertopic_topics(docs: List[str], lang: str, embedding_model: Optional[str],
                     min_topic_size: int = 5):
    """Fit BERTopic and return (topics_info, doc_topic_ids, doc_topic_probs)."""
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    stop = sorted(C.get_stopwords(lang))
    vectorizer = CountVectorizer(stop_words=stop,
                                 token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
    kwargs = dict(vectorizer_model=vectorizer, min_topic_size=min_topic_size,
                  calculate_probabilities=True, verbose=False)
    if embedding_model:
        kwargs["embedding_model"] = embedding_model  # e.g. a multilingual ST model

    model = BERTopic(**kwargs)
    topic_ids, probs = model.fit_transform(docs)

    info = model.get_topic_info()
    topics_info = []
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        words = [w for w, _ in (model.get_topic(tid) or [])][:10]
        name = row.get("Name", _topic_name(words))
        topics_info.append({"topic_id": tid, "top_words": words, "name": name})

    # per-doc confidence
    import numpy as np
    if probs is not None and getattr(probs, "ndim", 1) == 2:
        doc_probs = probs.max(axis=1).tolist()
    else:
        doc_probs = [1.0] * len(docs)
    return topics_info, list(topic_ids), doc_probs


# --------------------------------------------------------------------------- #
# Qwen (LLM) backend — discover themes with an LLM, then assign chunks
# --------------------------------------------------------------------------- #

_QWEN_DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def _qwen_build_prompt(document_text: str, num_topics: int, lang: str) -> str:
    """Prompt Qwen to return the main topics as a strict JSON array."""
    lang_hint = {"de": "German", "en": "English", "fr": "French",
                 "it": "Italian", "es": "Spanish"}.get(lang, "the transcript's language")
    return (
        "You are a topic-modelling expert analysing a single transcript.\n\n"
        f"Identify the {num_topics} main topics (themes) discussed in the transcript "
        "below. For each topic give a short human-readable label and 4-8 "
        f"representative keywords. Write the labels and keywords in {lang_hint}.\n\n"
        "Return ONLY a JSON array, with no explanation and no markdown fences, "
        'exactly in this form:\n'
        '[{"label": "short topic name", "keywords": ["word1","word2","word3","word4"]}]\n\n'
        "Transcript:\n\"\"\"\n" + document_text + "\n\"\"\"\n"
    )


_PREAMBLE_RE = re.compile(
    r"^\s*(so\b|here\b|these\b|the following|okay\b|sure\b|as an ai|"
    r"the (main )?topics|below|i (have|will|would)|let me|based on)", re.I)


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>…</think> reasoning and stray tags."""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"</?think>", "", text, flags=re.I)
    return text.strip()


def _repair_truncated_array(frag: str) -> Optional[str]:
    """Close a JSON array truncated mid-way by cutting to the last complete object."""
    last = frag.rfind("}")
    return frag[:last + 1] + "]" if last != -1 else None


def _extract_json_array(text: str):
    """Return a parsed list from ``text``, repairing a truncated array if needed."""
    text = re.sub(r"```(?:json)?", "", text).replace("```", "")
    start = text.find("[")
    if start == -1:
        return None
    frag = text[start:]
    end = frag.rfind("]")
    for cand in ([frag[:end + 1]] if end != -1 else []) + [frag, _repair_truncated_array(frag)]:
        if not cand:
            continue
        try:
            data = json.loads(cand)
        except Exception:
            continue
        if isinstance(data, dict):
            data = data.get("topics") or data.get("results") or []
        if isinstance(data, list) and data:
            return data
    return None


def _valid_topic(t: dict) -> bool:
    """Reject preambles / sentence-like 'topics' that a small model may emit."""
    label = (t.get("label") or "").strip()
    kws = [k for k in (t.get("keywords") or [])
           if k and re.search(r"[^\W\d_]", str(k))]
    if not label or _PREAMBLE_RE.match(label):
        return False
    if len(label.split()) > 8:            # topic labels are short, not sentences
        return False
    return len(kws) >= 2 or len(label.split()) <= 4


def _qwen_parse_topics(response: str, num_topics: int) -> List[dict]:
    """Pull a list of {'label','keywords'} out of an LLM response (JSON-first)."""
    if not response:
        return []
    text = _strip_thinking(response)

    topics: List[dict] = []
    data = _extract_json_array(text)
    if data is not None:
        for item in data:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("name") or "").strip()
            kws = item.get("keywords") or item.get("words") or []
            if isinstance(kws, str):
                kws = [w.strip() for w in re.split(r"[,;/]", kws) if w.strip()]
            kws = [str(w).strip() for w in kws if str(w).strip()]
            if label or kws:
                topics.append({"label": label or _topic_name(kws), "keywords": kws})
    else:
        # last resort: strict "Label: kw, kw, kw" lines (needs >=2 keywords)
        for line in text.splitlines():
            line = line.strip(" -*0123456789.\t")
            if ":" not in line:
                continue
            lab, _, rest = line.partition(":")
            kws = [w.strip() for w in re.split(r"[,;/]", rest) if w.strip()]
            if lab.strip() and len(kws) >= 2:
                topics.append({"label": lab.strip(), "keywords": kws})

    return topics[:num_topics]


def _qwen_discover(_generate, document_text: str, num_topics: int, lang: str,
                   attempts: int = 3) -> List[dict]:
    """Ask the LLM for topics, retrying with a stricter prompt, validating output."""
    base = _qwen_build_prompt(document_text, num_topics, lang)
    stricter = base + ("\n\nIMPORTANT: Output ONLY the JSON array. Begin your reply "
                       "with '[' and end with ']'. Do not write any text, reasoning, "
                       "or explanation before or after the JSON.")
    salvage: List[dict] = []
    for i in range(max(1, attempts)):
        resp = _generate(base if i == 0 else stricter)
        parsed = _qwen_parse_topics(resp, num_topics)
        good = [t for t in parsed if _valid_topic(t)]
        if len(good) >= 2:
            return good
        salvage = salvage or good
    if salvage:
        return salvage
    raise RuntimeError(
        "Qwen did not return usable topics. Qwen3-0.6B is often too small for "
        "structured JSON output — try a bigger model, e.g. "
        "--topic-qwen-model Qwen/Qwen3-1.7B (or Qwen3-4B), or fall back to "
        "--method bertopic / --method lda.")


def _assign_by_keywords(docs: List[str], topics: List[dict]
                        ) -> Tuple[List[int], List[float]]:
    """Assign each doc to the topic whose keywords best match its text.

    Deterministic, needs no extra model calls. Returns (topic_ids, scores) where
    an unmatched doc gets id ``-1`` (folded into the 'misc' bucket downstream).
    """
    kw_sets: List[set] = []
    for t in topics:
        kws = {k.lower() for k in (t.get("keywords") or []) if k}
        # include meaningful words from the label itself
        kws |= {w.lower() for w in re.findall(r"[^\W\d_]+", t.get("label", ""))
                if len(w) > 2}
        kw_sets.append(kws)

    ids, scores = [], []
    for d in docs:
        dl = d.lower()
        hits = [sum(1 for k in kset if k and k in dl) for kset in kw_sets]
        total = sum(hits)
        if total == 0:
            ids.append(-1)
            scores.append(0.0)
            continue
        best = max(range(len(hits)), key=lambda i: hits[i])
        ids.append(best)
        scores.append(round(hits[best] / total, 4))
    return ids, scores


def _qwen_generate_factory(model_id: str, gpu_index: Optional[int],
                           max_new_tokens: int = 1024, temperature: float = 0.2):
    """Load Qwen once and return a ``generate(prompt) -> str`` closure.

    Works with or without ``accelerate``: if it's installed we let it place the
    model via ``device_map`` (needed for large / multi-GPU models); otherwise we
    load plainly and ``.to()`` the chosen device — fine for small models like
    ``Qwen3-0.6B`` and avoids the hard ``accelerate`` dependency.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Pick a device explicitly so we never ask for CUDA on a CPU-only box.
    if gpu_index is not None and torch.cuda.is_available():
        device = f"cuda:{gpu_index}"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # fp16/bf16 weights don't run on CPU; force float32 there.
    dtype = torch.float32 if device == "cpu" else "auto"

    if C.has_module("accelerate"):
        # let accelerate handle placement / offload
        device_map = f"cuda:{gpu_index}" if gpu_index is not None else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device_map, trust_remote_code=True)
    else:
        # no accelerate: plain load, then move to the device ourselves
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True)
        model = model.to(device)

    def _generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        # Qwen3 defaults to "thinking" mode which prepends <think>…</think>;
        # turn it off when the tokenizer supports it (older ones don't).
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=temperature > 0.0,
                                 temperature=temperature if temperature > 0 else 1.0)
        gen = out[0][inputs.input_ids.shape[1]:]
        return _strip_thinking(tokenizer.decode(gen, skip_special_tokens=True))

    return _generate


def _qwen_topics(docs: List[str], lang: str, num_topics: int,
                 model_id: Optional[str] = None, gpu_index: Optional[int] = None,
                 top_words: int = 10, char_budget: int = 6000, _generate=None):
    """LLM topic modelling: Qwen discovers the themes, chunks are then assigned.

    ``_generate`` can be injected (a callable prompt->str) to unit-test without
    loading a model; when None, a Qwen model is loaded via
    :func:`_qwen_generate_factory`.
    """
    model_id = model_id or _QWEN_DEFAULT_MODEL

    # 1) build a representative sample of the document within a character budget
    sample, used = [], 0
    for d in docs:
        if used + len(d) > char_budget and sample:
            break
        sample.append(d)
        used += len(d)
    document_text = "\n".join(sample)

    # 2) discover topics with the LLM (retries + validation inside)
    if _generate is None:
        print(f"  [topics] loading Qwen model '{model_id}' ...")
        _generate = _qwen_generate_factory(model_id, gpu_index)
    topics = _qwen_discover(_generate, document_text, num_topics, lang)
    print(f"  [topics] Qwen proposed {len(topics)} topic(s): "
          + "; ".join(t["label"] for t in topics[:6]))

    # 3) shape into the common topics_info structure
    topics_info = []
    for tid, t in enumerate(topics):
        kws = (t.get("keywords") or [])[:top_words] or [t["label"]]
        topics_info.append({"topic_id": tid, "top_words": kws,
                            "name": t["label"] or _topic_name(kws)})

    # 4) assign each chunk to a discovered topic (keyword overlap)
    doc_ids, doc_scores = _assign_by_keywords(docs, topics)
    return topics_info, doc_ids, doc_scores


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run_topic_modeling(json_path: str, output_json: Optional[str] = None,
                       csv_path: Optional[str] = None, method: str = "auto",
                       num_topics: int = 8, unit: str = "chunk",
                       chunk_size: int = 5, top_words: int = 10,
                       embedding_model: Optional[str] = None,
                       qwen_model: Optional[str] = None,
                       gpu_index: Optional[int] = None) -> Dict:
    data = C.load_transcript(json_path)
    segments = C.get_segments(data)
    if not segments:
        print("  [topics] No segments found; nothing to do.")
        return {"topics": 0}

    lang = C.dominant_language(segments)
    docs, mapping = _prepare_documents(segments, unit, chunk_size)
    print(f"  [topics] {len(docs)} documents (unit={unit}"
          + (f", chunk_size={chunk_size}" if unit == "chunk" else "") + f"), lang={lang}")

    # ---- choose & run method -------------------------------------------------
    order = [method] if method != "auto" else ["bertopic", "lda"]
    topics_info = doc_topic_ids = doc_topic_probs = None
    used = None
    last_err = None
    for m in order:
        try:
            if m == "bertopic":
                if not (C.has_module("bertopic") and C.has_module("sentence_transformers")):
                    raise RuntimeError("bertopic/sentence-transformers not installed")
                topics_info, doc_topic_ids, doc_topic_probs = _bertopic_topics(
                    docs, lang, embedding_model)
            elif m in ("lda", "nmf"):
                topics_info, doc_topic_ids, doc_topic_probs = _sklearn_topics(
                    docs, m, num_topics, lang, top_words=top_words)
            elif m == "qwen":
                if not C.has_module("transformers"):
                    raise RuntimeError("transformers/torch not installed (needed for Qwen)")
                topics_info, doc_topic_ids, doc_topic_probs = _qwen_topics(
                    docs, lang, num_topics, model_id=qwen_model,
                    gpu_index=gpu_index, top_words=top_words)
            else:
                raise ValueError(f"Unknown method: {m}")
            used = m
            break
        except Exception as exc:
            last_err = exc
            if method != "auto":
                raise
            print(f"  [topics] method '{m}' unavailable: {exc}")
    if used is None:
        raise RuntimeError(f"No topic-modelling method available. Last error: {last_err}")
    print(f"  [topics] method = {used}, discovered {len(topics_info)} topics")

    # ---- map topics back onto every segment ---------------------------------
    id_to_name = {t["topic_id"]: t["name"] for t in topics_info}
    seg_topic_counts: Counter = Counter()
    rep_docs: "defaultdict[int, List[Tuple[float, str]]]" = defaultdict(list)

    for doc_i, seg_idxs in enumerate(mapping):
        tid = doc_topic_ids[doc_i]
        prob = float(doc_topic_probs[doc_i]) if doc_topic_probs else None
        label = id_to_name.get(tid, "misc")
        rep_docs[tid].append((prob or 0.0, docs[doc_i]))
        for si in seg_idxs:
            segments[si]["topic_id"] = int(tid)
            segments[si]["topic_label"] = label
            if prob is not None:
                segments[si]["topic_score"] = round(prob, 4)
            seg_topic_counts[tid] += 1

    # segments that weren't in any doc (shouldn't happen, but be safe)
    for seg in segments:
        seg.setdefault("topic_id", -1)
        seg.setdefault("topic_label", "misc")

    # ---- write topics CSV (BERTopic / json_to_eaf compatible) ---------------
    base = C.base_name_from_json(json_path)
    out_dir = os.path.dirname(output_json) if output_json else os.path.dirname(json_path)
    csv_path = csv_path or os.path.join(out_dir, f"{base}_topics.csv")
    json_out_topics = os.path.join(out_dir, f"{base}_topics.json")

    rows = []
    for t in sorted(topics_info, key=lambda x: seg_topic_counts.get(x["topic_id"], 0),
                    reverse=True):
        tid = t["topic_id"]
        reps = [d for _, d in sorted(rep_docs.get(tid, []), reverse=True)[:3]]
        rows.append({
            "Topic": tid,
            "Count": seg_topic_counts.get(tid, 0),
            "Name": t["name"],
            "Top_Words": ", ".join(t["top_words"]),
            "Representative_Docs": repr(reps),   # str(list) -> json_to_eaf ast.literal_eval
        })

    os.makedirs(out_dir, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Topic", "Count", "Name",
                                                "Top_Words", "Representative_Docs"])
        writer.writeheader()
        writer.writerows(rows)

    with open(json_out_topics, "w", encoding="utf-8") as fh:
        json.dump({"method": used, "unit": unit, "language": lang,
                   "topics": rows}, fh, ensure_ascii=False, indent=2)

    # ---- write enriched transcript JSON -------------------------------------
    out_json = output_json or json_path
    C.save_transcript(data, out_json)

    print(f"  [topics] wrote {os.path.basename(csv_path)} and per-segment labels")
    for r in rows[:6]:
        print(f"        #{r['Topic']:>2} ({r['Count']:>3} seg): {r['Top_Words'][:70]}")

    return {"topics": len(rows), "method": used, "csv": csv_path,
            "topics_json": json_out_topics, "json": out_json,
            "distribution": {id_to_name.get(k, str(k)): v
                             for k, v in seg_topic_counts.items()}}


def main():
    ap = argparse.ArgumentParser(description="Topic modelling for WhisperX JSON")
    ap.add_argument("json_file", help="Path to *_whisperx.json")
    ap.add_argument("--output_json", default=None, help="Where to write enriched JSON (default: overwrite)")
    ap.add_argument("--csv", default=None, help="Where to write topics CSV (default: <name>_topics.csv)")
    ap.add_argument("--method", default="auto",
                    choices=["auto", "lda", "nmf", "bertopic", "qwen"],
                    help="Topic method (default: auto -> bertopic, lda)")
    ap.add_argument("--num_topics", type=int, default=8, help="Topic count for lda/nmf/qwen")
    ap.add_argument("--unit", default="chunk",
                    choices=["chunk", "segment", "speaker"],
                    help="Document unit (default: chunk)")
    ap.add_argument("--chunk_size", type=int, default=5, help="Segments per chunk (unit=chunk)")
    ap.add_argument("--top_words", type=int, default=10, help="Words listed per topic")
    ap.add_argument("--embedding_model", default=None,
                    help="Sentence-Transformers model for BERTopic (e.g. a multilingual one)")
    ap.add_argument("--qwen_model", default=None,
                    help=f"Qwen model id for --method qwen (default: {_QWEN_DEFAULT_MODEL})")
    ap.add_argument("--gpu_index", type=int, default=None, help="GPU index for Qwen")
    args = ap.parse_args()

    if not os.path.exists(args.json_file):
        print(f"ERROR: file not found: {args.json_file}")
        sys.exit(1)
    run_topic_modeling(args.json_file, output_json=args.output_json, csv_path=args.csv,
                       method=args.method, num_topics=args.num_topics, unit=args.unit,
                       chunk_size=args.chunk_size, top_words=args.top_words,
                       embedding_model=args.embedding_model,
                       qwen_model=args.qwen_model, gpu_index=args.gpu_index)


if __name__ == "__main__":
    main()
