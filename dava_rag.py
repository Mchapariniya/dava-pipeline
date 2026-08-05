#!/usr/bin/env python3
"""
dava_rag.py  —  local, multilingual, semantic RETRIEVAL over one transcript.
================================================================================
This module is the heart of the RAG feature and it is deliberately LLM-free and
network-free. It turns a WhisperX transcript into searchable, timestamped
passages and answers "which parts of the audio are about X?" by *meaning*, across
German / English / French (a French question can match a German passage, because
the embeddings are multilingual).

What lives here
---------------
- chunking            : build_windows()  — overlapping time windows (45s / 10s),
                        each keeping its start/end so answers can cite audio times
- embeddings          : SentenceTransformerEmbedder (default, multilingual, local)
                        + HashingEmbedder (dependency-free, used for tests/offline)
- vector store        : InMemoryStore (numpy cosine, perfect for one document)
                        + ChromaStore (persistent, matches your existing pipeline)
- retriever           : Retriever.index() / Retriever.search() -> List[Passage]
- no-LLM answer       : ExtractiveAnswerer — returns the top passages as evidence
- shared prompt bits  : build_grounded_prompt(), resolve_target_lang(), parse_cited()
                        (imported by the SEPARATE ollama_generator.py so the prompt
                         shaping stays in one place)

Nothing here calls Ollama or any API. Generation is a separate module.

Default embedding model
-----------------------
sentence-transformers/paraphrase-multilingual-mpnet-base-v2  (local, ~420 MB,
same one your film pipeline uses). For stronger cross-lingual retrieval you can
swap in "intfloat/multilingual-e5-large" or "BAAI/bge-m3" — see EMBED_PRESETS.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Optional upgrades. e5/bge want the query/passage prefixes shown here.
EMBED_PRESETS = {
    "mpnet": {"model": DEFAULT_EMBED_MODEL, "query_prefix": "", "passage_prefix": ""},
    "e5":    {"model": "intfloat/multilingual-e5-large",
              "query_prefix": "query: ", "passage_prefix": "passage: "},
    "bge-m3": {"model": "BAAI/bge-m3", "query_prefix": "", "passage_prefix": ""},
}

LANG_NAMES = {"de": "German", "en": "English", "fr": "French"}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def fmt_ts(seconds: float) -> str:
    """Seconds -> 'MM:SS' or 'H:MM:SS'."""
    seconds = max(0.0, float(seconds))
    s = int(seconds) % 60
    m = int(seconds) // 60 % 60
    h = int(seconds) // 3600
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _f(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def seg_start(s: dict) -> float:
    return _f(s.get("start"))


def seg_end(s: dict) -> float:
    e = _f(s.get("end"))
    return e if e > seg_start(s) else seg_start(s) + 0.2


def seg_text(s: dict) -> str:
    return (s.get("text") or "").strip()


def seg_speaker(s: dict) -> str:
    for k in ("speaker", "speakerNumber", "speakerID", "speaker_id"):
        v = s.get(k)
        if v not in (None, ""):
            return str(v)
    return ""


def seg_lang(s: dict) -> str:
    for k in ("detected_language", "language", "lang"):
        v = s.get(k)
        if v:
            return str(v)
    return ""


def load_segments(src) -> list:
    """Accept a path, or an already-loaded dict/list; return the segment list."""
    raw = src if isinstance(src, (dict, list)) else json.load(open(src, encoding="utf-8"))
    return raw.get("segments", raw) if isinstance(raw, dict) else raw


# --------------------------------------------------------------------------- #
# passages (the unit of retrieval; always carries audio timestamps)
# --------------------------------------------------------------------------- #
@dataclass
class Passage:
    id: str
    text: str
    start: float                 # seconds
    end: float                   # seconds
    speaker: str = ""
    langs: List[str] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)   # source segment indices
    score: float = 0.0           # similarity, filled in at search time

    @property
    def start_fmt(self) -> str:
        return fmt_ts(self.start)

    @property
    def end_fmt(self) -> str:
        return fmt_ts(self.end)

    def to_metadata(self) -> dict:
        return {"text": self.text, "start": self.start, "end": self.end,
                "speaker": self.speaker, "langs": ",".join(self.langs),
                "indices": json.dumps(self.indices)}

    @classmethod
    def from_metadata(cls, pid: str, m: dict, score: float = 0.0) -> "Passage":
        langs = [x for x in (m.get("langs", "") or "").split(",") if x]
        try:
            idx = json.loads(m.get("indices", "[]"))
        except Exception:
            idx = []
        return cls(id=pid, text=m.get("text", ""), start=_f(m.get("start")),
                   end=_f(m.get("end")), speaker=m.get("speaker", ""),
                   langs=langs, indices=idx, score=score)


def build_windows(segments, window_secs: float = 45.0,
                  overlap_secs: float = 10.0) -> List[Passage]:
    """Group consecutive segments into overlapping time windows, mirroring your
    film pipeline's windowing but producing Passage objects that keep timestamps,
    speaker(s) and detected language(s)."""
    segs = load_segments(segments)
    passages: List[Passage] = []
    i, n, w = 0, len(segs), 0
    while i < n:
        win_start = seg_start(segs[i])
        win_end = win_start + window_secs
        idx = [j for j in range(i, n) if seg_start(segs[j]) < win_end]
        if not idx:
            i += 1
            continue
        text = " ".join(seg_text(segs[j]) for j in idx).strip()
        if text:
            spk = [seg_speaker(segs[j]) for j in idx if seg_speaker(segs[j])]
            langs = [seg_lang(segs[j]) for j in idx if seg_lang(segs[j])]
            main_spk = max(set(spk), key=spk.count) if spk else ""
            passages.append(Passage(
                id=f"win_{w}", text=text,
                start=seg_start(segs[idx[0]]), end=seg_end(segs[idx[-1]]),
                speaker=main_spk, langs=sorted(set(langs)), indices=idx))
            w += 1
        nxt = win_start + window_secs - overlap_secs
        while i < n and seg_start(segs[i]) < nxt:
            i += 1
    return passages


# --------------------------------------------------------------------------- #
# language detection (heuristic; uses langdetect if it happens to be installed)
# --------------------------------------------------------------------------- #
_MARKERS = {
    "de": (" der die das und ist nicht ein eine mit auch aber sehr wenn weil "
           "ich du wir bedürfnis kinder eltern erziehung ").split(),
    "fr": (" le la les une est dans pour avec vous nous cette mais très parce "
           "pourquoi enfants parents besoin éducation c'est qu'il ").split(),
    "en": (" the and is of to you what how why with this that children parents "
           "need education because ").split(),
}


def detect_language(text: str) -> str:
    """Return 'de' | 'en' | 'fr'. Best-effort; defaults to 'en'."""
    t = (text or "").lower()
    if not t.strip():
        return "en"
    try:
        from langdetect import detect
        code = detect(t)
        if code in LANG_NAMES:
            return code
    except Exception:
        pass
    toks = re.findall(r"[a-zà-ÿ']+", t)
    if not toks:
        return "en"
    tokset = set(toks)
    score = {lg: sum(1 for w in tokset if w in set(m)) for lg, m in _MARKERS.items()}
    if any(ch in t for ch in "äöüß"):
        score["de"] += 2
    if any(ch in t for ch in "éàçèêâîô") or "c'est" in t or "qu'" in t:
        score["fr"] += 2
    best = max(score, key=score.get)
    return best if score[best] > 0 else "en"


def resolve_target_lang(target_lang: str, question: str) -> str:
    """'auto' -> detect from the question; otherwise validate de/en/fr."""
    tl = (target_lang or "auto").lower()
    return tl if tl in LANG_NAMES else detect_language(question)


# --------------------------------------------------------------------------- #
# embedders
# --------------------------------------------------------------------------- #
class HashingEmbedder:
    """Dependency-free deterministic embedder (hashed bag-of-words). NOT semantic
    — it exists so the pipeline is testable/offline without downloading a model.
    Swap in SentenceTransformerEmbedder for real semantic search."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", (t or "").lower()):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self.dim
                vecs[i, h] += 1.0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class SentenceTransformerEmbedder:
    """Real multilingual embeddings via sentence-transformers (local). Lazy import
    so importing this module never requires torch."""

    def __init__(self, model: str = DEFAULT_EMBED_MODEL, device: Optional[str] = None,
                 query_prefix: str = "", passage_prefix: str = "", batch_size: int = 64):
        self.model_name = model
        self.device = device
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        prefix = self.query_prefix if is_query else self.passage_prefix
        inputs = [prefix + (t or "") for t in texts]
        embs = self._load().encode(inputs, batch_size=self.batch_size,
                                   show_progress_bar=False,
                                   normalize_embeddings=True)
        return np.asarray(embs, dtype=np.float32)


# --------------------------------------------------------------------------- #
# vector stores
# --------------------------------------------------------------------------- #
class InMemoryStore:
    """Cosine search over an in-memory matrix. Ideal for a single transcript
    (hundreds/thousands of passages) — instant, zero dependencies."""

    def __init__(self):
        self._ids: List[str] = []
        self._mat: Optional[np.ndarray] = None
        self._meta: dict = {}

    def add(self, ids, embeddings, metadatas=None):
        emb = np.asarray(embeddings, dtype=np.float32)
        self._mat = emb if self._mat is None else np.vstack([self._mat, emb])
        self._ids.extend(ids)
        if metadatas:
            for i, m in zip(ids, metadatas):
                self._meta[i] = m

    def search(self, query_emb, k: int = 5) -> List[Tuple[str, float, Optional[dict]]]:
        if self._mat is None or not self._ids:
            return []
        q = np.asarray(query_emb, dtype=np.float32).reshape(-1)
        sims = self._mat @ q            # rows are L2-normalized -> dot == cosine
        k = min(k, len(self._ids))
        top = np.argsort(-sims)[:k]
        return [(self._ids[j], float(sims[j]), self._meta.get(self._ids[j])) for j in top]

    def count(self) -> int:
        return len(self._ids)


class ChromaStore:
    """Persistent store using ChromaDB — matches your existing pipeline so an
    index can be reused across sessions/documents."""

    def __init__(self, db_path: str = "./chroma_db",
                 collection: str = "dava_transcript", reset: bool = True):
        import chromadb
        self.client = chromadb.PersistentClient(path=db_path)
        if reset:
            try:
                self.client.delete_collection(collection)
            except Exception:
                pass
        try:
            self.col = self.client.get_collection(collection)
        except Exception:
            self.col = self.client.create_collection(
                name=collection, metadata={"hnsw:space": "cosine"})

    def add(self, ids, embeddings, metadatas=None, documents=None):
        self.col.add(ids=list(ids),
                     embeddings=[e.tolist() if hasattr(e, "tolist") else e for e in embeddings],
                     metadatas=metadatas, documents=documents)

    def search(self, query_emb, k: int = 5) -> List[Tuple[str, float, Optional[dict]]]:
        q = query_emb.tolist() if hasattr(query_emb, "tolist") else list(query_emb)
        res = self.col.query(query_embeddings=[q], n_results=k,
                             include=["metadatas", "distances"])
        ids = res["ids"][0]
        dists = res.get("distances", [[0] * len(ids)])[0]
        metas = res.get("metadatas", [[{}] * len(ids)])[0]
        # cosine space -> distance = 1 - similarity
        return [(i, 1.0 - float(d), m) for i, d, m in zip(ids, dists, metas)]

    def count(self) -> int:
        return self.col.count()


# --------------------------------------------------------------------------- #
# retriever
# --------------------------------------------------------------------------- #
class Retriever:
    """Ties windowing + an embedder + a vector store together."""

    def __init__(self, embedder=None, store=None,
                 window_secs: float = 45.0, overlap_secs: float = 10.0):
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.store = store or InMemoryStore()
        self.window_secs = window_secs
        self.overlap_secs = overlap_secs
        self._passages: dict = {}   # id -> Passage (in-session hydration)

    def index(self, segments) -> int:
        passages = build_windows(segments, self.window_secs, self.overlap_secs)
        if not passages:
            return 0
        embs = self.embedder.encode([p.text for p in passages], is_query=False)
        self.store.add(ids=[p.id for p in passages], embeddings=embs,
                       metadatas=[p.to_metadata() for p in passages])
        self._passages = {p.id: p for p in passages}
        return len(passages)

    def search(self, query: str, k: int = 5) -> List[Passage]:
        q = self.embedder.encode([query], is_query=True)[0]
        out: List[Passage] = []
        for pid, score, meta in self.store.search(q, k=k):
            base = self._passages.get(pid)
            if base is not None:
                p = Passage(**{**base.__dict__})
                p.score = score
            elif meta is not None:
                p = Passage.from_metadata(pid, meta, score)
            else:
                continue
            out.append(p)
        return out


# --------------------------------------------------------------------------- #
# shared prompt shaping (used by ExtractiveAnswerer AND ollama_generator.py)
# --------------------------------------------------------------------------- #
@dataclass
class GenResult:
    answer: str
    cited: List[int] = field(default_factory=list)   # indices into the passage list
    provider: str = ""
    target_lang: str = ""


def build_context_block(passages: List[Passage]) -> str:
    parts = []
    for i, p in enumerate(passages, 1):
        lang = f" ({'/'.join(p.langs)})" if p.langs else ""
        spk = f" speaker {p.speaker}" if p.speaker else ""
        parts.append(f"[{i}] [{p.start_fmt} – {p.end_fmt}]{spk}{lang}\n{p.text}")
    return "\n\n".join(parts)


def build_grounded_prompt(question: str, passages: List[Passage],
                          target_lang: str) -> Tuple[str, str]:
    """Return (system, user) prompt strings for a grounded, single-document QA turn
    in the chosen language. Shared so every generation backend behaves identically."""
    lang_name = LANG_NAMES.get(target_lang, "English")
    system = (
        "You are a careful research assistant answering questions about a single "
        "transcribed audio/video document. You are given excerpts from its "
        "transcript, each with its timestamp. Follow these rules strictly:\n"
        "1. Answer using ONLY the provided excerpts. Do not add outside knowledge.\n"
        f"2. Write your entire answer in {lang_name}, regardless of the language of "
        "the excerpts or the question.\n"
        "3. Cite the timestamps you relied on, written as [MM:SS] or [H:MM:SS], so "
        "the user can jump to that moment in the audio.\n"
        f"4. If the excerpts do not contain the answer, say so plainly in {lang_name} "
        "— do not speculate.\n"
        "5. Be accurate and concise."
    )
    user = (f"Question: {question}\n\n"
            f"Transcript excerpts (use these to answer and cite their timestamps):\n\n"
            f"{build_context_block(passages)}\n\n"
            f"Reminder: write your entire answer in {lang_name}, even though the "
            f"excerpts may be in another language.")
    return system, user


_TS_RE = re.compile(r"\[?(\d{1,2}):(\d{2})(?::(\d{2}))?\]?")


def parse_cited(answer: str, passages: List[Passage]) -> List[int]:
    """Best-effort: which passages did the answer reference? Matches explicit
    [n] markers and any timestamps that fall inside a passage's span."""
    cited = set()
    for m in re.finditer(r"\[(\d{1,2})\]", answer or ""):
        n = int(m.group(1)) - 1
        if 0 <= n < len(passages):
            cited.add(n)
    for m in _TS_RE.finditer(answer or ""):
        h = int(m.group(3) is not None and m.group(1) or 0)
        mm = int(m.group(1)) if m.group(3) is None else int(m.group(2))
        ss = int(m.group(2)) if m.group(3) is None else int(m.group(3))
        secs = h * 3600 + mm * 60 + ss
        for i, p in enumerate(passages):
            if p.start - 1.0 <= secs <= p.end + 1.0:
                cited.add(i)
    return sorted(cited)


# --------------------------------------------------------------------------- #
# no-LLM answer (retrieval only) — always available, fully offline
# --------------------------------------------------------------------------- #
class ExtractiveAnswerer:
    """Returns the top passages themselves as the 'answer' — the honest,
    zero-model baseline. It cannot translate, so the evidence stays in its source
    language; the timestamps are exact."""

    def generate(self, question: str, passages: List[Passage],
                 target_lang: str = "auto", top: int = 3) -> GenResult:
        lang = resolve_target_lang(target_lang, question)
        chosen = passages[:top]
        if not chosen:
            msg = {"de": "Dazu wurde nichts im Transkript gefunden.",
                   "fr": "Rien trouvé à ce sujet dans la transcription.",
                   "en": "Nothing about this was found in the transcript."}[lang]
            return GenResult(answer=msg, cited=[], provider="extractive", target_lang=lang)
        lines = [f"[{p.start_fmt} – {p.end_fmt}]"
                 + (f" {p.speaker}" if p.speaker else "") + f"\n{p.text}"
                 for p in chosen]
        return GenResult(answer="\n\n".join(lines), cited=list(range(len(chosen))),
                         provider="extractive", target_lang=lang)
