#!/usr/bin/env python3
"""
sentiment.py — segment-level sentiment analysis over a WhisperX transcript.

Writes three fields onto every non-empty segment:

* ``sentiment``        — "positive" | "neutral" | "negative"
* ``sentiment_score``  — signed float in [-1, 1]  (>0 positive, <0 negative)
* ``sentiment_scores`` — per-class probabilities (when the backend gives them)

Backends (auto-selected, override with ``--backend``)
-----------------------------------------------------
* ``transformers`` — multilingual sentiment model (default
  ``cardiffnlp/twitter-xlm-roberta-base-sentiment``). Best quality; needs
  ``transformers``/``torch`` + model download.
* ``lexicon``      — dependency-free bilingual (DE+EN) lexicon scorer with
  negation and intensifier handling. Always available.

Public entry point: :func:`run_sentiment`.
"""
from __future__ import annotations

import os
import sys
import argparse
from collections import Counter
from typing import Dict, List, Optional

try:
    from . import common as C
    from .lexicons import sentiment_lexicon as LEX
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis import common as C
    from analysis.lexicons import sentiment_lexicon as LEX

_LABELS = ["negative", "neutral", "positive"]


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

class _TransformersSentiment:
    DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    # normalise the various label spellings different models use
    _NORM = {
        "positive": "positive", "pos": "positive", "label_2": "positive",
        "negative": "negative", "neg": "negative", "label_0": "negative",
        "neutral": "neutral", "neu": "neutral", "label_1": "neutral",
        # star-rating models (nlptown) -> collapse to 3 classes
        "1 star": "negative", "2 stars": "negative", "3 stars": "neutral",
        "4 stars": "positive", "5 stars": "positive",
    }

    def __init__(self, model: Optional[str] = None, gpu_index: Optional[int] = None):
        from transformers import pipeline
        self.model_name = model or self.DEFAULT_MODEL
        self.pipe = pipeline(
            "text-classification",
            model=self.model_name,
            top_k=None,                      # return all class scores
            device=C.transformers_device_index(gpu_index),
            truncation=True,
        )

    def __call__(self, text: str) -> Dict:
        preds = self.pipe(text)
        if preds and isinstance(preds[0], list):   # top_k=None -> list-of-list
            preds = preds[0]
        scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        for p in preds:
            lab = self._NORM.get(str(p["label"]).lower())
            if lab:
                scores[lab] += float(p["score"])
        label = max(scores, key=scores.get)
        signed = scores["positive"] - scores["negative"]
        return {"sentiment": label,
                "sentiment_score": round(signed, 4),
                "sentiment_scores": {k: round(v, 4) for k, v in scores.items()}}


class _LexiconSentiment:
    """Bilingual lexicon scorer with negation + intensifier handling."""

    def __init__(self, *_, pos_thr: float = 0.15, neg_thr: float = -0.15, **__):
        self.model_name = "lexicon-de-en"
        self.pos_thr = pos_thr
        self.neg_thr = neg_thr

    def __call__(self, text: str) -> Dict:
        toks = C._TOKEN_RE.findall(text.lower())
        raw = 0.0
        polar = 0
        for i, tok in enumerate(toks):
            base = 0.0
            if tok in LEX.POSITIVE:
                base = 1.0
            elif tok in LEX.NEGATIVE:
                base = -1.0
            if base == 0.0:
                continue

            # intensifier immediately before?
            mult = 1.0
            if i > 0 and toks[i - 1] in LEX.INTENSIFIERS:
                mult = LEX.INTENSIFIERS[toks[i - 1]]

            # negation within the previous 2 tokens flips polarity
            window = toks[max(0, i - 2):i]
            if any(w in LEX.NEGATIONS for w in window):
                base = -base

            raw += base * mult
            polar += 1

        if polar == 0:
            return {"sentiment": "neutral", "sentiment_score": 0.0,
                    "sentiment_scores": {"negative": 0.0, "neutral": 1.0, "positive": 0.0}}

        score = max(-1.0, min(1.0, raw / polar))
        if score >= self.pos_thr:
            label = "positive"
        elif score <= self.neg_thr:
            label = "negative"
        else:
            label = "neutral"

        # pseudo class-probabilities from the signed score (for a consistent schema)
        pos = max(0.0, score)
        neg = max(0.0, -score)
        neu = 1.0 - abs(score)
        return {"sentiment": label, "sentiment_score": round(score, 4),
                "sentiment_scores": {"negative": round(neg, 4),
                                     "neutral": round(neu, 4),
                                     "positive": round(pos, 4)}}


def _build_backend(backend: str, model: Optional[str], gpu_index: Optional[int]):
    order = [backend] if backend != "auto" else ["transformers", "lexicon"]
    last_err = None
    for name in order:
        try:
            if name == "transformers":
                if not C.has_module("transformers"):
                    raise RuntimeError("transformers not installed")
                impl = _TransformersSentiment(model=model, gpu_index=gpu_index)
            elif name == "lexicon":
                impl = _LexiconSentiment()
            else:
                raise ValueError(f"Unknown backend: {name}")
            print(f"  [sentiment] backend = {name}  (model = {getattr(impl, 'model_name', '?')})")
            return impl
        except Exception as exc:
            last_err = exc
            if backend != "auto":
                raise
            print(f"  [sentiment] backend '{name}' unavailable: {exc}")
    raise RuntimeError(f"No sentiment backend available. Last error: {last_err}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run_sentiment(json_path: str, output_json: Optional[str] = None,
                  backend: str = "auto", model: Optional[str] = None,
                  gpu_index: Optional[int] = None) -> Dict:
    data = C.load_transcript(json_path)
    segments = C.get_segments(data)
    if not segments:
        print("  [sentiment] No segments found; nothing to do.")
        return {"segments": 0}

    clf = _build_backend(backend, model, gpu_index)

    tally: Counter = Counter()
    scores: List[float] = []
    for idx, seg in C.iter_sentences(segments):
        text = (seg.get("text") or "").strip()
        try:
            res = clf(text)
        except Exception as exc:
            print(f"    [sentiment] segment {idx} failed: {exc}")
            res = {"sentiment": "neutral", "sentiment_score": 0.0,
                   "sentiment_scores": {"negative": 0.0, "neutral": 1.0, "positive": 0.0}}
        seg.update(res)
        tally[res["sentiment"]] += 1
        scores.append(res["sentiment_score"])

    out_json = output_json or json_path
    C.save_transcript(data, out_json)

    mean = round(sum(scores) / len(scores), 4) if scores else 0.0
    print(f"  [sentiment] {sum(tally.values())} segments  "
          f"(pos={tally['positive']}, neu={tally['neutral']}, neg={tally['negative']}, "
          f"mean_score={mean})")
    return {"segments": sum(tally.values()), "distribution": dict(tally),
            "mean_score": mean, "json": out_json,
            "backend": getattr(clf, "model_name", backend)}


def main():
    ap = argparse.ArgumentParser(description="Sentiment analysis for WhisperX JSON")
    ap.add_argument("json_file", help="Path to *_whisperx.json")
    ap.add_argument("--output_json", default=None, help="Where to write enriched JSON (default: overwrite)")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "transformers", "lexicon"],
                    help="Sentiment backend (default: auto -> transformers, lexicon)")
    ap.add_argument("--model", default=None, help="Model name/path (transformers backend)")
    ap.add_argument("--gpu_index", type=int, default=None, help="GPU index (transformers backend)")
    args = ap.parse_args()

    if not os.path.exists(args.json_file):
        print(f"ERROR: file not found: {args.json_file}")
        sys.exit(1)
    run_sentiment(args.json_file, output_json=args.output_json,
                  backend=args.backend, model=args.model, gpu_index=args.gpu_index)


if __name__ == "__main__":
    main()
