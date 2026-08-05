#!/usr/bin/env python3
"""
emotion.py — text-based emotion recognition over a WhisperX transcript.

The original pipeline has an *audio* emotion stage (emotion2vec, needs the WAV).
This module instead classifies emotion from the transcribed *text*, so it runs
directly on the JSON with no audio and no GPU. It writes the same fields the
rest of the pipeline and ``json_to_eaf.py`` already understand:

* ``emotion``        — one of: angry, disgusted, fearful, happy, neutral, sad,
                       surprised  (plus "unknown" for empty segments)
* ``emotion_id``     — matching numeric id (0..8, same map as the audio stage)
* ``emotion_score``  — confidence in [0, 1]  (None when nothing was detected)
* ``emotion_scores`` — per-emotion distribution (when available)

Backends (auto-selected, override with ``--backend``)
-----------------------------------------------------
* ``transformers`` — text emotion classifier. Default
  ``j-hartmann/emotion-english-distilroberta-base`` (7 classes that map cleanly
  onto the pipeline's scheme). For German-dominant corpora you may prefer a
  multilingual/German model via ``--model``; the label map below already
  understands the common label spellings.
* ``lexicon``      — dependency-free bilingual (DE+EN) keyword scorer. Always
  available; neutral by default, fires on emotion words.

Public entry point: :func:`run_emotion`.
"""
from __future__ import annotations

import os
import sys
import argparse
from collections import Counter
from typing import Dict, List, Optional

try:
    from . import common as C
    from .lexicons import emotion_lexicon as LEX
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis import common as C
    from analysis.lexicons import emotion_lexicon as LEX

EMOTION_ID = LEX.EMOTION_ID
ID_EMOTION = LEX.ID_EMOTION


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

class _TransformersEmotion:
    DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

    # map model label spellings -> pipeline emotion names
    _NORM = {
        "anger": "angry", "angry": "angry",
        "disgust": "disgusted", "disgusted": "disgusted",
        "fear": "fearful", "fearful": "fearful",
        "joy": "happy", "happy": "happy", "happiness": "happy",
        "neutral": "neutral",
        "sadness": "sad", "sad": "sad",
        "surprise": "surprised", "surprised": "surprised",
        "love": "happy", "optimism": "happy",  # extra labels some models emit
    }

    def __init__(self, model: Optional[str] = None, gpu_index: Optional[int] = None):
        from transformers import pipeline
        self.model_name = model or self.DEFAULT_MODEL
        self.pipe = pipeline(
            "text-classification",
            model=self.model_name,
            top_k=None,
            device=C.transformers_device_index(gpu_index),
            truncation=True,
        )

    def __call__(self, text: str) -> Dict:
        preds = self.pipe(text)
        if preds and isinstance(preds[0], list):
            preds = preds[0]
        scores: Counter = Counter()
        for p in preds:
            name = self._NORM.get(str(p["label"]).lower())
            if name:
                scores[name] += float(p["score"])
        if not scores:
            return _neutral()
        winner = max(scores, key=scores.get)
        return {"emotion": winner, "emotion_id": EMOTION_ID.get(winner, 4),
                "emotion_score": round(float(scores[winner]), 4),
                "emotion_scores": {k: round(v, 4) for k, v in scores.items()}}


class _LexiconEmotion:
    """Keyword scorer over the bilingual emotion lexicon."""

    def __init__(self, *_, **__):
        self.model_name = "lexicon-de-en"

    def __call__(self, text: str) -> Dict:
        toks = C._TOKEN_RE.findall(text.lower())
        hits: Counter = Counter()
        for i, tok in enumerate(toks):
            # skip if negated within the previous 2 tokens
            window = toks[max(0, i - 2):i]
            negated = any(w in _NEGATIONS for w in window)
            if negated:
                continue
            for emo, words in LEX.EMOTION_WORDS.items():
                if tok in words:
                    hits[emo] += 1
                    break
        if not hits:
            return _neutral()
        total = sum(hits.values())
        winner = hits.most_common(1)[0][0]
        return {"emotion": winner, "emotion_id": EMOTION_ID.get(winner, 4),
                "emotion_score": round(hits[winner] / total, 4),
                "emotion_scores": {k: round(v / total, 4) for k, v in hits.items()}}


# small negation set (reuse the sentiment one if present, else a minimal list)
try:
    from .lexicons.sentiment_lexicon import NEGATIONS as _NEGATIONS
except Exception:  # pragma: no cover
    _NEGATIONS = {"nicht", "kein", "keine", "nie", "not", "no", "never"}


def _neutral() -> Dict:
    return {"emotion": "neutral", "emotion_id": EMOTION_ID["neutral"],
            "emotion_score": None, "emotion_scores": {}}


def _build_backend(backend: str, model: Optional[str], gpu_index: Optional[int]):
    order = [backend] if backend != "auto" else ["transformers", "lexicon"]
    last_err = None
    for name in order:
        try:
            if name == "transformers":
                if not C.has_module("transformers"):
                    raise RuntimeError("transformers not installed")
                impl = _TransformersEmotion(model=model, gpu_index=gpu_index)
            elif name == "lexicon":
                impl = _LexiconEmotion()
            else:
                raise ValueError(f"Unknown backend: {name}")
            print(f"  [emotion] backend = {name}  (model = {getattr(impl, 'model_name', '?')})")
            return impl
        except Exception as exc:
            last_err = exc
            if backend != "auto":
                raise
            print(f"  [emotion] backend '{name}' unavailable: {exc}")
    raise RuntimeError(f"No emotion backend available. Last error: {last_err}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run_emotion(json_path: str, output_json: Optional[str] = None,
                backend: str = "auto", model: Optional[str] = None,
                gpu_index: Optional[int] = None) -> Dict:
    data = C.load_transcript(json_path)
    segments = C.get_segments(data)
    if not segments:
        print("  [emotion] No segments found; nothing to do.")
        return {"segments": 0}

    clf = _build_backend(backend, model, gpu_index)

    tally: Counter = Counter()
    for idx, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        if not text:
            seg.update(_neutral())
            seg["emotion"] = "unknown"
            seg["emotion_id"] = EMOTION_ID["unknown"]
            continue
        try:
            res = clf(text)
        except Exception as exc:
            print(f"    [emotion] segment {idx} failed: {exc}")
            res = _neutral()
        seg.update(res)
        tally[res["emotion"]] += 1

    out_json = output_json or json_path
    C.save_transcript(data, out_json)

    print(f"  [emotion] {sum(tally.values())} segments  "
          + ", ".join(f"{k}={v}" for k, v in tally.most_common()))
    return {"segments": sum(tally.values()), "distribution": dict(tally),
            "json": out_json, "backend": getattr(clf, "model_name", backend)}


def main():
    ap = argparse.ArgumentParser(description="Text-based emotion recognition for WhisperX JSON")
    ap.add_argument("json_file", help="Path to *_whisperx.json")
    ap.add_argument("--output_json", default=None, help="Where to write enriched JSON (default: overwrite)")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "transformers", "lexicon"],
                    help="Emotion backend (default: auto -> transformers, lexicon)")
    ap.add_argument("--model", default=None, help="Model name/path (transformers backend)")
    ap.add_argument("--gpu_index", type=int, default=None, help="GPU index (transformers backend)")
    args = ap.parse_args()

    if not os.path.exists(args.json_file):
        print(f"ERROR: file not found: {args.json_file}")
        sys.exit(1)
    run_emotion(args.json_file, output_json=args.output_json,
                backend=args.backend, model=args.model, gpu_index=args.gpu_index)


if __name__ == "__main__":
    main()
