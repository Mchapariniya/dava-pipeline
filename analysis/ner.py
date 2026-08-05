#!/usr/bin/env python3
"""
ner.py — Named Entity Recognition over a WhisperX transcript.

Runs per-segment NER and produces two things:

1. A sidecar TSV ``<name>.entities.tsv`` whose columns are exactly what
   ``json_to_eaf.py`` reads (``sent_id, entity, label, start, end``) plus a few
   convenience columns (``score, speaker, seg_start, seg_end``). ``sent_id`` is
   1-based and equals ``segment_index + 1`` so ELAN annotations line up with
   the transcription tier.

2. An ``entities`` list written onto every segment in the JSON, so the
   dashboard (and anything else reading the JSON) can show entities inline.

Backends (auto-selected, override with ``--backend``)
-----------------------------------------------------
* ``transformers`` — a Hugging Face token-classification pipeline. Best quality;
  needs ``transformers``/``torch`` and model download. Default model is
  multilingual so it copes with the German + code-switching content.
* ``spacy``        — spaCy pipeline (e.g. ``xx_ent_wiki_sm`` multilingual or a
  language-specific model). No GPU needed; models install from GitHub.
* ``regex``        — dependency-free fallback (capitalised multi-word spans).
  Low precision, but guarantees the stage always produces output.

The public entry point is :func:`run_ner`.
"""
from __future__ import annotations

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional

try:
    from . import common as C
except ImportError:  # allow running as a script: python analysis/ner.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis import common as C


# --------------------------------------------------------------------------- #
# Label normalisation — collapse the many tag sets onto PER / LOC / ORG / MISC
# --------------------------------------------------------------------------- #
_LABEL_MAP = {
    "PER": "PER", "PERSON": "PER", "PERS": "PER", "PP": "PER", "PPER": "PER",
    "LOC": "LOC", "LOCATION": "LOC", "GPE": "LOC", "FAC": "LOC", "GEO": "LOC",
    "ORG": "ORG", "ORGANIZATION": "ORG", "ORGANISATION": "ORG", "NORP": "ORG",
    "MISC": "MISC", "MISCELLANEOUS": "MISC", "PRODUCT": "MISC", "EVENT": "MISC",
    "WORK_OF_ART": "MISC", "LAW": "MISC", "LANGUAGE": "MISC",
}


def normalize_label(label: str) -> str:
    if not label:
        return "MISC"
    label = label.upper().lstrip("BI-").strip()  # drop BIO prefixes
    return _LABEL_MAP.get(label, label if label.isalpha() else "MISC")


# --------------------------------------------------------------------------- #
# Backend implementations — each returns list[{text,label,start,end,score}]
# for a single piece of text (char offsets relative to that text).
# --------------------------------------------------------------------------- #

class _TransformersNER:
    """HF token-classification pipeline (multilingual by default)."""

    DEFAULT_MODEL = "Babelscape/wikineural-multilingual-ner"

    def __init__(self, model: Optional[str] = None, gpu_index: Optional[int] = None):
        from transformers import pipeline  # noqa: E402  (lazy)
        self.model_name = model or self.DEFAULT_MODEL
        device = C.transformers_device_index(gpu_index)
        self.pipe = pipeline(
            "token-classification",
            model=self.model_name,
            aggregation_strategy="simple",
            device=device,
        )

    def __call__(self, text: str) -> List[dict]:
        out = []
        for ent in self.pipe(text):
            out.append({
                "text": ent.get("word", "").strip(),
                "label": normalize_label(ent.get("entity_group", ent.get("entity", ""))),
                "start": int(ent.get("start", 0)),
                "end": int(ent.get("end", 0)),
                "score": round(float(ent.get("score", 0.0)), 4),
            })
        return out


class _SpacyNER:
    """spaCy pipeline. ``model`` may be a name (loaded) or a blank lang code."""

    def __init__(self, model: Optional[str] = None, lang: str = "xx"):
        import spacy  # noqa: E402  (lazy)
        candidates = [m for m in (model, f"{C.spacy_lang_code(lang)}_core_news_sm",
                                  "xx_ent_wiki_sm") if m]
        self.nlp = None
        errors = []
        for name in candidates:
            try:
                self.nlp = spacy.load(name)
                self.model_name = name
                break
            except Exception as exc:  # model not installed -> try next
                errors.append(f"{name}: {exc}")
        if self.nlp is None:
            raise RuntimeError("No spaCy model could be loaded. Tried:\n  "
                               + "\n  ".join(errors)
                               + "\nInstall one, e.g.:  python -m spacy download xx_ent_wiki_sm")

    def __call__(self, text: str) -> List[dict]:
        doc = self.nlp(text)
        return [{
            "text": ent.text.strip(),
            "label": normalize_label(ent.label_),
            "start": ent.start_char,
            "end": ent.end_char,
            "score": None,  # spaCy small models don't expose per-entity scores
        } for ent in doc.ents]


class _RegexNER:
    """Dependency-free fallback: sequences of Capitalised Words as entities.

    Purposely conservative and label-agnostic (everything -> MISC). Only used
    when neither transformers nor spaCy is available, so the pipeline still
    yields a usable (if coarse) entity file.
    """

    import re as _re
    _PAT = _re.compile(r"\b([A-ZÄÖÜ][\wäöüß]+(?:\s+[A-ZÄÖÜ][\wäöüß]+){0,3})")
    _COMMON = {"Ich", "Der", "Die", "Das", "Und", "Aber", "Wenn", "Was", "Wer",
               "Wie", "Wir", "Sie", "Es", "Er", "The", "And", "But", "If"}

    def __init__(self, *_, **__):
        self.model_name = "regex-capitalised"

    def __call__(self, text: str) -> List[dict]:
        out = []
        for m in self._PAT.finditer(text):
            span = m.group(1).strip()
            # skip a leading sentence-initial capital that is just a stop word
            first = span.split()[0]
            if first in self._COMMON and len(span.split()) == 1:
                continue
            out.append({"text": span, "label": "MISC",
                        "start": m.start(1), "end": m.end(1), "score": None})
        return out


def _build_backend(backend: str, model: Optional[str], lang: str,
                   gpu_index: Optional[int]):
    """Instantiate the requested backend, auto-selecting when ``backend='auto'``."""
    order = ([backend] if backend != "auto"
             else ["transformers", "spacy", "regex"])
    last_err = None
    for name in order:
        try:
            if name == "transformers":
                if not C.has_module("transformers"):
                    raise RuntimeError("transformers not installed")
                impl = _TransformersNER(model=model, gpu_index=gpu_index)
            elif name == "spacy":
                if not C.has_module("spacy"):
                    raise RuntimeError("spacy not installed")
                impl = _SpacyNER(model=model, lang=lang)
            elif name == "regex":
                impl = _RegexNER()
            else:
                raise ValueError(f"Unknown backend: {name}")
            print(f"  [ner] backend = {name}  (model = {getattr(impl, 'model_name', '?')})")
            return impl
        except Exception as exc:
            last_err = exc
            if backend != "auto":
                raise
            print(f"  [ner] backend '{name}' unavailable: {exc}")
    raise RuntimeError(f"No NER backend available. Last error: {last_err}")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def run_ner(json_path: str, output_json: Optional[str] = None,
            tsv_path: Optional[str] = None, backend: str = "auto",
            model: Optional[str] = None, gpu_index: Optional[int] = None,
            min_score: float = 0.0) -> Dict:
    """Run NER over ``json_path``.

    Writes the enriched JSON (segments gain an ``entities`` list) and a TSV
    sidecar. Returns a small summary dict.
    """
    data = C.load_transcript(json_path)
    segments = C.get_segments(data)
    if not segments:
        print("  [ner] No segments found; nothing to do.")
        return {"entities": 0}

    lang = C.dominant_language(segments)
    ner = _build_backend(backend, model, lang, gpu_index)

    base = C.base_name_from_json(json_path)
    out_dir = Path(output_json).parent if output_json else Path(json_path).parent
    tsv_path = tsv_path or str(out_dir / f"{base}.entities.tsv")

    rows: List[dict] = []
    total = 0
    for idx, seg in C.iter_sentences(segments):
        text = (seg.get("text") or "").strip()
        try:
            ents = ner(text)
        except Exception as exc:
            print(f"    [ner] segment {idx} failed: {exc}")
            ents = []

        # filter by score when the backend supplies one
        ents = [e for e in ents
                if (e.get("score") is None or e["score"] >= min_score)
                and e.get("text")]

        seg["entities"] = ents  # attach to JSON for the dashboard / downstream
        total += len(ents)

        for e in ents:
            rows.append({
                "sent_id": idx + 1,          # 1-based, matches json_to_eaf
                "entity": e["text"],
                "label": e["label"],
                "start": e["start"],          # char offset within the segment
                "end": e["end"],
                "score": "" if e["score"] is None else e["score"],
                "speaker": seg.get("speaker") or "",
                "seg_start": seg.get("start", 0.0),
                "seg_end": seg.get("end", 0.0),
            })

    # write TSV sidecar (consumed by json_to_eaf.py)
    fieldnames = ["sent_id", "entity", "label", "start", "end",
                  "score", "speaker", "seg_start", "seg_end"]
    Path(tsv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(tsv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    # write enriched JSON
    out_json = output_json or json_path
    C.save_transcript(data, out_json)

    # quick label tally for the log
    from collections import Counter
    tally = Counter(r["label"] for r in rows)
    print(f"  [ner] {total} entities across {len(segments)} segments "
          f"-> {os.path.basename(tsv_path)}")
    if tally:
        print("        by type: " + ", ".join(f"{k}={v}" for k, v in tally.most_common()))

    return {"entities": total, "tsv": tsv_path, "json": out_json,
            "by_type": dict(tally), "backend": getattr(ner, "model_name", backend)}


def main():
    ap = argparse.ArgumentParser(description="Named Entity Recognition for WhisperX JSON")
    ap.add_argument("json_file", help="Path to *_whisperx.json")
    ap.add_argument("--output_json", default=None, help="Where to write enriched JSON (default: overwrite)")
    ap.add_argument("--tsv", default=None, help="Where to write entities TSV (default: <name>.entities.tsv)")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "transformers", "spacy", "regex"],
                    help="NER backend (default: auto -> transformers, spacy, regex)")
    ap.add_argument("--model", default=None, help="Model name/path for the chosen backend")
    ap.add_argument("--gpu_index", type=int, default=None, help="GPU index (transformers backend)")
    ap.add_argument("--min_score", type=float, default=0.0, help="Drop entities below this score")
    args = ap.parse_args()

    if not os.path.exists(args.json_file):
        print(f"ERROR: file not found: {args.json_file}")
        sys.exit(1)

    run_ner(args.json_file, output_json=args.output_json, tsv_path=args.tsv,
            backend=args.backend, model=args.model, gpu_index=args.gpu_index,
            min_score=args.min_score)


if __name__ == "__main__":
    main()
