#!/usr/bin/env python3
"""
run_analysis.py — one-shot post-transcription analysis for a WhisperX JSON.

Runs the enrichment stages in order and (optionally) renders the diagrams::

    NER  ->  sentiment  ->  emotion  ->  topic modelling  ->  visualisations

All stages enrich the *same* JSON in place (a copy is made first if
``--out_json`` points elsewhere), so at the end you have one JSON carrying
entities, sentiment, emotion and topic labels — plus the sidecar files the ELAN
exporter reads (``<name>.entities.tsv``, ``<name>_topics.csv``) and a ``figures/``
folder of PNGs.

Examples
--------
    # everything, auto-selecting the best available backend for each stage
    python -m analysis.run_analysis output/json/episode_whisperx.json

    # force offline backends (no model downloads), custom figures dir
    python -m analysis.run_analysis ep_whisperx.json \\
        --ner-backend spacy --sentiment-backend lexicon \\
        --emotion-backend lexicon --topic-method lda --figures_dir out/figs

    # skip a stage
    python -m analysis.run_analysis ep_whisperx.json --skip-emotion

    # batch a whole directory of *_whisperx.json files
    python -m analysis.run_analysis --batch_dir output/json
"""
from __future__ import annotations

import os
import sys
import time
import json
import shutil
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional

try:
    from . import common as C
    from . import ner, sentiment, emotion, topic_modeling, visualize
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis import common as C
    from analysis import ner, sentiment, emotion, topic_modeling, visualize


def analyze_file(json_path: str, out_json: Optional[str] = None,
                 figures_dir: Optional[str] = None, opts: Optional[dict] = None) -> Dict:
    """Run the enabled stages on one JSON file. Returns a summary dict."""
    opts = opts or {}
    if not os.path.exists(json_path):
        print(f"ERROR: file not found: {json_path}")
        return {"error": "not found"}

    # Work on a single evolving file so stages compose.
    work = out_json or json_path
    if out_json and os.path.abspath(out_json) != os.path.abspath(json_path):
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(json_path, out_json)

    print("=" * 68)
    print(f"Analysing: {os.path.basename(json_path)}")
    if work != json_path:
        print(f"Output   : {work}")
    print("=" * 68)

    summary: Dict[str, dict] = {}
    t0 = time.time()

    if not opts.get("skip_ner"):
        try:
            summary["ner"] = ner.run_ner(
                work, output_json=work, backend=opts.get("ner_backend", "auto"),
                model=opts.get("ner_model"), gpu_index=opts.get("gpu_index"),
                min_score=opts.get("ner_min_score", 0.0))
        except Exception as exc:
            print(f"  [ner] FAILED: {exc}"); traceback.print_exc(); summary["ner"] = {"error": str(exc)}

    if not opts.get("skip_sentiment"):
        try:
            summary["sentiment"] = sentiment.run_sentiment(
                work, output_json=work, backend=opts.get("sentiment_backend", "auto"),
                model=opts.get("sentiment_model"), gpu_index=opts.get("gpu_index"))
        except Exception as exc:
            print(f"  [sentiment] FAILED: {exc}"); traceback.print_exc(); summary["sentiment"] = {"error": str(exc)}

    if not opts.get("skip_emotion"):
        try:
            summary["emotion"] = emotion.run_emotion(
                work, output_json=work, backend=opts.get("emotion_backend", "auto"),
                model=opts.get("emotion_model"), gpu_index=opts.get("gpu_index"))
        except Exception as exc:
            print(f"  [emotion] FAILED: {exc}"); traceback.print_exc(); summary["emotion"] = {"error": str(exc)}

    if not opts.get("skip_topics"):
        try:
            summary["topics"] = topic_modeling.run_topic_modeling(
                work, output_json=work, method=opts.get("topic_method", "auto"),
                num_topics=opts.get("num_topics", 8), unit=opts.get("topic_unit", "chunk"),
                chunk_size=opts.get("chunk_size", 5),
                embedding_model=opts.get("topic_embedding_model"))
        except Exception as exc:
            print(f"  [topics] FAILED: {exc}"); traceback.print_exc(); summary["topics"] = {"error": str(exc)}

    if not opts.get("skip_viz"):
        try:
            figs = figures_dir or os.path.join(os.path.dirname(work) or ".", "figures")
            summary["figures"] = visualize.run_visualizations(
                work, out_dir=figs, top_n=opts.get("top_n", 20))
        except Exception as exc:
            print(f"  [viz] FAILED: {exc}"); traceback.print_exc(); summary["figures"] = {"error": str(exc)}

    elapsed = round(time.time() - t0, 2)
    print("-" * 68)
    print(f"Done in {elapsed}s -> {work}")
    print("-" * 68)

    # drop a small machine-readable manifest next to the JSON
    base = C.base_name_from_json(work)
    manifest = os.path.join(os.path.dirname(work) or ".", f"{base}_analysis_summary.json")
    try:
        with open(manifest, "w", encoding="utf-8") as fh:
            json.dump({"input": json_path, "output": work, "elapsed_s": elapsed,
                       "stages": summary}, fh, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass

    summary["_elapsed_s"] = elapsed
    summary["_output"] = work
    return summary


def _find_jsons(directory: str) -> List[str]:
    import glob
    files = sorted(glob.glob(os.path.join(directory, "*_whisperx.json"))) or \
            sorted(glob.glob(os.path.join(directory, "*.json")))
    return files


def build_opts(args) -> dict:
    return {
        "skip_ner": args.skip_ner, "skip_sentiment": args.skip_sentiment,
        "skip_emotion": args.skip_emotion, "skip_topics": args.skip_topics,
        "skip_viz": args.skip_viz,
        "ner_backend": args.ner_backend, "ner_model": args.ner_model,
        "ner_min_score": args.ner_min_score,
        "sentiment_backend": args.sentiment_backend, "sentiment_model": args.sentiment_model,
        "emotion_backend": args.emotion_backend, "emotion_model": args.emotion_model,
        "topic_method": args.topic_method, "num_topics": args.num_topics,
        "topic_unit": args.topic_unit, "chunk_size": args.chunk_size,
        "topic_embedding_model": args.topic_embedding_model,
        "gpu_index": args.gpu_index, "top_n": args.top_n,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Run NER + sentiment + emotion + topic modelling + diagrams on a WhisperX JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("json_file", nargs="?", help="Path to *_whisperx.json")
    ap.add_argument("--batch_dir", default=None, help="Process every *_whisperx.json in this directory")
    ap.add_argument("--out_json", default=None, help="Write enriched JSON here (default: in place)")
    ap.add_argument("--figures_dir", default=None, help="Directory for PNGs (default: <json_dir>/figures)")

    # stage toggles
    ap.add_argument("--skip-ner", dest="skip_ner", action="store_true")
    ap.add_argument("--skip-sentiment", dest="skip_sentiment", action="store_true")
    ap.add_argument("--skip-emotion", dest="skip_emotion", action="store_true")
    ap.add_argument("--skip-topics", dest="skip_topics", action="store_true")
    ap.add_argument("--skip-viz", dest="skip_viz", action="store_true")

    # per-stage backends / models
    ap.add_argument("--ner-backend", dest="ner_backend", default="auto",
                    choices=["auto", "transformers", "spacy", "regex"])
    ap.add_argument("--ner-model", dest="ner_model", default=None)
    ap.add_argument("--ner-min-score", dest="ner_min_score", type=float, default=0.0)
    ap.add_argument("--sentiment-backend", dest="sentiment_backend", default="auto",
                    choices=["auto", "transformers", "lexicon"])
    ap.add_argument("--sentiment-model", dest="sentiment_model", default=None)
    ap.add_argument("--emotion-backend", dest="emotion_backend", default="auto",
                    choices=["auto", "transformers", "lexicon"])
    ap.add_argument("--emotion-model", dest="emotion_model", default=None)
    ap.add_argument("--topic-method", dest="topic_method", default="auto",
                    choices=["auto", "lda", "nmf", "bertopic"])
    ap.add_argument("--num-topics", dest="num_topics", type=int, default=8)
    ap.add_argument("--topic-unit", dest="topic_unit", default="chunk",
                    choices=["chunk", "segment", "speaker"])
    ap.add_argument("--chunk-size", dest="chunk_size", type=int, default=5)
    ap.add_argument("--topic-embedding-model", dest="topic_embedding_model", default=None)

    ap.add_argument("--gpu-index", dest="gpu_index", type=int, default=None)
    ap.add_argument("--top-n", dest="top_n", type=int, default=20)

    args = ap.parse_args()
    opts = build_opts(args)

    if args.batch_dir:
        files = _find_jsons(args.batch_dir)
        if not files:
            print(f"No *_whisperx.json files found in {args.batch_dir}")
            sys.exit(1)
        print(f"Batch: {len(files)} file(s) in {args.batch_dir}\n")
        for jf in files:
            out = None
            if args.out_json:  # treat as a directory in batch mode
                out = os.path.join(args.out_json, os.path.basename(jf))
            analyze_file(jf, out_json=out, figures_dir=args.figures_dir, opts=opts)
            print()
    elif args.json_file:
        analyze_file(args.json_file, out_json=args.out_json,
                     figures_dir=args.figures_dir, opts=opts)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
