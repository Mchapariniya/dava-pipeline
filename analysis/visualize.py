#!/usr/bin/env python3
"""
visualize.py — diagrams for an (enriched) WhisperX transcript.

Generates PNG charts from the JSON, focused on the two the brief calls out —
**most-frequent words** and **topics** — plus a few more that become possible
once the NER/sentiment/emotion stages have run. Everything uses matplotlib (+
wordcloud) and works fully offline on CPU.

Charts produced by :func:`run_visualizations`
---------------------------------------------
* ``<name>_wordcloud.png``          — word cloud of the whole transcript
* ``<name>_top_words.png``          — top-N frequent words (bar)
* ``<name>_topics.png``             — topic sizes + top words (bar)
* ``<name>_sentiment_timeline.png`` — sentiment score over time (line)
* ``<name>_sentiment_dist.png``     — sentiment distribution (bar)
* ``<name>_emotion_dist.png``       — emotion distribution (bar)
* ``<name>_entities.png``           — top entities & entity-type split
* ``<name>_speaker_time.png``       — speaking time per speaker

Charts whose inputs are missing (e.g. no sentiment fields yet) are skipped.

Public entry point: :func:`run_visualizations`. Individual ``plot_*`` functions
are reusable and return the saved path (or None if skipped).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    from . import common as C
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis import common as C

# ---- shared look & feel ---------------------------------------------------- #
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

# A font that renders German umlauts (bundled with matplotlib).
def _unicode_font_path() -> Optional[str]:
    for name in ("DejaVu Sans", "DejaVuSans"):
        try:
            return fm.findfont(fm.FontProperties(family=name), fallback_to_default=True)
        except Exception:
            continue
    return None

_FONT = _unicode_font_path()

# Colour helpers (colour-blind-friendly-ish, consistent across charts).
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
            "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
_SENTIMENT_COLORS = {"positive": "#55A868", "neutral": "#8C8C8C", "negative": "#C44E52"}
_EMOTION_COLORS = {
    "happy": "#F2C14E", "sad": "#4C72B0", "angry": "#C44E52", "fearful": "#8172B3",
    "surprised": "#DD8452", "disgusted": "#55A868", "neutral": "#B0B0B0",
    "unknown": "#D9D9D9", "other": "#CCCCCC",
}


def _save(fig, path: str) -> str:
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Word frequency charts
# --------------------------------------------------------------------------- #

def plot_wordcloud(segments, out_path, lang=None, max_words=150) -> Optional[str]:
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  [viz] wordcloud not installed; skipping word cloud "
              "(pip install wordcloud)")
        return None
    freqs = C.word_frequencies(segments, lang=lang)
    if not freqs:
        return None
    wc = WordCloud(width=1200, height=600, background_color="white",
                   max_words=max_words, font_path=_FONT, colormap="viridis",
                   prefer_horizontal=0.9)
    wc.generate_from_frequencies(freqs)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most frequent words", fontsize=14, pad=10)
    return _save(fig, out_path)


def plot_top_words(segments, out_path, lang=None, top_n=20) -> Optional[str]:
    freqs = C.word_frequencies(segments, lang=lang, top_n=top_n)
    if not freqs:
        return None
    words, counts = zip(*freqs.most_common(top_n))
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.32)))
    y = range(len(words))
    ax.barh(y, counts, color=_PALETTE[0])
    ax.set_yticks(list(y))
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title(f"Top {top_n} words")
    for i, c in enumerate(counts):
        ax.text(c, i, f" {c}", va="center", fontsize=9)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Topic charts
# --------------------------------------------------------------------------- #

def plot_topics(segments, out_path, topics_json: Optional[str] = None) -> Optional[str]:
    """Bar of topic sizes annotated with each topic's top words."""
    # Prefer the topics.json table; else reconstruct from per-segment labels.
    topics = None
    if topics_json and os.path.exists(topics_json):
        import json
        with open(topics_json, encoding="utf-8") as fh:
            topics = json.load(fh).get("topics")
    if not topics:
        counts = Counter(seg.get("topic_label") for seg in segments
                         if seg.get("topic_label") and seg.get("topic_label") != "misc")
        if not counts:
            return None
        topics = [{"Name": name, "Count": cnt, "Top_Words": name.replace("_", ", ")}
                  for name, cnt in counts.most_common()]

    topics = [t for t in topics if t.get("Count", 0) > 0][:12]
    if not topics:
        return None
    names = [t.get("Top_Words", t.get("Name", "")) for t in topics]
    counts = [t.get("Count", 0) for t in topics]
    labels = [", ".join(str(n).split(", ")[:4]) for n in names]

    fig, ax = plt.subplots(figsize=(10, max(4, len(topics) * 0.5)))
    y = range(len(labels))
    ax.barh(y, counts, color=[_PALETTE[i % len(_PALETTE)] for i in range(len(labels))])
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Segments assigned")
    ax.set_title("Discovered topics (size & keywords)")
    for i, c in enumerate(counts):
        ax.text(c, i, f" {c}", va="center", fontsize=9)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Sentiment / emotion charts
# --------------------------------------------------------------------------- #

def plot_sentiment_timeline(segments, out_path, smooth=9) -> Optional[str]:
    pts = [(float(s.get("start", 0.0)), float(s.get("sentiment_score", 0.0)))
           for s in segments if "sentiment_score" in s]
    if len(pts) < 3:
        return None
    xs, ys = zip(*pts)
    xs_min = [x / 60.0 for x in xs]  # minutes

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(xs_min, ys, color="#BBBBBB", lw=0.8, alpha=0.7, label="per segment")
    # rolling mean
    if len(ys) >= smooth:
        import numpy as np
        kernel = np.ones(smooth) / smooth
        roll = np.convolve(ys, kernel, mode="same")
        ax.plot(xs_min, roll, color=_PALETTE[3], lw=2.0, label=f"rolling mean ({smooth})")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Sentiment  (-1 … +1)")
    ax.set_title("Sentiment over time")
    ax.legend(loc="upper right", fontsize=9)
    return _save(fig, out_path)


def plot_sentiment_dist(segments, out_path) -> Optional[str]:
    counts = Counter(s.get("sentiment") for s in segments if s.get("sentiment"))
    if not counts:
        return None
    order = ["positive", "neutral", "negative"]
    labels = [o for o in order if o in counts] + \
             [k for k in counts if k not in order]
    vals = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color=[_SENTIMENT_COLORS.get(k, "#888") for k in labels])
    ax.set_ylabel("Segments")
    ax.set_title("Sentiment distribution")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    return _save(fig, out_path)


def plot_emotion_dist(segments, out_path, include_neutral=False) -> Optional[str]:
    counts = Counter(s.get("emotion") for s in segments if s.get("emotion"))
    if not include_neutral:
        for k in ("neutral", "unknown", "other"):
            counts.pop(k, None)
    if not counts:
        return None
    labels, vals = zip(*counts.most_common())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals, color=[_EMOTION_COLORS.get(k, "#888") for k in labels])
    ax.set_ylabel("Segments")
    ax.set_title("Emotion distribution" +
                 ("" if include_neutral else " (excluding neutral)"))
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Entity & speaker charts
# --------------------------------------------------------------------------- #

def plot_entities(segments, out_path, top_n=15) -> Optional[str]:
    ent_counts: Counter = Counter()
    type_counts: Counter = Counter()
    for s in segments:
        for e in s.get("entities", []) or []:
            txt = (e.get("text") or "").strip()
            if txt:
                ent_counts[txt] += 1
                type_counts[e.get("label", "MISC")] += 1
    if not ent_counts:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(4, top_n * 0.32)),
                                   gridspec_kw={"width_ratios": [2, 1]})
    # top entities
    ents, cnts = zip(*ent_counts.most_common(top_n))
    y = range(len(ents))
    ax1.barh(y, cnts, color=_PALETTE[2])
    ax1.set_yticks(list(y)); ax1.set_yticklabels(ents)
    ax1.invert_yaxis(); ax1.set_xlabel("Mentions")
    ax1.set_title(f"Top {top_n} named entities")
    # entity types
    tlabels, tvals = zip(*type_counts.most_common())
    ax2.bar(tlabels, tvals, color=[_PALETTE[i] for i in range(len(tlabels))])
    ax2.set_title("Entity types")
    ax2.set_ylabel("Count")
    for i, v in enumerate(tvals):
        ax2.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    return _save(fig, out_path)


def plot_speaker_time(segments, out_path) -> Optional[str]:
    dur: "defaultdict[str, float]" = defaultdict(float)
    for s in segments:
        spk = s.get("speaker") or "UNKNOWN"
        dur[spk] += C.segment_duration(s)
    dur = {k: v for k, v in dur.items() if v > 0}
    if len(dur) < 2:
        return None
    items = sorted(dur.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    mins = [v / 60.0 for _, v in items]
    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.4)))
    y = range(len(labels))
    ax.barh(y, mins, color=[_PALETTE[i % len(_PALETTE)] for i in range(len(labels))])
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    ax.invert_yaxis(); ax.set_xlabel("Speaking time (minutes)")
    ax.set_title("Speaking time per speaker")
    for i, m in enumerate(mins):
        ax.text(m, i, f" {m:.1f}m", va="center", fontsize=9)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

def run_visualizations(json_path: str, out_dir: Optional[str] = None,
                       top_n: int = 20) -> Dict[str, str]:
    data = C.load_transcript(json_path)
    segments = C.get_segments(data)
    if not segments:
        print("  [viz] No segments; nothing to plot.")
        return {}

    base = C.base_name_from_json(json_path)
    out_dir = out_dir or os.path.join(os.path.dirname(json_path), "figures")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    lang = C.dominant_language(segments)
    topics_json = os.path.join(os.path.dirname(json_path), f"{base}_topics.json")

    jobs = {
        "wordcloud":          lambda p: plot_wordcloud(segments, p, lang=lang),
        "top_words":          lambda p: plot_top_words(segments, p, lang=lang, top_n=top_n),
        "topics":             lambda p: plot_topics(segments, p, topics_json=topics_json),
        "sentiment_timeline": lambda p: plot_sentiment_timeline(segments, p),
        "sentiment_dist":     lambda p: plot_sentiment_dist(segments, p),
        "emotion_dist":       lambda p: plot_emotion_dist(segments, p),
        "entities":           lambda p: plot_entities(segments, p),
        "speaker_time":       lambda p: plot_speaker_time(segments, p),
    }

    produced: Dict[str, str] = {}
    for name, fn in jobs.items():
        path = os.path.join(out_dir, f"{base}_{name}.png")
        try:
            result = fn(path)
            if result:
                produced[name] = result
                print(f"  [viz] {name:20s} -> {os.path.basename(result)}")
            else:
                print(f"  [viz] {name:20s} -- skipped (no data)")
        except Exception as exc:
            print(f"  [viz] {name:20s} !! failed: {exc}")
    return produced


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate diagrams from a WhisperX JSON")
    ap.add_argument("json_file", help="Path to *_whisperx.json (ideally already enriched)")
    ap.add_argument("--out_dir", default=None, help="Directory for PNGs (default: <json_dir>/figures)")
    ap.add_argument("--top_n", type=int, default=20, help="How many top words to show")
    args = ap.parse_args()
    if not os.path.exists(args.json_file):
        print(f"ERROR: file not found: {args.json_file}")
        sys.exit(1)
    run_visualizations(args.json_file, out_dir=args.out_dir, top_n=args.top_n)


if __name__ == "__main__":
    main()
