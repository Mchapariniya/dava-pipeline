#!/usr/bin/env python3
"""
elan_annotations.py
===================
Turn your pipeline's per-segment sentiment / emotion / topic results into extra
tiers on the ELAN timeline (and, optionally, into the .eaf itself so they open in
ELAN proper). Works alongside elan_viz.py.

Expected input
--------------
Per-segment annotations, most naturally the transcript JSON with extra keys on
each segment, e.g.:

    {"segments": [
        {"start": 61.7, "end": 63.0, "text": "...",
         "sentiment": "neutral",              # or {"label": "neutral", "score": .9}
         "emotion":   "sadness",
         "topic":     "Erziehung und Bedürfnisse"},
        ...
    ]}

If your labels live in a separate file or under different keys, that's fine —
pass the keys via `keys={"sentiment": "...", ...}`, or hand `make_label_tier` a
plain list of labels aligned to the segments (one per segment, in order).

Coloring
--------
- sentiment / emotion tiers: each block is coloured by its label
  (positive/negative/neutral; joy, anger, sadness, fear, surprise, …).
- topic tiers: coloured categorically, one stable colour per distinct topic.
Adjacent segments with the same label are merged into one block by default, so
you get readable spans instead of hundreds of identical one-word blocks.

Quick start
-----------
    import elan_viz, elan_annotations
    data = elan_annotations.build_timeline_data(
        eaf_path="output/session.eaf",              # base tiers (or transcript_json=...)
        annotations_json="output/session.json",     # per-segment sentiment/emotion/topic
        show=("sentiment", "emotion", "topic"))
    html = elan_viz.render_elan_html(data, audio_src="output/session.wav")
"""

import json
import re
import xml.etree.ElementTree as ET

import elan_viz


# --------------------------------------------------------------------------- #
# colour maps  (Python side, so the renderer stays untouched)
# --------------------------------------------------------------------------- #
SENTIMENT_COLORS = {
    "positive": "#43b581", "negative": "#d9534f", "neutral": "#7f8aa3",
    "mixed": "#e0a53f", "very positive": "#2f9e5b", "very negative": "#b0392c",
    # emotions share the same tier mechanism
    "joy": "#f2c94c", "happiness": "#f2c94c", "happy": "#f2c94c",
    "anger": "#d9534f", "angry": "#d9534f", "sadness": "#4a90d9", "sad": "#4a90d9",
    "fear": "#9b6bd0", "surprise": "#e07a8b", "disgust": "#7cbf3f",
    "love": "#e06ba0", "trust": "#3fbfae", "anticipation": "#d98f3f",
}
CATEGORICAL = [
    "#f2c94c", "#7cbf3f", "#4a90d9", "#9b6bd0", "#e07a8b", "#3fbfae",
    "#d98f3f", "#6b78e0", "#43b581", "#c0653f", "#5aa0c0", "#b06bd0",
]


def _sentiment_color(value: str):
    return SENTIMENT_COLORS.get(value.strip().lower())


def _categorical_color(value: str) -> str:
    h = 0
    for ch in value:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return CATEGORICAL[h % len(CATEGORICAL)]


def _color_for(value: str, mode: str):
    if mode == "categorical":
        return _categorical_color(value)
    if mode == "sentiment":
        return _sentiment_color(value)   # may be None -> renderer falls back
    return None


# --------------------------------------------------------------------------- #
# label handling
# --------------------------------------------------------------------------- #
def _label_str(x) -> str:
    """Coerce a label (string, number, or dict like {'label':..,'score':..}) to text."""
    if x is None:
        return ""
    if isinstance(x, dict):
        for k in ("label", "value", "sentiment", "emotion", "topic", "name", "tag"):
            if k in x and x[k] not in (None, ""):
                return str(x[k]).strip()
        return ""
    return str(x).strip()


def _resolve_labels(segments, spec):
    """Turn a label spec into a list aligned to `segments`.
    spec may be: a key string (read segment['raw'][key]), a list/tuple (by index),
    a dict keyed by index, or a callable(seg, idx)."""
    if callable(spec):
        return [spec(s, i) for i, s in enumerate(segments)]
    if isinstance(spec, str):
        return [(s.get("raw") or {}).get(spec) for s in segments]
    if isinstance(spec, (list, tuple)):
        return [spec[i] if i < len(spec) else None for i in range(len(segments))]
    if isinstance(spec, dict):
        return [spec.get(i, spec.get(str(i))) for i in range(len(segments))]
    return [None] * len(segments)


def transcript_segments(src):
    """Normalise a WhisperX transcript (path, or already-loaded dict/list) to
    [{start_ms, end_ms, text, raw}]."""
    raw = src if isinstance(src, (dict, list)) else json.load(open(src, encoding="utf-8"))
    segs = raw.get("segments", raw) if isinstance(raw, dict) else raw
    out = []
    for s in segs:
        start = int(round(float(s.get("start", 0)) * 1000))
        end = int(round(float(s.get("end", 0)) * 1000))
        if end <= start:
            end = start + 200
        out.append({"start": start, "end": end,
                    "text": (s.get("text") or "").strip(), "raw": s})
    return out


# --------------------------------------------------------------------------- #
# tier construction
# --------------------------------------------------------------------------- #
def make_label_tier(segments, labels, tier_id, color_mode="solid",
                    participant=None, merge_adjacent=True, skip_values=()):
    """Build a tier dict (renderer-compatible) from segments + a list of labels.

    color_mode: "sentiment" (colour by label map), "categorical" (one colour per
    distinct label), or "solid" (use the tier's default colour)."""
    skip = {str(s).strip().lower() for s in skip_values}
    items = []
    for seg, lab in zip(segments, labels):
        v = _label_str(lab)
        if not v or v.lower() in skip:
            continue
        items.append({"start": seg["start"], "end": seg["end"], "value": v})

    if merge_adjacent:
        merged = []
        for it in items:
            if merged and merged[-1]["value"].lower() == it["value"].lower():
                merged[-1]["end"] = it["end"]
            else:
                merged.append(dict(it))
        items = merged

    for it in items:
        it["id"] = None
        it["ref"] = None
        col = _color_for(it["value"], color_mode)
        if col:
            it["color"] = col

    return {"tier_id": tier_id, "participant": participant, "type": color_mode,
            "parent": None, "color_mode": color_mode, "annotations": items}


def attach_tiers(data, extra_tiers, position="after_words"):
    """Return a copy of `data` with `extra_tiers` inserted.
    position: "after_words" (below Transcription/Words, above speakers),
    "top", or "bottom"."""
    tiers = list(data.get("tiers", []))
    if position == "after_words":
        anchor = -1
        for i, t in enumerate(tiers):
            if (t.get("tier_id") or "").lower() in ("transcription", "transcript", "words"):
                anchor = i
        idx = anchor + 1 if anchor >= 0 else 0
    elif position == "top":
        idx = 0
    else:
        idx = len(tiers)
    tiers[idx:idx] = extra_tiers

    dur = data.get("duration_ms", 0)
    for t in extra_tiers:
        for a in t["annotations"]:
            dur = max(dur, a["end"])
    return {"media": data.get("media"), "tiers": tiers, "duration_ms": dur}


_DEMO = {
    "sentiment": ["positive", "neutral", "negative", "neutral", "positive", "neutral"],
    "emotion": ["joy", "neutral", "sadness", "surprise", "neutral", "anger", "fear"],
    "topic": ["Erziehung und Bedürfnisse", "Eltern und Bedürfnisse",
              "Social Media und Erziehung", "Autoritäre Erziehung und Bedürfnisse"],
}


def _demo_tiers(segments, show, merge_adjacent=True):
    """Deterministic PLACEHOLDER tiers, purely to preview the look. Not real analysis."""
    out = []
    for name in show:
        cyc = _DEMO.get(name)
        if not cyc:
            continue
        labels = [cyc[(i // 2) % len(cyc)] for i in range(len(segments))]
        mode = "categorical" if name == "topic" else "sentiment"
        out.append(make_label_tier(segments, labels,
                                   tier_id=name.capitalize() + " (demo)",
                                   color_mode=mode, merge_adjacent=merge_adjacent))
    return out


def build_timeline_data(eaf_path=None, transcript_json=None, annotations_json=None,
                        show=("sentiment", "emotion", "topic"), keys=None,
                        merge_adjacent=True, demo=False, position="after_words"):
    """One call to assemble the full timeline: base tiers + annotation tiers.

    Base tiers come from `eaf_path` if given, else from a transcript JSON.
    Annotation tiers are built from per-segment labels in `annotations_json`
    (or `transcript_json`). Set demo=True to overlay placeholder tiers instead."""
    keys = keys or {}
    if eaf_path:
        data = elan_viz.parse_eaf(eaf_path)
    elif transcript_json:
        data = elan_viz.from_whisperx_json(transcript_json)
    elif annotations_json:
        data = elan_viz.from_whisperx_json(annotations_json)
    else:
        return {"media": None, "tiers": [], "duration_ms": 0}

    seg_src = annotations_json or transcript_json
    segs = transcript_segments(seg_src) if seg_src else None

    extra = []
    if segs and demo:
        extra = _demo_tiers(segs, show, merge_adjacent)
    elif segs:
        for name in show:
            labels = _resolve_labels(segs, keys.get(name, name))
            if any(_label_str(l) for l in labels):
                mode = "categorical" if name == "topic" else "sentiment"
                extra.append(make_label_tier(segs, labels, tier_id=name.capitalize(),
                                             color_mode=mode,
                                             merge_adjacent=merge_adjacent))

    return attach_tiers(data, extra, position=position) if extra else data


# --------------------------------------------------------------------------- #
# optional: write the extra tiers back into the .eaf (opens in ELAN proper)
# --------------------------------------------------------------------------- #
def _loc(tag):
    return tag.rsplit("}", 1)[-1]


def write_eaf_with_tiers(eaf_in: str, new_tiers, eaf_out: str,
                         ltype_id: str = "dava_annotation") -> str:
    """Add `new_tiers` (as produced by make_label_tier) to an existing .eaf and
    save to eaf_out. Tiers are inserted with time-aligned annotations and a
    top-level linguistic type, in the child order ELAN expects. Validated with
    this repo's parser; give the result a quick check in ELAN itself."""
    tree = ET.parse(eaf_in)
    root = tree.getroot()
    kids = list(root)

    time_order = next((c for c in kids if _loc(c.tag) == "TIME_ORDER"), None)
    if time_order is None:
        time_order = ET.Element("TIME_ORDER")
        root.insert(1, time_order)
        kids = list(root)

    # next free numeric ids for time slots and annotations
    def _max_num(attr):
        m = 0
        for el in root.iter():
            v = el.get(attr) if hasattr(el, "get") else None
            if v:
                g = re.search(r"(\d+)", v)
                if g:
                    m = max(m, int(g.group(1)))
        return m
    ts_n = _max_num("TIME_SLOT_ID")
    a_n = _max_num("ANNOTATION_ID")

    # ensure a top-level (time-alignable, unconstrained) linguistic type exists
    have_type = any(_loc(c.tag) == "LINGUISTIC_TYPE" and c.get("LINGUISTIC_TYPE_ID") == ltype_id
                    for c in kids)
    if not have_type:
        lt = ET.Element("LINGUISTIC_TYPE")
        lt.set("LINGUISTIC_TYPE_ID", ltype_id)
        lt.set("TIME_ALIGNABLE", "true")
        lt.set("GRAPHIC_REFERENCES", "false")
        last_type_idx = max((i for i, c in enumerate(kids)
                             if _loc(c.tag) == "LINGUISTIC_TYPE"), default=None)
        last_tier_idx = max((i for i, c in enumerate(kids)
                            if _loc(c.tag) == "TIER"), default=None)
        pos = (last_type_idx + 1) if last_type_idx is not None else \
              (last_tier_idx + 1) if last_tier_idx is not None else len(kids)
        root.insert(pos, lt)
        kids = list(root)

    # build the new TIER elements (+ their time slots)
    new_tier_els = []
    for tier in new_tiers:
        tel = ET.Element("TIER")
        tel.set("TIER_ID", tier["tier_id"])
        tel.set("LINGUISTIC_TYPE_REF", ltype_id)
        if tier.get("participant"):
            tel.set("PARTICIPANT", tier["participant"])
        for a in tier["annotations"]:
            ts_n += 1
            ts1 = f"ts{ts_n}"
            ts_n += 1
            ts2 = f"ts{ts_n}"
            for tsid, val in ((ts1, a["start"]), (ts2, a["end"])):
                slot = ET.SubElement(time_order, "TIME_SLOT")
                slot.set("TIME_SLOT_ID", tsid)
                slot.set("TIME_VALUE", str(int(val)))
            a_n += 1
            wrap = ET.SubElement(tel, "ANNOTATION")
            al = ET.SubElement(wrap, "ALIGNABLE_ANNOTATION")
            al.set("ANNOTATION_ID", f"a{a_n}")
            al.set("TIME_SLOT_REF1", ts1)
            al.set("TIME_SLOT_REF2", ts2)
            ET.SubElement(al, "ANNOTATION_VALUE").text = a["value"]
        new_tier_els.append(tel)

    # insert TIERs right after the last existing TIER (before LINGUISTIC_TYPEs)
    last_tier_idx = max((i for i, c in enumerate(kids) if _loc(c.tag) == "TIER"),
                        default=None)
    insert_at = (last_tier_idx + 1) if last_tier_idx is not None else len(kids)
    for offset, tel in enumerate(new_tier_els):
        root.insert(insert_at + offset, tel)

    tree.write(eaf_out, encoding="UTF-8", xml_declaration=True)
    return eaf_out
