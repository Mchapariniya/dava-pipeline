#!/usr/bin/env python3
"""
elan_viz.py
===========
Parse an ELAN .eaf file (or a WhisperX JSON) and render an interactive,
ELAN-style tier timeline as a single self-contained HTML string.

The renderer intentionally mirrors ELAN's own visual grammar so a linguistic
audience feels at home:
  - one horizontal lane per tier, tier names in a fixed left gutter
  - annotation blocks positioned by time, with the annotation text readable
    inside each block (full text + timecodes on hover)
  - a zoomable time ruler (Fit / zoom in / zoom out)
  - click an annotation (or the ruler) to seek the audio; a playhead tracks
    playback and the view auto-scrolls to follow it

No third-party dependencies: parsing uses the stdlib, and the HTML is plain
CSS/JS so it embeds cleanly inside a Streamlit `components.html` iframe, a Dash
`html.Iframe`, a Flask template, or a standalone file.

Public API
----------
  parse_eaf(path)            -> dict   # {media, tiers, duration_ms}
  from_whisperx_json(path)   -> dict   # same shape, tiers grouped by speaker
  render_elan_html(data, audio_src=None, title=..., height_px=560) -> str

CLI
---
  # standalone HTML preview from an .eaf, embedding a local audio file:
  python elan_viz.py session.eaf --audio session.mp3 --out session_elan.html
  # or straight from a WhisperX transcript (auto-detected by .json extension):
  python elan_viz.py transcript.json --audio session.wav --out preview.html
"""

import argparse
import base64
import json
import mimetypes
import os
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict


# --------------------------------------------------------------------------- #
# ELAN (.eaf) parsing  -- namespace-agnostic, tolerant of exporter variations
# --------------------------------------------------------------------------- #
def _local(tag: str) -> str:
    """Strip any XML namespace, returning the local element name."""
    return tag.rsplit("}", 1)[-1]


def _iter(elem, name):
    return (c for c in elem.iter() if _local(c.tag) == name)


def _direct(elem, name):
    return [c for c in list(elem) if _local(c.tag) == name]


def parse_eaf(path: str) -> dict:
    """Parse an ELAN .eaf file into {media, tiers, duration_ms}.

    Fully supports time-aligned (ALIGNABLE) annotations. Reference (REF)
    annotations inherit timing from the annotation they point to, which covers
    the common symbolic-association case; deeply nested subdivisions without
    their own time slots are best-effort.
    """
    root = ET.parse(path).getroot()

    media = None
    for md in _iter(root, "MEDIA_DESCRIPTOR"):
        media = {
            "url": md.get("MEDIA_URL"),
            "relative": md.get("RELATIVE_MEDIA_URL"),
            "mime": md.get("MIME_TYPE"),
        }
        break

    # time-slot id -> milliseconds (value may be missing for unaligned slots)
    ts = {}
    for slot in _iter(root, "TIME_SLOT"):
        v = slot.get("TIME_VALUE")
        ts[slot.get("TIME_SLOT_ID")] = int(v) if v is not None else None

    tiers = []
    ann_index = {}  # annotation_id -> record (for resolving REF annotations)

    for tier in _iter(root, "TIER"):
        anns = []
        for wrap in _direct(tier, "ANNOTATION"):
            for a in list(wrap):
                kind = _local(a.tag)
                aid = a.get("ANNOTATION_ID")
                value = ""
                for v in a:
                    if _local(v.tag) == "ANNOTATION_VALUE":
                        value = (v.text or "").strip()
                        break
                if kind == "ALIGNABLE_ANNOTATION":
                    rec = {
                        "id": aid, "value": value, "ref": None,
                        "start": ts.get(a.get("TIME_SLOT_REF1")),
                        "end": ts.get(a.get("TIME_SLOT_REF2")),
                    }
                elif kind == "REF_ANNOTATION":
                    rec = {"id": aid, "value": value,
                           "ref": a.get("ANNOTATION_REF"),
                           "start": None, "end": None}
                else:
                    continue
                anns.append(rec)
                if aid:
                    ann_index[aid] = rec
        tiers.append({
            "tier_id": tier.get("TIER_ID"),
            "participant": tier.get("PARTICIPANT") or tier.get("ANNOTATOR"),
            "type": tier.get("LINGUISTIC_TYPE_REF"),
            "parent": tier.get("PARENT_REF"),
            "annotations": anns,
        })

    # resolve REF annotations to their referenced annotation's timing
    def _resolve(rec, depth=0):
        if rec["start"] is not None:
            return rec["start"], rec["end"]
        ref = rec.get("ref")
        if ref and ref in ann_index and depth < 25:
            return _resolve(ann_index[ref], depth + 1)
        return None, None

    duration = 0
    for t in tiers:
        for rec in t["annotations"]:
            if rec["start"] is None:
                rec["start"], rec["end"] = _resolve(rec)
        t["annotations"] = [a for a in t["annotations"]
                            if a["start"] is not None and a["end"] is not None]
        t["annotations"].sort(key=lambda a: a["start"])
        for a in t["annotations"]:
            duration = max(duration, a["end"])

    return {"media": media, "tiers": tiers, "duration_ms": duration}


def from_whisperx_json(path: str) -> dict:
    """Build the same {media, tiers, duration_ms} shape directly from a WhisperX
    transcript, grouping segments into one tier per speaker. Handy for previewing
    before/without an actual .eaf export."""
    raw = json.load(open(path, encoding="utf-8"))
    segs = raw.get("segments", raw if isinstance(raw, list) else [])
    tiers = OrderedDict()
    duration = 0
    for s in segs:
        spk = s.get("speaker") or "UNASSIGNED"
        start = int(round(float(s.get("start", 0)) * 1000))
        end = int(round(float(s.get("end", 0)) * 1000))
        if end <= start:
            end = start + 200  # keep zero/negative-length blocks visible
        tiers.setdefault(spk, []).append(
            {"id": None, "value": (s.get("text") or "").strip(),
             "ref": None, "start": start, "end": end})
        duration = max(duration, end)
    tier_list = [{"tier_id": k, "participant": k, "type": "transcription",
                  "parent": None, "annotations": v} for k, v in sorted(tiers.items())]
    return {"media": None, "tiers": tier_list, "duration_ms": duration}


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #
_TEMPLATE = r"""
<div id="elan-app" class="elan-root" style="height:__HEIGHT__px;">
  <div class="elan-toolbar">
    <span class="elan-title">__TITLE__</span>
    <span class="elan-spacer"></span>
    <div class="elan-zoom">
      <button class="elan-btn" data-act="out" title="Zoom out">&#8722;</button>
      <button class="elan-btn" data-act="in"  title="Zoom in">&#43;</button>
      <button class="elan-btn" data-act="fit" title="Fit to width">Fit</button>
    </div>
    <audio id="elan-audio" class="elan-audio" controls preload="metadata"></audio>
  </div>
  <div class="elan-body">
    <div class="elan-gutter" id="elan-gutter"></div>
    <div class="elan-scroll" id="elan-scroll">
      <div class="elan-canvas" id="elan-canvas">
        <div class="elan-ruler" id="elan-ruler"></div>
        <div class="elan-lanes" id="elan-lanes"></div>
        <div class="elan-playhead" id="elan-playhead"></div>
      </div>
    </div>
  </div>
</div>
<style>
  .elan-root{--bg:#0e1117;--panel:#161b26;--panel2:#1b2130;--line:#2a3142;
    --text:#e6e9f0;--muted:#8b93a7;--accent:#ff4b4b;--ruler-h:26px;--row-h:38px;
    display:flex;flex-direction:column;background:var(--bg);color:var(--text);
    border:1px solid var(--line);border-radius:10px;overflow:hidden;
    font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
  .elan-root *{box-sizing:border-box;}
  .elan-toolbar{display:flex;align-items:center;gap:12px;padding:8px 12px;
    background:var(--panel);border-bottom:1px solid var(--line);}
  .elan-title{font-weight:650;font-size:14px;letter-spacing:.2px;}
  .elan-spacer{flex:1;}
  .elan-zoom{display:flex;gap:6px;}
  .elan-btn{background:var(--panel2);color:var(--text);border:1px solid var(--line);
    border-radius:6px;min-width:30px;height:28px;padding:0 9px;cursor:pointer;
    font-size:14px;line-height:1;}
  .elan-btn:hover{border-color:var(--accent);}
  .elan-btn:focus-visible{outline:2px solid var(--accent);outline-offset:1px;}
  .elan-audio{height:30px;max-width:340px;}
  .elan-body{flex:1;display:flex;min-height:0;}
  .elan-gutter{width:172px;flex:0 0 172px;background:var(--panel);
    border-right:1px solid var(--line);overflow:hidden;}
  .elan-gutter-head{height:var(--ruler-h);display:flex;align-items:center;
    padding:0 10px;font-size:11px;color:var(--muted);text-transform:uppercase;
    letter-spacing:.7px;border-bottom:1px solid var(--line);}
  .elan-gutter-row{height:var(--row-h);display:flex;flex-direction:column;
    justify-content:center;padding:0 10px 0 9px;border-bottom:1px solid var(--line);
    border-left:3px solid transparent;}
  .elan-tier-name{font-size:12.5px;font-weight:600;white-space:nowrap;
    overflow:hidden;text-overflow:ellipsis;}
  .elan-tier-meta{font-size:10.5px;color:var(--muted);white-space:nowrap;
    overflow:hidden;text-overflow:ellipsis;}
  .elan-scroll{flex:1;overflow-x:auto;overflow-y:hidden;position:relative;}
  .elan-canvas{position:relative;min-width:100%;}
  .elan-ruler{position:relative;height:var(--ruler-h);
    background:var(--panel2);border-bottom:1px solid var(--line);cursor:pointer;}
  .elan-tick{position:absolute;top:0;height:100%;border-left:1px solid var(--line);}
  .elan-tick>span{position:absolute;left:4px;top:5px;font-size:10.5px;
    color:var(--muted);white-space:nowrap;font-variant-numeric:tabular-nums;}
  .elan-lane{position:relative;height:var(--row-h);border-bottom:1px solid var(--line);}
  .elan-lane.alt{background:rgba(255,255,255,.018);}
  .elan-block{position:absolute;top:5px;height:calc(var(--row-h) - 10px);
    border-radius:4px;padding:0 6px;display:flex;align-items:center;
    overflow:hidden;cursor:pointer;border:1px solid rgba(0,0,0,.35);
    box-shadow:0 1px 2px rgba(0,0,0,.35);transition:filter .1s;}
  .elan-block:hover{filter:brightness(1.15);}
  .elan-block-label{font-size:11.5px;color:#0c0f16;font-weight:600;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
    text-shadow:0 1px 0 rgba(255,255,255,.15);}
  .elan-playhead{position:absolute;top:var(--ruler-h);width:2px;left:0;
    background:var(--accent);pointer-events:none;
    box-shadow:0 0 6px rgba(255,75,75,.8);}
  .elan-playhead:before{content:"";position:absolute;top:-6px;left:-4px;
    border:5px solid transparent;border-top-color:var(--accent);}
  .elan-empty{padding:24px;color:var(--muted);font-size:13px;}
</style>
<script>
(function(){
  const DATA = __DATA__;
  const AUDIO_SRC = __AUDIO__;
  const ROW_H = 38, RULER_H = 26, GUTTER_W = 172;
  const PALETTE = ["#f2c94c","#7cbf3f","#4a90d9","#9b6bd0","#e07a8b",
                   "#3fbfae","#d98f3f","#6b78e0","#43b581","#c0653f"];
  const SENTI = {positive:"#43b581",negative:"#d9534f",neutral:"#7f8aa3",
    joy:"#f2c94c",happiness:"#f2c94c",anger:"#d9534f",angry:"#d9534f",
    sadness:"#4a90d9",sad:"#4a90d9",fear:"#9b6bd0",surprise:"#e07a8b",
    disgust:"#7cbf3f"};

  const $ = id => document.getElementById(id);
  const audio = $("elan-audio"), scroll = $("elan-scroll"),
        canvas = $("elan-canvas"), ruler = $("elan-ruler"),
        lanes = $("elan-lanes"), gutter = $("elan-gutter"),
        playhead = $("elan-playhead");

  const tiers = (DATA.tiers || []).filter(t => t.annotations && t.annotations.length);
  const dur = Math.max((DATA.duration_ms || 0) / 1000, 0.001);
  let pxPerSec = 40;

  if (AUDIO_SRC) { audio.src = AUDIO_SRC; } else { audio.style.display = "none"; }

  function fmt(t){
    if (t < 0) t = 0;
    const ms = Math.floor((t % 1) * 1000);
    const s = Math.floor(t) % 60, m = Math.floor(t / 60) % 60, h = Math.floor(t / 3600);
    const p = (n, w=2) => String(n).padStart(w, "0");
    const base = (h > 0 ? h + ":" + p(m) : m) + ":" + p(s);
    return base + "." + p(ms, 3);
  }
  function hexRgba(hex, a){
    const n = parseInt(hex.slice(1), 16);
    return "rgba(" + (n>>16 & 255) + "," + (n>>8 & 255) + "," + (n & 255) + "," + a + ")";
  }
  function blockColor(tierColor, value){
    const s = SENTI[(value || "").trim().toLowerCase()];
    return s || tierColor;
  }

  if (!tiers.length){
    lanes.innerHTML = '<div class="elan-empty">No time-aligned annotations found in this file.</div>';
    return;
  }

  // --- static structure (gutter + lanes + blocks) --------------------------
  function buildStatic(){
    gutter.innerHTML = "";
    const head = document.createElement("div");
    head.className = "elan-gutter-head"; head.textContent = "tiers";
    gutter.appendChild(head);

    lanes.innerHTML = "";
    tiers.forEach((t, i) => {
      const color = PALETTE[i % PALETTE.length];

      const g = document.createElement("div");
      g.className = "elan-gutter-row";
      g.style.borderLeftColor = color;
      const nm = document.createElement("div");
      nm.className = "elan-tier-name"; nm.textContent = t.tier_id || "(tier)";
      const mt = document.createElement("div");
      mt.className = "elan-tier-meta";
      mt.textContent = (t.participant && t.participant !== t.tier_id ? t.participant + " · " : "")
                       + t.annotations.length + " ann";
      g.appendChild(nm); g.appendChild(mt); gutter.appendChild(g);

      const lane = document.createElement("div");
      lane.className = "elan-lane" + (i % 2 ? " alt" : "");
      t.annotations.forEach(a => {
        const b = document.createElement("div");
        b.className = "elan-block";
        b.dataset.start = a.start; b.dataset.end = a.end;
        b.style.background = a.color || blockColor(color, a.value);
        const lab = document.createElement("span");
        lab.className = "elan-block-label"; lab.textContent = a.value || "";
        b.appendChild(lab);
        b.setAttribute("title", fmt(a.start/1000) + " – " + fmt(a.end/1000)
                                + (a.value ? "\n" + a.value : ""));
        b.addEventListener("click", () => {
          if (AUDIO_SRC){ audio.currentTime = a.start/1000; audio.play().catch(()=>{}); }
        });
        lane.appendChild(b);
      });
      lanes.appendChild(lane);
    });
    playhead.style.height = (tiers.length * ROW_H) + "px";
  }

  // --- time-dependent layout (positions, ruler, playhead) ------------------
  const STEPS = [0.25,0.5,1,2,5,10,15,30,60,120,300,600,900];
  function niceStep(){
    const want = 92 / pxPerSec;              // aim: a label ~every 92px
    for (const s of STEPS) if (s >= want) return s;
    return STEPS[STEPS.length - 1];
  }
  function layout(){
    const w = Math.max(dur * pxPerSec, scroll.clientWidth - 4);
    canvas.style.width = w + "px";
    ruler.style.width = w + "px";
    lanes.style.width = w + "px";
    lanes.querySelectorAll(".elan-block").forEach(b => {
      const s = +b.dataset.start / 1000, e = +b.dataset.end / 1000;
      b.style.left = (s * pxPerSec) + "px";
      b.style.width = Math.max((e - s) * pxPerSec, 3) + "px";
    });
    ruler.innerHTML = "";
    const step = niceStep();
    for (let t = 0; t <= dur + step; t += step){
      const tick = document.createElement("div");
      tick.className = "elan-tick"; tick.style.left = (t * pxPerSec) + "px";
      const s = document.createElement("span"); s.textContent = fmt(t);
      tick.appendChild(s); ruler.appendChild(tick);
    }
    updatePlayhead();
  }
  function updatePlayhead(){
    const t = (audio && !isNaN(audio.currentTime)) ? audio.currentTime : 0;
    playhead.style.left = (t * pxPerSec) + "px";
  }
  function follow(){
    const x = (audio.currentTime || 0) * pxPerSec;
    const view = scroll.clientWidth;
    if (x < scroll.scrollLeft + 40 || x > scroll.scrollLeft + view - 40)
      scroll.scrollLeft = x - view * 0.4;
  }
  function fit(){
    pxPerSec = Math.max((scroll.clientWidth - 8) / dur, 2);
    layout();
  }

  // --- events --------------------------------------------------------------
  document.querySelector(".elan-zoom").addEventListener("click", e => {
    const act = e.target.dataset.act;
    if (act === "in")  pxPerSec = Math.min(pxPerSec * 1.6, 600);
    else if (act === "out") pxPerSec = Math.max(pxPerSec / 1.6, 2);
    else if (act === "fit") return fit();
    else return;
    layout();
  });
  ruler.addEventListener("click", e => {
    if (!AUDIO_SRC) return;
    const r = ruler.getBoundingClientRect();
    audio.currentTime = Math.max(0, (e.clientX - r.left) / pxPerSec);
  });
  if (AUDIO_SRC){
    audio.addEventListener("timeupdate", () => { updatePlayhead(); follow(); });
    audio.addEventListener("seeked", updatePlayhead);
  }
  window.addEventListener("resize", updatePlayhead);

  buildStatic();
  requestAnimationFrame(fit);   // fit once we know the real viewport width
})();
</script>
"""


def render_elan_html(data: dict, audio_src: str = None,
                     title: str = "ELAN annotation timeline",
                     height_px: int = 560) -> str:
    """Render the parsed ELAN/transcript data to a self-contained HTML string.

    audio_src may be a URL, a relative path served by your app, or a
    `data:` URI (see embed_audio_datauri). If None, the player is hidden and the
    timeline is view-only.
    """
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    audio_js = json.dumps(audio_src) if audio_src else "null"
    return (_TEMPLATE
            .replace("__DATA__", payload)
            .replace("__AUDIO__", audio_js)
            .replace("__TITLE__", _esc(title))
            .replace("__HEIGHT__", str(int(height_px))))


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def embed_audio_datauri(path: str) -> str:
    """Read a local audio file and return a base64 `data:` URI so playback works
    fully offline inside a sandboxed iframe. Convenient for a demo; note that a
    ~30 MB MP3 becomes ~40 MB of HTML, so for long files prefer serving a URL."""
    mime = mimetypes.guess_type(path)[0] or "audio/mpeg"
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# --------------------------------------------------------------------------- #
# CLI -> standalone HTML file
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Render an ELAN .eaf (or WhisperX JSON) "
                                             "as an interactive HTML timeline.")
    ap.add_argument("input", help="Path to a .eaf file or a WhisperX .json transcript.")
    ap.add_argument("--audio", default=None, help="Audio/video file or URL to sync with.")
    ap.add_argument("--embed-audio", action="store_true",
                    help="Base64-embed --audio into the HTML (offline demo; large files).")
    ap.add_argument("--out", default=None, help="Output .html (default: <input>_elan.html)")
    ap.add_argument("--title", default="ELAN annotation timeline")
    ap.add_argument("--height", type=int, default=620)
    args = ap.parse_args()

    if args.input.lower().endswith(".json"):
        data = from_whisperx_json(args.input)
    else:
        data = parse_eaf(args.input)

    n_ann = sum(len(t["annotations"]) for t in data["tiers"])
    print(f"[elan] tiers={len(data['tiers'])}  annotations={n_ann}  "
          f"duration={data['duration_ms']/1000:.1f}s")

    audio_src = None
    if args.audio:
        audio_src = embed_audio_datauri(args.audio) if args.embed_audio else args.audio

    html = render_elan_html(data, audio_src=audio_src,
                            title=args.title, height_px=args.height)
    out = args.out or (os.path.splitext(args.input)[0] + "_elan.html")
    full = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>" + _esc(args.title) + "</title>"
            "<style>body{margin:0;background:#0e1117;padding:16px;}</style></head>"
            "<body>" + html + "</body></html>")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(full)
    print(f"[elan] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
