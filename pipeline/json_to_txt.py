#!/usr/bin/env python3
"""
Convert the pipeline's JSON output (<name>_whisperx.json) into readable
plain-text (.txt) and subtitle (.srt) transcripts.

Usage:
    # one file
    python json_to_txt.py path/to/name_whisperx.json

    # a whole folder of JSONs (e.g. the pipeline's json/ output dir)
    python json_to_txt.py path/to/json_dir

Outputs <name>.txt and <name>.srt next to each JSON (or use --out_dir).
Works whether or not speaker labels are present.
"""
import argparse
import json
import os
import glob


def _fmt_ts(seconds: float) -> str:
    """Seconds -> SRT timestamp HH:MM:SS,mmm."""
    if seconds is None:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def convert_one(json_path: str, out_dir: str | None = None) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", []) or []
    base = os.path.splitext(os.path.basename(json_path))[0]
    if base.endswith("_whisperx"):
        base = base[: -len("_whisperx")]

    target_dir = out_dir or os.path.dirname(json_path) or "."
    os.makedirs(target_dir, exist_ok=True)
    txt_path = os.path.join(target_dir, f"{base}.txt")
    srt_path = os.path.join(target_dir, f"{base}.srt")

    # --- plain text ---
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            spk = seg.get("speaker")
            f.write(f"{spk}: {text}\n" if spk else f"{text}\n")

    # --- srt ---
    with open(srt_path, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            spk = seg.get("speaker")
            line = f"{spk}: {text}" if spk else text
            f.write(f"{idx}\n")
            f.write(f"{_fmt_ts(seg.get('start'))} --> {_fmt_ts(seg.get('end'))}\n")
            f.write(f"{line}\n\n")
            idx += 1

    lang = data.get("language")
    print(f"[ok] {os.path.basename(json_path)} -> {base}.txt / {base}.srt"
          + (f"  (language={lang})" if lang else ""))


def main():
    ap = argparse.ArgumentParser(description="JSON transcript -> .txt/.srt")
    ap.add_argument("path", help="A *_whisperx.json file OR a directory of them")
    ap.add_argument("--out_dir", default=None, help="Where to write outputs (default: next to each JSON)")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*_whisperx.json"))) \
                or sorted(glob.glob(os.path.join(args.path, "*.json")))
        if not files:
            print(f"No JSON files found in {args.path}")
            return
        for jf in files:
            convert_one(jf, args.out_dir)
    else:
        convert_one(args.path, args.out_dir)


if __name__ == "__main__":
    main()
