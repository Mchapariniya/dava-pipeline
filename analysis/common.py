#!/usr/bin/env python3
"""
common.py — Shared helpers for the DAVA analysis package.

Everything the individual analysis stages (NER, sentiment, emotion, topic
modelling, visualisation) need in one place: reading/writing the WhisperX
JSON, pulling out sentences, guessing the dominant language, tokenising for
word-frequency work, stop-word lists, and small helpers for picking the best
available model backend.

Design notes
------------
* The WhisperX JSON produced by the pipeline looks like::

      {"segments": [{"start","end","text","words","speaker",
                     "detected_language", ...}, ...],
       "word_segments": [...]}

  A "sentence" here == one segment. That matches how ``json_to_eaf.py`` and
  ``qwen_topic_detector.py`` already treat the data, so downstream ELAN export
  stays consistent (segment index i == sent_id i+1 in the NER TSV).

* ``detected_language`` is present per-segment but is noisy (Whisper's
  language head fires on short/musical segments). We therefore compute a
  *document* dominant language by majority vote weighted by character length,
  and expose it for model/stop-word selection, while still keeping the raw
  per-segment codes available for anyone who wants them.
"""
from __future__ import annotations

import json
import re
import importlib.util
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple, Iterator, Iterable

# --------------------------------------------------------------------------- #
# JSON I/O
# --------------------------------------------------------------------------- #

def load_transcript(json_path: str | Path) -> dict:
    """Load a WhisperX JSON file and return the parsed dict."""
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_transcript(data: dict, json_path: str | Path) -> None:
    """Write the (possibly enriched) transcript dict back to disk as UTF-8."""
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def get_segments(data: dict | list) -> List[dict]:
    """Accept either a bare list of segments or a {'segments': [...]} dict."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("segments", []) or []
    return []


def base_name_from_json(json_path: str | Path) -> str:
    """`foo_whisperx.json` -> `foo`  (mirrors json_to_txt / json_to_eaf)."""
    stem = Path(json_path).stem
    if stem.endswith("_whisperx"):
        stem = stem[: -len("_whisperx")]
    return stem


# --------------------------------------------------------------------------- #
# Sentence / text extraction
# --------------------------------------------------------------------------- #

def iter_sentences(segments: List[dict]) -> Iterator[Tuple[int, dict]]:
    """Yield (segment_index, segment) for every segment that has real text.

    The index is the position in the ORIGINAL segments list so that callers
    can write it back onto the same segment (important for EAF alignment).
    """
    for idx, seg in enumerate(segments):
        if (seg.get("text") or "").strip():
            yield idx, seg


def sentence_texts(segments: List[dict]) -> List[Tuple[int, str]]:
    """List of (index, cleaned_text) for non-empty segments."""
    return [(i, (s.get("text") or "").strip()) for i, s in iter_sentences(segments)]


def document_text(segments: List[dict]) -> str:
    """All segment text joined into one string (whole-document analyses)."""
    return " ".join(t for _, t in sentence_texts(segments))


# --------------------------------------------------------------------------- #
# Language handling
# --------------------------------------------------------------------------- #

# ISO-639-1 -> spaCy blank-pipeline code, for stop-word / lemmatiser lookups.
_SPACY_LANG = {
    "en": "en", "de": "de", "fr": "fr", "es": "es", "it": "it", "pt": "pt",
    "nl": "nl", "ru": "ru", "pl": "pl", "ro": "ro", "el": "el", "sv": "sv",
    "da": "da", "no": "nb", "fi": "fi", "cs": "cs", "uk": "uk", "ca": "ca",
    "tr": "tr", "ja": "ja", "zh": "zh", "ar": "ar", "ko": "ko",
}


def dominant_language(segments: List[dict], default: str = "en") -> str:
    """Best guess at the document's main language.

    Strategy: majority vote over ``detected_language`` weighted by the number
    of characters in each segment (so a 40-word German paragraph outweighs a
    two-word mis-detected snippet). Falls back to ``langdetect`` on the full
    text if the per-segment codes are missing/unusable.
    """
    weights: Counter = Counter()
    for seg in segments:
        lang = seg.get("detected_language")
        text = (seg.get("text") or "").strip()
        if lang and text:
            weights[lang] += len(text)

    if weights:
        return weights.most_common(1)[0][0]

    # Fallback: langdetect over the whole document.
    try:
        from langdetect import detect  # type: ignore
        text = document_text(segments)
        if text.strip():
            return detect(text)
    except Exception:
        pass
    return default


def language_distribution(segments: List[dict]) -> Dict[str, int]:
    """Char-weighted distribution of detected languages (for the dashboard)."""
    weights: Counter = Counter()
    for seg in segments:
        lang = seg.get("detected_language") or "unknown"
        weights[lang] += len((seg.get("text") or "").strip())
    return dict(weights.most_common())


def spacy_lang_code(iso: str) -> str:
    """Map an ISO-639-1 code to the closest spaCy blank-pipeline code."""
    return _SPACY_LANG.get((iso or "").lower(), "xx")


# --------------------------------------------------------------------------- #
# Tokenisation & stop-words (for word clouds / topic modelling)
# --------------------------------------------------------------------------- #

# Compact but practical stop-word lists. English is also available from
# scikit-learn, but we bundle both so the package has no hard dependency on a
# particular sklearn version and so German is always covered.
_STOP_EN = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs
them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your
yours yourself yourselves just like get got really thing things one two also
""".split())

_STOP_DE = set("""
aber alle allem allen aller alles als also am an ander andere anderem anderen
anderer anderes anderm andern anderr anders auch auf aus bei bin bis bist da
damit dann der den des dem die das dass daß derselbe derselben denselben
desselben demselben dieselbe dieselben dasselbe dazu dein deine deinem deinen
deiner deines denn derer dessen dich dir du dies diese diesem diesen dieser
dieses doch dort durch ein eine einem einen einer eines einig einige einigem
einigen einiger einiges einmal er ihn ihm es etwas euer eure eurem euren eurer
eures für gegen gewesen hab habe haben hat hatte hatten hier hin hinter ich
mich mir ihr ihre ihrem ihren ihrer ihres euch im in indem ins ist jede jedem
jeden jeder jedes jene jenem jenen jener jenes jetzt kann kein keine keinem
keinen keiner keines können könnte machen man manche manchem manchen mancher
manches mein meine meinem meinen meiner meines mit muss musste nach nicht
nichts noch nun nur ob oder ohne sehr sein seine seinem seinen seiner seines
selbst sich sie sind so solche solchem solchen solcher solches soll sollte
sondern sonst über um und uns unse unsem unsen unser unses unter viel vom von
vor während war waren warst was weg weil weiter welche welchem welchen welcher
welches wenn werde werden wie wieder will wir wird wirst wo wollen wollte
würde würden zu zum zur zwar zwischen mal schon ja halt eben ganz gar immer
etwa dabei dafür dadurch daran darauf gibt gibts geht mehr wurde wurden
""".split())

_STOP_MULTI = _STOP_EN | _STOP_DE

_STOP_BY_LANG = {"en": _STOP_EN, "de": _STOP_DE}

# Keep only tokens that are alphabetic (Unicode-aware) and reasonably long.
_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def get_stopwords(lang: Optional[str] = None, extra: Optional[Iterable[str]] = None) -> set:
    """Stop-word set for ``lang`` (falls back to the combined DE+EN set)."""
    words = set(_STOP_BY_LANG.get((lang or "").lower(), _STOP_MULTI))
    if extra:
        words |= {w.lower() for w in extra}
    return words


def tokenize(text: str, lang: Optional[str] = None, min_len: int = 3,
             stopwords: Optional[set] = None) -> List[str]:
    """Lower-cased alphabetic tokens with stop-words and short tokens removed.

    Used by the word-cloud and topic-modelling code. Not a linguistic
    tokeniser — deliberately simple and fast so it runs on CPU over long
    transcripts without extra dependencies.
    """
    stop = stopwords if stopwords is not None else get_stopwords(lang)
    out: List[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if len(tok) >= min_len and tok not in stop:
            out.append(tok)
    return out


def word_frequencies(segments: List[dict], lang: Optional[str] = None,
                     min_len: int = 3, top_n: Optional[int] = None) -> Counter:
    """Counter of word -> frequency across all segment text."""
    lang = lang or dominant_language(segments)
    stop = get_stopwords(lang)
    counter: Counter = Counter()
    for _, text in sentence_texts(segments):
        counter.update(tokenize(text, lang=lang, min_len=min_len, stopwords=stop))
    if top_n:
        return Counter(dict(counter.most_common(top_n)))
    return counter


# --------------------------------------------------------------------------- #
# Backend / dependency helpers
# --------------------------------------------------------------------------- #

def has_module(name: str) -> bool:
    """True if an importable module ``name`` is installed (no import side-effects)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def torch_device(gpu_index: Optional[int] = None) -> str:
    """Return a device string: 'cuda:N' if a GPU is available, else 'cpu'.

    Safe to call when torch is not installed (returns 'cpu').
    """
    if not has_module("torch"):
        return "cpu"
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            if gpu_index is not None:
                return f"cuda:{gpu_index}"
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def transformers_device_index(gpu_index: Optional[int] = None) -> int:
    """HF ``pipeline(device=...)`` wants an int (-1 == CPU, >=0 == that GPU)."""
    dev = torch_device(gpu_index)
    if dev.startswith("cuda"):
        try:
            return int(dev.split(":")[1])
        except (IndexError, ValueError):
            return 0
    return -1


# --------------------------------------------------------------------------- #
# Speaker helpers (used by the dashboard & visualisations)
# --------------------------------------------------------------------------- #

def speakers(segments: List[dict]) -> List[str]:
    """Sorted list of distinct speaker labels present in the transcript."""
    found = {seg.get("speaker") for seg in segments if seg.get("speaker")}
    return sorted(found)


def segment_duration(seg: dict) -> float:
    """End - start in seconds (>= 0)."""
    return max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
