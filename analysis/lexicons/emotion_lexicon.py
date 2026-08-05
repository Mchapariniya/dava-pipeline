#!/usr/bin/env python3
"""
emotion_lexicon.py — compact bilingual (German + English) emotion lexicon.

Backs the *offline fallback* of ``analysis/emotion.py``. Emotion keys match the
label/ID scheme already used by the pipeline and ``json_to_eaf.py``::

    0 angry   1 disgusted   2 fearful   3 happy
    4 neutral 5 other       6 sad       7 surprised   8 unknown

``neutral`` is the default when no emotion words are found, so it has no word
list here. ``other``/``unknown`` are reserved for no-detection cases.
"""

# emotion label -> set of trigger words (lower-cased)
EMOTION_WORDS = {
    "angry": {
        # German
        "wut", "wütend", "ärger", "ärgerlich", "sauer", "zorn", "zornig",
        "genervt", "aggressiv", "gereizt", "empört", "aufgebracht", "rage",
        "frust", "frustriert", "schreien", "geschrien", "brüllen", "toben",
        "explodieren", "hass", "hassen", "streit", "streiten", "provozieren",
        # English
        "angry", "anger", "mad", "furious", "rage", "annoyed", "irritated",
        "outraged", "aggressive", "frustrated", "frustration", "yell", "yelled",
        "scream", "screamed", "hate", "hatred", "hostile", "fuming", "livid",
    },
    "disgusted": {
        # German
        "ekel", "eklig", "ekelhaft", "widerlich", "abstoßend", "angewidert",
        "abscheu", "abscheulich", "übel", "würg", "igitt", "verabscheuen",
        # English
        "disgust", "disgusted", "disgusting", "gross", "revolting", "repulsive",
        "nasty", "sick", "sickening", "yuck", "nauseating", "loathe", "vile",
    },
    "fearful": {
        # German
        "angst", "ängstlich", "furcht", "fürchten", "erschrocken", "panik",
        "panisch", "sorge", "sorgen", "besorgt", "nervös", "unsicher",
        "bedroht", "gefahr", "gefährlich", "schreck", "scheu", "zittern",
        "verängstigt", "beunruhigt", "bange", "horror",
        # English
        "fear", "fearful", "afraid", "scared", "frightened", "terrified",
        "panic", "panicked", "worried", "worry", "anxious", "anxiety", "nervous",
        "threatened", "danger", "dangerous", "dread", "terror", "alarmed",
        "uneasy", "insecure",
    },
    "happy": {
        # German
        "glück", "glücklich", "freude", "freuen", "gefreut", "froh", "fröhlich",
        "spaß", "lachen", "gelacht", "lächeln", "zufrieden", "begeistert",
        "dankbar", "liebe", "lieben", "geliebt", "genießen", "genossen",
        "wunderbar", "toll", "super", "schön", "stolz", "heiter", "vergnügt",
        "entzückt", "strahlen", "jubeln", "euphorisch",
        # English
        "happy", "happiness", "joy", "joyful", "glad", "cheerful", "fun",
        "laugh", "laughed", "smile", "smiling", "content", "delighted",
        "excited", "grateful", "love", "loved", "enjoy", "enjoyed", "wonderful",
        "great", "proud", "pleased", "thrilled", "elated", "ecstatic", "merry",
    },
    "sad": {
        # German
        "traurig", "trauer", "weinen", "geweint", "tränen", "schmerz", "leid",
        "leiden", "verzweifelt", "verzweiflung", "einsam", "allein", "elend",
        "niedergeschlagen", "deprimiert", "hoffnungslos", "kummer", "betrübt",
        "melancholisch", "wehmut", "verloren", "schwermütig", "erschöpft",
        # English
        "sad", "sadness", "cry", "cried", "crying", "tears", "pain", "grief",
        "suffer", "suffering", "desperate", "despair", "lonely", "alone",
        "miserable", "depressed", "hopeless", "sorrow", "heartbroken", "gloomy",
        "melancholy", "unhappy", "down", "lost", "mourning", "exhausted",
    },
    "surprised": {
        # German
        "überrascht", "überraschung", "erstaunt", "staunen", "verblüfft",
        "verwundert", "unerwartet", "plötzlich", "schockiert", "schock",
        "sprachlos", "fassungslos", "wow", "krass", "unglaublich", "wahnsinn",
        # English
        "surprised", "surprise", "astonished", "amazed", "shocked", "shock",
        "stunned", "unexpected", "suddenly", "speechless", "wow", "incredible",
        "unbelievable", "startled", "astounded", "flabbergasted", "whoa",
    },
}

# label -> numeric id used across the pipeline & EAF export
EMOTION_ID = {
    "angry": 0, "disgusted": 1, "fearful": 2, "happy": 3,
    "neutral": 4, "other": 5, "sad": 6, "surprised": 7, "unknown": 8,
}
ID_EMOTION = {v: k for k, v in EMOTION_ID.items()}
