#!/usr/bin/env python3
"""
sentiment_lexicon.py — compact bilingual (German + English) polarity lexicon.

This backs the *offline fallback* of ``analysis/sentiment.py`` (the production
path is a transformer model). It is deliberately small and high-frequency:
enough to give sensible polarity on conversational speech without any model
download. Words are stored lower-cased and matched against lower-cased tokens.

Coverage priorities: everyday evaluative adjectives/nouns/verbs and
intensifiers that show up in spoken German and English.
"""

POSITIVE = {
    # --- German ---
    "gut", "gute", "guter", "gutes", "besser", "beste", "bestes", "toll",
    "super", "prima", "schön", "schöne", "wunderbar", "wunderschön", "großartig",
    "herrlich", "fantastisch", "perfekt", "positiv", "richtig", "wichtig",
    "wertvoll", "hilfreich", "liebe", "lieben", "liebevoll", "geliebt", "glück",
    "glücklich", "froh", "freude", "freuen", "gefreut", "dankbar", "danke",
    "vertrauen", "sicher", "geborgen", "geborgenheit", "gesund", "stark",
    "erfolg", "erfolgreich", "gelungen", "spaß", "lustig", "angenehm", "ruhig",
    "entspannt", "gelassen", "zufrieden", "stolz", "mut", "mutig", "hoffnung",
    "hoffnungsvoll", "einfach", "leicht", "klar", "warm", "herzlich", "nett",
    "freundlich", "respekt", "respektvoll", "verständnis", "geduld", "geduldig",
    "unterstützung", "fördern", "wachsen", "lernen", "verbindung", "nähe",
    "sinnvoll", "wohl", "wohlfühlen", "genießen", "genossen", "lächeln",
    "lachen", "gelacht", "wertschätzung", "harmonie", "friedlich", "gerne",
    # --- English ---
    "good", "great", "better", "best", "nice", "wonderful", "beautiful",
    "amazing", "awesome", "excellent", "perfect", "positive", "right",
    "important", "valuable", "helpful", "love", "loving", "loved", "happy",
    "happiness", "glad", "joy", "joyful", "grateful", "thanks", "thank",
    "trust", "safe", "secure", "healthy", "strong", "success", "successful",
    "fun", "funny", "pleasant", "calm", "relaxed", "content", "proud", "brave",
    "courage", "hope", "hopeful", "easy", "clear", "warm", "kind", "friendly",
    "respect", "respectful", "understanding", "patience", "patient", "support",
    "supportive", "grow", "growth", "learn", "connection", "meaningful",
    "enjoy", "enjoyed", "smile", "laugh", "laughed", "peaceful", "wonderfully",
    "brilliant", "lovely", "delighted", "cheerful", "confident", "gentle",
}

NEGATIVE = {
    # --- German ---
    "schlecht", "schlechter", "schlimm", "schlimmer", "schrecklich",
    "furchtbar", "fürchterlich", "grausam", "böse", "falsch", "problem",
    "probleme", "schwierig", "schwer", "hart", "streng", "wut", "wütend",
    "ärger", "ärgerlich", "sauer", "genervt", "frust", "frustriert",
    "frustration", "angst", "ängstlich", "sorge", "sorgen", "besorgt", "traurig",
    "trauer", "weinen", "geweint", "schmerz", "schmerzhaft", "verletzt",
    "verletzen", "leid", "leiden", "müde", "erschöpft", "erschöpfung",
    "überfordert", "stress", "gestresst", "krank", "schwach", "einsam",
    "allein", "verzweifelt", "verzweiflung", "hoffnungslos", "hilflos",
    "unsicher", "verwirrt", "chaos", "chaotisch", "kritik", "kritisch",
    "streit", "streiten", "gestritten", "konflikt", "schuld", "schuldig",
    "scham", "peinlich", "enttäuscht", "enttäuschung", "hass", "hassen",
    "gehasst", "nervig", "anstrengend", "unangenehm", "gefährlich", "gefahr",
    "kaputt", "vermeiden", "leider", "kompliziert", "eskaliert", "eskalation",
    "schreien", "geschrien", "brüllen", "drohen", "strafe", "bestrafen",
    # --- English ---
    "bad", "worse", "worst", "terrible", "horrible", "awful", "cruel", "evil",
    "wrong", "problem", "problems", "difficult", "hard", "strict", "anger",
    "angry", "annoyed", "annoying", "frustrated", "frustration", "fear",
    "afraid", "scared", "worry", "worried", "sad", "sadness", "cry", "cried",
    "pain", "painful", "hurt", "suffer", "suffering", "tired", "exhausted",
    "exhaustion", "overwhelmed", "stress", "stressed", "sick", "weak", "lonely",
    "alone", "desperate", "hopeless", "helpless", "insecure", "confused",
    "chaos", "chaotic", "criticism", "critical", "conflict", "guilt", "guilty",
    "shame", "embarrassed", "disappointed", "disappointment", "hate", "hated",
    "dangerous", "danger", "broken", "avoid", "unfortunately", "complicated",
    "escalate", "escalated", "scream", "screamed", "yell", "yelled", "threaten",
    "punish", "punishment", "upset", "miserable", "anxious", "nervous", "fail",
    "failure", "struggle", "struggling",
}

# Words that negate/flip the polarity of a following token (within a small window).
NEGATIONS = {
    # German
    "nicht", "kein", "keine", "keinen", "keiner", "keines", "keinem", "nie",
    "niemals", "nichts", "kaum", "ohne", "weder", "noch",
    # English
    "not", "no", "never", "none", "without", "nor", "n't", "cannot", "cant",
    "dont", "doesnt", "didnt", "wont", "isnt", "arent", "hardly", "barely",
}

# Intensifiers that scale the magnitude of a nearby polar word.
INTENSIFIERS = {
    # German
    "sehr": 1.5, "total": 1.5, "extrem": 1.7, "wirklich": 1.3, "richtig": 1.3,
    "ziemlich": 1.2, "besonders": 1.4, "unglaublich": 1.6, "absolut": 1.5,
    "voll": 1.3, "so": 1.2, "mega": 1.6, "furchtbar": 1.5, "kaum": 0.5,
    "ein bisschen": 0.6, "etwas": 0.7,
    # English
    "very": 1.5, "really": 1.3, "extremely": 1.7, "incredibly": 1.6,
    "absolutely": 1.5, "totally": 1.5, "especially": 1.4, "quite": 1.2,
    "so": 1.2, "super": 1.5, "slightly": 0.6, "somewhat": 0.7, "a bit": 0.6,
}
