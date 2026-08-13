# DAVA analysis package

Post-transcription language analysis over the WhisperX JSON. Every stage has a
best-quality, model-based backend and a dependency-free offline fallback, and they
all read and write the same JSON, so `../json_to_eaf.py` (ELAN export) and the
dashboard keep working whichever backend runs. The dashboard's `Run analysis`
button calls this package, so what you see there is exactly what these scripts
produce on the command line.

| Module              | Does                          | Backends                          |
| ------------------- | ----------------------------- | --------------------------------- |
| `ner.py`            | Named Entity Recognition      | transformers, spaCy, regex        |
| `sentiment.py`      | sentiment (pos / neu / neg)   | transformers, lexicon             |
| `emotion.py`        | emotion from text             | transformers, lexicon             |
| `topic_modeling.py` | topic modelling               | BERTopic, LDA, NMF, Qwen          |
| `visualize.py`      | word clouds and charts        | matplotlib, wordcloud             |
| `run_analysis.py`   | runs all of the above         | orchestrator + CLI                |
| `common.py`         | shared JSON and helper code   | used by every module              |

## What it writes

The analysis enriches every segment of the transcript in place and saves charts
alongside it. Each segment gains a `sentiment` label and score, an `emotion` label
and score, a `topic_id` and `topic_label`, and a list of `entities`. Figures are
written to a `figures/` folder next to the JSON. These are the fields the ELAN
export and the dashboard tabs read.

## Backends and offline mode

With only the packages in `requirements.txt` installed, the analysis runs fully
offline: spaCy for names, lexicons for sentiment and emotion, and scikit-learn
(LDA or NMF) for topics. When `sentence-transformers` and `transformers` are
present (they are in the project's `dava_env`), the higher-quality transformer
backends switch on automatically. `auto` picks the best available backend for each
stage; you can also force a specific one.

## Quick use

```bash
pip install -r requirements.txt
python -m spacy download xx_ent_wiki_sm

# everything at once (auto-selects the best available backend per stage)
python -m analysis.run_analysis path/to/episode_whisperx.json

# useful options
python -m analysis.run_analysis episode_whisperx.json \
    --ner-backend transformers \
    --topic-method lda --num-topics 8 \
    --skip-emotion
```

Run this from the repository root so that the `analysis` package is importable.

## Topic methods

`--topic-method` accepts `auto`, `lda`, `nmf`, `bertopic`, or `qwen`. `lda` and
`nmf` are classic statistical methods; `bertopic` groups passages by meaning using
embeddings; `qwen` asks a local Qwen language model to read the transcript and name
the topics directly. Note that the `qwen` topic method runs through the
`transformers` library (it downloads the model set in the dashboard's Advanced
settings), which is separate from the Ollama model used by the Ask feature.

Full project documentation is in `../ANALYSIS.md`, and a from-scratch guide with a
glossary and model descriptions is in `../docs/GUIDE.md`.
