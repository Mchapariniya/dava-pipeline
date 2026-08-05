# DAVA analysis package

Post-transcription NLP over the WhisperX JSON. Every stage has a best-quality
(model-based) backend and a dependency-free offline fallback, and they all read/
write the same JSON so `../json_to_eaf.py` (ELAN export) keeps working.

| Module              | Does                          | Backends                          |
| ------------------- | ----------------------------- | --------------------------------- |
| `ner.py`            | Named Entity Recognition      | transformers · spaCy · regex      |
| `sentiment.py`      | sentiment (pos/neu/neg)       | transformers · lexicon            |
| `emotion.py`        | emotion from text             | transformers · lexicon            |
| `topic_modeling.py` | topic modelling               | BERTopic · LDA · NMF              |
| `visualize.py`      | word clouds & charts          | matplotlib · wordcloud            |
| `run_analysis.py`   | runs all of the above         | —                                 |

## Quick use

```bash
pip install -r requirements.txt
python -m spacy download xx_ent_wiki_sm

# everything at once (auto-selects the best available backend per stage)
python -m analysis.run_analysis path/to/episode_whisperx.json

# or a single stage
python -m analysis.ner            episode_whisperx.json --backend spacy
python -m analysis.topic_modeling episode_whisperx.json --method lda --num_topics 8
```

Full documentation: `../ANALYSIS.md`.
