"""
DAVA analysis package
======================

Post-transcription NLP stages that run on the WhisperX JSON:

* :mod:`analysis.ner`            — Named Entity Recognition
* :mod:`analysis.sentiment`      — sentiment analysis
* :mod:`analysis.emotion`        — text-based emotion recognition
* :mod:`analysis.topic_modeling` — topic modelling (BERTopic / LDA / NMF)
* :mod:`analysis.visualize`      — word clouds & charts
* :mod:`analysis.run_analysis`   — one-shot orchestrator for all of the above

Each stage has a ``transformers`` (or model-based) backend for production use
and a dependency-free fallback so the whole pipeline runs on CPU with no model
downloads. All stages read/write the same JSON schema so the existing
``json_to_eaf.py`` ELAN export keeps working unchanged.
"""

__all__ = [
    "common", "ner", "sentiment", "emotion", "topic_modeling",
    "visualize", "run_analysis",
]

__version__ = "1.0.0"
