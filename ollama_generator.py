#!/usr/bin/env python3
"""
ollama_generator.py  —  local answer generation via Ollama.
================================================================================
Kept deliberately separate from the retrieval core (dava_rag.py) and from the
dashboard. Its ONLY job: take a question + already-retrieved passages and produce
a grounded answer in the requested language, using a local Ollama model. No API
key, no internet after the model is pulled.

Contract (so it's interchangeable with dava_rag.ExtractiveAnswerer):
    generate(question: str, passages: List[Passage], target_lang="auto") -> GenResult

Prompt shaping and language handling are imported from dava_rag, so every backend
answers identically — this module only owns the HTTP transport to Ollama.

Setup:
    ollama serve
    ollama pull qwen2.5:7b        # strong multilingual (de/en/fr); your default
    # bigger/alternatives: qwen2.5:14b, llama3.1:8b, aya:8b (very multilingual)
"""

from __future__ import annotations

import time
from typing import List, Tuple

import requests

from dava_rag import (Passage, GenResult, build_grounded_prompt,
                      resolve_target_lang, parse_cited)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"


class OllamaGenerator:
    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_OLLAMA_URL,
                 temperature: float = 0.2, num_predict: int = 1024,
                 timeout: int = 180, max_retries: int = 2):
        self.model = model
        self.url = url.rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout
        self.max_retries = max_retries

    # -- readiness check (nice for the dashboard to show a clear message) ----- #
    def available(self) -> Tuple[bool, str]:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=5)
            r.raise_for_status()
        except requests.exceptions.ConnectionError:
            return False, f"Ollama is not running at {self.url}. Start it with `ollama serve`."
        except Exception as e:
            return False, f"Could not reach Ollama at {self.url}: {e}"
        names = [m.get("name", "") for m in r.json().get("models", [])]
        base = self.model.split(":")[0]
        if not any(base in n for n in names):
            return False, (f"Model '{self.model}' isn't pulled. Run `ollama pull {self.model}`. "
                           f"Available: {names or 'none'}")
        return True, f"Ollama ready — {self.model}"

    # -- the generate() contract --------------------------------------------- #
    def generate(self, question: str, passages: List[Passage],
                 target_lang: str = "auto") -> GenResult:
        lang = resolve_target_lang(target_lang, question)
        if not passages:
            return GenResult(answer="No relevant passages were retrieved.",
                             cited=[], provider=f"ollama:{self.model}", target_lang=lang)

        system, user = build_grounded_prompt(question, passages, lang)
        payload = {
            "model": self.model,
            # /api/chat reads the system prompt from a role:"system" message, NOT from
            # a top-level "system" field (that one is only for /api/generate). Sending
            # it as a message is what makes the language + grounding rules take effect.
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.num_predict},
        }

        answer = ""
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
                r.raise_for_status()
                answer = (r.json().get("message", {}) or {}).get("content", "").strip()
                break
            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                else:
                    answer = "The local model timed out. Try a smaller model or a lower top-k."
            except requests.exceptions.ConnectionError:
                answer = f"Could not reach Ollama at {self.url}. Is `ollama serve` running?"
                break
            except Exception as e:
                answer = f"Generation failed: {e}"
                break

        return GenResult(answer=answer, cited=parse_cited(answer, passages),
                         provider=f"ollama:{self.model}", target_lang=lang)


# quick manual smoke test:  python ollama_generator.py "your question" transcript.json
if __name__ == "__main__":
    import sys
    from dava_rag import Retriever, SentenceTransformerEmbedder
    if len(sys.argv) < 3:
        print("usage: python ollama_generator.py \"question\" transcript.json [de|en|fr]")
        raise SystemExit(1)
    question, transcript = sys.argv[1], sys.argv[2]
    lang = sys.argv[3] if len(sys.argv) > 3 else "auto"

    gen = OllamaGenerator()
    ok, msg = gen.available()
    print(msg)
    if not ok:
        raise SystemExit(1)

    r = Retriever(embedder=SentenceTransformerEmbedder())
    print(f"indexed {r.index(transcript)} passages")
    passages = r.search(question, k=5)
    res = gen.generate(question, passages, target_lang=lang)
    print("\n=== ANSWER ===\n" + res.answer)
    print("\n=== REFERENCES ===")
    for i, p in enumerate(passages):
        mark = "*" if i in res.cited else " "
        print(f" {mark} [{p.start_fmt}–{p.end_fmt}] {p.speaker}  (score {p.score:.3f})")
