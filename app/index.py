"""Chunking and vector search over medical documents.

Uses TF-IDF vectors by default so the project runs offline with no GPU.
Set EMBEDDINGS=sbert to use sentence-transformers dense embeddings instead.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Chunk:
    doc_id: str
    title: str
    source: str
    text: str


def load_docs(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_docs(docs: list[dict], max_words: int = 120, overlap: int = 30) -> list[Chunk]:
    """Split each document into overlapping word windows."""
    chunks: list[Chunk] = []
    step = max_words - overlap
    for d in docs:
        words = d["text"].split()
        for start in range(0, max(len(words), 1), step):
            piece = " ".join(words[start:start + max_words])
            if piece:
                chunks.append(Chunk(d["id"], d["title"], d.get("source", ""), piece))
            if start + max_words >= len(words):
                break
    return chunks


class _TfidfEncoder:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")

    def fit(self, texts):
        self.vec.fit(texts)

    def encode(self, texts) -> np.ndarray:
        m = self.vec.transform(texts).toarray()
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)


class _SbertEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        pass

    def encode(self, texts) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True))


class VectorIndex:
    def __init__(self, chunks: list[Chunk], backend: str | None = None):
        backend = backend or os.getenv("EMBEDDINGS", "tfidf")
        self.chunks = chunks
        self.encoder = _SbertEncoder() if backend == "sbert" else _TfidfEncoder()
        texts = [f"{c.title}. {c.text}" for c in chunks]
        self.encoder.fit(texts)
        self.matrix = self.encoder.encode(texts)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        q = self.encoder.encode([query])[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


def build_index(path: str | Path, backend: str | None = None) -> VectorIndex:
    return VectorIndex(chunk_docs(load_docs(path)), backend)
