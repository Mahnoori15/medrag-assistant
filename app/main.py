"""FastAPI service: POST /ask with a question, get a cited answer."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .generate import answer
from .index import build_index

DATA_PATH = os.getenv("DATA_PATH", str(Path(__file__).parent.parent / "data" / "sample_docs.jsonl"))

app = FastAPI(title="MedRAG Assistant", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    k: int = Field(4, ge=1, le=10)


@lru_cache
def get_index():
    return build_index(DATA_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(get_index().chunks)}


@app.post("/ask")
def ask(req: AskRequest):
    hits = get_index().search(req.question, k=req.k)
    return answer(req.question, hits)
