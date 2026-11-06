"""Retrieval evaluation: hit rate@k and mean reciprocal rank (MRR).

Usage: python -m eval.run_eval --k 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.index import build_index

ROOT = Path(__file__).parent.parent


def evaluate(k: int = 3, data=ROOT / "data" / "sample_docs.jsonl", backend=None) -> dict:
    index = build_index(data, backend)
    cases = [json.loads(l) for l in open(ROOT / "eval" / "eval_set.jsonl") if l.strip()]
    hits, rr = 0, 0.0
    for c in cases:
        ids = [ch.doc_id for ch, _ in index.search(c["question"], k=k)]
        if c["expected_doc"] in ids:
            hits += 1
            rr += 1 / (ids.index(c["expected_doc"]) + 1)
    return {"cases": len(cases), f"hit_rate@{k}": round(hits / len(cases), 3), "mrr": round(rr / len(cases), 3)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--backend", default=None, help="tfidf (default) or sbert")
    a = p.parse_args()
    print(json.dumps(evaluate(a.k, backend=a.backend), indent=2))
