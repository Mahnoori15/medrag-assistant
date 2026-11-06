"""Download PubMed abstracts into data/pubmed_docs.jsonl.

Usage:
    python -m app.ingest "atrial fibrillation anticoagulation" --max 200

Uses NCBI's free E-utilities API. Please keep to NCBI's usage guidelines
(max 3 requests/second without an API key).
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def search_ids(query: str, max_results: int) -> list[str]:
    q = urllib.parse.urlencode({"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"})
    return json.loads(_get(f"{BASE}/esearch.fcgi?{q}"))["esearchresult"]["idlist"]


def fetch_abstracts(ids: list[str]) -> list[dict]:
    docs = []
    for i in range(0, len(ids), 100):
        batch = ",".join(ids[i:i + 100])
        root = ET.fromstring(_get(f"{BASE}/efetch.fcgi?db=pubmed&id={batch}&retmode=xml"))
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//PMID")
            title = art.findtext(".//ArticleTitle") or ""
            abstract = " ".join("".join(a.itertext()) for a in art.iter("AbstractText"))
            if abstract:
                docs.append({
                    "id": f"PMID:{pmid}",
                    "title": title,
                    "source": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "text": abstract,
                })
        time.sleep(0.4)
    return docs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--max", type=int, default=200)
    p.add_argument("--out", default="data/pubmed_docs.jsonl")
    a = p.parse_args()
    docs = fetch_abstracts(search_ids(a.query, a.max))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    print(f"Saved {len(docs)} abstracts to {a.out}")


if __name__ == "__main__":
    main()
