# MedRAG Assistant

A retrieval-augmented generation (RAG) API that answers medical literature questions **with citations**, and refuses when the sources don't contain the answer.

![CI](https://github.com/Mahnoori15/medrag-assistant/actions/workflows/ci.yml/badge.svg)

> For research and education only. Not medical advice.

## Why I built it
Clinicians and researchers need quick answers they can verify. Plain LLMs can make up facts. This project grounds every answer in retrieved sources, cites them, and says "I could not find this" instead of guessing. Those are the core safety habits for clinical AI.

## Features
- **Ingestion** of PubMed abstracts through NCBI's free E-utilities API (`app/ingest.py`)
- **Chunking + vector search**: TF-IDF by default (runs offline on CPU), or dense `sentence-transformers` embeddings with `EMBEDDINGS=sbert`
- **Grounded generation**: an LLM (Claude via the Anthropic API) answers only from the retrieved passages, with inline citations `[1]`
- **Safety guardrails**: a relevance threshold refuses off-topic questions, the prompt blocks diagnosis or personal advice, and every response carries a disclaimer
- **Offline mode**: with no API key, it returns the best matching passage (extractive), so it runs anywhere
- **Evaluation**: retrieval hit rate@k and MRR on a labelled question set (`eval/`)
- **Engineering**: FastAPI, pytest, Docker, GitHub Actions CI

## Architecture
```
question ─► FastAPI /ask ─► vector search (top-k chunks) ─► relevance check
                                                              │
                                     no match ◄───────────────┤
                                                              ▼
                                         LLM with sources ─► answer + citations + disclaimer
```

## Quick start
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://localhost:8000/docs
```

Ask a question:
```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "How is atrial fibrillation diagnosed?"}'
```

Use an LLM for full answers:
```bash
export ANTHROPIC_API_KEY=your_key
```

Use real PubMed data:
```bash
python -m app.ingest "atrial fibrillation anticoagulation" --max 200
DATA_PATH=data/pubmed_docs.jsonl uvicorn app.main:app
```

Docker:
```bash
docker build -t medrag-assistant .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY medrag-assistant
```

## Evaluation
```bash
python -m eval.run_eval --k 3
```
| Retriever | Dataset | hit rate@3 | MRR |
|---|---|---|---|
| TF-IDF | sample set (10 questions) | 1.00 | 1.00 |

*Next step: build a larger PubMed question set and compare TF-IDF with dense embeddings.*

## Tests
```bash
pytest -q
```

## Roadmap
- [ ] Larger evaluation set from PubMed, comparing TF-IDF with dense embeddings
- [ ] Answer faithfulness checks (does every sentence trace back to a source?)
- [ ] Simple web UI (Streamlit)
- [ ] Deploy to AWS or Azure

## Data note
`data/sample_docs.jsonl` holds short, author-written summaries of common conditions, used for demos and tests. They are not clinical guidance.
