"""Answer generation with citations and a safety fallback.

If ANTHROPIC_API_KEY is set, an LLM writes the answer using only the
retrieved passages. Otherwise the app returns the best matching passages
(extractive mode), so it still works without any API key.
"""
from __future__ import annotations

import os

from .index import Chunk

DISCLAIMER = "For research and education only. Not medical advice."
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.08"))

SYSTEM_PROMPT = """You are a careful medical literature assistant.
Answer ONLY using the numbered sources provided. Cite sources inline like [1], [2].
If the sources do not contain the answer, say you could not find it in the sources.
Never give a diagnosis or personal treatment advice. Keep answers under 150 words."""


def _format_sources(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{i}] {c.title}\n{c.text}" for i, (c, _) in enumerate(hits, 1))


def _citations(hits):
    return [
        {"n": i, "doc_id": c.doc_id, "title": c.title, "source": c.source, "score": round(s, 3)}
        for i, (c, s) in enumerate(hits, 1)
    ]


def answer(question: str, hits: list[tuple[Chunk, float]]) -> dict:
    relevant = [h for h in hits if h[1] >= MIN_SCORE]
    if not relevant:
        return {
            "answer": "I could not find this in the indexed sources.",
            "mode": "no_match",
            "citations": [],
            "disclaimer": DISCLAIMER,
        }

    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=os.getenv("LLM_MODEL", "claude-haiku-5-5"),
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Sources:\n{_format_sources(relevant)}\n\nQuestion: {question}",
            }],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        mode = "llm"
    else:
        best = relevant[0][0]
        text = f"Most relevant passage [1]: {best.text}"
        mode = "extractive"

    return {"answer": text, "mode": mode, "citations": _citations(relevant), "disclaimer": DISCLAIMER}
