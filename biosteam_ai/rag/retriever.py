"""Retrieval layer for grounding the assistant's explanations.

The corpus combines two sources:
  1. Curated markdown knowledge (BioSTEAM overview, TEA glossary, process
     background) under rag/knowledge/, chunked by top-level heading.
  2. Auto-generated documentation of the live model registry (one chunk per
     model, parameter, and metric) so descriptions stay in sync with the code.

Retrieval uses TF-IDF + cosine similarity via scikit-learn. This needs no
embedding API and no extra credentials. The Retriever interface is deliberately
simple (search(query, k)) so it can later be swapped for a vector/embedding
backend without touching callers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..models import REGISTRY

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


@dataclass
class Document:
    id: str
    title: str
    text: str
    source: str


def _chunk_markdown(path: Path) -> list[Document]:
    """Split a markdown file into one Document per top-level (#) heading."""
    raw = path.read_text()
    docs: list[Document] = []
    # Split on level-1 headings while keeping the heading text.
    parts = re.split(r"^# (.+)$", raw, flags=re.MULTILINE)
    # parts = [preamble, heading1, body1, heading2, body2, ...]
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if not body:
            continue
        docs.append(
            Document(
                id=f"{path.stem}#{i // 2}",
                title=title,
                text=f"{title}. {body}",
                source=path.name,
            )
        )
    return docs


def _registry_docs() -> list[Document]:
    """Generate documentation chunks from the live model registry."""
    docs: list[Document] = []
    for spec in REGISTRY.values():
        docs.append(
            Document(
                id=f"registry/{spec.key}",
                title=f"{spec.name} model",
                text=f"{spec.name} (model key '{spec.key}'). {spec.description}",
                source="model registry",
            )
        )
        for p in spec.parameters.values():
            lo, hi = p.bounds
            docs.append(
                Document(
                    id=f"registry/{spec.key}/param/{p.name}",
                    title=f"{spec.name}: parameter '{p.name}'",
                    text=(
                        f"In the {spec.name} model, the adjustable parameter "
                        f"'{p.name}' ({p.units}) is: {p.description} "
                        f"Allowed range {lo} to {hi}."
                    ),
                    source="model registry",
                )
            )
        for m in spec.metrics.values():
            docs.append(
                Document(
                    id=f"registry/{spec.key}/metric/{m.name}",
                    title=f"{spec.name}: metric '{m.name}'",
                    text=(
                        f"In the {spec.name} model, the output metric "
                        f"'{m.name}' ({m.units}) is: {m.description}"
                    ),
                    source="model registry",
                )
            )
    return docs


def build_corpus() -> list[Document]:
    docs: list[Document] = []
    for md in sorted(KNOWLEDGE_DIR.glob("*.md")):
        docs.extend(_chunk_markdown(md))
    docs.extend(_registry_docs())
    return docs


class Retriever:
    def __init__(self, documents: list[Document] | None = None):
        self.documents = documents if documents is not None else build_corpus()
        if not self.documents:
            raise RuntimeError("Knowledge corpus is empty.")
        self._vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2)
        )
        self._matrix = self._vectorizer.fit_transform(
            [d.text for d in self.documents]
        )

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = self._vectorizer.transform([query])
        scores = cosine_similarity(q, self._matrix)[0]
        ranked = sorted(
            range(len(self.documents)), key=lambda i: scores[i], reverse=True
        )
        results = []
        for i in ranked[:k]:
            if scores[i] <= 0:
                continue
            d = self.documents[i]
            results.append(
                {
                    "title": d.title,
                    "source": d.source,
                    "score": round(float(scores[i]), 4),
                    "text": d.text,
                }
            )
        return results


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Return a process-wide singleton retriever (built once)."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
