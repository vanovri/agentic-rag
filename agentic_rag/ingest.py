"""Load PDF/MD/TXT, chunk, embed locally, store in a FAISS index."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .config import Settings

EXTS = {".pdf", ".md", ".txt", ".docx"}


def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    if path.suffix.lower() == ".docx":
        import docx
        d = docx.Document(str(path))
        rows = [" | ".join(c.text for c in r.cells) for tb in d.tables for r in tb.rows]
        return "\n\n".join([p.text for p in d.paragraphs if p.text.strip()] + rows)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    """Paragraph/sentence-aware chunking with character overlap."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    pieces = [p.strip() for p in re.split(r"\n\s*\n|(?<=[.!?])\s+", text) if p.strip()]
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 1 > size:
            chunks.append(cur)
            cur = cur[-overlap:].split(" ", 1)[-1] if overlap else ""
        cur = (cur + " " + p).strip()
        while len(cur) > size * 1.5:  # hard split very long pieces
            chunks.append(cur[:size]); cur = cur[size - overlap:]
    if cur:
        chunks.append(cur)
    return chunks


class Embedder:
    def __init__(self, model: str):
        from fastembed import TextEmbedding
        from .config import ROOT
        self.m = TextEmbedding(model_name=model, cache_dir=str(ROOT / ".cache" / "fastembed"))

    def embed(self, texts: list[str]) -> np.ndarray:
        v = np.array(list(self.m.embed(texts)), dtype="float32")
        return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9)


def collect(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        out += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS) if p.is_dir() else [p]
    return out


def ingest(paths: list[Path], s: Settings | None = None, embedder=None) -> int:
    import faiss
    s = s or Settings()
    embedder = embedder or Embedder(s.embed_model)
    chunks = []
    skipped = []
    for f in collect(paths):
        try:
            text = load_text(f)
        except Exception as e:  # e.g. OneDrive cloud-only placeholder, corrupt PDF
            skipped.append(f"{f.name}: {e}"); continue
        if not text.strip():
            skipped.append(f"{f.name}: no extractable text"); continue
        for i, c in enumerate(chunk_text(text, s.chunk_size, s.chunk_overlap)):
            chunks.append({"id": f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', f.stem)}#{i}", "source": f.name, "text": c})
    if not chunks:
        raise ValueError("no documents found")
    vecs = embedder.embed([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    s.index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(s.index_dir / "faiss.index"))
    (s.index_dir / "chunks.json").write_text(json.dumps(chunks, indent=1))
    for x in skipped:
        print("SKIPPED", x)
    return len(chunks)
