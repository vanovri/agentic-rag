import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentic_rag.config import Settings  # noqa: E402
from agentic_rag.ingest import ingest  # noqa: E402
from agentic_rag.tools import Retriever  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


class HashEmbedder:
    """Offline bag-of-words hashing embedder (no model download)."""
    def embed(self, texts):
        out = np.zeros((len(texts), 256), dtype="float32")
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.strip(".,()").encode()).hexdigest(), 16) % 256] += 1
        return out / np.linalg.norm(out, axis=1, keepdims=True).clip(1e-9)


@pytest.fixture
def retriever(tmp_path):
    s = Settings(index_dir=tmp_path / "idx")
    emb = HashEmbedder()
    ingest([SAMPLE], s, embedder=emb)
    return Retriever(s, embedder=emb), s
