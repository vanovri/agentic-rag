"""Agent tools: vector search (FAISS), BM25 keyword search, safe calculator."""
from __future__ import annotations

import ast
import json
import math
import operator as op
import re

from rank_bm25 import BM25Okapi

from .config import Settings


def _tok(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


class Retriever:
    def __init__(self, s: Settings | None = None, embedder=None):
        import faiss
        self.s = s or Settings()
        self.chunks = json.loads((self.s.index_dir / "chunks.json").read_text())
        self.index = faiss.read_index(str(self.s.index_dir / "faiss.index"))
        self.bm25 = BM25Okapi([_tok(c["text"]) for c in self.chunks])
        self._embedder = embedder

    @property
    def embedder(self):
        if self._embedder is None:
            from .ingest import Embedder
            self._embedder = Embedder(self.s.embed_model)
        return self._embedder

    def vector_search(self, query: str, k: int | None = None) -> list[dict]:
        k = min(k or self.s.top_k, len(self.chunks))
        D, I = self.index.search(self.embedder.embed([query]), k)
        return [{**self.chunks[i], "score": round(float(d), 4)} for d, i in zip(D[0], I[0]) if i >= 0]

    def bm25_search(self, query: str, k: int | None = None) -> list[dict]:
        k = k or self.s.top_k
        sc = self.bm25.get_scores(_tok(query))
        top = sorted(range(len(sc)), key=lambda i: -sc[i])[:k]
        return [{**self.chunks[i], "score": round(float(sc[i]), 4)} for i in top if sc[i] > 0]


_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
        ast.Mod: op.mod, ast.FloorDiv: op.floordiv, ast.USub: op.neg, ast.UAdd: op.pos}
_FUNCS = {k: getattr(math, k) for k in ("sqrt", "log", "log10", "exp", "sin", "cos", "tan")} | {
    "abs": abs, "round": round, "min": min, "max": max, "pi": math.pi, "e": math.e}


def calculator(expr: str) -> float:
    """Evaluate arithmetic safely (AST whitelist, no eval)."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name) and n.id in _FUNCS:
            return _FUNCS[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"unsupported expression: {ast.dump(n)[:80]}")
    return ev(ast.parse(expr.replace("^", "**"), mode="eval").body)
