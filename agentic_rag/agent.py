"""Agent loop: plan -> tools -> draft -> verify claims -> re-retrieve -> final cited answer."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import Settings
from .llm import LLM, parse_json
from .tools import Retriever, calculator

NOT_FOUND = "Not found: the indexed documents do not contain enough support to answer this question."

PLAN_PROMPT = """You are a research agent answering a question from a local document collection.
Tools:
- vector_search(query): semantic search over document chunks
- bm25_search(query): exact keyword search (good for names, numbers, rare terms)
- calculator(expression): arithmetic, e.g. "3.67 * 0.8"
- answer: stop using tools and write the answer
Question: {question}
Observations so far:
{obs}
Steps remaining: {left}. Reply with ONLY JSON: {{"thought": "...", "action": "vector_search|bm25_search|calculator|answer", "input": "..."}}"""

DRAFT_PROMPT = """Answer the question using ONLY the chunks below. Split the answer into atomic factual claims, each citing chunk ids it relies on.
If the chunks do not answer the question, return an empty claims list.
Question: {question}
Calculator results: {calc}
Chunks:
{ctx}
Reply with ONLY JSON: {{"claims": [{{"text": "...", "citations": ["chunk-id"]}}]}}"""

VERIFY_PROMPT = """You are a strict fact checker. For each claim decide whether it is fully supported by the cited chunk text (or calculator results). Unsupported if any part is missing or contradicted.
Calculator results: {calc}
Chunks:
{ctx}
Claims:
{claims}
Reply with ONLY JSON: {{"verdicts": [{{"index": 0, "supported": true, "reason": "..."}}]}}"""


@dataclass
class Result:
    question: str
    answer: str
    found: bool
    claims: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)


class Agent:
    def __init__(self, llm: LLM, retriever: Retriever, s: Settings | None = None):
        self.llm, self.r, self.s = llm, retriever, s or Settings()

    def _ask(self, prompt: str) -> dict:
        try:
            return parse_json(self.llm.complete(prompt))
        except Exception as e:  # one retry with a nudge
            return parse_json(self.llm.complete(prompt + f"\n(Previous reply was invalid: {e}. Output valid JSON only.)"))

    @staticmethod
    def _ctx(chunks: dict) -> str:
        return "\n\n".join(f"[{cid}] ({c['source']}) {c['text']}" for cid, c in chunks.items()) or "(none)"

    def run(self, question: str) -> Result:
        trace, chunks, calc = [], {}, []

        def add(hits):
            for h in hits:
                chunks.setdefault(h["id"], h)

        # 1) tool-choosing loop
        for step in range(self.s.max_tool_steps):
            obs = "\n".join(f"- {t['tool']}({t['input']!r}) -> {t['summary']}" for t in trace) or "(none)"
            d = self._ask(PLAN_PROMPT.format(question=question, obs=obs, left=self.s.max_tool_steps - step))
            act, arg = d.get("action", "answer"), str(d.get("input", question))
            if act == "answer":
                trace.append({"step": "plan", "tool": "answer", "input": "", "thought": d.get("thought", ""), "summary": "stop"})
                break
            if act in ("vector_search", "bm25_search"):
                hits = getattr(self.r, act)(arg)
                add(hits)
                trace.append({"step": "tool", "tool": act, "input": arg, "thought": d.get("thought", ""),
                              "results": hits, "summary": ", ".join(h["id"] for h in hits) or "no hits"})
            elif act == "calculator":
                try:
                    out = calculator(arg)
                except Exception as e:
                    out = f"error: {e}"
                calc.append(f"{arg} = {out}")
                trace.append({"step": "tool", "tool": "calculator", "input": arg, "thought": d.get("thought", ""), "summary": str(out)})
        if not chunks:  # always ground on something
            hits = self.r.vector_search(question)
            add(hits)
            trace.append({"step": "tool", "tool": "vector_search", "input": question, "thought": "fallback retrieval",
                          "results": hits, "summary": ", ".join(h["id"] for h in hits)})

        # 2) draft -> verify -> (re-retrieve + redraft) loop
        for rnd in range(self.s.max_verify_rounds + 1):
            draft = self._ask(DRAFT_PROMPT.format(question=question, calc=calc or "none", ctx=self._ctx(chunks)))
            claims = [c for c in draft.get("claims", []) if c.get("text")]
            for c in claims:  # citations must point to retrieved chunks
                c["citations"] = [x for x in c.get("citations", []) if x in chunks]
            trace.append({"step": "draft", "round": rnd, "claims": json.loads(json.dumps(claims))})
            if not claims:
                verdicts = []
            else:
                cited = {cid: chunks[cid] for c in claims for cid in c["citations"]} or chunks
                v = self._ask(VERIFY_PROMPT.format(calc=calc or "none", ctx=self._ctx(cited),
                              claims="\n".join(f"{i}. {c['text']} (cites: {', '.join(c['citations']) or 'none'})" for i, c in enumerate(claims))))
                vmap = {int(x.get("index", -1)): x for x in v.get("verdicts", [])}
                verdicts = []
                for i, c in enumerate(claims):
                    x = vmap.get(i, {})
                    raw_sup = x.get("supported")
                    is_sup = raw_sup is True or (isinstance(raw_sup, str) and raw_sup.strip().lower() == "true")
                    ok = is_sup and bool(c["citations"])
                    c["supported"], c["reason"] = ok, x.get("reason", "no verdict" if c["citations"] else "no valid citation")
                    verdicts.append(ok)
            trace.append({"step": "verify", "round": rnd, "verdicts": [{"claim": c["text"], "supported": c["supported"], "reason": c["reason"]} for c in claims]})
            bad = [c for c in claims if not c["supported"]]
            if claims and not bad:
                break
            if rnd < self.s.max_verify_rounds:  # re-retrieve for unsupported claims (or the question)
                for q in [c["text"] for c in bad] or [question]:
                    for tool in ("vector_search", "bm25_search"):
                        hits = getattr(self.r, tool)(q)
                        new = [h for h in hits if h["id"] not in chunks]
                        add(hits)
                        trace.append({"step": "re-retrieve", "tool": tool, "input": q, "results": hits,
                                      "summary": f"{len(new)} new chunk(s)"})

        # 3) final answer: only verified claims, inline citations
        good = [c for c in claims if c.get("supported")]
        if not good:
            return Result(question, NOT_FOUND, False, claims, trace)
        ans = " ".join(f"{c['text'].rstrip('.')}. " + "".join(f"[{x}]" for x in c["citations"]) for c in good)
        dropped = len(claims) - len(good)
        if dropped:
            ans += f"\n\n({dropped} unverified claim(s) omitted.)"
        sources = sorted({chunks[x]["source"] for c in good for x in c["citations"]})
        ans += "\n\nSources: " + ", ".join(sources)
        return Result(question, ans, True, claims, trace)
