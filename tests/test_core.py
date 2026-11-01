import json

import pytest

from agentic_rag.agent import NOT_FOUND, Agent
from agentic_rag.ingest import chunk_text
from agentic_rag.llm import MockLLM, parse_json
from agentic_rag.tools import calculator


def test_chunking_overlap_and_size():
    text = " ".join(f"Sentence number {i} about biochar." for i in range(100))
    ch = chunk_text(text, size=200, overlap=40)
    assert len(ch) > 5 and all(len(c) <= 300 for c in ch)


def test_calculator():
    assert calculator("44/12 * 3") == pytest.approx(11.0)
    assert calculator("sqrt(16) + 2^3") == 12
    with pytest.raises(ValueError):
        calculator("__import__('os')")


def test_parse_json_tolerates_fences():
    assert parse_json('Sure:\n```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}


def test_retrieval(retriever):
    r, _ = retriever
    assert any("03_biochar_permanence" in h["id"] for h in r.bm25_search("H/Corg ratio"))
    assert r.vector_search("pyrolysis temperature")


def _id(r, needle):
    return next(c["id"] for c in r.chunks if needle in c["text"])


def test_agent_happy_path(retriever):
    r, s = retriever
    cid = _id(r, "0.7")
    llm = MockLLM([
        json.dumps({"thought": "keyword", "action": "bm25_search", "input": "H/Corg ratio"}),
        json.dumps({"action": "answer", "input": ""}),
        json.dumps({"claims": [{"text": "Standards often require H/Corg below 0.7", "citations": [cid]}]}),
        json.dumps({"verdicts": [{"index": 0, "supported": True, "reason": "stated"}]}),
    ])
    res = Agent(llm, r, s).run("What H/Corg ratio is required?")
    assert res.found and f"[{cid}]" in res.answer
    assert [t["tool"] for t in res.trace if t["step"] == "tool"] == ["bm25_search"]


def test_agent_reretrieves_then_not_found(retriever):
    r, s = retriever
    llm = MockLLM([
        json.dumps({"action": "vector_search", "input": "price of biochar"}),
        json.dumps({"action": "answer"}),
        json.dumps({"claims": [{"text": "Biochar costs $5 per tonne", "citations": ["bogus#0"]}]}),
        json.dumps({"claims": [{"text": "Biochar costs $5 per tonne", "citations": [r.chunks[0]["id"]]}]}),
        json.dumps({"verdicts": [{"index": 0, "supported": False, "reason": "price not mentioned"}]}),
    ])
    res = Agent(llm, r, s).run("What is the price of biochar?")
    assert not res.found and res.answer == NOT_FOUND
    assert any(t["step"] == "re-retrieve" for t in res.trace)


def test_agent_calculator_and_partial_drop(retriever):
    r, s = retriever
    cid = _id(r, "3.67")
    llm = MockLLM([
        json.dumps({"action": "calculator", "input": "2 * 44/12"}),
        json.dumps({"action": "bm25_search", "input": "CO2-equivalent 44/12"}),
        json.dumps({"action": "answer"}),
        json.dumps({"claims": [{"text": "Multiply carbon by 44/12", "citations": [cid]},
                               {"text": "Biochar is blue", "citations": [cid]}]}),
        json.dumps({"verdicts": [{"index": 0, "supported": True}, {"index": 1, "supported": False}]}),
        json.dumps({"claims": [{"text": "Multiply carbon by 44/12", "citations": [cid]}]}),
        json.dumps({"verdicts": [{"index": 0, "supported": True, "reason": "ok"}]}),
    ])
    res = Agent(llm, r, s).run("2 t carbon is how much CO2?")
    assert res.found and "blue" not in res.answer
    assert any(t["tool"] == "calculator" and t["summary"].startswith("7.33") for t in res.trace if t["step"] == "tool")
