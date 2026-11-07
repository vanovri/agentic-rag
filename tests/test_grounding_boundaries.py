"""Offline grounding, AST calculator, ingestion, and CLI boundary tests."""
import json
from unittest.mock import patch

import pytest

from agentic_rag import cli
from agentic_rag.agent import Agent
from agentic_rag.config import Settings
from agentic_rag.ingest import ingest
from agentic_rag.llm import MockLLM, make_llm, parse_json
from agentic_rag.tools import Retriever, calculator


@pytest.mark.parametrize(
    "expression",
    ["(1).__class__", "[1, 2]", "sum([1, 2])", '"a" + "b"', "open('x')"],
)
def test_calculator_rejects_non_arithmetic_ast(expression):
    with pytest.raises(ValueError):
        calculator(expression)


def test_calculator_supports_unary_ops_constants_and_math_functions():
    assert calculator("-3 + +5") == 2
    assert calculator("log10(100) + min(4, max(1, 2))") == 4.0
    assert calculator("round(pi, 2)") == 3.14


def test_invalid_provider_response_is_retried_once(retriever):
    retrieval, settings = retriever
    llm = MockLLM(["not json at all", '{"action":"answer"}'])
    assert Agent(llm, retrieval, settings)._ask("plan") == {"action": "answer"}
    assert len(llm.prompts) == 2
    assert "Previous reply was invalid" in llm.prompts[1]


def test_calculator_runtime_error_is_recorded_in_trace_without_aborting(retriever):
    retrieval, settings = retriever
    settings.max_verify_rounds = 0
    cid = retrieval.vector_search("biochar")[0]["id"]
    llm = MockLLM(
        [
            '{"action":"calculator","input":"1 / 0"}',
            '{"action":"answer"}',
            json.dumps({"claims": [{"text": "Grounded statement", "citations": [cid]}]}),
            '{"verdicts":[{"index":0,"supported":true,"reason":"supported"}]}',
        ]
    )
    result = Agent(llm, retrieval, settings).run("biochar")
    assert result.found
    calc_steps = [s for s in result.trace if s.get("tool") == "calculator"]
    assert len(calc_steps) == 1
    assert "error:" in calc_steps[0]["summary"]


def test_fallback_retrieval_keeps_only_retrieved_citations(retriever):
    retrieval, settings = retriever
    settings.max_tool_steps = 1
    settings.max_verify_rounds = 0
    hits = retrieval.vector_search("biochar")
    citation = hits[0]["id"]
    llm = MockLLM(
        [
            '{"action":"answer"}',
            json.dumps(
                {"claims": [{"text": "Grounded fixture claim", "citations": ["invented#9", citation]}]}
            ),
            '{"verdicts":[{"index":0,"supported":true,"reason":"fixture"}]}',
        ]
    )
    result = Agent(llm, retrieval, settings).run("biochar")
    assert result.found
    assert result.claims[0]["citations"] == [citation]
    assert "invented#9" not in result.answer
    assert "Sources:" in result.answer
    assert any(step.get("thought") == "fallback retrieval" for step in result.trace)


def test_ingest_filters_extensions_skips_blank_files_and_rejects_empty_corpus(tmp_path):
    from conftest import HashEmbedder

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "empty.txt").write_text("   \n  ", encoding="utf-8")
    (docs / "ignored.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    settings = Settings(index_dir=tmp_path / "idx")
    with pytest.raises(ValueError, match="no documents found"):
        ingest([docs], settings, embedder=HashEmbedder())

    (docs / "note.md").write_text("Biochar stores stable pyrogenic carbon in soil.", encoding="utf-8")
    (docs / "other1.txt").write_text("Solar panels generate electricity on sunny days.", encoding="utf-8")
    (docs / "other2.txt").write_text("Wind turbines rotate blades across coastal ridges.", encoding="utf-8")
    count = ingest([docs], settings, embedder=HashEmbedder())
    assert count == 3
    loaded = Retriever(settings, embedder=HashEmbedder())
    hits = loaded.bm25_search("pyrogenic carbon")
    assert hits and hits[0]["id"] == "note#0"


def test_cli_ingest_and_ask_brief_json_offline(tmp_path, capsys, monkeypatch):
    from conftest import HashEmbedder

    docs = tmp_path / "corpus"
    docs.mkdir()
    (docs / "guide.txt").write_text("Pyrolysis converts biomass into stable biochar.", encoding="utf-8")
    (docs / "other1.txt").write_text("Hydroelectric dams regulate river water reservoirs.", encoding="utf-8")
    (docs / "other2.txt").write_text("Geothermal wells extract subsurface steam heat.", encoding="utf-8")
    idx_dir = tmp_path / "index"
    monkeypatch.setenv("INDEX_DIR", str(idx_dir))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("agentic_rag.ingest.Embedder", lambda _model: HashEmbedder()):
        cli.main(["ingest", str(docs)])
    ingest_out = capsys.readouterr().out
    assert "Indexed 3 chunks" in ingest_out

    mock_llm = MockLLM(
        [
            '{"action":"bm25_search","input":"pyrolysis"}',
            '{"action":"answer"}',
            '{"claims":[{"text":"Pyrolysis converts biomass into stable biochar","citations":["guide#0"]}]}',
            '{"verdicts":[{"index":0,"supported":true,"reason":"exact match"}]}',
        ]
    )
    with (
        patch("agentic_rag.llm.make_llm", return_value=mock_llm),
        patch("agentic_rag.ingest.Embedder", lambda _model: HashEmbedder()),
    ):
        cli.main(["ask", "What does pyrolysis do?", "--brief"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["found"] is True
    assert payload["claims"][0]["citations"] == ["guide#0"]
    assert payload["tools"] == ["bm25_search(pyrolysis)"]


def test_make_llm_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        make_llm(Settings(provider="unsupported-provider"))


@pytest.mark.xfail(
    strict=True,
    reason="Current parse_json counts braces inside quoted strings; TECH_DEBT.md records raw_decode fix.",
)
def test_quoted_closing_brace_is_not_an_object_delimiter():
    assert parse_json('{"text":"}"}') == {"text": "}"}


def test_string_false_supported_verdict_is_rejected(retriever):
    retrieval, settings = retriever
    settings.max_verify_rounds = 0
    cid = retrieval.vector_search("biochar")[0]["id"]
    llm = MockLLM(
        [
            '{"action":"answer"}',
            json.dumps({"claims": [{"text": "Hallucinated claim", "citations": [cid]}]}),
            '{"verdicts":[{"index":0,"supported":"false","reason":"contradicted by source"}]}',
        ]
    )
    result = Agent(llm, retrieval, settings).run("biochar")
    assert result.found is False
    assert result.claims[0]["supported"] is False
