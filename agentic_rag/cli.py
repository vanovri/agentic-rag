import argparse
import json
from pathlib import Path

from .config import Settings


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agentic-rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest"); i.add_argument("paths", nargs="+", type=Path)
    q = sub.add_parser("ask"); q.add_argument("question")
    q.add_argument("--provider", choices=["openai", "grok-cli"]); q.add_argument("--trace", action="store_true")
    q.add_argument("--json", action="store_true")
    q.add_argument("--brief", action="store_true", help="compact JSON: answer, found, claims, tools used")
    a = ap.parse_args(argv)
    s = Settings()
    if a.cmd == "ingest":
        from .ingest import ingest
        print(f"Indexed {ingest(a.paths, s)} chunks into {s.index_dir}")
        return
    if a.provider:
        s.provider = a.provider
    from .agent import Agent
    from .llm import make_llm
    from .tools import Retriever
    res = Agent(make_llm(s), Retriever(s), s).run(a.question)
    if a.brief:
        print(json.dumps({"found": res.found, "answer": res.answer,
                          "claims": [{k: c.get(k) for k in ("text", "citations", "supported")} for c in res.claims],
                          "tools": [f"{t['tool']}({t['input']})" for t in res.trace if t["step"] in ("tool", "re-retrieve")]}, indent=1)); return
    if a.json:
        print(json.dumps(res.__dict__, indent=1, default=str)); return
    if a.trace:
        for t in res.trace:
            if t["step"] in ("tool", "re-retrieve", "plan"):
                print(f"[{t['step']}] {t['tool']}({t['input']!r}) -> {t['summary']}")
            elif t["step"] == "verify":
                for v in t["verdicts"]:
                    print(f"[verify r{t['round']}] {'OK ' if v['supported'] else 'BAD'} {v['claim']} -- {v['reason']}")
        print("-" * 60)
    print(res.answer)


if __name__ == "__main__":
    main()
