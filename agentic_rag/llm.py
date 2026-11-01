"""LLM providers: OpenAI-compatible API, headless Grok CLI, and a mock for tests."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from typing import Callable, Protocol

from .config import Settings


class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...


class OpenAICompatLLM:
    def __init__(self, s: Settings):
        from openai import OpenAI
        self.client = OpenAI(base_url=s.base_url, api_key=s.api_key)
        self.model = s.model

    def complete(self, prompt: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""


class GrokCLILLM:
    """Shells out to the logged-in Grok CLI (uses the SuperGrok subscription, no API key)."""

    def __init__(self, s: Settings):
        self.bin, self.timeout = s.grok_bin, s.grok_timeout

    def complete(self, prompt: str) -> str:
        prompt = ("You are a pure text-completion backend. Do NOT use any tools, files or shell. "
                  "Answer only from the text below.\n\n" + prompt)
        with tempfile.TemporaryDirectory() as cwd:  # empty cwd: nothing for the CLI to read
            p = subprocess.run([self.bin, "--no-auto-update", "-p", prompt, "--output-format", "json"],
                               cwd=cwd, capture_output=True, text=True, timeout=self.timeout)
        if p.returncode != 0:
            raise RuntimeError(f"grok CLI failed ({p.returncode}): {p.stderr[-500:]}")
        try:
            return json.loads(p.stdout)["text"]
        except (json.JSONDecodeError, KeyError):
            return p.stdout


class MockLLM:
    """Deterministic LLM for tests: a function or list of canned responses."""

    def __init__(self, responder: Callable[[str], str] | list[str]):
        self.responder, self.prompts = responder, []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if callable(self.responder):
            return self.responder(prompt)
        return self.responder.pop(0)


def make_llm(s: Settings | None = None) -> LLM:
    s = s or Settings()
    if s.provider == "openai":
        return OpenAICompatLLM(s)
    if s.provider == "grok-cli":
        return GrokCLILLM(s)
    raise ValueError(f"unknown provider {s.provider}")


def parse_json(text: str) -> dict:
    """Extract the first JSON object from an LLM reply (tolerates code fences / prose)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON in: {text[:200]}")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            return json.loads(text[start:i + 1])
    raise ValueError(f"unbalanced JSON in: {text[:200]}")
