import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER") or ("openai" if os.getenv("OPENAI_API_KEY") else "grok-cli"))
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.x.ai/v1"))
    api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    model: str = field(default_factory=lambda: os.getenv("MODEL", "grok-4"))
    grok_bin: str = field(default_factory=lambda: os.getenv("GROK_BIN") or (str(Path.home() / ".local/bin/grok") if (Path.home() / ".local/bin/grok").exists() else "grok"))
    grok_timeout: int = field(default_factory=lambda: int(os.getenv("GROK_TIMEOUT", "300")))
    index_dir: Path = field(default_factory=lambda: Path(os.getenv("INDEX_DIR", ROOT / "index")))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 4
    max_tool_steps: int = 3
    max_verify_rounds: int = 1
