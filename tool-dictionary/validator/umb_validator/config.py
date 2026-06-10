"""Config loading + the seed corpus. SPEC v2 §12 / §2.

The shipped `config.toml` carries the four §13 recommended defaults baked in;
this module parses it into typed pydantic models with the same defaults so the
harness is usable even without an on-disk config (tests, dry-runs).
"""

from __future__ import annotations

import math
import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

# --- SPEC §2: the curated 30-server seed corpus -----------------------------
# 15 already-shipped seed TOMLs + 15 expansion candidates.
SEED_SERVERS_SHIPPED: list[str] = [
    "brave-search", "fetch", "filesystem", "gdrive", "github-actions",
    "github", "gitlab", "memory", "playwright", "postgres", "puppeteer",
    "sequential-thinking", "slack", "sqlite", "time",
]
SEED_SERVERS_EXPANSION: list[str] = [
    "aws", "notion", "linear", "jira", "discord", "mongodb", "redis",
    "elasticsearch", "kubernetes", "terraform", "chroma", "weaviate",
    "qdrant", "obsidian", "everart",
]
SEED_SERVERS: list[str] = SEED_SERVERS_SHIPPED + SEED_SERVERS_EXPANSION


class GatewayConfig(BaseModel):
    # Your OpenAI-compatible gateway (e.g. vLLM, an AI gateway) — override in
    # config.toml for your deployment.
    base_url: str = "http://localhost:8000/v1"
    discover_models: bool = True
    model_exclude: list[str] = Field(default_factory=list)
    model_pin: list[str] = Field(default_factory=list)
    generator_model: str = "qwen3.5-35b-a3b"
    # Extra OpenAI-compatible request-body params merged into every chat
    # completion. The default disables thinking/reasoning mode: a
    # thinking-capable model (e.g. a qwen3.5-122b) otherwise reasons
    # its way out of `tool_choice="required"` and returns `tool_calls=[]`,
    # making it unusable as a gate/jury model. Sending `enable_thinking=false`
    # to a model whose chat template lacks the variable is harmless — Jinja
    # ignores unused context vars. Set to {} to disable, or override per
    # deployment if a backend needs different kwargs.
    chat_template_kwargs: dict[str, object] = Field(
        default_factory=lambda: dict[str, object](enable_thinking=False)
    )


class JuryConfig(BaseModel):
    # DECISION #1: strict 3-of-4 quorum.
    n: int = 4
    quorum_q: int = 3
    models: list[str] = Field(
        default_factory=lambda: ["qwen3.5-35b-a3b", "glm-4.7-flash", "qwen3.5-4b"]
    )


class OracleConfig(BaseModel):
    gen_positive: int = 45
    gen_negative: int = 30
    oracle_min_positive: int = 25
    oracle_min_negative: int = 15
    max_gen_rounds: int = 2
    low_conf_agreement: float = 0.85


class GatesConfig(BaseModel):
    local_pp_tolerance: float = 3.0
    local_k_fraction: float = 0.75  # DECISION #3
    cloud_rel_floor: float = 0.95
    token_reduction_min: float = 50.0
    positive_weight: float = 0.6
    negative_weight: float = 0.4
    shortener_max_retries: int = 5
    cloud_shortener_max_retries: int = 3

    def local_k(self, n: int) -> int:
        """K = ceil(local_k_fraction * N), clamped to [1, N]."""
        if n <= 0:
            return 0
        return max(1, min(n, math.ceil(self.local_k_fraction * n)))

    def min_gateable_local(self, roster_size: int) -> int:
        """Minimum local models that must support forced tool-use for the
        local gate to be VALID (SPEC §2 / §13 K-of-N anti-fabrication rule).

        A LOCAL_GATE_PASS requires K genuine per-model passes, where K is
        computed from the FULL configured roster — NOT from the shrunken
        subset of models that happened to accept forced tool-use. If fewer
        models than K can be gated (some/all backends 400 on forced
        tool-use), a pass is impossible without fabricating it from an
        under-strength jury, so the tool MUST go NEEDS_MANUAL instead.
        """
        return self.local_k(roster_size)


class SessionPoolConfig(BaseModel):
    # backend_a/b/c are opaque per-backend pool keys (see
    # integration/gateway.py); size each cap to the matching inference
    # host's session ceiling.
    max_inflight_local: int = 24
    backend_a: int = 14
    backend_b: int = 18
    backend_c: int = 6
    max_concurrent_tools: int = 4
    max_concurrent_umb_dev: int = 2


class BudgetConfig(BaseModel):
    max_run_wall_seconds: int = 39600
    max_daily_cost_usd: float = 5.00
    max_run_cost_usd: float = 2.00
    cloud_rps: int = 2


class ResearchConfig(BaseModel):
    max_web_docs_per_tool: int = 3
    max_artifact_chars: int = 8000
    distractor_window: int = 8  # DECISION #4: candidate + 7 distractors
    github_api_base: str = "https://api.github.com"


class CloudConfig(BaseModel):
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4o-mini"


class PathsConfig(BaseModel):
    state_dir: str = "~/.umb-validator"
    umb_dev_bin: str = "/usr/local/bin/umb-dev"
    pending_dir: str = ""
    tool_dictionary_dir: str = ""

    def state_path(self) -> Path:
        return Path(os.path.expanduser(self.state_dir))


class NtfyConfig(BaseModel):
    host: str = "https://ntfy.universalmcpbridge.app"
    log_topic: str = ""
    alert_topic: str = ""


class Config(BaseModel):
    """Top-level harness config — mirrors config.toml's section layout."""

    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    jury: JuryConfig = Field(default_factory=JuryConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    gates: GatesConfig = Field(default_factory=GatesConfig)
    session_pool: SessionPoolConfig = Field(default_factory=SessionPoolConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ntfy: NtfyConfig = Field(default_factory=NtfyConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load config from a TOML file. Missing file -> all-defaults config
        (the four §13 recommended values). A partial file overlays defaults
        section-by-section."""
        if path is None:
            for cand in (
                "/etc/umb-validator/config.toml",
                str(Path(__file__).resolve().parents[1] / "config.toml"),
            ):
                if Path(cand).is_file():
                    path = cand
                    break
        if path is None or not Path(path).is_file():
            return cls()
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        return cls.model_validate(raw)

    def resolve_dict_dir(self) -> Path:
        """The live `tool-dictionary/` directory."""
        if self.paths.tool_dictionary_dir:
            return Path(os.path.expanduser(self.paths.tool_dictionary_dir))
        # validator/ lives INSIDE tool-dictionary/ -> parent is the dict dir.
        return Path(__file__).resolve().parents[2]

    def resolve_pending_dir(self) -> Path:
        """The `_pending/` directory the harness writes proposals to."""
        if self.paths.pending_dir:
            return Path(os.path.expanduser(self.paths.pending_dir))
        return self.resolve_dict_dir() / "_pending"
