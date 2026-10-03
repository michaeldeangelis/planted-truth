"""Model backends. All return (text, usage) for a list of chat messages."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import requests


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    usd: float = 0.0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.usd += other.usd


class Backend:
    name = "base"

    def chat(self, system: str, messages: list[dict]) -> tuple[str, Usage]:
        raise NotImplementedError


@dataclass
class MockBackend(Backend):
    """Replays a fixed script of responses. Used by tests."""
    script: list[str] = field(default_factory=list)
    name: str = "mock"

    def chat(self, system, messages):
        i = sum(1 for m in messages if m["role"] == "assistant")
        return (self.script[i] if i < len(self.script) else "DONE"), Usage()


@dataclass
class OllamaBackend(Backend):
    model: str = "qwen2.5-coder:7b"
    seed: int = 0
    temperature: float = 0.7
    num_ctx: int = 12288
    host: str = "http://localhost:11434"

    @property
    def name(self):
        return f"ollama:{self.model}"

    def chat(self, system, messages):
        r = requests.post(f"{self.host}/api/chat", timeout=600, json={
            "model": self.model, "stream": False,
            "messages": [{"role": "system", "content": system}] + messages,
            "options": {"seed": self.seed, "temperature": self.temperature, "num_ctx": self.num_ctx},
        })
        r.raise_for_status()
        j = r.json()
        return j["message"]["content"], Usage(j.get("prompt_eval_count", 0), j.get("eval_count", 0))


class BudgetError(RuntimeError):
    pass


@dataclass
class AnthropicBackend(Backend):
    """Paid backend. Refuses to run without an approved budget id and enforces a hard USD cap.

    Prices are per million tokens and must be set from the current pricing page.
    """
    model: str
    price_in: float
    price_out: float
    price_cache_read: float
    cap_usd: float
    spent: Usage = field(default_factory=Usage)
    temperature: float = 0.7
    max_tokens: int = 2048

    def __post_init__(self):
        if not os.environ.get("OCB_BUDGET_APPROVED"):
            raise BudgetError("Set OCB_BUDGET_APPROVED=<approved budget id> to use a paid backend.")
        import anthropic
        self.client = anthropic.Anthropic()

    @property
    def name(self):
        return f"anthropic:{self.model}"

    def chat(self, system, messages):
        if self.spent.usd >= self.cap_usd:
            raise BudgetError(f"cap ${self.cap_usd} reached (${self.spent.usd:.2f})")
        msgs = [dict(m) for m in messages]
        # Cache the growing prefix: mark the last message as a cache breakpoint.
        msgs[-1] = {"role": msgs[-1]["role"], "content": [
            {"type": "text", "text": msgs[-1]["content"], "cache_control": {"type": "ephemeral"}}]}
        r = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=msgs)
        u = r.usage
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        usd = ((u.input_tokens + 1.25 * cache_write) * self.price_in
               + cache_read * self.price_cache_read + u.output_tokens * self.price_out) / 1e6
        usage = Usage(u.input_tokens + cache_write, u.output_tokens, cache_read, usd)
        self.spent.add(usage)
        return "".join(b.text for b in r.content if b.type == "text"), usage
