import logging
import threading
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Pricing per million tokens (input, output) as of 2026-04
# https://docs.anthropic.com/en/docs/about-claude/models
# https://openai.com/api/pricing/
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic (per 1M tokens: input, output)
    "claude-opus-4-6":            (15.00, 75.00),
    "claude-opus-4-5-20251101":   (15.00, 75.00),
    "claude-sonnet-4-6":          (3.00,  15.00),
    "claude-sonnet-4-5-20250929": (3.00,  15.00),
    "claude-haiku-4-5-20251001":  (0.80,  4.00),
    # OpenAI (per 1M tokens: input, output)
    "gpt-4o":                     (2.50,  10.00),
    "gpt-4o-mini":                (0.15,  0.60),
    "o3":                         (10.00, 40.00),
    "o3-mini":                    (1.10,  4.40),
    "o4-mini":                    (1.10,  4.40),
}


@dataclass
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0


class CostTracker:
    """Thread-safe token and cost tracker across all API calls."""

    def __init__(self) -> None:
        self._usage: dict[str, ModelUsage] = {}
        self._lock = threading.Lock()

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            if model not in self._usage:
                self._usage[model] = ModelUsage()
            u = self._usage[model]
            u.input_tokens += input_tokens
            u.output_tokens += output_tokens
            u.api_calls += 1

    def _cost_for_model(self, model: str, usage: ModelUsage) -> float:
        input_price, output_price = PRICING.get(model, (0.0, 0.0))
        return (
            usage.input_tokens * input_price / 1_000_000
            + usage.output_tokens * output_price / 1_000_000
        )

    def total_cost(self) -> float:
        with self._lock:
            return sum(
                self._cost_for_model(model, usage)
                for model, usage in self._usage.items()
            )

    def summary(self) -> dict:
        with self._lock:
            per_model = {}
            total = 0.0
            for model, usage in sorted(self._usage.items()):
                cost = self._cost_for_model(model, usage)
                total += cost
                per_model[model] = {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "api_calls": usage.api_calls,
                    "cost_usd": round(cost, 4),
                }
            return {
                "per_model": per_model,
                "total_cost_usd": round(total, 4),
            }

    def log_summary(self) -> None:
        s = self.summary()
        log.info("=== Cost Summary ===")
        for model, info in s["per_model"].items():
            log.info(
                "  %s: %d calls, %d in / %d out tokens, $%.4f",
                model, info["api_calls"],
                info["input_tokens"], info["output_tokens"],
                info["cost_usd"],
            )
        log.info("  Total: $%.4f", s["total_cost_usd"])
