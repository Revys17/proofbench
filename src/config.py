from dataclasses import dataclass
from enum import Enum


class PromptLevel(Enum):
    """Controls how much strategic guidance the generator receives."""
    MINIMAL = "minimal"    # Scoring formula + tool/format only
    STANDARD = "standard"  # + scoring details (reliability matters)
    DETAILED = "detailed"  # + full strategy section with prescriptive advice


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    display_name: str
    provider: str = "anthropic"  # "anthropic" or "openai"
    max_tokens: int = 16384


@dataclass(frozen=True)
class EvalConfig:
    generator_model: ModelConfig
    solver_models: tuple[ModelConfig, ...]
    rounds: int = 3
    solver_max_calls: int = 10
    generator_max_calls: int = 20
    attempts_during_loop: int = 3
    attempts_reeval: int = 3
    lean_timeout_seconds: int = 120
    lean_project_path: str = "lean_solver"
    output_dir: str = "results"
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    seed: int | None = None
    max_concurrent_api: int = 20
    summarize_rounds: bool = False
    prompt_level: PromptLevel = PromptLevel.STANDARD


KNOWN_MODELS: dict[str, ModelConfig] = {
    # Anthropic
    "opus": ModelConfig("claude-opus-4-6", "opus-4.6"),
    "opus-4.5": ModelConfig("claude-opus-4-5-20251101", "opus-4.5"),
    "opus-4.1": ModelConfig("claude-opus-4-1-20250805", "opus-4.1"),
    "opus-4": ModelConfig("claude-opus-4-20250514", "opus-4"),
    "sonnet": ModelConfig("claude-sonnet-4-6", "sonnet-4.6"),
    "sonnet-4.5": ModelConfig("claude-sonnet-4-5-20250929", "sonnet-4.5"),
    "sonnet-4": ModelConfig("claude-sonnet-4-20250514", "sonnet-4"),
    "haiku": ModelConfig("claude-haiku-4-5-20251001", "haiku-4.5"),
    # OpenAI
    # Note - I don't have a working OpenAI API key atm so these are not supported - should be fairly easy to add though if necessary.
    # "gpt-5.4": ModelConfig("gpt-5.4", "gpt-5.4", provider="openai"),
    # "gpt-5.4-mini": ModelConfig("gpt-5.4-mini", "gpt-5.4-mini", provider="openai"),
    # "gpt-5-mini": ModelConfig("gpt-5-mini", "gpt-5-mini", provider="openai"),
    # "gpt-5": ModelConfig("gpt-5", "gpt-5", provider="openai"),
    # "gpt-5.4-pro": ModelConfig("gpt-5.4-pro", "gpt-5.4-pro", provider="openai"),
    # "gpt-5.4-nano": ModelConfig("gpt-5.4-nano", "gpt-5.4-nano", provider="openai"),
    # "gpt-5-nano": ModelConfig("gpt-5-nano", "gpt-5-nano", provider="openai"),
    # "gpt-4.1": ModelConfig("gpt-4.1", "gpt-4.1", provider="openai"),
}

DEFAULT_GENERATORS = ("opus",)
DEFAULT_SOLVERS = ("opus", "sonnet", "haiku")
