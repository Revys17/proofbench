from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    display_name: str
    max_tokens: int = 16384


@dataclass(frozen=True)
class EvalConfig:
    generator_model: ModelConfig
    solver_models: tuple[ModelConfig, ...]
    rounds: int = 3
    solver_max_calls: int = 10
    generator_max_calls: int = 20
    attempts_during_loop: int = 1
    attempts_reeval: int = 3
    lean_timeout_seconds: int = 120
    lean_project_path: str = "lean_solver"
    output_dir: str = "results"
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    seed: int | None = None


KNOWN_MODELS: dict[str, ModelConfig] = {
    "opus": ModelConfig("claude-opus-4-6", "opus-4.6"),
    "opus-4.5": ModelConfig("claude-opus-4-5-20250501", "opus-4.5"),
    "sonnet": ModelConfig("claude-sonnet-4-6", "sonnet-4.6"),
    "sonnet-4.5": ModelConfig("claude-sonnet-4-5-20250514", "sonnet-4.5"),
    "haiku": ModelConfig("claude-haiku-4-5-20251001", "haiku-4.5"),
}

DEFAULT_GENERATOR = "opus"
DEFAULT_SOLVERS = ("opus", "sonnet", "haiku")
