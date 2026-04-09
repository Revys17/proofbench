import pytest

from src.config import (
    EvalConfig,
    ModelConfig,
    PromptLevel,
    KNOWN_MODELS,
    DEFAULT_GENERATORS,
    DEFAULT_SOLVERS,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_defaults(self):
        m = ModelConfig("test-model", "test")
        assert m.provider == "anthropic"
        assert m.max_tokens == 16384

    def test_custom_provider(self):
        m = ModelConfig("gpt-5", "gpt-5", provider="openai")
        assert m.provider == "openai"

    def test_frozen(self):
        m = ModelConfig("test", "test")
        with pytest.raises(AttributeError):
            m.model_id = "other"  # type: ignore[misc]


class TestEvalConfig:
    """Tests for EvalConfig defaults and constraints."""

    def _minimal_config(self, **overrides):
        defaults = dict(
            generator_model=ModelConfig("gen", "gen"),
            solver_models=(ModelConfig("s1", "s1"),),
        )
        defaults.update(overrides)
        return EvalConfig(**defaults)

    def test_defaults(self):
        cfg = self._minimal_config()
        assert cfg.rounds == 3
        assert cfg.solver_max_calls == 10
        assert cfg.generator_max_calls == 20
        assert cfg.attempts_during_loop == 3
        assert cfg.attempts_reeval == 3
        assert cfg.lean_timeout_seconds == 120
        assert cfg.prior_alpha == 1.0
        assert cfg.prior_beta == 1.0
        assert cfg.prompt_level == PromptLevel.STANDARD

    def test_frozen(self):
        cfg = self._minimal_config()
        with pytest.raises(AttributeError):
            cfg.rounds = 10  # type: ignore[misc]

    def test_custom_values(self):
        cfg = self._minimal_config(
            rounds=10,
            solver_max_calls=20,
            prompt_level=PromptLevel.DETAILED,
        )
        assert cfg.rounds == 10
        assert cfg.solver_max_calls == 20
        assert cfg.prompt_level == PromptLevel.DETAILED


class TestKnownModels:
    """Tests for the model registry."""

    def test_default_generators_in_registry(self):
        for key in DEFAULT_GENERATORS:
            assert key in KNOWN_MODELS, f"Default generator '{key}' not in KNOWN_MODELS"

    def test_default_solvers_in_registry(self):
        for key in DEFAULT_SOLVERS:
            assert key in KNOWN_MODELS, f"Default solver '{key}' not in KNOWN_MODELS"

    def test_all_models_have_required_fields(self):
        for key, model in KNOWN_MODELS.items():
            assert model.model_id, f"{key} missing model_id"
            assert model.display_name, f"{key} missing display_name"
            assert model.provider in ("anthropic", "openai"), f"{key} unknown provider"

    def test_no_duplicate_model_ids(self):
        ids = [m.model_id for m in KNOWN_MODELS.values()]
        assert len(ids) == len(set(ids)), "Duplicate model_ids in KNOWN_MODELS"


class TestPromptLevel:
    """Tests for PromptLevel enum."""

    def test_values(self):
        assert PromptLevel.MINIMAL.value == "minimal"
        assert PromptLevel.STANDARD.value == "standard"
        assert PromptLevel.DETAILED.value == "detailed"

    def test_all_levels_count(self):
        assert len(PromptLevel) == 3
