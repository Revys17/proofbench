import pytest

from src.config import KNOWN_MODELS, ModelConfig
from main import resolve_model, build_config, parse_args


class TestResolveModel:
    """Tests for model key/ID resolution."""

    def test_known_key(self):
        model = resolve_model("opus")
        assert model.model_id == KNOWN_MODELS["opus"].model_id

    def test_full_model_id(self):
        model = resolve_model("claude-opus-4-6")
        assert model.model_id == "claude-opus-4-6"
        assert model.display_name == "claude-opus-4-6"

    def test_unknown_id_creates_config(self):
        model = resolve_model("some-future-model")
        assert model.model_id == "some-future-model"
        assert model.display_name == "some-future-model"
        assert model.provider == "anthropic"  # default provider


class TestBuildConfig:
    """Tests for CLI args -> config construction."""

    def _make_args(self, **overrides):
        """Create a namespace mimicking parsed CLI args."""
        defaults = {
            "generator_models": ["opus"],
            "solver_models": ["opus", "sonnet", "haiku"],
            "rounds": 3,
            "solver_max_calls": 10,
            "generator_max_calls": 20,
            "attempts_during_loop": 3,
            "attempts_reeval": 10,
            "lean_timeout": 120,
            "lean_project_path": "lean_solver",
            "output_dir": "results",
            "seed": None,
            "max_concurrent_api": 20,
            "summarize_rounds": False,
            "prompt_level": "standard",
        }
        defaults.update(overrides)

        import argparse
        return argparse.Namespace(**defaults)

    def test_generator_in_solver_list(self):
        """Generator model must always appear in solver list."""
        args = self._make_args(
            generator_models=["opus"],
            solver_models=["sonnet", "haiku"],
        )
        generators, config = build_config(args)
        solver_ids = {sm.model_id for sm in config.solver_models}
        assert generators[0].model_id in solver_ids

    def test_deduplication(self):
        """Duplicate model keys should be collapsed."""
        args = self._make_args(
            generator_models=["opus", "opus"],
            solver_models=["sonnet", "sonnet", "haiku"],
        )
        generators, config = build_config(args)
        assert len(generators) == 1
        solver_ids = [sm.model_id for sm in config.solver_models]
        assert len(solver_ids) == len(set(solver_ids))

    def test_multiple_generators(self):
        args = self._make_args(generator_models=["opus", "sonnet"])
        generators, config = build_config(args)
        assert len(generators) == 2
        gen_ids = {g.model_id for g in generators}
        assert KNOWN_MODELS["opus"].model_id in gen_ids
        assert KNOWN_MODELS["sonnet"].model_id in gen_ids

    def test_config_values_match_args(self):
        args = self._make_args(rounds=7, solver_max_calls=15)
        _, config = build_config(args)
        assert config.rounds == 7
        assert config.solver_max_calls == 15


class TestParseArgs:
    """Tests for CLI argument parsing defaults."""

    def test_default_preset(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py"])
        args = parse_args()
        assert args.rounds == 3
        assert args.attempts_during_loop == 3
        assert args.attempts_reeval == 10

    def test_full_run_preset(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--full-run"])
        args = parse_args()
        assert args.rounds == 10
        assert args.attempts_during_loop == 10
        assert args.attempts_reeval == 30

    def test_explicit_overrides_full_run(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--full-run", "--rounds", "5"])
        args = parse_args()
        assert args.rounds == 5  # explicit wins
        assert args.attempts_reeval == 30  # full-run default
