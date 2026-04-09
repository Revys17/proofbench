import pytest

from src.config import EvalConfig, ModelConfig
from src.generator import RoundResult, SolveResult, TheoremSubmission
from src.orchestrator import _build_anon_map, _score_round_results


class TestBuildAnonMap:
    """Tests for model anonymization."""

    def test_self_is_model_self(self):
        gen = ModelConfig("claude-opus-4-6", "opus")
        solvers = (
            ModelConfig("claude-opus-4-6", "opus"),
            ModelConfig("claude-sonnet-4-6", "sonnet"),
        )
        anon = _build_anon_map(gen, solvers)
        assert anon["claude-opus-4-6"] == "model_self"

    def test_opponents_get_letter_names(self):
        gen = ModelConfig("claude-opus-4-6", "opus")
        solvers = (
            ModelConfig("claude-opus-4-6", "opus"),
            ModelConfig("claude-sonnet-4-6", "sonnet"),
            ModelConfig("claude-haiku-4-5-20251001", "haiku"),
        )
        anon = _build_anon_map(gen, solvers)
        opponent_names = {v for k, v in anon.items() if v != "model_self"}
        assert opponent_names == {"model_a", "model_b"}

    def test_all_solvers_mapped(self):
        gen = ModelConfig("gen", "gen")
        solvers = (
            ModelConfig("gen", "gen"),
            ModelConfig("s1", "s1"),
            ModelConfig("s2", "s2"),
        )
        anon = _build_anon_map(gen, solvers)
        for sm in solvers:
            assert sm.model_id in anon

    def test_no_duplicates_in_values(self):
        gen = ModelConfig("gen", "gen")
        solvers = (
            ModelConfig("gen", "gen"),
            ModelConfig("s1", "s1"),
            ModelConfig("s2", "s2"),
            ModelConfig("s3", "s3"),
        )
        anon = _build_anon_map(gen, solvers)
        values = list(anon.values())
        assert len(values) == len(set(values))

    def test_generator_not_in_solvers(self):
        """Generator is always mapped as model_self, even if not in solver list."""
        gen = ModelConfig("gen", "gen")
        solvers = (ModelConfig("s1", "s1"),)
        anon = _build_anon_map(gen, solvers)
        assert anon["gen"] == "model_self"
        assert anon["s1"] == "model_a"


class TestScoreRoundResults:
    """Tests for scoring round results from loop data."""

    def _make_round_result(
        self,
        round_number: int,
        self_succ: int,
        self_att: int,
        opp_results: dict[str, tuple[int, int]],
    ) -> RoundResult:
        solve_results = [
            SolveResult("model_self", self_succ > 0, self_succ, self_att),
        ]
        for name, (succ, att) in opp_results.items():
            solve_results.append(SolveResult(name, succ > 0, succ, att))
        return RoundResult(
            round_number=round_number,
            submission=TheoremSubmission("thm", "proof", "import M", round_number),
            solve_results=tuple(solve_results),
        )

    def _make_config(self, **overrides) -> EvalConfig:
        defaults = dict(
            generator_model=ModelConfig("gen", "gen"),
            solver_models=(ModelConfig("gen", "gen"), ModelConfig("s1", "s1")),
        )
        defaults.update(overrides)
        return EvalConfig(**defaults)

    def test_single_round(self):
        rr = self._make_round_result(1, 8, 10, {"model_a": (2, 10)})
        config = self._make_config(seed=42)
        scores = _score_round_results([rr], config)
        assert len(scores) == 1
        assert scores[0].theorem_id == "round_1"
        assert scores[0].gap_score > 0.5

    def test_multiple_rounds_scored_independently(self):
        rr1 = self._make_round_result(1, 10, 10, {"model_a": (0, 10)})
        rr2 = self._make_round_result(2, 0, 10, {"model_a": (10, 10)})
        config = self._make_config(seed=42)
        scores = _score_round_results([rr1, rr2], config)
        assert len(scores) == 2
        assert scores[0].gap_score > scores[1].gap_score

    def test_empty_rounds(self):
        config = self._make_config()
        scores = _score_round_results([], config)
        assert scores == []

    def test_uses_successes_not_solved_flag(self):
        """Scoring should use successes/attempts, not the boolean solved flag."""
        # solved=True but only 1/10 successes — gap should be weak
        rr = self._make_round_result(1, 1, 10, {"model_a": (0, 10)})
        config = self._make_config(seed=42)
        scores = _score_round_results([rr], config)
        # With Beta(2,10) vs Beta(1,11), gap should be modest
        assert scores[0].gap_score < 0.9
