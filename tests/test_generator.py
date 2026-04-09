import pytest

from src.config import PromptLevel
from src.generator import (
    SolveResult,
    TheoremSubmission,
    RoundResult,
    _build_solve_feedback,
    _build_budget_exhausted_feedback,
)


def _make_solve_results(
    self_succ: int,
    self_att: int,
    opponent_rates: dict[str, tuple[int, int]] | None = None,
) -> tuple[SolveResult, ...]:
    """Helper to build SolveResult tuples."""
    results = [
        SolveResult(
            anonymized_name="model_self",
            solved=self_succ > 0,
            successes=self_succ,
            attempts=self_att,
        ),
    ]
    if opponent_rates:
        for name, (succ, att) in opponent_rates.items():
            results.append(SolveResult(
                anonymized_name=name,
                solved=succ > 0,
                successes=succ,
                attempts=att,
            ))
    return tuple(results)


class TestBuildSolveFeedback:
    """Tests for the solve feedback message builder."""

    def test_minimal_level_no_diagnostics(self):
        """MINIMAL level should show results but no diagnostics or summary."""
        results = _make_solve_results(0, 10, {"model_a": (5, 10)})
        feedback = _build_solve_feedback(results, 1, 3, PromptLevel.MINIMAL)
        assert "model_self: 0/10" in feedback
        assert "model_a: 5/10" in feedback
        # MINIMAL: no diagnostic labels
        assert "PROBLEM" not in feedback
        assert "WARNING" not in feedback
        assert "STRONG" not in feedback

    def test_standard_level_has_summary_line(self):
        """STANDARD level should include a one-line gap summary."""
        results = _make_solve_results(8, 10, {"model_a": (2, 10)})
        feedback = _build_solve_feedback(results, 1, 3, PromptLevel.STANDARD)
        assert "Your model: 80%" in feedback
        assert "best opponent: 20%" in feedback
        # STANDARD: no prescriptive diagnostic labels
        assert "STRONG" not in feedback
        assert "PROBLEM" not in feedback

    def test_detailed_self_zero(self):
        """DETAILED level, self at 0% -> PROBLEM diagnostic."""
        results = _make_solve_results(0, 10, {"model_a": (5, 10)})
        feedback = _build_solve_feedback(results, 1, 3, PromptLevel.DETAILED)
        assert "PROBLEM" in feedback

    def test_detailed_self_low(self):
        """DETAILED level, self below 50% -> WARNING."""
        results = _make_solve_results(3, 10, {"model_a": (1, 10)})
        feedback = _build_solve_feedback(results, 1, 3, PromptLevel.DETAILED)
        assert "WARNING" in feedback

    def test_detailed_strong_gap(self):
        """DETAILED level, self >= 80% and opponents <= 30% -> STRONG."""
        results = _make_solve_results(9, 10, {"model_a": (2, 10)})
        feedback = _build_solve_feedback(results, 1, 3, PromptLevel.DETAILED)
        assert "STRONG" in feedback

    def test_detailed_decent_gap(self):
        """DETAILED level, moderate gap -> DECENT."""
        results = _make_solve_results(6, 10, {"model_a": (4, 10)})
        feedback = _build_solve_feedback(results, 1, 3, PromptLevel.DETAILED)
        assert "DECENT" in feedback

    def test_detailed_no_gap(self):
        """DETAILED level, opponents match self -> NO GAP."""
        results = _make_solve_results(5, 10, {"model_a": (6, 10)})
        feedback = _build_solve_feedback(results, 1, 3, PromptLevel.DETAILED)
        assert "NO GAP" in feedback

    def test_proceed_message_mid_round(self):
        """Should include 'Proceed to Round N+1' when not the last round."""
        results = _make_solve_results(5, 10, {"model_a": (3, 10)})
        feedback = _build_solve_feedback(results, 2, 5, PromptLevel.MINIMAL)
        assert "Proceed to Round 3" in feedback

    def test_no_proceed_on_last_round(self):
        """Should NOT include proceed message on the final round."""
        results = _make_solve_results(5, 10, {"model_a": (3, 10)})
        feedback = _build_solve_feedback(results, 3, 3, PromptLevel.MINIMAL)
        assert "Proceed to Round" not in feedback

    def test_round_header(self):
        results = _make_solve_results(5, 10)
        feedback = _build_solve_feedback(results, 2, 5, PromptLevel.MINIMAL)
        assert "Round 2/5 solver results:" in feedback


class TestBuildBudgetExhaustedFeedback:
    """Tests for the budget-exhausted feedback message builder."""

    def test_minimal_no_advice(self):
        feedback = _build_budget_exhausted_feedback(
            "type mismatch", 3, 2, PromptLevel.MINIMAL,
        )
        assert "Round 3" in feedback
        assert "SKIPPED" in feedback
        assert "type mismatch" in feedback
        assert "2 round(s) remaining" in feedback
        # MINIMAL: no prescriptive advice
        assert "Simplify" not in feedback

    def test_detailed_includes_advice(self):
        feedback = _build_budget_exhausted_feedback(
            "type mismatch", 3, 2, PromptLevel.DETAILED,
        )
        assert "Simplify" in feedback

    def test_zero_remaining(self):
        feedback = _build_budget_exhausted_feedback(
            "error", 5, 0, PromptLevel.MINIMAL,
        )
        assert "0 round(s) remaining" in feedback


class TestDataclasses:
    """Tests for frozen dataclasses in generator module."""

    def test_theorem_submission_frozen(self):
        ts = TheoremSubmission("thm", "proof", "import M", 1)
        with pytest.raises(AttributeError):
            ts.proof = "new"  # type: ignore[misc]

    def test_solve_result_frozen(self):
        sr = SolveResult("model_self", True, 5, 10)
        with pytest.raises(AttributeError):
            sr.solved = False  # type: ignore[misc]

    def test_round_result_frozen(self):
        ts = TheoremSubmission("thm", "proof", "import M", 1)
        sr = (SolveResult("model_self", True, 5, 10),)
        rr = RoundResult(1, ts, sr)
        with pytest.raises(AttributeError):
            rr.round_number = 2  # type: ignore[misc]
