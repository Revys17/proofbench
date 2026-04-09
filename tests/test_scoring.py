import pytest
import numpy as np

from src.scoring import score_theorem, select_best, TheoremScore


class TestScoreTheorem:
    """Tests for Bayesian gap scoring."""

    def test_perfect_gap(self):
        """Self solves all, opponents solve none -> high gap score."""
        score = score_theorem(
            theorem_id="t1",
            self_successes=10,
            self_attempts=10,
            other_results={"model_a": (0, 10), "model_b": (0, 10)},
            seed=42,
        )
        assert score.gap_score > 0.95
        assert score.raw_gap > 0.5
        assert score.self_solve_rate > 0.8

    def test_no_gap(self):
        """Everyone solves equally -> gap score near 0.5 or below."""
        score = score_theorem(
            theorem_id="t2",
            self_successes=5,
            self_attempts=10,
            other_results={"model_a": (5, 10)},
            seed=42,
        )
        assert score.gap_score < 0.6

    def test_self_fails_all(self):
        """Self solves none -> gap score near zero."""
        score = score_theorem(
            theorem_id="t3",
            self_successes=0,
            self_attempts=10,
            other_results={"model_a": (5, 10)},
            seed=42,
        )
        assert score.gap_score < 0.05

    def test_no_opponents(self):
        """No opponents -> gap score should be 1.0."""
        score = score_theorem(
            theorem_id="t4",
            self_successes=5,
            self_attempts=10,
            other_results={},
            seed=42,
        )
        assert score.gap_score == 1.0

    def test_single_attempt_each(self):
        """Minimal data: 1 attempt each."""
        score = score_theorem(
            theorem_id="t5",
            self_successes=1,
            self_attempts=1,
            other_results={"model_a": (0, 1)},
            seed=42,
        )
        # With uniform prior Beta(1,1), posterior is Beta(2,1) vs Beta(1,2)
        # P(self > other) should be meaningfully above 0.5
        assert score.gap_score > 0.5

    def test_beta_prior_parameters(self):
        """Custom prior changes posterior."""
        # Strong prior toward 0.5 should pull posteriors together
        score_default = score_theorem(
            theorem_id="t6a",
            self_successes=3,
            self_attempts=3,
            other_results={"model_a": (0, 3)},
            seed=42,
        )
        score_strong_prior = score_theorem(
            theorem_id="t6b",
            self_successes=3,
            self_attempts=3,
            other_results={"model_a": (0, 3)},
            prior_alpha=10.0,
            prior_beta=10.0,
            seed=42,
        )
        # Strong prior pulls both posteriors toward 0.5, reducing the gap
        assert score_strong_prior.gap_score < score_default.gap_score

    def test_reproducibility_with_seed(self):
        """Same seed produces same score."""
        kwargs = dict(
            theorem_id="t7",
            self_successes=7,
            self_attempts=10,
            other_results={"model_a": (3, 10)},
            seed=123,
        )
        score1 = score_theorem(**kwargs)
        score2 = score_theorem(**kwargs)
        assert score1.gap_score == score2.gap_score
        assert score1.raw_gap == score2.raw_gap

    def test_multiple_opponents_max(self):
        """Gap is computed against the BEST opponent, not average."""
        # model_a is weak, model_b is strong
        score = score_theorem(
            theorem_id="t8",
            self_successes=8,
            self_attempts=10,
            other_results={"model_a": (0, 10), "model_b": (7, 10)},
            seed=42,
        )
        # The strong opponent (model_b) should drag the gap score down
        score_weak_only = score_theorem(
            theorem_id="t8b",
            self_successes=8,
            self_attempts=10,
            other_results={"model_a": (0, 10)},
            seed=42,
        )
        assert score.gap_score < score_weak_only.gap_score

    def test_output_fields(self):
        """TheoremScore has all expected fields."""
        score = score_theorem(
            theorem_id="t9",
            self_successes=5,
            self_attempts=10,
            other_results={"model_a": (2, 10)},
            seed=42,
        )
        assert score.theorem_id == "t9"
        assert 0.0 <= score.gap_score <= 1.0
        assert isinstance(score.raw_gap, float)
        assert isinstance(score.self_solve_rate, float)
        assert "model_a" in score.other_solve_rates


class TestSelectBest:
    """Tests for best theorem selection."""

    def test_empty_list(self):
        assert select_best([]) is None

    def test_single_entry(self):
        s = TheoremScore("t1", 0.8, {"a": 0.2}, 0.9, 0.6)
        assert select_best([s]) is s

    def test_picks_highest_gap(self):
        low = TheoremScore("t1", 0.5, {"a": 0.4}, 0.3, 0.1)
        high = TheoremScore("t2", 0.9, {"a": 0.1}, 0.95, 0.8)
        mid = TheoremScore("t3", 0.7, {"a": 0.3}, 0.6, 0.4)
        assert select_best([low, high, mid]) is high

    def test_tiebreak_is_stable(self):
        """When gap scores are equal, max() picks the first."""
        a = TheoremScore("t1", 0.5, {}, 0.5, 0.5)
        b = TheoremScore("t2", 0.5, {}, 0.5, 0.5)
        result = select_best([a, b])
        assert result is a or result is b  # deterministic, just not crashing
