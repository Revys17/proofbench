import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TheoremScore:
    theorem_id: str
    self_solve_rate: float
    other_solve_rates: dict[str, float]
    gap_score: float
    raw_gap: float


def score_theorem(
    *,
    theorem_id: str,
    self_successes: int,
    self_attempts: int,
    other_results: dict[str, tuple[int, int]],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 10_000,
    seed: int | None = None,
) -> TheoremScore:
    """Compute Bayesian gap score for a single theorem.

    Args:
        self_successes: Number of successful solves by model_self.
        self_attempts: Total attempts by model_self.
        other_results: {anonymized_name: (successes, attempts)} for opponents.
        prior_alpha, prior_beta: Beta prior parameters.
        n_samples: Monte Carlo sample count.

    Returns:
        TheoremScore with gap_score = P(self_rate > max(opponent_rates)).
    """
    rng = np.random.default_rng(seed)

    self_a = prior_alpha + self_successes
    self_b = prior_beta + self_attempts - self_successes
    self_mean = self_a / (self_a + self_b)
    self_samples = rng.beta(self_a, self_b, size=n_samples)

    log.info(
        "Scoring %s: model_self %d/%d → Beta(%.1f, %.1f), mean=%.3f",
        theorem_id, self_successes, self_attempts, self_a, self_b, self_mean,
    )

    other_means: dict[str, float] = {}
    other_sample_arrays: list[np.ndarray] = []

    for name, (succ, att) in other_results.items():
        a = prior_alpha + succ
        b = prior_beta + att - succ
        mean = a / (a + b)
        other_means[name] = mean
        other_sample_arrays.append(rng.beta(a, b, size=n_samples))
        log.info(
            "  %s: %d/%d → Beta(%.1f, %.1f), mean=%.3f",
            name, succ, att, a, b, mean,
        )

    if other_sample_arrays:
        max_other = np.maximum.reduce(other_sample_arrays)
        gap_score = float(np.mean(self_samples > max_other))
        raw_gap = self_mean - max(other_means.values())
    else:
        gap_score = 1.0
        raw_gap = self_mean

    log.info(
        "  → gap_score=%.3f (P(self > max(others))), raw_gap=%.3f",
        gap_score, raw_gap,
    )

    return TheoremScore(
        theorem_id=theorem_id,
        self_solve_rate=self_mean,
        other_solve_rates=other_means,
        gap_score=gap_score,
        raw_gap=raw_gap,
    )


def select_best(scores: list[TheoremScore]) -> TheoremScore | None:
    """Return the theorem with the highest gap score."""
    if not scores:
        return None
    return max(scores, key=lambda s: s.gap_score)
