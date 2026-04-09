import asyncio
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

from .config import EvalConfig, ModelConfig
from .generator import RoundResult, SolveResult, TheoremSubmission, run_generator
from .lean import LeanCompiler
from .models import LLMClient
from .scoring import TheoremScore, score_theorem, select_best
from .solver import SolverAttemptResult, run_solver

log = logging.getLogger(__name__)


def _build_anon_map(
    generator_model: ModelConfig,
    solver_models: tuple[ModelConfig, ...],
) -> dict[str, str]:
    """Map model_id -> anonymized name. Shuffle opponent order."""
    anon: dict[str, str] = {}
    others: list[str] = []

    for sm in solver_models:
        if sm.model_id == generator_model.model_id and sm.model_id not in anon:
            anon[sm.model_id] = "model_self"
        elif sm.model_id not in anon:
            others.append(sm.model_id)

    random.shuffle(others)
    for i, model_id in enumerate(others):
        anon[model_id] = f"model_{chr(ord('a') + i)}"

    return anon


async def _run_solver_attempts(
    config: EvalConfig,
    model: ModelConfig,
    llm: LLMClient,
    lean: LeanCompiler,
    submission: TheoremSubmission,
    attempts: int,
) -> tuple[int, int]:
    """Run solver for N attempts in parallel. Return (successes, attempts)."""
    if attempts == 1:
        result = await run_solver(
            config=config,
            model=model,
            llm=llm,
            lean=lean,
            theorem_statement=submission.theorem_statement,
            imports=submission.imports,
        )
        solved_str = "SOLVED" if result.solved else "FAILED"
        log.info(
            "  %s attempt 1/1: %s (%d calls used)",
            model.display_name, solved_str, result.calls_used,
        )
        return (1 if result.solved else 0, 1)

    async def _single_attempt(idx: int) -> SolverAttemptResult:
        result = await run_solver(
            config=config,
            model=model,
            llm=llm,
            lean=lean,
            theorem_statement=submission.theorem_statement,
            imports=submission.imports,
        )
        solved_str = "SOLVED" if result.solved else "FAILED"
        log.info(
            "  %s attempt %d/%d: %s (%d calls used)",
            model.display_name, idx + 1, attempts, solved_str, result.calls_used,
        )
        return result

    results = await asyncio.gather(*[_single_attempt(i) for i in range(attempts)])
    successes = sum(1 for r in results if r.solved)
    return (successes, attempts)


async def run_eval(config: EvalConfig) -> dict:
    """Run the full ProofBench evaluation."""
    llm = LLMClient.create()
    lean = LeanCompiler(
        project_path=config.lean_project_path,
        timeout=config.lean_timeout_seconds,
    )
    anon_map = _build_anon_map(config.generator_model, config.solver_models)

    log.info("Starting eval: generator=%s, solvers=%s",
             config.generator_model.display_name,
             [sm.display_name for sm in config.solver_models])
    log.info("Anonymization: %s", {v: k for k, v in anon_map.items()})

    # Solve callback for the generator loop
    async def solve_callback(submission: TheoremSubmission) -> tuple[SolveResult, ...]:
        tasks = [
            _run_solver_attempts(
                config, sm, llm, lean, submission,
                attempts=config.attempts_during_loop,
            )
            for sm in config.solver_models
        ]
        results = await asyncio.gather(*tasks)
        return tuple(
            SolveResult(
                anonymized_name=anon_map[sm.model_id],
                solved=successes > 0,
                attempts=att,
            )
            for sm, (successes, att) in zip(config.solver_models, results)
        )

    # Phase 1: Generator loop
    log.info("=== Phase 1: Generator loop (%d rounds) ===", config.rounds)
    round_results = await run_generator(config, llm, lean, solve_callback)

    if not round_results:
        log.warning("Generator produced no valid theorems.")
        return _build_output(config, anon_map, round_results, [], None, None)

    # Score each round's theorem from loop data
    loop_scores = _score_round_results(round_results, config)
    best_loop = select_best(loop_scores)
    log.info("Best theorem from loop: %s (gap=%.3f)",
             best_loop.theorem_id if best_loop else "none",
             best_loop.gap_score if best_loop else 0.0)

    # Phase 2: Re-evaluate the best submission
    best_round = next(
        (rr for rr in round_results
         if f"round_{rr.round_number}" == (best_loop.theorem_id if best_loop else "")),
        None,
    )

    final_score: TheoremScore | None = None
    if best_round is not None:
        log.info("=== Phase 2: Re-evaluation (%d attempts) ===", config.attempts_reeval)
        final_score = await _reeval_theorem(config, llm, lean, anon_map, best_round)
        log.info("Final score: gap=%.3f, raw_gap=%.3f",
                 final_score.gap_score, final_score.raw_gap)

    return _build_output(config, anon_map, round_results, loop_scores, best_loop, final_score)


def _score_round_results(
    round_results: list[RoundResult],
    config: EvalConfig,
) -> list[TheoremScore]:
    """Score each round's theorem from the loop solve data."""
    scores = []
    for rr in round_results:
        self_result = next(
            (sr for sr in rr.solve_results if sr.anonymized_name == "model_self"),
            None,
        )
        other_results = {
            sr.anonymized_name: (1 if sr.solved else 0, sr.attempts)
            for sr in rr.solve_results
            if sr.anonymized_name != "model_self"
        }
        scores.append(score_theorem(
            theorem_id=f"round_{rr.round_number}",
            self_successes=1 if (self_result and self_result.solved) else 0,
            self_attempts=self_result.attempts if self_result else 1,
            other_results=other_results,
            prior_alpha=config.prior_alpha,
            prior_beta=config.prior_beta,
            seed=config.seed,
        ))
    return scores


async def _reeval_theorem(
    config: EvalConfig,
    llm: LLMClient,
    lean: LeanCompiler,
    anon_map: dict[str, str],
    round_result: RoundResult,
) -> TheoremScore:
    """Re-evaluate a theorem with more solver attempts (no early stopping)."""
    tasks = [
        _run_solver_attempts(
            config, sm, llm, lean, round_result.submission,
            attempts=config.attempts_reeval,
        )
        for sm in config.solver_models
    ]
    results = await asyncio.gather(*tasks)

    self_successes = 0
    self_attempts = 0
    other_results: dict[str, tuple[int, int]] = {}

    for sm, (successes, attempts) in zip(config.solver_models, results):
        name = anon_map[sm.model_id]
        if name == "model_self":
            self_successes = successes
            self_attempts = attempts
        else:
            other_results[name] = (successes, attempts)

    return score_theorem(
        theorem_id=f"round_{round_result.round_number}",
        self_successes=self_successes,
        self_attempts=self_attempts,
        other_results=other_results,
        prior_alpha=config.prior_alpha,
        prior_beta=config.prior_beta,
        seed=config.seed,
    )


def _build_output(
    config: EvalConfig,
    anon_map: dict[str, str],
    round_results: list[RoundResult],
    loop_scores: list[TheoremScore],
    best_loop: TheoremScore | None,
    final_score: TheoremScore | None,
) -> dict:
    """Build the JSON output dict."""
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "generator": config.generator_model.model_id,
            "solvers": [sm.model_id for sm in config.solver_models],
            "rounds": config.rounds,
            "solver_max_calls": config.solver_max_calls,
            "attempts_during_loop": config.attempts_during_loop,
            "attempts_reeval": config.attempts_reeval,
        },
        "anonymization": {v: k for k, v in anon_map.items()},
        "rounds": [
            {
                "round": rr.round_number,
                "theorem_statement": rr.submission.theorem_statement,
                "proof": rr.submission.proof,
                "imports": rr.submission.imports,
                "solver_results": [
                    {"model": sr.anonymized_name, "solved": sr.solved, "attempts": sr.attempts}
                    for sr in rr.solve_results
                ],
            }
            for rr in round_results
        ],
        "loop_scores": [
            {
                "theorem_id": s.theorem_id,
                "gap_score": s.gap_score,
                "raw_gap": s.raw_gap,
                "self_solve_rate": s.self_solve_rate,
                "other_solve_rates": s.other_solve_rates,
            }
            for s in loop_scores
        ],
        "best_submission": best_loop.theorem_id if best_loop else None,
        "final_score": {
            "theorem_id": final_score.theorem_id,
            "gap_score": final_score.gap_score,
            "raw_gap": final_score.raw_gap,
            "self_solve_rate": final_score.self_solve_rate,
            "other_solve_rates": final_score.other_solve_rates,
        } if final_score else None,
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_{ts}.json"
    output_path.write_text(json.dumps(output, indent=2))
    log.info("Results written to %s", output_path)

    return output
