import asyncio
import contextvars
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

from .config import EvalConfig, ModelConfig
from .costs import CostTracker
from .generator import RoundResult, SolveResult, TheoremSubmission, run_generator
from .lean import LeanCompiler
from .models import LLMClient, create_client
from .progress import ProgressTracker
from .scoring import TheoremScore, score_theorem, select_best
from .solver import SolverAttemptResult, run_solver

log = logging.getLogger(__name__)

# Tracks which generator is active in the current async task.
_current_generator: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_current_generator", default="",
)


class _GeneratorTagFilter(logging.Filter):
    """Injects 'generator' field into every log record from the contextvar."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.generator = _current_generator.get()  # type: ignore[attr-defined]
        return True


class _GeneratorMatchFilter(logging.Filter):
    """Only passes records whose generator tag matches."""

    def __init__(self, generator_name: str) -> None:
        super().__init__()
        self._name = generator_name

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "generator", "") == self._name


class ClientRegistry:
    """Lazily creates one LLMClient per provider."""

    def __init__(self, max_concurrent: int = 20, cost_tracker: CostTracker | None = None) -> None:
        self._clients: dict[str, LLMClient] = {}
        self._max_concurrent = max_concurrent
        self.cost_tracker = cost_tracker or CostTracker()

    def get(self, provider: str) -> LLMClient:
        if provider not in self._clients:
            self._clients[provider] = create_client(provider, self._max_concurrent, self.cost_tracker)
        return self._clients[provider]

    def for_model(self, model: ModelConfig) -> LLMClient:
        return self.get(model.provider)


def _build_anon_map(
    generator_model: ModelConfig,
    solver_models: tuple[ModelConfig, ...],
) -> dict[str, str]:
    """Map model_id -> anonymized name. Shuffle opponent order."""
    anon: dict[str, str] = {generator_model.model_id: "model_self"}
    others: list[str] = []

    for sm in solver_models:
        if sm.model_id not in anon:
            others.append(sm.model_id)

    random.shuffle(others)
    for i, model_id in enumerate(others):
        anon[model_id] = f"model_{chr(ord('a') + i)}"

    return anon


def _build_anon_display_map(
    anon_map: dict[str, str],
    solver_models: tuple[ModelConfig, ...],
) -> dict[str, dict[str, str]]:
    """Build a mapping from anonymized names to model_id and display_name."""
    id_to_display = {sm.model_id: sm.display_name for sm in solver_models}
    return {
        anon_name: {
            "model_id": model_id,
            "display_name": id_to_display.get(model_id, model_id),
        }
        for model_id, anon_name in anon_map.items()
    }


async def _run_solver_attempts(
    config: EvalConfig,
    model: ModelConfig,
    clients: ClientRegistry,
    lean: LeanCompiler,
    submission: TheoremSubmission,
    attempts: int,
) -> tuple[int, int]:
    """Run solver for N attempts in parallel. Return (successes, attempts)."""
    llm = clients.for_model(model)

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


async def run_eval(
    config: EvalConfig,
    clients: ClientRegistry | None = None,
    lean: LeanCompiler | None = None,
    *,
    tracker: ProgressTracker | None = None,
    anon_map: dict[str, str] | None = None,
    resume_state: dict | None = None,
) -> dict:
    """Run the ProofBench evaluation for a single generator model.

    For resume support:
        tracker: shared ProgressTracker for events and checkpointing
        anon_map: pre-built anonymization map (restored from checkpoint)
        resume_state: generator state dict from checkpoint
    """
    if clients is None:
        clients = ClientRegistry(config.max_concurrent_api)
    if lean is None:
        lean = LeanCompiler(
            project_path=config.lean_project_path,
            timeout=config.lean_timeout_seconds,
        )
    if anon_map is None:
        anon_map = _build_anon_map(config.generator_model, config.solver_models)

    gen_id = config.generator_model.model_id

    log.info("Starting eval: generator=%s, solvers=%s",
             config.generator_model.display_name,
             [sm.display_name for sm in config.solver_models])
    log.info("Anonymization: %s", {v: k for k, v in anon_map.items()})

    # Solve callback for the generator loop
    async def solve_callback(submission: TheoremSubmission) -> tuple[SolveResult, ...]:
        tasks = [
            _run_solver_attempts(
                config, sm, clients, lean, submission,
                attempts=config.attempts_during_loop,
            )
            for sm in config.solver_models
        ]
        results = await asyncio.gather(*tasks)
        solve_results = tuple(
            SolveResult(
                anonymized_name=anon_map[sm.model_id],
                solved=successes > 0,
                successes=successes,
                attempts=att,
            )
            for sm, (successes, att) in zip(config.solver_models, results)
        )
        if tracker:
            for sr in solve_results:
                await tracker.emit(
                    "solver_completed", generator=gen_id,
                    round=submission.round_number,
                    model=sr.anonymized_name, solved=sr.solved,
                    successes=sr.successes, attempts=sr.attempts,
                )
        return solve_results

    # Resume support: reconstruct prior state
    start_round = 0
    initial_messages = None
    prior_results: list[RoundResult] | None = None

    if resume_state and resume_state.get("completed_rounds"):
        prior_results = _reconstruct_round_results(resume_state["completed_rounds"])
        start_round = len(prior_results)
        initial_messages = resume_state.get("messages")
        log.info("Resuming from round %d with %d prior results",
                 start_round + 1, len(prior_results))

    # Checkpoint callbacks
    async def on_round_complete(rr: RoundResult, messages: list) -> None:
        if not tracker:
            return
        round_data = {
            "round_number": rr.round_number,
            "submission": {
                "theorem_statement": rr.submission.theorem_statement,
                "proof": rr.submission.proof,
                "imports": rr.submission.imports,
            },
            "solve_results": [
                {"anonymized_name": sr.anonymized_name, "solved": sr.solved,
                 "successes": sr.successes, "attempts": sr.attempts}
                for sr in rr.solve_results
            ],
        }
        await tracker.emit(
            "round_scored", generator=gen_id,
            round=rr.round_number,
        )
        await tracker.save_round(gen_id, round_data, messages)

    async def on_round_skipped(round_number: int, messages: list) -> None:
        if not tracker:
            return
        await tracker.emit(
            "round_skipped", generator=gen_id,
            round=round_number, reason="budget_exhausted",
        )
        await tracker.save_skipped_round(gen_id, round_number, messages)

    # Phase 1: Generator loop
    if tracker:
        await tracker.update_generator(gen_id, status="in_progress", phase="generator_loop")
        await tracker.emit("eval_started", generator=gen_id)

    # Build anon → display_name mapping for log output
    id_to_display = {sm.model_id: sm.display_name for sm in config.solver_models}
    anon_to_display = {
        anon_name: id_to_display.get(model_id, model_id)
        for model_id, anon_name in anon_map.items()
    }

    log.info("=== Phase 1: Generator loop (%d rounds) ===", config.rounds)
    generator_llm = clients.for_model(config.generator_model)
    round_results = await run_generator(
        config, generator_llm, lean, solve_callback,
        start_round=start_round,
        initial_messages=initial_messages,
        prior_results=prior_results,
        on_round_complete=on_round_complete,
        on_round_skipped=on_round_skipped,
        anon_to_display=anon_to_display,
    )

    if not round_results:
        log.warning("Generator produced no valid theorems.")
        if tracker:
            await tracker.mark_generator_complete(gen_id)
            await tracker.emit("eval_completed", generator=gen_id, final_gap=0.0)
        return _build_generator_output(config, anon_map, round_results, [], None, None,
                                       cost=await clients.cost_tracker.summary())

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
        if tracker:
            await tracker.update_generator(gen_id, phase="reeval")
            await tracker.emit(
                "reeval_started", generator=gen_id,
                theorem_id=best_loop.theorem_id if best_loop else "",
                attempts=config.attempts_reeval,
            )

        log.info("=== Phase 2: Re-evaluation (%d attempts) ===", config.attempts_reeval)
        final_score = await _reeval_theorem(config, clients, lean, anon_map, best_round)
        log.info("Final score: gap=%.3f, raw_gap=%.3f",
                 final_score.gap_score, final_score.raw_gap)

    if tracker:
        await tracker.mark_generator_complete(gen_id)
        await tracker.emit(
            "eval_completed", generator=gen_id,
            final_gap=final_score.gap_score if final_score else 0.0,
        )

    return _build_generator_output(config, anon_map, round_results, loop_scores, best_loop, final_score,
                                   cost=await clients.cost_tracker.summary())


async def run_multi_eval(
    generator_models: list[ModelConfig],
    base_config: EvalConfig,
    *,
    tracker: ProgressTracker | None = None,
) -> dict:
    """Run ProofBench evaluation for multiple generator models concurrently.

    Each generator gets an independent eval run with the same solver set.
    Returns a combined output with per-generator results and overall comparison.

    For resume: pass a ProgressTracker loaded from a checkpoint.
    """
    clients = ClientRegistry(base_config.max_concurrent_api)
    lean = LeanCompiler(
        project_path=base_config.lean_project_path,
        timeout=base_config.lean_timeout_seconds,
    )

    # Set up per-generator log files
    log_dir = Path(base_config.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = tracker.timestamp if tracker else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Add the tag filter to every handler (not the root logger) so that
    # propagated records get the 'generator' field before formatting.
    # Filters on a parent *logger* are skipped when child loggers propagate
    # records — only handler-level filters run reliably.
    root = logging.getLogger()
    tag_filter = _GeneratorTagFilter()

    file_handlers: list[logging.FileHandler] = []
    for g in generator_models:
        log_path = log_dir / f"{g.display_name}_{ts}.log"
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        fh.addFilter(tag_filter)
        fh.addFilter(_GeneratorMatchFilter(g.display_name))
        fh.setLevel(logging.DEBUG)
        root.addHandler(fh)
        file_handlers.append(fh)

    # Update console formatter to include generator prefix; save originals for restore
    original_formatters: list[tuple[logging.Handler, logging.Formatter | None]] = []
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and handler not in file_handlers:
            original_formatters.append((handler, handler.formatter))
            handler.addFilter(tag_filter)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-8s [%(generator)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            ))

    # Build anonymization maps upfront (or restore from checkpoint)
    anon_maps: dict[str, dict[str, str]] = {}
    for g in generator_models:
        saved = tracker.get_generator_state(g.model_id) if tracker else None
        if saved and saved.get("anon_map"):
            anon_maps[g.model_id] = saved["anon_map"]
        else:
            anon_maps[g.model_id] = _build_anon_map(g, base_config.solver_models)

    # Create tracker if not resuming
    if tracker is None:
        tracker = ProgressTracker(base_config.output_dir, ts)
        tracker.init_state(base_config, [g.model_id for g in generator_models], anon_maps)

    async def _run_one(generator: ModelConfig) -> dict:
        _current_generator.set(generator.display_name)
        gen_id = generator.model_id

        # Check if already complete from checkpoint
        saved = tracker.get_generator_state(gen_id)
        if saved and saved.get("status") == "complete" and saved.get("final_score"):
            log.info("Generator %s already complete, skipping", generator.display_name)
            return saved.get("_output", _build_skipped_output(generator, saved))

        # Ensure this generator's model is in the solver list
        solver_ids = {sm.model_id for sm in base_config.solver_models}
        if generator.model_id in solver_ids:
            solvers = base_config.solver_models
        else:
            solvers = (generator, *base_config.solver_models)

        config = EvalConfig(
            generator_model=generator,
            solver_models=solvers,
            rounds=base_config.rounds,
            solver_max_calls=base_config.solver_max_calls,
            generator_max_calls=base_config.generator_max_calls,
            attempts_during_loop=base_config.attempts_during_loop,
            attempts_reeval=base_config.attempts_reeval,
            lean_timeout_seconds=base_config.lean_timeout_seconds,
            lean_project_path=base_config.lean_project_path,
            output_dir=base_config.output_dir,
            prior_alpha=base_config.prior_alpha,
            prior_beta=base_config.prior_beta,
            seed=base_config.seed,
            max_concurrent_api=base_config.max_concurrent_api,
            summarize_rounds=base_config.summarize_rounds,
            prompt_level=base_config.prompt_level,
        )

        log.info(">>> Starting generator: %s <<<", generator.display_name)
        try:
            return await run_eval(
                config, clients=clients, lean=lean,
                tracker=tracker,
                anon_map=anon_maps[gen_id],
                resume_state=saved if saved and saved.get("rounds_completed", 0) > 0 else None,
            )
        except Exception:
            log.exception("Generator %s failed", generator.display_name)
            return {
                "generator": generator.model_id,
                "error": f"Generator {generator.display_name} failed",
                "final_score": None,
            }

    generator_results = await asyncio.gather(
        *[_run_one(g) for g in generator_models]
    )

    # Clean up: remove file handlers, remove tag filter, restore console formatters
    for fh in file_handlers:
        fh.close()
        root.removeHandler(fh)
    for handler in root.handlers:
        handler.removeFilter(tag_filter)
    for handler, original_fmt in original_formatters:
        handler.setFormatter(original_fmt)

    log.info("Per-generator logs written to %s/", log_dir)

    await clients.cost_tracker.log_summary()
    output = _build_multi_output(base_config, generator_models, list(generator_results),
                                 cost=await clients.cost_tracker.summary(),
                                 timestamp=ts)

    await tracker.finalize(output)
    return output


def _reconstruct_round_results(saved_rounds: list[dict]) -> list[RoundResult]:
    """Reconstruct RoundResult objects from checkpoint data."""
    results = []
    for rd in saved_rounds:
        submission = TheoremSubmission(
            theorem_statement=rd["submission"]["theorem_statement"],
            proof=rd["submission"]["proof"],
            imports=rd["submission"]["imports"],
            round_number=rd["round_number"],
        )
        solve_results = tuple(
            SolveResult(
                anonymized_name=sr["anonymized_name"],
                solved=sr["solved"],
                successes=sr.get("successes", 1 if sr["solved"] else 0),
                attempts=sr["attempts"],
            )
            for sr in rd["solve_results"]
        )
        results.append(RoundResult(rd["round_number"], submission, solve_results))
    return results


def _build_skipped_output(generator: ModelConfig, saved: dict) -> dict:
    """Build a minimal output dict for a generator that was already complete."""
    return {
        "generator": generator.model_id,
        "generator_display_name": generator.display_name,
        "final_score": saved.get("final_score"),
    }


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
            sr.anonymized_name: (sr.successes, sr.attempts)
            for sr in rr.solve_results
            if sr.anonymized_name != "model_self"
        }
        scores.append(score_theorem(
            theorem_id=f"round_{rr.round_number}",
            self_successes=self_result.successes if self_result else 0,
            self_attempts=self_result.attempts if self_result else 1,
            other_results=other_results,
            prior_alpha=config.prior_alpha,
            prior_beta=config.prior_beta,
            seed=config.seed,
        ))
    return scores


async def _reeval_theorem(
    config: EvalConfig,
    clients: ClientRegistry,
    lean: LeanCompiler,
    anon_map: dict[str, str],
    round_result: RoundResult,
) -> TheoremScore:
    """Re-evaluate a theorem with more solver attempts (no early stopping)."""
    tasks = [
        _run_solver_attempts(
            config, sm, clients, lean, round_result.submission,
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


def _build_generator_output(
    config: EvalConfig,
    anon_map: dict[str, str],
    round_results: list[RoundResult],
    loop_scores: list[TheoremScore],
    best_loop: TheoremScore | None,
    final_score: TheoremScore | None,
    cost: dict | None = None,
) -> dict:
    """Build the output dict for a single generator run."""
    output = {
        "generator": config.generator_model.model_id,
        "generator_display_name": config.generator_model.display_name,
        "anonymization": _build_anon_display_map(anon_map, config.solver_models),
        "rounds": [
            {
                "round": rr.round_number,
                "theorem_statement": rr.submission.theorem_statement,
                "proof": rr.submission.proof,
                "imports": rr.submission.imports,
                "solver_results": [
                    {"model": sr.anonymized_name, "solved": sr.solved, "successes": sr.successes, "attempts": sr.attempts}
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
    if cost:
        output["cost"] = cost
    return output


def _build_multi_output(
    config: EvalConfig,
    generator_models: list[ModelConfig],
    generator_results: list[dict],
    cost: dict | None = None,
    timestamp: str | None = None,
) -> dict:
    """Build combined output for a multi-generator eval."""
    # Find the generator with the best final gap score
    best_generator: dict | None = None
    best_gap = -1.0
    for gr in generator_results:
        fs = gr.get("final_score")
        if fs and fs["gap_score"] > best_gap:
            best_gap = fs["gap_score"]
            best_generator = gr

    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "generators": [g.model_id for g in generator_models],
            "solvers": [sm.model_id for sm in config.solver_models],
            "rounds": config.rounds,
            "solver_max_calls": config.solver_max_calls,
            "attempts_during_loop": config.attempts_during_loop,
            "attempts_reeval": config.attempts_reeval,
            "prompt_level": config.prompt_level.value,
        },
        "generators": generator_results,
        "best_generator": {
            "model": best_generator["generator"],
            "gap_score": best_gap,
            "theorem_id": best_generator["final_score"]["theorem_id"],
        } if best_generator and best_generator.get("final_score") else None,
    }
    if cost:
        output["cost"] = cost

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"eval_{ts}.json"
    output_path.write_text(json.dumps(output, indent=2))
    log.info("Results written to %s", output_path)

    return output
