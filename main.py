import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

from src.config import (
    DEFAULT_GENERATORS,
    DEFAULT_SOLVERS,
    KNOWN_MODELS,
    EvalConfig,
    ModelConfig,
)
from src.orchestrator import run_multi_eval


def _available_model_keys() -> list[str]:
    """Return keys of uncommented (available) models in KNOWN_MODELS."""
    return list(KNOWN_MODELS.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ProofBench: AI eval where a generator crafts Lean 4 theorems",
    )
    parser.add_argument(
        "--full-run", action="store_true",
        help="Production preset: all available models as generators and solvers, "
             "10 rounds, 10 attempts/loop, 30 for re-eval. "
             "Individual flags override these defaults.",
    )
    parser.add_argument(
        "--rounds", type=int, default=None,
        help="Number of generator rounds (default: 3, full-run: 10)",
    )
    parser.add_argument(
        "--generator-models", type=str, nargs="+", default=None,
        help=f"Generator model key(s) or full ID(s) (known: {', '.join(KNOWN_MODELS)})",
    )
    parser.add_argument(
        "--solver-models", type=str, nargs="+", default=None,
        help=f"Solver model keys or full IDs (known: {', '.join(KNOWN_MODELS)})",
    )
    parser.add_argument(
        "--solver-max-calls", type=int, default=None,
        help="Max submit_proof calls per solver attempt (default: 10)",
    )
    parser.add_argument(
        "--generator-max-calls", type=int, default=None,
        help="Max propose_theorem calls per generator round (default: 20)",
    )
    parser.add_argument(
        "--attempts-during-loop", type=int, default=None,
        help="Solver attempts per theorem during generator loop (default: 3, full-run: 10)",
    )
    parser.add_argument(
        "--attempts-reeval", type=int, default=None,
        help="Solver attempts per theorem during re-evaluation (default: 10, full-run: 30)",
    )
    parser.add_argument(
        "--lean-timeout", type=int, default=120,
        help="Lean compilation timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--lean-project-path", type=str, default="lean_solver",
        help="Path to Lean project directory (default: lean_solver)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for scoring reproducibility (default: random)",
    )
    parser.add_argument(
        "--max-concurrent-api", type=int, default=20,
        help="Max concurrent API calls per provider (default: 20)",
    )
    parser.add_argument(
        "--summarize-rounds", action="store_true",
        help="Compact conversation history between rounds by asking the generator to summarize",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Apply defaults: --full-run sets production values, smoke-test otherwise.
    # Explicit flags always win.
    available = _available_model_keys()
    if args.full_run:
        defaults = {
            "generator_models": available,
            "solver_models": available,
            "rounds": 10,
            "solver_max_calls": 10,
            "generator_max_calls": 20,
            "attempts_during_loop": 10,
            "attempts_reeval": 30,
        }
    else:
        defaults = {
            "generator_models": list(DEFAULT_GENERATORS),
            "solver_models": list(DEFAULT_SOLVERS),
            "rounds": 3,
            "solver_max_calls": 10,
            "generator_max_calls": 20,
            "attempts_during_loop": 3,
            "attempts_reeval": 10,
        }

    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    return args


def resolve_model(key_or_id: str) -> ModelConfig:
    """Resolve a model key (e.g. 'opus') or full model ID to a ModelConfig."""
    if key_or_id in KNOWN_MODELS:
        return KNOWN_MODELS[key_or_id]
    return ModelConfig(model_id=key_or_id, display_name=key_or_id)


def build_config(args: argparse.Namespace) -> tuple[list[ModelConfig], EvalConfig]:
    """Build generator list and base EvalConfig from CLI args.

    Returns (generator_models, base_config). base_config.generator_model is set
    to the first generator — run_multi_eval overrides it per generator.
    """
    # Deduplicate generators
    gen_seen: dict[str, ModelConfig] = {}
    for g in args.generator_models:
        model = resolve_model(g)
        if model.model_id not in gen_seen:
            gen_seen[model.model_id] = model
    generators = list(gen_seen.values())

    # Deduplicate solvers
    solver_seen: dict[str, ModelConfig] = {}
    for s in args.solver_models:
        model = resolve_model(s)
        if model.model_id not in solver_seen:
            solver_seen[model.model_id] = model

    # Ensure all generator models are in the solver list
    for g in generators:
        if g.model_id not in solver_seen:
            solver_seen[g.model_id] = g

    solvers = tuple(solver_seen.values())

    base_config = EvalConfig(
        generator_model=generators[0],
        solver_models=solvers,
        rounds=args.rounds,
        solver_max_calls=args.solver_max_calls,
        generator_max_calls=args.generator_max_calls,
        attempts_during_loop=args.attempts_during_loop,
        attempts_reeval=args.attempts_reeval,
        lean_timeout_seconds=args.lean_timeout,
        lean_project_path=args.lean_project_path,
        output_dir=args.output_dir,
        seed=args.seed,
        max_concurrent_api=args.max_concurrent_api,
        summarize_rounds=args.summarize_rounds,
    )

    return generators, base_config


def configure_logging(verbose: bool) -> None:
    """Set up logging, silencing noisy HTTP libraries."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("httpcore", "httpcore.http11", "httpcore.connection",
                 "httpx", "anthropic._base_client"):
        logging.getLogger(name).setLevel(logging.WARNING)


def print_summary(result: dict) -> None:
    """Print human-readable summary of eval results."""
    generators = result.get("generators", [])

    if not generators:
        print("\nNo results.")
        return

    print(f"\n{'='*60}")
    print(f"ProofBench Results — {len(generators)} generator(s)")
    print(f"{'='*60}")

    for gr in generators:
        name = gr.get("generator_display_name", gr.get("generator", "unknown"))
        fs = gr.get("final_score")
        if gr.get("error"):
            print(f"\n  {name}: FAILED — {gr['error']}")
        elif fs:
            print(f"\n  {name}:")
            print(f"    Best theorem:  {fs['theorem_id']}")
            print(f"    Gap score:     {fs['gap_score']:.3f}")
            print(f"    Raw gap:       {fs['raw_gap']:.3f}")
            print(f"    Self rate:     {fs['self_solve_rate']:.3f}")
            for oname, rate in fs["other_solve_rates"].items():
                print(f"    {oname:14s} {rate:.3f}")
        else:
            print(f"\n  {name}: no valid theorems produced")

    best = result.get("best_generator")
    if best:
        print(f"\n{'─'*60}")
        print(f"  Best generator: {best['model']} "
              f"(gap={best['gap_score']:.3f}, theorem={best['theorem_id']})")

    cost = result.get("cost")
    if cost:
        print(f"\n{'─'*60}")
        print(f"  Cost: ${cost['total_cost_usd']:.4f}")
        for model, info in cost["per_model"].items():
            print(f"    {model}: {info['api_calls']} calls, "
                  f"{info['input_tokens']}in/{info['output_tokens']}out, "
                  f"${info['cost_usd']:.4f}")
    print(f"{'='*60}")


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    configure_logging(args.verbose)

    generators, base_config = build_config(args)

    result = asyncio.run(run_multi_eval(generators, base_config))

    print_summary(result)

    if not result.get("best_generator"):
        sys.exit(1)


if __name__ == "__main__":
    main()
