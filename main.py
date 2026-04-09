import argparse
import asyncio
import json
import logging
import sys

from dotenv import load_dotenv

from src.config import (
    DEFAULT_GENERATOR,
    DEFAULT_SOLVERS,
    KNOWN_MODELS,
    EvalConfig,
    ModelConfig,
)
from src.orchestrator import run_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ProofBench: AI eval where a generator crafts Lean 4 theorems",
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="Number of generator rounds (default: 3)",
    )
    parser.add_argument(
        "--generator-model", type=str, default=DEFAULT_GENERATOR,
        help=f"Generator model key or full ID (known: {', '.join(KNOWN_MODELS)})",
    )
    parser.add_argument(
        "--solver-models", type=str, nargs="+", default=list(DEFAULT_SOLVERS),
        help=f"Solver model keys or full IDs (known: {', '.join(KNOWN_MODELS)})",
    )
    parser.add_argument(
        "--solver-max-calls", type=int, default=10,
        help="Max submit_proof calls per solver attempt (default: 10)",
    )
    parser.add_argument(
        "--generator-max-calls", type=int, default=20,
        help="Max propose_theorem calls per generator round (default: 20)",
    )
    parser.add_argument(
        "--attempts-during-loop", type=int, default=1,
        help="Solver attempts per theorem during generator loop (default: 1)",
    )
    parser.add_argument(
        "--attempts-reeval", type=int, default=3,
        help="Solver attempts per theorem during re-evaluation (default: 3)",
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
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def resolve_model(key_or_id: str) -> ModelConfig:
    """Resolve a model key (e.g. 'opus') or full model ID to a ModelConfig."""
    if key_or_id in KNOWN_MODELS:
        return KNOWN_MODELS[key_or_id]
    return ModelConfig(model_id=key_or_id, display_name=key_or_id)


def build_config(args: argparse.Namespace) -> EvalConfig:
    generator = resolve_model(args.generator_model)
    # Deduplicate by model_id, preserving order
    seen: dict[str, ModelConfig] = {}
    for s in args.solver_models:
        model = resolve_model(s)
        if model.model_id not in seen:
            seen[model.model_id] = model

    # Ensure generator model is in the solver list
    if generator.model_id not in seen:
        seen = {generator.model_id: generator, **seen}

    solvers = tuple(seen.values())

    return EvalConfig(
        generator_model=generator,
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
    )


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


def main() -> None:
    load_dotenv()
    args = parse_args()
    configure_logging(args.verbose)

    config = build_config(args)
    result = asyncio.run(run_eval(config))

    # Print summary
    final = result.get("final_score")
    if final:
        print(f"\n{'='*50}")
        print(f"Best theorem: {final['theorem_id']}")
        print(f"Gap score:    {final['gap_score']:.3f}")
        print(f"Raw gap:      {final['raw_gap']:.3f}")
        print(f"Self rate:    {final['self_solve_rate']:.3f}")
        for name, rate in final["other_solve_rates"].items():
            print(f"  {name}:    {rate:.3f}")
        print(f"{'='*50}")
    else:
        print("\nNo valid theorems were produced.")
        sys.exit(1)


if __name__ == "__main__":
    main()
