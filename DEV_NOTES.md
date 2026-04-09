# Development Notes

This file contains the description of the development process for this project.

## LLM Usage

I worked with Claude Opus 4.6 and Gemini 3.1 Pro to develop the project plan, and used Claude Code heavily during implementation. You can see the initial implementation plan that Claude Opus came up with below. Once implemented, we iterated until we got to the current version of the repo.

<details>
<summary>ProofBench: Implementation Plan</summary>

## Context

Build an AI eval where a **generator** model crafts Lean 4 theorems that a fresh copy of itself can solve but other models can't. The generator iterates based on solver feedback, and the best submission is scored via Bayesian gap analysis. End-to-end MVP.

## File Structure (existing stubs in `src/`)

```
proofbench/
├── lean_solver/              # Existing Lean 4.29.0 + Mathlib v4.29.0
├── src/
│   ├── __init__.py
│   ├── config.py             # Frozen dataclasses
│   ├── lean.py               # Lean compilation: assemble file, compile, parse
│   ├── models.py             # Async Anthropic SDK wrapper with retry
│   ├── generator.py          # Generator agentic loop
│   ├── solver.py             # Solver agentic loop
│   ├── scoring.py            # Beta-Binomial Bayesian gap scoring
│   ├── orchestrator.py       # Core eval loop (no arg parsing)
│   └── prompts.py            # System prompts
├── main.py                   # Arg parsing → build config → call orchestrator
├── pyproject.toml
└── .env
```

## Tool Design (simplified)

### Solver tool: `submit_proof`
- Input: `proof` (string) — just the proof body, not the full file
- The **harness** assembles: `{imports}\n\n{theorem_statement}\n{proof}`
- Compiles the assembled file. Checks: no errors AND no `sorry` in the proof.
- Success → return success, solver is done
- Failure → return compilation errors to solver for retry
- Budget = number of `submit_proof` calls (not LLM turns)

### Generator tool: `propose_theorem`
- Input: `theorem_statement` (string), `proof` (string), `imports` (string, optional)
- Harness assembles full file, compiles it
- If compilation fails → return errors to generator for iteration
- If compilation succeeds → **auto-submit** to solvers, return anonymized results
- This means one tool does both "check my proof" and "submit to solvers"

## Implementation Steps

### Step 1: `pyproject.toml` + dependencies
- Add: `anthropic>=0.52.0`, `python-dotenv>=1.1.0`, `scipy>=1.15.0`

### Step 2: `src/config.py`
```python
@dataclass(frozen=True)
class ModelConfig:
    model_id: str           # "claude-sonnet-4-6-20250514"
    display_name: str       # "sonnet-4.6" (logging only)
    max_tokens: int = 16384

@dataclass(frozen=True)
class EvalConfig:
    generator_model: ModelConfig
    solver_models: list[ModelConfig]  # must include generator's model
    rounds: int = 3
    solver_max_calls: int = 10        # max submit_proof calls per solver attempt
    generator_max_calls: int = 20     # max propose_theorem calls per round
    attempts_during_loop: int = 1
    attempts_reeval: int = 3
    lean_timeout_seconds: int = 120
    lean_project_path: str = "lean_solver"
    output_dir: str = "results"
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
```

### Step 3: `src/lean.py`
- `LeanResult(frozen)`: `success: bool`, `stdout: str`, `stderr: str`, `return_code: int`
- `LeanCompiler`:
  - `__init__(project_path, timeout)` — computes `LEAN_PATH` from `.lake/packages/*/lib/lean` dirs
  - `check(lean_code: str) -> LeanResult` — writes `LeanSolver/Check_{uuid8}.lean`, runs `lake env lean` with computed `LEAN_PATH`, cleans up file
  - `assemble(imports, theorem_statement, proof) -> str` — builds complete `.lean` file content
  - `has_sorry(code) -> bool` — checks if proof contains `sorry`
  - `asyncio.Semaphore(4)` for concurrency limit

Why `lake env lean` over `lake build`: `lake build` holds a workspace lock that serializes all builds. `lake env lean` runs the lean binary directly with include paths, enabling parallel compilation across solvers.

### Step 4: `src/models.py`
- `LLMClient`: wraps `anthropic.AsyncAnthropic`
  - `create()` classmethod
  - `send(model, system, messages, tools, max_tokens) -> Message`
  - Retry with exponential backoff on `RateLimitError` (3 attempts)

### Step 5: `src/prompts.py`
- `GENERATOR_SYSTEM`: Design theorems exploiting capability gaps. Avoid trivially-searchable theorems. Explains `propose_theorem` workflow.
- `SOLVER_SYSTEM`: Prove the given theorem by replacing `sorry`. Use search tactics, term proofs. Explains `submit_proof` workflow.

### Step 6: `src/solver.py`
- `SolverAttemptResult(frozen)`: `solved`, `proof_code`, `error`, `calls_used`
- `run_solver(config, model, llm, lean, theorem_statement, imports) -> SolverAttemptResult`
- Tool: `submit_proof(proof: str)`
  - Harness calls `lean.assemble(imports, theorem_statement, proof)` → `lean.check()` + `lean.has_sorry(proof)`
  - Success → return immediately
  - Failure → return errors, continue loop
- Budget = `config.solver_max_calls` submit_proof invocations
- Fresh conversation per attempt

### Step 7: `src/generator.py`
- `TheoremSubmission(frozen)`: `theorem_statement`, `proof`, `imports`, `round_number`
- `SolveResult(frozen)`: `anonymized_name`, `solved`, `attempts`
- `RoundResult(frozen)`: `round_number`, `submission`, `solve_results`
- `run_generator(config, llm, lean, solve_callback) -> list[RoundResult]`
- Tool: `propose_theorem(theorem_statement, proof, imports?)`
  - Harness assembles + compiles
  - Fail → return errors, generator iterates (uses a call)
  - Pass → auto-submit to solvers via `solve_callback`, return anonymized results, advance round
- Budget = `config.generator_max_calls` per round (covers both failed and successful proposals)
- Single persistent conversation across all rounds (learns from feedback)
- Anonymization: generator sees `model_self`, `model_a`, `model_b`

### Step 8: `src/scoring.py`
- `TheoremScore(frozen)`: `theorem_id`, `self_solve_rate`, `other_solve_rates`, `gap_score`, `raw_gap`
- `score_theorem(self_successes, self_attempts, other_results: dict[str, (succ, att)], prior_alpha, prior_beta) -> TheoremScore`
  - Posterior: `Beta(alpha + s, beta + n - s)` per model
  - Gap = `P(self > max(others))` via 10k Monte Carlo samples from posteriors
- `select_best(scores) -> TheoremScore` — highest gap score

### Step 9: `src/orchestrator.py`
- `run_eval(config: EvalConfig) -> dict`
- Flow:
  1. Load `.env`, create `LLMClient`, `LeanCompiler`
  2. Build anonymization map (shuffled): generator model → `model_self`, others → `model_a`, `model_b`, ...
  3. Define `solve_callback`: fans out to all solvers via `asyncio.gather`
  4. Run `run_generator(config, llm, lean, solve_callback)` → `round_results`
  5. Score each round's theorem during the loop (with `attempts_during_loop` data)
  6. Select best submission by gap score
  7. Re-evaluate best: run all solvers with `attempts_reeval` attempts each (run ALL attempts, no early stop)
  8. Compute final gap score from re-eval data
  9. Write JSON output

### Step 10: `main.py`
- `argparse` with: `--rounds`, `--solver-max-calls`, `--generator-max-calls`, `--lean-timeout`, `--output-dir`, `--generator-model`, `--solver-models`
- Build `EvalConfig`, call `asyncio.run(run_eval(config))`
- Default models: Opus 4.6 generator, [Opus 4.6, Sonnet 4.6, Haiku 4.5] solvers

## Scoring System

**Per-theorem during loop** (1 attempt each):
- Each solver: s/n → posterior Beta(1+s, 1+n-s)
- Gap score = P(self_rate > max(opponent_rates)) via Monte Carlo
- Best theorem = highest gap score across rounds

**Re-evaluation** (3 attempts, no early stopping):
- Run ALL attempts for every solver on the best theorem
- Recompute posteriors with more data → tighter estimates → reliable final score

## Verification
```bash
# Unit: test lean compilation
uv run python -c "import asyncio; from src.lean import LeanCompiler; ..."

# Smoke test (1 round, cheap model)
uv run python main.py --rounds 1 --generator-model claude-haiku-4-5-20250414

# Full run
uv run python main.py --rounds 3
```

## Critical Files
- `src/lean.py` — most infrastructure-critical, must handle LEAN_PATH correctly
- `src/generator.py` — most complex agentic loop
- `src/orchestrator.py` — ties everything together
- `lean_solver/lakefile.toml` — do not modify

</details>

## Design

I decided to go with the simplest design that would get results as quickly as possible:
* Custom prompting for each model (generator and solver)
* Custom LLM client and agentic loop rather than something like `claude -p`
* Local Lean install rather than Docker

## Results

We ran 9 evals across different configurations. Full JSON results are under `results/`.

### Key finding: Opus and Sonnet cannot differentiate each other

In head-to-head runs (opus vs sonnet, no haiku), both models solve nearly every theorem the other generates. Final gap scores cluster around 0.50 (pure noise), with raw gaps of 0.0 in most rounds. Neither model can reliably craft theorems that the other fails to prove.

| Run | Config | Opus gap | Sonnet gap | Winner | Notes |
|-----|--------|----------|------------|--------|-------|
| `233829` | 2-solver, 3 rnd, detailed | 0.502 | 0.503 | tied | All rounds 3/3 for both |
| `232028` | 2-solver, 3 rnd, minimal | 0.507 | 0.494 | opus (marginal) | All rounds 3/3 for both |
| `232709` | 2-solver, 3 rnd, detailed | 0.503 | 0.236 | opus | Sonnet's best theorem backfired in reeval |
| `234857` | 2-solver, 10 rnd, minimal, full budget | 0.504 | 0.030 | opus | Sonnet's round_10 reversed in reeval (8/10 loop -> 22/32 reeval) |

In the full 10-round run (`234857`), Opus generated 10 theorems spanning number theory, algebra, and combinatorics. Sonnet solved 10/10 on 9 of them. The only partial gap was Opus's round_10 (`24 | p^2 - 1` for primes >= 5, solved 8/10 by opus vs 10/10 by sonnet) — a *negative* gap. Sonnet similarly failed to find separation: its best theorem (`exists_prime_between_n_and_factorial_plus_one`) showed a small loop gap (8/10 self vs 7/10 opus) that collapsed in reeval.

### Haiku provides the only real gaps

The only run that found a genuine gap was `eval_213139` (opus as sole generator against opus+sonnet+haiku):

| Round | Theorem | Opus | Sonnet | Haiku |
|-------|---------|------|--------|-------|
| 1 | `(3^(2n+1) + 2^(2n+1)) % 5 = 0` | 1/1 | 1/1 | 0/1 |
| 2 | `16 | 5^n - 4n - 1` | 1/1 | 0/1 | 0/1 |
| 3 | `64 | 3^(2n+2) - 8n - 9` | 1/1 | 0/1 | 0/1 |

Re-eval on round_2: opus 80%, sonnet 40%, haiku 20%. **Final gap: 0.919** — the highest gap score across all runs.

In the 3-solver multi-generator runs, Sonnet consistently outscored Opus as a generator because Sonnet's theorems tripped haiku while Opus's theorems were solved by everyone:

| Run | Opus gap | Sonnet gap | Winner |
|-----|----------|------------|--------|
| `223650` (10 rnd) | 0.303 | 0.500 | sonnet |
| `230918` (3 rnd) | 0.332 | 0.497 | sonnet |

### Reeval reversals

A recurring pattern: theorems that show a gap in the loop (3-10 attempts) lose that gap in re-evaluation (10-30 attempts). This happened in `eval_232709` (Sonnet's fibonacci identity: loop gap 0.985 -> reeval gap 0.236) and `eval_234857` (Sonnet's prime existence: loop gap 0.680 -> reeval gap 0.030). The loop sample sizes are too small for reliable selection of the "best" theorem, leading the re-eval to contradict the loop signal.

### Cost

The full 10-round run (`234857`, 10 attempts/loop, 30 reeval) cost **$76.33** ($63.70 opus, $12.63 sonnet). A typical 3-round run costs $6-11. Opus API costs are ~5x sonnet due to token pricing differences.

For full results, see the run results JSON files under `results/`.

## Limitations

* We limit ourselves to only the Claude family of models - OpenAI support is implemented but not tested because I don't have a working OpenAI API key at the moment.
* The generator model budget was somewhat underspecified in the problem statement. I interpreted this as **the generator gets a fixed number of submissions per round and a fixed number of rounds per eval.** If instead there should be a token limit, we would have to implement this.
* The eval results depend heavily on which models you choose to use as solvers, and the main `gap_score` result doesn't mean much when any of the solvers are generally stronger than the generator. The per-opponent results are more meaningful. Gap scores are also not directly comparable across generators if they face a different effective opponent pool.
* I didn't have time to run a `--full-run` run - doing so would take on the order of 6 hours or so by my estimate. I ran several test runs with low budgets to iron things out and a few runs with larger budgets, but only using Opus & Sonnet 4.6 as generators/solvers, and sometimes Haiku 4.5 as a solver.

## Extensions

* Token-based generator/solver budgets rather than attempt-based budgets
* Larger runs take a while - a run dashboard would help with monitoring run status
* An Elo-style ranking across models would complement the gap score — the gap score measures a generator's ability to find problems that specifically differentiate it from opponents, while Elo would measure overall relative math proving strength.
* Use prompting to intentionally diversify the math domains that the generators operate in to check whether math capabilities are jagged across domains.
