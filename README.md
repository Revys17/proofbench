# ProofBench

An AI evaluation where a **generator** model crafts Lean 4 theorems that a fresh copy of itself can solve, but other models can't. The generator iterates based on solver feedback, and the best submission is scored via Bayesian gap analysis.

Most benchmarks are static and eventually saturate. This one scales with model capability — stronger models craft harder problems.

## How It Works

1. The **generator** model creates a Lean 4 theorem and proves it.
2. The proof is stripped. The bare theorem statement is sent to **solver** models (including a fresh copy of the generator).
3. Each solver independently tries to produce a proof. Either it type-checks or it doesn't — no partial credit.
4. The generator sees anonymized results (`model_self` solved, `model_a` failed, etc.) and iterates.
5. After all rounds, the best submission (highest gap score) is re-evaluated with more attempts for a reliable final score.

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [elan](https://github.com/leanprover/elan) (Lean version manager)

### Installation

```bash
uv sync
```

### Lean Project

The `lean_solver/` directory contains a Lean 4.29.0 project with Mathlib v4.29.0. On first setup:

```bash
cd lean_solver
lake exe cache get   # Downloads pre-compiled Mathlib (~5GB, takes a few minutes)
cd ..
```

### API Keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...          # Only needed if using OpenAI models
```

The `.env` file overrides any keys in your shell environment.

## Usage

```bash
uv run python main.py [OPTIONS]
```

### Quick Start

```bash
# Smoke test (1 round, cheap model)
uv run python main.py --rounds 1 --generator-models haiku --solver-models haiku sonnet

# Standard run (Opus generates, all models solve)
uv run python main.py --generator-models opus --solver-models sonnet haiku

# Multi-generator comparison
uv run python main.py --generator-models opus sonnet --solver-models opus sonnet haiku

# Full production run (all available models, high attempt counts)
uv run python main.py --full-run -v
```

### CLI Parameters

`--full-run` is a production preset that selects all available models and higher attempt counts. Individual flags override the preset. Example: `uv run python main.py --full-run --rounds 5 -v`

| Parameter | Default | `--full-run` | Description |
|---|---|---|---|
| `--full-run` | off | — | Enable production preset (see this column for values). |
| `--rounds N` | `3` | `10` | Number of theorems the generator submits per eval. Each round = create theorem, verify proof, fan out to solvers, receive feedback. |
| `--generator-models M [M ...]` | `opus` | all available | Which model(s) act as generator. Each gets an independent eval run. Use model keys (`opus`, `sonnet`, `haiku`) or full IDs (`claude-opus-4-6`). |
| `--solver-models M [M ...]` | `opus sonnet haiku` | all available | Which models solve the theorems. The generator's own model is auto-included as `model_self`. |
| `--solver-max-calls N` | `10` | `10` | Max `submit_proof` tool calls per solver attempt. This is the solver's budget — each call compiles a proof and returns success/failure. |
| `--generator-max-calls N` | `20` | `20` | Max `propose_theorem` tool calls per generator round. Failed compilations count against this budget. |
| `--attempts-during-loop N` | `3` | `10` | How many independent solver attempts per model during the generator loop. More attempts = better P(solve) estimates for selecting the best theorem, but more API cost. |
| `--attempts-reeval N` | `10` | `30` | How many independent solver attempts per model during final re-evaluation of the best theorem. More = tighter Bayesian posteriors = more reliable gap score. |
| `--lean-timeout N` | `120` | `120` | Seconds before a Lean compilation is killed. |
| `--lean-project-path PATH` | `lean_solver` | `lean_solver` | Path to the Lean 4 project directory. |
| `--output-dir PATH` | `results` | `results` | Where JSON result files and per-generator logs are written. |
| `--seed N` | random | random | Random seed for scoring reproducibility (Monte Carlo sampling). |
| `--max-concurrent-api N` | `20` | `20` | Max concurrent API calls per provider. Controls parallelism. |
| `--summarize-rounds` | off | off | At each round boundary, ask the generator to summarize its work so far, then replace the conversation history with that summary. Reduces token cost on long runs at the expense of one extra API call per round. |
| `--verbose`, `-v` | off | off | Enable debug logging from `src.*` modules. |

### Known Model Keys

| Key | Model ID | Provider |
|---|---|---|
| `opus` | `claude-opus-4-6` | Anthropic |
| `opus-4.5` | `claude-opus-4-5-20251101` | Anthropic |
| `opus-4.1` | `claude-opus-4-1-20250805` | Anthropic |
| `opus-4` | `claude-opus-4-20250514` | Anthropic |
| `sonnet` | `claude-sonnet-4-6` | Anthropic |
| `sonnet-4.5` | `claude-sonnet-4-5-20250929` | Anthropic |
| `sonnet-4` | `claude-sonnet-4-20250514` | Anthropic |
| `haiku` | `claude-haiku-4-5-20251001` | Anthropic |
| `gpt-5.4` | `gpt-5.4` | OpenAI |
| `gpt-5.4-mini` | `gpt-5.4-mini` | OpenAI |
| `gpt-5-mini` | `gpt-5-mini` | OpenAI |
| `gpt-5` | `gpt-5` | OpenAI |
| `gpt-5.4-pro` | `gpt-5.4-pro` | OpenAI |
| `gpt-5.4-nano` | `gpt-5.4-nano` | OpenAI |
| `gpt-5-nano` | `gpt-5-nano` | OpenAI |
| `gpt-4.1` | `gpt-4.1` | OpenAI |

You can also pass full model IDs directly (e.g. `--solver-models claude-sonnet-4-6`).

> **Note:** OpenAI models are supported in code but currently commented out in the model registry (`src/config.py`). To enable them, uncomment the entries and set `OPENAI_API_KEY` in `.env`.

## Architecture

```
proofbench/
├── main.py              # CLI entry point, arg parsing, config building
├── src/
│   ├── config.py        # ModelConfig, EvalConfig dataclasses, model registry
│   ├── models.py        # Provider-agnostic LLM client (Anthropic + OpenAI)
│   ├── lean.py          # Lean compilation via `lake env lean`, proof caching
│   ├── prompts.py       # System prompts for generator and solver
│   ├── generator.py     # Generator agentic loop (propose_theorem tool)
│   ├── solver.py        # Solver agentic loop (submit_proof tool)
│   ├── scoring.py       # Bayesian gap scoring (Beta posteriors, Monte Carlo)
│   ├── costs.py         # Per-model token/cost tracking
│   └── orchestrator.py  # Eval orchestration, solver fanout, re-evaluation
└── lean_solver/         # Lean 4.29.0 + Mathlib v4.29.0 project
```

### Key Design Decisions

- **`lake env lean`** for compilation instead of `lake build` — avoids workspace lock, enables parallel proof checking across solvers.
- **Proof cache** — SHA-256 hash of lean code to cached result. Same proof never compiled twice.
- **Provider abstraction** — `AnthropicClient` and `OpenAIClient` normalize tool call formats behind a common `LLMClient` interface. Solver/generator code is provider-agnostic.
- **`ClientRegistry`** — lazily creates one LLM client per provider, shared across the eval.
- **Parallel execution** — solvers for a theorem run in parallel across models AND across attempts within a model (via `asyncio.gather`).
- **Anonymization** — generator sees `model_self` for its own fresh copy, `model_a`/`model_b`/etc. for opponents. Opponent order is shuffled per eval.

## Scoring

Each theorem gets a **gap score**: the probability that the generator's model has a higher true solve rate than the best opponent.

1. Each model's solve rate is modeled as a Beta random variable with a uniform prior: `Beta(1, 1)`.
2. After observing `s` successes in `n` attempts, the posterior is `Beta(1 + s, 1 + n - s)`.
3. The gap score = `P(self_rate > max(opponent_rates))`, estimated by drawing 10,000 Monte Carlo samples from each posterior.

Example: if `model_self` solves 3/3 and the best opponent solves 1/3:
- self ~ `Beta(4, 1)`, opponent ~ `Beta(2, 3)`
- gap score ~ 0.92 (high confidence self is better)

The generator's best theorem (highest gap score during the loop) is re-evaluated with more attempts for a reliable final score.

## Output

Results are written to `results/eval_YYYYMMDD_HHMMSS.json` containing:

- Per-round theorem statements, proofs, and solver results
- Loop-phase gap scores for each round
- Final re-evaluation gap score for the best theorem
- Cost breakdown (tokens and USD per model)
- Full anonymization mapping

When running multiple generators, per-generator log files are written to `results/logs/`.
