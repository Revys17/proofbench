# ProofBench — Programming Exercise

A shortcoming of most AI evaluations is that they saturate, which limits their useful lifetime. The exercise here is to make a particular kind of evaluation that won't saturate easily because it scales up as AI improves. In particular, the AI model being tested is itself put to the task of creating the evaluation — in this case, to come up with a math problem that it would be capable of solving, but that other models cannot.

## Description

Make an evaluation where a model tries to come up with a Lean 4 math problem that a fresh instance of itself can solve, but that other models can't. The problem is a formal theorem statement, and "solving" it means producing a proof that Lean's type checker accepts.

**The setup:**

1. A model (call it the **generator**) crafts a theorem and proves it.
2. The proof is stripped away. The bare theorem statement is handed to several models (the **solvers**), including a fresh copy of the generator's own model.
3. Each solver independently tries to produce a proof. Either the proof type-checks or it doesn't — no partial credit.
4. The generator sees the results: who solved it, who didn't.
5. The generator can submit more problems, adjusting difficulty based on what happened.

The challenge for the generator is finding the right difficulty. Too easy and all solvers nail it. Too hard and even a fresh copy of itself fails. The score comes from whichever submission had the biggest gap between "my model can solve this" and "other models can't."

Most benchmarks are static and eventually saturate. This one scales with model capability — stronger models craft harder problems.

## Goal

Build the generator loop, the solver loop, the Lean compilation layer, and the scoring logic. If you can't finish in the time, just make as much progress as possible. Prefer to have code that works and does something (even if it falls short of the overall requirements) to a more ambitious approach that doesn't run.

## Core Requirements

### 1. Generator Agent

An agentic loop where the generator model:

- Creates Lean 4 theorem statements and proves them
- Submits problems for evaluation against solver models
- Sees solver results and iterates to improve difficulty
- Is scored on its best submission (the one with the biggest gap between its solve rate and opponents')

The generator should have a tool that:

- Compiles the proof to verify it's valid
- If the proof doesn't compile or has `sorry`: returns the compiler errors (acts as a development tool)
- If the proof compiles clean: strips the proof, runs solver agents on the theorem, and returns solve rates

### 2. Solver Agent

An agentic loop where a solver model attempts to prove a theorem:

- Receives a theorem statement (no proof)
- Has a tool `submit_proof(proof)` — provides only the proof body
- The harness assembles the full file (imports + theorem + proof) and compiles it
- Success = compiles with no errors and no `sorry`
- Gets a fixed step budget (tool call limit)

### 3. Lean Compilation

- Use a Lean 4 + Mathlib installation
- Solvers can use `exact?`, `apply?`, `rw?` as search tactics

### 4. Scoring

- Each submission gets a gap score: P(generator's solve rate > best opponent's solve rate), estimated with Bayesian inference (Beta posteriors)
- The generator's best submission (highest gap score) is selected
- After the generator finishes, the harness re-evaluates the best submission with more solver attempts for a reliable final score

### 5. Configuration

- Support multiple solver models (anonymized as `model_a`, `model_b`, etc.)
- The generator's own model also runs as a solver (`model_self`)
- Configurable step limits for generator and solver
- Configurable number of solver attempts per model

## What Full Success Looks Like

- The generator creates theorems, submits them, sees results, and iterates
- At least some submissions show a gap (generator solves, some opponents don't)
- Final evaluation confirms the gap with more attempts
- JSON results file with all submissions, solve rates, and scores
- High code quality, good design decisions

## Bonus Features

- **Identify a gap between similar models** — you find math problems that achieve a high gap score among models with similar math capabilities
- **Solver runtime** — the eval provides solver models with at least 10 attempts for finding the proof (you may want to reduce this number while iterating on a design)
- **Fast Lean compilation** — find a way to speed this up
- **Parallel solver execution** — run multiple solver models/attempts concurrently
- **Adaptive feedback** — guide the generator based on results (too hard, too easy, need more certainty)
- **Multiple generator models** — run several models as generator and compare results
