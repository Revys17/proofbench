import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .config import EvalConfig, PromptLevel
from .lean import LeanCompiler
from .models import LLMClient, ToolDef, ToolResult
from .prompts import GENERATOR_SYSTEMS

log = logging.getLogger(__name__)

PROPOSE_THEOREM_TOOL: ToolDef = {
    "name": "propose_theorem",
    "description": (
        "Propose a theorem with its proof. The harness compiles it. "
        "If compilation fails, you get the error back. "
        "If it succeeds (no errors, no sorry), it is automatically submitted "
        "to solver models and you receive their anonymized results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "theorem_statement": {
                "type": "string",
                "description": (
                    "The theorem declaration WITHOUT the proof. "
                    "E.g. 'theorem my_thm (n : Nat) : n + 0 = n'"
                ),
            },
            "proof": {
                "type": "string",
                "description": "The proof, e.g. ':= by simp'",
            },
            "imports": {
                "type": "string",
                "description": "Lean import statements. Default: 'import Mathlib'",
            },
        },
        "required": ["theorem_statement", "proof"],
    },
}


@dataclass(frozen=True)
class TheoremSubmission:
    theorem_statement: str
    proof: str
    imports: str
    round_number: int


@dataclass(frozen=True)
class SolveResult:
    anonymized_name: str
    solved: bool
    successes: int
    attempts: int


@dataclass(frozen=True)
class RoundResult:
    round_number: int
    submission: TheoremSubmission
    solve_results: tuple[SolveResult, ...]


SolveCallback = Callable[[TheoremSubmission], Awaitable[tuple[SolveResult, ...]]]
RoundCallback = Callable[[RoundResult, list[Any]], Awaitable[None]]
SkipCallback = Callable[[int, list[Any]], Awaitable[None]]


def _build_solve_feedback(
    solve_results: tuple[SolveResult, ...],
    current_round: int,
    total_rounds: int,
    prompt_level: PromptLevel,
) -> str:
    """Build the tool-result message after a successful compilation + solver run."""
    self_result = next(
        (sr for sr in solve_results if sr.anonymized_name == "model_self"),
        None,
    )
    opponent_results = [
        sr for sr in solve_results if sr.anonymized_name != "model_self"
    ]
    self_rate = (
        (self_result.successes / self_result.attempts)
        if self_result and self_result.attempts > 0
        else 0.0
    )
    best_opp_rate = max(
        (sr.successes / sr.attempts for sr in opponent_results if sr.attempts > 0),
        default=0.0,
    )

    lines = [f"Round {current_round}/{total_rounds} solver results:"]
    for sr in solve_results:
        rate = sr.successes / sr.attempts if sr.attempts > 0 else 0.0
        lines.append(
            f"  {sr.anonymized_name}: {sr.successes}/{sr.attempts} "
            f"({rate:.0%} solve rate)"
        )

    # Standard: add a one-line gap summary (factual, no advice)
    if prompt_level == PromptLevel.STANDARD:
        lines.append("")
        lines.append(
            f"Your model: {self_rate:.0%}, best opponent: {best_opp_rate:.0%}."
        )

    # Detailed: full diagnostic feedback with prescriptive advice
    if prompt_level == PromptLevel.DETAILED:
        lines.append("")
        if self_rate == 0:
            lines.append(
                "PROBLEM: Your own model solved 0% — this scores zero regardless "
                "of opponent results. Significantly simplify your next theorem. "
                "A theorem you reliably solve (>80%) that opponents also solve "
                "is better than one nobody can solve."
            )
        elif self_rate < 0.5:
            lines.append(
                f"WARNING: Your model's solve rate ({self_rate:.0%}) is too low. "
                f"Even if opponents fail, low self-rate means uncertain posteriors "
                f"and a weak gap score. Target 80%+ self-solve rate. Simplify."
            )
        elif self_rate >= 0.8 and best_opp_rate <= 0.3:
            lines.append(
                f"STRONG: Your model at {self_rate:.0%}, best opponent at "
                f"{best_opp_rate:.0%}. This is a high-quality submission."
            )
        elif self_rate >= 0.5 and best_opp_rate < self_rate:
            gap = self_rate - best_opp_rate
            lines.append(
                f"DECENT: Your model at {self_rate:.0%}, best opponent at "
                f"{best_opp_rate:.0%} (gap: {gap:.0%}). Try to increase your "
                f"model's reliability while keeping opponent rate low."
            )
        elif best_opp_rate >= self_rate:
            lines.append(
                f"NO GAP: Best opponent ({best_opp_rate:.0%}) matches or beats "
                f"your model ({self_rate:.0%}). The theorem doesn't differentiate. "
                f"Try a different mathematical area or proof strategy."
            )

    if current_round < total_rounds:
        lines.append(f"\nProceed to Round {current_round + 1}.")

    return "\n".join(lines)


def _build_budget_exhausted_feedback(
    error_msg: str,
    skipped_round: int,
    remaining_rounds: int,
    prompt_level: PromptLevel,
) -> str:
    """Build the tool-result message when the call budget is exhausted."""
    lines = [
        f"Compilation FAILED. Call budget exhausted — Round {skipped_round} "
        f"is SKIPPED (no theorem submitted, scores zero).",
        error_msg,
        "",
        f"You have {remaining_rounds} round(s) remaining.",
    ]
    if prompt_level == PromptLevel.DETAILED:
        lines.append(
            "Simplify your approach — target a theorem you can prove "
            "in fewer attempts. A submitted theorem that everyone solves "
            "is better than no submission at all."
        )
    return "\n".join(lines)


async def run_generator(
    config: EvalConfig,
    llm: LLMClient,
    lean: LeanCompiler,
    solve_callback: SolveCallback,
    *,
    start_round: int = 0,
    initial_messages: list[Any] | None = None,
    prior_results: list[RoundResult] | None = None,
    on_round_complete: RoundCallback | None = None,
    on_round_skipped: SkipCallback | None = None,
    anon_to_display: dict[str, str] | None = None,
) -> list[RoundResult]:
    """Run the generator for `config.rounds` rounds, returning results per round.

    Args:
        anon_to_display: maps anonymized names to display names for log output

    For resume support:
        start_round: round to resume from (0-indexed, number of completed rounds)
        initial_messages: conversation history to restore
        prior_results: RoundResults from completed rounds
        on_round_complete: called after each successful round with (result, messages)
        on_round_skipped: called after each skipped round with (round_number, messages)
    """
    system_prompt = GENERATOR_SYSTEMS[config.prompt_level]
    tools = [PROPOSE_THEOREM_TOOL]

    if initial_messages is not None:
        messages: list[Any] = initial_messages
    else:
        messages = [{
            "role": "user",
            "content": (
                f"Design and submit {config.rounds} theorem(s), one per round. "
                f"You have {config.generator_max_calls} propose_theorem calls per round "
                f"(failed compilations count against this budget). "
                f"Each theorem should be solvable by a fresh copy of your model "
                f"but not by other models. Start with Round 1."
            ),
        }]

    round_results: list[RoundResult] = list(prior_results) if prior_results else []
    current_round = start_round
    calls_this_round = 0

    if start_round > 0:
        log.info("Resuming generator from round %d/%d", start_round + 1, config.rounds)

    while current_round < config.rounds:
        _round_completed_this_iter = False
        _round_skipped_this_iter = False

        response = await llm.send(
            model=config.generator_model.model_id,
            system=system_prompt,
            messages=messages,
            tools=tools,
            max_tokens=config.generator_model.max_tokens,
        )

        messages.extend(llm.format_assistant(response))

        if not response.tool_calls:
            if current_round >= config.rounds:
                break
            messages.append({
                "role": "user",
                "content": "Please propose a theorem using the propose_theorem tool.",
            })
            continue

        tool_results: list[ToolResult] = []
        for tc in response.tool_calls:
            if tc.name != "propose_theorem":
                tool_results.append(ToolResult(
                    tool_call_id=tc.id,
                    content="Unknown tool. Use propose_theorem.",
                    is_error=True,
                ))
                continue

            calls_this_round += 1
            theorem_statement = tc.input.get("theorem_statement", "")
            proof = tc.input.get("proof", "")
            imports = tc.input.get("imports", "import Mathlib")

            full_code = LeanCompiler.assemble(imports, theorem_statement, proof)
            result = await lean.check(full_code)

            if not result.success:
                remaining = config.generator_max_calls - calls_this_round
                error_msg = result.stdout or result.stderr or "Compilation failed."
                if result.has_sorry:
                    error_msg = "Proof contains sorry — not accepted.\n" + error_msg

                if calls_this_round >= config.generator_max_calls:
                    skipped_round = current_round + 1
                    remaining_rounds = config.rounds - skipped_round
                    tool_results.append(ToolResult(
                        tool_call_id=tc.id,
                        content=_build_budget_exhausted_feedback(
                            error_msg, skipped_round, remaining_rounds,
                            config.prompt_level,
                        ),
                    ))
                    log.warning(
                        "Generator exhausted call budget for round %d without a valid proof "
                        "(%d calls used, no theorem submitted)",
                        skipped_round,
                        calls_this_round,
                    )
                    current_round += 1
                    calls_this_round = 0
                    _round_skipped_this_iter = True
                else:
                    tool_results.append(ToolResult(
                        tool_call_id=tc.id,
                        content=(
                            f"Compilation FAILED ({remaining} calls remaining this round):\n"
                            f"{error_msg}"
                        ),
                    ))
                continue

            # Proof compiled — submit to solvers
            current_round += 1
            _round_completed_this_iter = True
            submission = TheoremSubmission(
                theorem_statement=theorem_statement,
                proof=proof,
                imports=imports,
                round_number=current_round,
            )
            log.info("Round %d: theorem compiled, submitting to solvers", current_round)

            solve_results = await solve_callback(submission)
            round_results.append(RoundResult(current_round, submission, solve_results))

            solver_summary = ", ".join(
                f"{(anon_to_display or {}).get(sr.anonymized_name, sr.anonymized_name)}: "
                f"{sr.successes}/{sr.attempts} solved"
                for sr in solve_results
            )
            log.info("Round %d results: %s", current_round, solver_summary)

            tool_results.append(ToolResult(
                tool_call_id=tc.id,
                content=_build_solve_feedback(
                    solve_results, current_round, config.rounds,
                    config.prompt_level,
                ),
            ))
            calls_this_round = 0

        messages.extend(llm.format_tool_results(tool_results))

        # After a successful submission, optionally compact the conversation
        # by asking the generator to summarize what it's learned so far.
        if (config.summarize_rounds
                and round_results
                and round_results[-1].round_number == current_round
                and current_round < config.rounds):
            summary = await _request_summary(config, llm, messages)
            messages = [
                messages[0],
                {"role": "user", "content": summary},
            ]

        # Checkpoint callbacks — only fire when a round actually completed or
        # was skipped THIS iteration (not on every subsequent loop turn).
        if _round_completed_this_iter:
            if on_round_complete:
                await on_round_complete(round_results[-1], messages)
        elif _round_skipped_this_iter:
            if on_round_skipped:
                await on_round_skipped(current_round, messages)

    return round_results


_SUMMARY_PROMPT = (
    "Before continuing, write a brief summary of your work so far: "
    "what theorems you tried, what compilation errors you hit, what "
    "the solver results were, and what strategies you want to try next. "
    "This summary will replace the conversation history to save context, "
    "so include everything you need to remember."
)


async def _request_summary(
    config: EvalConfig,
    llm: LLMClient,
    messages: list[Any],
) -> str:
    """Ask the generator to summarize its work so far."""
    summary_messages = [*messages, {"role": "user", "content": _SUMMARY_PROMPT}]
    response = await llm.send(
        model=config.generator_model.model_id,
        system=GENERATOR_SYSTEMS[config.prompt_level],
        messages=summary_messages,
        tools=[],
        max_tokens=config.generator_model.max_tokens,
    )
    return response.text or "No summary provided."
