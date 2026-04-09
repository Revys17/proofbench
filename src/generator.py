import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from anthropic.types import ToolParam

from .config import EvalConfig
from .lean import LeanCompiler
from .models import LLMClient
from .prompts import GENERATOR_SYSTEM

log = logging.getLogger(__name__)

PROPOSE_THEOREM_TOOL: ToolParam = {
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
    attempts: int


@dataclass(frozen=True)
class RoundResult:
    round_number: int
    submission: TheoremSubmission
    solve_results: tuple[SolveResult, ...]


SolveCallback = Callable[[TheoremSubmission], Awaitable[tuple[SolveResult, ...]]]


async def run_generator(
    config: EvalConfig,
    llm: LLMClient,
    lean: LeanCompiler,
    solve_callback: SolveCallback,
) -> list[RoundResult]:
    """Run the generator for `config.rounds` rounds, returning results per round."""
    tools = [PROPOSE_THEOREM_TOOL]
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

    round_results: list[RoundResult] = []
    current_round = 0
    calls_this_round = 0

    while current_round < config.rounds:
        response = await llm.send(
            model=config.generator_model.model_id,
            system=GENERATOR_SYSTEM,
            messages=messages,
            tools=tools,
            max_tokens=config.generator_model.max_tokens,
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            if current_round >= config.rounds:
                break
            messages.append({
                "role": "user",
                "content": "Please propose a theorem using the propose_theorem tool.",
            })
            continue

        tool_results = []
        for block in tool_uses:
            if block.name != "propose_theorem":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Unknown tool. Use propose_theorem.",
                    "is_error": True,
                })
                continue

            calls_this_round += 1
            theorem_statement = block.input.get("theorem_statement", "")
            proof = block.input.get("proof", "")
            imports = block.input.get("imports", "import Mathlib")

            full_code = LeanCompiler.assemble(imports, theorem_statement, proof)
            result = await lean.check(full_code)

            if not result.success:
                remaining = config.generator_max_calls - calls_this_round
                error_msg = result.stdout or result.stderr or "Compilation failed."
                if result.has_sorry:
                    error_msg = "Proof contains sorry — not accepted.\n" + error_msg

                if calls_this_round >= config.generator_max_calls:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": (
                            f"Compilation FAILED. Call budget exhausted for this round.\n"
                            f"{error_msg}"
                        ),
                    })
                    log.warning(
                        "Generator exhausted call budget for round %d without a valid proof "
                        "(%d calls used, no theorem submitted)",
                        current_round + 1,
                        calls_this_round,
                    )
                    current_round += 1
                    calls_this_round = 0
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": (
                            f"Compilation FAILED ({remaining} calls remaining this round):\n"
                            f"{error_msg}"
                        ),
                    })
                continue

            # Proof compiled — submit to solvers
            current_round += 1
            submission = TheoremSubmission(
                theorem_statement=theorem_statement,
                proof=proof,
                imports=imports,
                round_number=current_round,
            )
            log.info("Round %d: theorem compiled, submitting to solvers", current_round)

            solve_results = await solve_callback(submission)
            round_results.append(RoundResult(current_round, submission, solve_results))

            self_result = next(
                (sr for sr in solve_results if sr.anonymized_name == "model_self"),
                None,
            )
            opponent_results = [
                sr for sr in solve_results if sr.anonymized_name != "model_self"
            ]
            self_solved = self_result.solved if self_result else False
            opponents_solved = sum(1 for sr in opponent_results if sr.solved)

            result_lines = [f"Round {current_round}/{config.rounds} solver results:"]
            for sr in solve_results:
                status = "SOLVED" if sr.solved else "FAILED"
                result_lines.append(f"  {sr.anonymized_name}: {status}")

            # Diagnostic feedback
            if self_solved and opponents_solved == 0:
                result_lines.append(
                    "\nExcellent — only your model solved it. This is a strong submission."
                )
            elif self_solved and opponents_solved < len(opponent_results):
                result_lines.append(
                    f"\nPartial gap — {opponents_solved}/{len(opponent_results)} opponents "
                    f"also solved it. Try to make it harder for them while keeping it "
                    f"solvable for your model."
                )
            elif self_solved and opponents_solved == len(opponent_results):
                result_lines.append(
                    "\nNo gap — everyone solved it. Increase difficulty significantly."
                )
            elif not self_solved:
                result_lines.append(
                    "\nYour own model failed — this scores zero. Decrease difficulty "
                    "so a fresh copy of your model can solve it."
                )

            if current_round < config.rounds:
                result_lines.append(
                    f"\nProceed to Round {current_round + 1}."
                )

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": "\n".join(result_lines),
            })
            calls_this_round = 0

        messages.append({"role": "user", "content": tool_results})

    return round_results
