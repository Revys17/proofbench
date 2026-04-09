import logging
from dataclasses import dataclass

from anthropic.types import ToolParam

from .config import EvalConfig, ModelConfig
from .lean import LeanCompiler
from .models import LLMClient
from .prompts import SOLVER_SYSTEM

log = logging.getLogger(__name__)

SUBMIT_PROOF_TOOL: ToolParam = {
    "name": "submit_proof",
    "description": (
        "Submit a proof for the given theorem. Provide ONLY the proof "
        "(starting with ':= by' or ':= ...'). The harness assembles the full "
        "file and compiles it. Returns compilation result."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "proof": {
                "type": "string",
                "description": "The proof term or tactic block, e.g. ':= by simp'",
            },
        },
        "required": ["proof"],
    },
}


@dataclass(frozen=True)
class SolverAttemptResult:
    solved: bool
    proof_code: str | None
    error: str | None
    calls_used: int


async def run_solver(
    config: EvalConfig,
    model: ModelConfig,
    llm: LLMClient,
    lean: LeanCompiler,
    theorem_statement: str,
    imports: str,
) -> SolverAttemptResult:
    """Single solver attempt: try to prove the theorem within the call budget."""
    prompt = (
        f"Prove the following Lean 4 theorem.\n\n"
        f"Imports:\n```lean\n{imports}\n```\n\n"
        f"Theorem:\n```lean\n{theorem_statement}\n```\n\n"
        f"Submit only the proof body using submit_proof."
    )
    messages = [{"role": "user", "content": prompt}]
    tools = [SUBMIT_PROOF_TOOL]

    calls_used = 0
    last_error: str | None = None

    for _ in range(config.solver_max_calls + config.solver_max_calls):
        # Allow up to 2x iterations (model may emit text turns without tool use)
        response = await llm.send(
            model=model.model_id,
            system=SOLVER_SYSTEM,
            messages=messages,
            tools=tools,
            max_tokens=model.max_tokens,
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            if response.stop_reason == "end_turn":
                break
            messages.append({
                "role": "user",
                "content": "Please use submit_proof to submit your proof.",
            })
            continue

        tool_results = []
        for block in tool_uses:
            if block.name != "submit_proof":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Unknown tool. Use submit_proof.",
                    "is_error": True,
                })
                continue

            calls_used += 1
            proof = block.input.get("proof", "")
            full_code = LeanCompiler.assemble(imports, theorem_statement, proof)
            result = await lean.check(full_code)

            if result.success:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Proof accepted! Compilation successful, no sorry.",
                })
                messages.append({"role": "user", "content": tool_results})
                return SolverAttemptResult(
                    solved=True,
                    proof_code=proof,
                    error=None,
                    calls_used=calls_used,
                )

            error_msg = result.stdout or result.stderr or "Compilation failed."
            if result.has_sorry:
                error_msg = "Proof contains sorry — not accepted.\n" + error_msg
            last_error = error_msg

            remaining = config.solver_max_calls - calls_used
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": (
                    f"Compilation FAILED ({remaining} attempts remaining):\n"
                    f"{error_msg}"
                ),
            })

            if calls_used >= config.solver_max_calls:
                messages.append({"role": "user", "content": tool_results})
                return SolverAttemptResult(
                    solved=False,
                    proof_code=proof,
                    error=last_error,
                    calls_used=calls_used,
                )

        messages.append({"role": "user", "content": tool_results})

    return SolverAttemptResult(
        solved=False,
        proof_code=None,
        error=last_error or "Budget exhausted without submitting a proof.",
        calls_used=calls_used,
    )
