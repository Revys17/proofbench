from src.config import PromptLevel
from src.prompts import GENERATOR_SYSTEMS, SOLVER_SYSTEM


class TestGeneratorSystems:
    """Tests for tiered generator prompts."""

    def test_all_levels_present(self):
        for level in PromptLevel:
            assert level in GENERATOR_SYSTEMS

    def test_minimal_is_shortest(self):
        assert len(GENERATOR_SYSTEMS[PromptLevel.MINIMAL]) < len(GENERATOR_SYSTEMS[PromptLevel.STANDARD])

    def test_detailed_is_longest(self):
        assert len(GENERATOR_SYSTEMS[PromptLevel.DETAILED]) > len(GENERATOR_SYSTEMS[PromptLevel.STANDARD])

    def test_all_contain_tool_name(self):
        for level, prompt in GENERATOR_SYSTEMS.items():
            assert "propose_theorem" in prompt, f"{level} missing tool name"

    def test_all_contain_preamble(self):
        for level, prompt in GENERATOR_SYSTEMS.items():
            assert "Lean 4 theorem designer" in prompt

    def test_detailed_has_strategy(self):
        assert "STRATEGY" in GENERATOR_SYSTEMS[PromptLevel.DETAILED]

    def test_minimal_no_strategy(self):
        assert "STRATEGY" not in GENERATOR_SYSTEMS[PromptLevel.MINIMAL]

class TestSolverSystem:
    """Tests for solver prompt."""

    def test_contains_tool_name(self):
        assert "submit_proof" in SOLVER_SYSTEM

    def test_contains_search_tactics(self):
        for tactic in ("exact?", "apply?", "rw?"):
            assert tactic in SOLVER_SYSTEM

    def test_mentions_mathlib(self):
        assert "Mathlib" in SOLVER_SYSTEM
