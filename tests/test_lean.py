from src.lean import LeanCompiler, LeanResult, SORRY_WARNING


class TestAssemble:
    """Tests for LeanCompiler.assemble static method."""

    def test_basic_assembly(self):
        result = LeanCompiler.assemble(
            "import Mathlib",
            "theorem foo : True",
            ":= by trivial",
        )
        assert result == "import Mathlib\n\ntheorem foo : True\n:= by trivial\n"

    def test_multiline_imports(self):
        imports = "import Mathlib.Tactic\nimport Mathlib.Data.Nat.Basic"
        result = LeanCompiler.assemble(imports, "theorem bar : 1 = 1", ":= rfl")
        lines = result.strip().split("\n")
        assert lines[0] == "import Mathlib.Tactic"
        assert lines[1] == "import Mathlib.Data.Nat.Basic"

    def test_empty_proof(self):
        result = LeanCompiler.assemble("import Mathlib", "theorem x : True", "")
        assert "theorem x : True" in result

    def test_preserves_whitespace_in_proof(self):
        proof = ":= by\n  simp\n  ring"
        result = LeanCompiler.assemble("import Mathlib", "theorem t : True", proof)
        assert proof in result


class TestAssembleSorry:
    """Tests for LeanCompiler.assemble_sorry static method."""

    def test_sorry_proof(self):
        result = LeanCompiler.assemble_sorry(
            "import Mathlib",
            "theorem foo (n : Nat) : n = n",
        )
        assert "sorry" in result
        assert "theorem foo" in result
        assert result.endswith(":= by sorry\n")


class TestLeanResult:
    """Tests for LeanResult dataclass."""

    def test_success_result(self):
        r = LeanResult(success=True, stdout="", stderr="", return_code=0, has_sorry=False)
        assert r.success is True
        assert r.has_sorry is False

    def test_sorry_detection(self):
        r = LeanResult(
            success=False, stdout=SORRY_WARNING, stderr="",
            return_code=0, has_sorry=True,
        )
        assert r.has_sorry is True
        assert r.success is False

    def test_frozen(self):
        r = LeanResult(success=True, stdout="", stderr="", return_code=0, has_sorry=False)
        with __import__("pytest").raises(AttributeError):
            r.success = False  # type: ignore[misc]
