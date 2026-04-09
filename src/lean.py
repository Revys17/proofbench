import asyncio
import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SORRY_WARNING = 'declaration uses `sorry`'


@dataclass(frozen=True)
class LeanResult:
    success: bool
    stdout: str
    stderr: str
    return_code: int
    has_sorry: bool


class LeanCompiler:
    def __init__(self, project_path: str, timeout: int = 120, max_concurrent: int = 4) -> None:
        self._project_path = Path(project_path).resolve()
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._elan_bin = Path.home() / ".elan" / "bin"
        self._lean_path: str | None = None
        self._cache: dict[str, LeanResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    async def _get_lean_path(self) -> str:
        """Compute LEAN_PATH via `lake env` (cached after first call)."""
        if self._lean_path is not None:
            return self._lean_path

        proc = await asyncio.create_subprocess_exec(
            str(self._elan_bin / "lake"), "env", "sh", "-c", "echo $LEAN_PATH",
            cwd=str(self._project_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._base_env(),
        )
        stdout, _ = await proc.communicate()
        self._lean_path = stdout.decode().strip()
        return self._lean_path

    def _base_env(self) -> dict[str, str]:
        return {
            "PATH": f"{self._elan_bin}:{os.environ.get('PATH', '')}",
            "HOME": str(Path.home()),
        }

    @staticmethod
    def assemble(imports: str, theorem_statement: str, proof: str) -> str:
        """Build a complete .lean file from parts."""
        return f"{imports}\n\n{theorem_statement}\n{proof}\n"

    @staticmethod
    def assemble_sorry(imports: str, theorem_statement: str) -> str:
        """Build a .lean file with sorry as the proof."""
        return f"{imports}\n\n{theorem_statement} := by sorry\n"

    async def check(self, lean_code: str) -> LeanResult:
        """Compile lean_code and return the result (cached by content hash)."""
        key = hashlib.sha256(lean_code.encode()).hexdigest()

        if key in self._cache:
            self._cache_hits += 1
            log.debug("Lean cache hit (%d hits, %d misses)", self._cache_hits, self._cache_misses)
            return self._cache[key]

        self._cache_misses += 1
        tag = uuid.uuid4().hex[:8]
        tmp_file = self._project_path / f"_check_{tag}.lean"
        tmp_file.write_text(lean_code, encoding="utf-8")

        try:
            result = await self._run_lean(tmp_file)
        finally:
            tmp_file.unlink(missing_ok=True)

        self._cache[key] = result
        return result

    @property
    def cache_stats(self) -> dict[str, int]:
        return {"hits": self._cache_hits, "misses": self._cache_misses}

    async def _run_lean(self, file_path: Path) -> LeanResult:
        lean_path = await self._get_lean_path()
        env = self._base_env()
        env["LEAN_PATH"] = lean_path

        async with self._semaphore:
            proc = await asyncio.create_subprocess_exec(
                str(self._elan_bin / "lean"), str(file_path),
                cwd=str(self._project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return LeanResult(
                    success=False,
                    stdout="",
                    stderr="TIMEOUT: compilation exceeded time limit",
                    return_code=-1,
                    has_sorry=False,
                )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        combined = stdout + stderr
        has_sorry = SORRY_WARNING in combined
        success = proc.returncode == 0 and not has_sorry

        return LeanResult(
            success=success,
            stdout=stdout,
            stderr=stderr,
            return_code=proc.returncode or 0,
            has_sorry=has_sorry,
        )
