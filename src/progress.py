import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import EvalConfig

log = logging.getLogger(__name__)


def config_hash(config: EvalConfig) -> str:
    """Deterministic hash of the eval config for resume verification."""
    # Hash the fields that affect eval behavior (not output_dir, etc.)
    data = json.dumps({
        "generator_model": config.generator_model.model_id,
        "solver_models": [sm.model_id for sm in config.solver_models],
        "rounds": config.rounds,
        "solver_max_calls": config.solver_max_calls,
        "generator_max_calls": config.generator_max_calls,
        "attempts_during_loop": config.attempts_during_loop,
        "attempts_reeval": config.attempts_reeval,
        "lean_timeout_seconds": config.lean_timeout_seconds,
    }, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


class ProgressTracker:
    """Unified progress tracking and crash recovery.

    Manages two files:
    - JSONL event log (append-only, for real-time monitoring)
    - State file (overwritten atomically, serves as both live state and checkpoint)
    """

    def __init__(self, output_dir: str, timestamp: str) -> None:
        self._output_dir = Path(output_dir)
        self._progress_dir = self._output_dir / "progress"
        self._progress_dir.mkdir(parents=True, exist_ok=True)

        self._events_path = self._progress_dir / f"eval_{timestamp}_events.jsonl"
        self._state_path = self._progress_dir / f"eval_{timestamp}_state.json"
        self._timestamp = timestamp
        self._start_time = time.monotonic()

        self._state: dict[str, Any] = {
            "status": "in_progress",
            "config_hash": "",
            "config": {},
            "started_at": datetime.now(timezone.utc).isoformat(),
            "progress": {
                "generators_completed": 0,
                "generators_total": 0,
                "elapsed_seconds": 0,
            },
            "generators": {},
        }
        self._lock = asyncio.Lock()

    @property
    def state_path(self) -> Path:
        return self._state_path

    @property
    def timestamp(self) -> str:
        return self._timestamp

    def init_state(
        self,
        config: EvalConfig,
        generator_ids: list[str],
        anon_maps: dict[str, dict[str, str]],
    ) -> None:
        """Initialize state at the start of a multi-generator eval."""
        self._state["config_hash"] = config_hash(config)
        def _serialize_model(m: "ModelConfig") -> dict:
            return {"model_id": m.model_id, "display_name": m.display_name, "provider": m.provider, "max_tokens": m.max_tokens}

        self._state["config"] = {
            "generator_model": _serialize_model(config.generator_model),
            "solver_models": [_serialize_model(sm) for sm in config.solver_models],
            "rounds": config.rounds,
            "solver_max_calls": config.solver_max_calls,
            "generator_max_calls": config.generator_max_calls,
            "attempts_during_loop": config.attempts_during_loop,
            "attempts_reeval": config.attempts_reeval,
            "lean_timeout_seconds": config.lean_timeout_seconds,
            "lean_project_path": config.lean_project_path,
            "output_dir": config.output_dir,
            "prior_alpha": config.prior_alpha,
            "prior_beta": config.prior_beta,
            "seed": config.seed,
            "max_concurrent_api": config.max_concurrent_api,
            "summarize_rounds": config.summarize_rounds,
            "prompt_level": config.prompt_level.value,
        }
        self._state["progress"]["generators_total"] = len(generator_ids)

        for gen_id in generator_ids:
            self._state["generators"][gen_id] = {
                "status": "pending",
                "phase": "waiting",
                "anon_map": anon_maps.get(gen_id, {}),
                "rounds_completed": 0,
                "rounds_total": config.rounds,
                "completed_rounds": [],
                "messages": None,
                "reeval": None,
                "final_score": None,
            }

    async def emit(self, event: str, **data: Any) -> None:
        """Append an event to the JSONL log."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        }
        line = json.dumps(entry, default=str) + "\n"
        async with self._lock:
            with open(self._events_path, "a") as f:
                f.write(line)

    async def update_generator(
        self,
        generator_id: str,
        save: bool = True,
        **fields: Any,
    ) -> None:
        """Update fields on a generator's state entry and optionally save."""
        async with self._lock:
            gen = self._state["generators"].get(generator_id)
            if gen is None:
                return
            gen.update(fields)
            self._state["progress"]["elapsed_seconds"] = int(
                time.monotonic() - self._start_time
            )
            if save:
                self._write_state()

    async def mark_generator_complete(self, generator_id: str) -> None:
        """Mark a generator as complete and update progress."""
        async with self._lock:
            gen = self._state["generators"].get(generator_id)
            if gen:
                gen["status"] = "complete"
                gen["phase"] = "done"
                gen["messages"] = None  # Drop messages, no longer needed
            self._state["progress"]["generators_completed"] = sum(
                1 for g in self._state["generators"].values()
                if g["status"] == "complete"
            )
            self._state["progress"]["elapsed_seconds"] = int(
                time.monotonic() - self._start_time
            )
            self._write_state()

    async def save_round(
        self,
        generator_id: str,
        round_data: dict,
        messages: list[Any],
        current_best: dict | None = None,
    ) -> None:
        """Save a completed round as a checkpoint."""
        async with self._lock:
            gen = self._state["generators"].get(generator_id)
            if gen is None:
                return
            gen["completed_rounds"].append(round_data)
            gen["rounds_completed"] = len(gen["completed_rounds"])
            gen["messages"] = messages
            if current_best:
                gen["current_best"] = current_best
            self._state["progress"]["elapsed_seconds"] = int(
                time.monotonic() - self._start_time
            )
            self._write_state()

    async def save_skipped_round(
        self,
        generator_id: str,
        round_number: int,
        messages: list[Any],
    ) -> None:
        """Record that a round was skipped (budget exhausted)."""
        async with self._lock:
            gen = self._state["generators"].get(generator_id)
            if gen is None:
                return
            gen["messages"] = messages
            self._state["progress"]["elapsed_seconds"] = int(
                time.monotonic() - self._start_time
            )
            self._write_state()

    async def finalize(self, final_output: dict) -> None:
        """Replace state with final output."""
        async with self._lock:
            self._state = {**final_output, "status": "completed"}
            self._write_state()

    def get_generator_state(self, generator_id: str) -> dict | None:
        """Get saved state for a generator (for resume)."""
        return self._state["generators"].get(generator_id)

    def get_all_generator_ids(self) -> list[str]:
        """Get all generator IDs from state."""
        return list(self._state["generators"].keys())

    def _write_state(self) -> None:
        """Atomically overwrite the state file."""
        data = json.dumps(self._state, indent=2, default=str)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._progress_dir), suffix=".tmp",
        )
        fd_closed = False
        try:
            os.write(fd, data.encode())
            os.close(fd)
            fd_closed = True
            os.rename(tmp_path, str(self._state_path))
        except Exception:
            if not fd_closed:
                os.close(fd)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    @classmethod
    def from_checkpoint(cls, state_path: str) -> "ProgressTracker":
        """Load a ProgressTracker from an existing state file for resume.

        Accepts a full path, a path relative to CWD, or just the filename
        (searches results/progress/ automatically).
        """
        path = Path(state_path)
        if not path.exists():
            # Try results/progress/ as a fallback
            fallback = Path("results") / "progress" / path.name
            if fallback.exists():
                path = fallback
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found: {state_path}\n"
                    f"Also checked: {fallback}"
                )

        with open(path) as f:
            state = json.load(f)

        # Extract timestamp from filename: eval_YYYYMMDD_HHMMSS_state.json
        stem = path.stem  # eval_YYYYMMDD_HHMMSS_state
        parts = stem.split("_")
        # timestamp is parts[1] + "_" + parts[2]
        timestamp = f"{parts[1]}_{parts[2]}"

        # State file lives in {output_dir}/progress/, so output_dir is the grandparent
        tracker = cls(
            output_dir=str(path.parent.parent),
            timestamp=timestamp,
        )
        tracker._state = state
        return tracker

    def verify_config(self, config: EvalConfig) -> bool:
        """Check that the config matches the checkpoint."""
        return config_hash(config) == self._state.get("config_hash", "")

    def get_stored_config(self) -> dict:
        """Get the config dict stored in the checkpoint."""
        return self._state.get("config", {})
