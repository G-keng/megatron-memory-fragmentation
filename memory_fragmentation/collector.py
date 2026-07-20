"""Small, copy-friendly PyTorch CUDA memory trace collector."""

from __future__ import annotations

import atexit
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Set


def _parse_ranks(value: str) -> Optional[Set[int]]:
    value = value.strip().lower()
    if value in {"", "none"}:
        return set()
    if value == "all":
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "snapshot"


class MemoryTraceCollector:
    """Collect full snapshots on selected ranks and light stats on every rank."""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        trace_ranks: Optional[str] = None,
        end_iteration: Optional[int] = None,
        max_entries: Optional[int] = None,
    ) -> None:
        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.output_dir = Path(output_dir or os.getenv("MEMORY_TRACE_DIR", "memory-traces"))
        ranks_value = trace_ranks if trace_ranks is not None else os.getenv("MEMORY_TRACE_RANKS", "0")
        self.trace_ranks = _parse_ranks(ranks_value)
        env_end = os.getenv("MEMORY_TRACE_END_ITER")
        self.end_iteration = end_iteration if end_iteration is not None else int(env_end or "20")
        self.max_entries = max_entries or int(os.getenv("MEMORY_TRACE_MAX_ENTRIES", "1000000"))
        self.emit_nvtx = os.getenv("MEMORY_TRACE_NVTX", "0") == "1"
        self.full_trace = self.trace_ranks is None or self.rank in self.trace_ranks
        self.started = False
        self.dumped = False
        self._atexit_registered = False

    def start(self) -> bool:
        """Start full allocator history on selected ranks before CUDA allocations."""
        import torch

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not torch.cuda.is_available():
            return False
        if not self.full_trace:
            return False

        record = torch.cuda.memory._record_memory_history
        kwargs: Dict[str, Any] = {
            "enabled": "all",
            "context": "alloc",
            "stacks": "all",
            "max_entries": self.max_entries,
        }
        try:
            if "global_record_annotations" in inspect.signature(record).parameters:
                kwargs["global_record_annotations"] = True
        except (TypeError, ValueError):
            # Some PyTorch builds expose a builtin without an inspectable signature.
            kwargs["global_record_annotations"] = True

        try:
            record(**kwargs)
        except TypeError:
            kwargs.pop("global_record_annotations", None)
            record(**kwargs)

        self.started = True
        if not self._atexit_registered:
            atexit.register(self._dump_at_exit)
            self._atexit_registered = True
        return True

    def sample(self, iteration: int, **metadata: object) -> Dict[str, Any]:
        """Append inexpensive iteration-level allocator and device memory statistics."""
        import torch

        row: Dict[str, Any] = {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "iteration": iteration,
            **metadata,
        }
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            stats = torch.cuda.memory_stats(device)
            row.update(
                {
                    "device": device,
                    "allocated_bytes": torch.cuda.memory_allocated(device),
                    "reserved_bytes": torch.cuda.memory_reserved(device),
                    "requested_bytes": stats.get("requested_bytes.all.current"),
                    "inactive_split_bytes": stats.get("inactive_split_bytes.all.current"),
                    "num_alloc_retries": stats.get("num_alloc_retries"),
                    "num_ooms": stats.get("num_ooms"),
                }
            )
            try:
                row["device_memory_used"] = torch.cuda.device_memory_used(device)
            except (AttributeError, RuntimeError):
                row["device_memory_used"] = None

        stats_path = self.output_dir / f"rank{self.rank}-memory-stats.jsonl"
        with stats_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

        if self.started and not self.dumped and iteration >= self.end_iteration:
            self.dump(reason=f"iter{iteration}")
        return row

    def dump(self, reason: str = "manual") -> Optional[Path]:
        """Dump a PyTorch-compatible snapshot once and stop recording history."""
        if not self.started or self.dumped:
            return None
        import torch

        path = self.output_dir / f"rank{self.rank}-{_safe_name(reason)}-snapshot.pickle"
        torch.cuda.memory._dump_snapshot(str(path))
        self.dumped = True
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
        except TypeError:
            torch.cuda.memory._record_memory_history(False)
        return path

    def dump_on_oom(self, exc: BaseException) -> Optional[Path]:
        """Dump only for CUDA OOM-like failures, then return the generated path."""
        if "out of memory" not in str(exc).lower():
            return None
        return self.dump("oom")

    def _dump_at_exit(self) -> None:
        if self.started and not self.dumped:
            try:
                self.dump("exit")
            except Exception:
                # Never mask the training process' original exit condition.
                pass


_COLLECTOR: Optional[MemoryTraceCollector] = None


def get_collector(**kwargs: object) -> MemoryTraceCollector:
    """Return the process-global collector used by thin Megatron instrumentation."""
    global _COLLECTOR
    if _COLLECTOR is None:
        _COLLECTOR = MemoryTraceCollector(**kwargs)
    return _COLLECTOR
