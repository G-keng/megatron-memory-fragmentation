"""Structured phase annotations shared by the collector and analyzer."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from typing import Dict, Iterator, Optional, Union


MEMORY_PHASES = (
    "init/distributed",
    "init/model_parameters",
    "init/optimizer",
    "init/checkpoint",
    "train/iteration",
    "pipeline/warmup/forward",
    "pipeline/steady/forward",
    "pipeline/steady/backward",
    "pipeline/cooldown/backward",
    "grad/finalize",
    "grad/sync",
    "optimizer/prepare",
    "optimizer/state_update",
    "param/update",
    "param/sync",
)

_FIELD_NAMES = {
    "iteration": "iter",
    "microbatch": "mb",
    "direction": "direction",
    "model_chunk": "chunk",
    "pp_stage": "pp",
    "tp_rank": "tp",
    "ep_rank": "ep",
}
_REVERSE_FIELD_NAMES = {value: key for key, value in _FIELD_NAMES.items()}
_CURRENT_METADATA: ContextVar[Dict[str, object]] = ContextVar(
    "memory_fragmentation_phase_metadata", default={}
)


def format_phase_name(phase: str, **metadata: object) -> str:
    """Encode a phase and its scalar metadata in a record_function-safe name."""
    if "|" in phase or "=" in phase:
        raise ValueError(f"invalid phase name: {phase!r}")
    fields = ["memtrace", f"phase={phase}"]
    for long_name, short_name in _FIELD_NAMES.items():
        value = metadata.get(long_name)
        if value is not None:
            text = str(value)
            if "|" in text or "=" in text:
                raise ValueError(f"invalid {long_name}: {value!r}")
            fields.append(f"{short_name}={text}")
    return "|".join(fields)


def parse_phase_name(name: str) -> Optional[Dict[str, Union[str, int]]]:
    """Parse a structured memtrace annotation name."""
    if not name.startswith("memtrace|"):
        return None
    result: Dict[str, Union[str, int]] = {}
    for field in name.split("|")[1:]:
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        key = _REVERSE_FIELD_NAMES.get(key, key)
        if key in {"iteration", "microbatch", "model_chunk", "pp_stage", "tp_rank", "ep_rank"}:
            try:
                result[key] = int(value)
                continue
            except ValueError:
                pass
        result[key] = value
    return result if "phase" in result else None


@contextmanager
def memory_phase(
    phase: str,
    *,
    iteration: Optional[int] = None,
    microbatch: Optional[int] = None,
    direction: Optional[str] = None,
    model_chunk: Optional[int] = None,
    pp_stage: Optional[int] = None,
    tp_rank: Optional[int] = None,
    ep_rank: Optional[int] = None,
    emit_nvtx: bool = False,
) -> Iterator[str]:
    """Annotate a logical training phase using record_function and optional NVTX."""
    metadata = dict(_CURRENT_METADATA.get())
    metadata.update(
        {
            key: value
            for key, value in {
                "iteration": iteration,
                "microbatch": microbatch,
                "direction": direction,
                "model_chunk": model_chunk,
                "pp_stage": pp_stage,
                "tp_rank": tp_rank,
                "ep_rank": ep_rank,
            }.items()
            if value is not None
        }
    )
    name = format_phase_name(phase, **metadata)
    token = _CURRENT_METADATA.set(metadata)

    # Import lazily so analysis-only use does not require importing torch.
    import torch

    try:
        with ExitStack() as stack:
            stack.enter_context(torch.profiler.record_function(name))
            nvtx_pushed = False
            if emit_nvtx and torch.cuda.is_available():
                try:
                    torch.cuda.nvtx.range_push(name)
                    nvtx_pushed = True
                except (AttributeError, RuntimeError):
                    nvtx_pushed = False
            try:
                yield name
            finally:
                if nvtx_pushed:
                    torch.cuda.nvtx.range_pop()
    finally:
        _CURRENT_METADATA.reset(token)
