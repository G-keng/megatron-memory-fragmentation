"""Lightweight CUDA allocator tracing helpers for LLM training experiments."""

from .collector import MemoryTraceCollector, get_collector
from .phases import MEMORY_PHASES, memory_phase, parse_phase_name

__version__ = "0.3.0"

__all__ = [
    "MEMORY_PHASES",
    "MemoryTraceCollector",
    "get_collector",
    "memory_phase",
    "parse_phase_name",
    "__version__",
]
