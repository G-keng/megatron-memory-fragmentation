"""Offline analysis for PyTorch native CUDA allocator snapshots.

The exact replay intentionally targets the experiment configuration used by this
workspace: native allocator, non-expandable segments, and default rounding.
Static snapshot metrics and stack-based phase attribution remain available when
the trace is incomplete or has wrapped its ring buffer.
"""

from __future__ import annotations

import bisect
import copy
import csv
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

from .phases import parse_phase_name


KIB = 1024
MIB = 1024 * KIB
MIN_BLOCK = 512
SMALL_LIMIT = 1 * MIB
DEFAULT_MAX_NON_SPLIT_ROUNDING = 20 * MIB


def load_snapshot(path: Path) -> Dict[str, Any]:
    """Load a trusted local PyTorch snapshot pickle."""
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict) or "segments" not in value:
        raise ValueError(f"not a PyTorch allocator snapshot: {path}")
    return value


def _pool_id(value: object) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return 0, 0


def _is_active(state: str) -> bool:
    return state in {"active_allocated", "active_pending_free", "active_awaiting_free"}


def _is_pending(state: str) -> bool:
    return state in {"active_pending_free", "active_awaiting_free"}


def format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024.0 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{amount:.2f} TiB"


def summarize_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Separate allocator cache, stranded blocks, pending frees, and rounding."""
    per_device: DefaultDict[int, Dict[str, Any]] = defaultdict(
        lambda: {
            "reserved_bytes": 0,
            "active_block_bytes": 0,
            "active_requested_bytes": 0,
            "internal_waste_bytes": 0,
            "releasable_cache_bytes": 0,
            "stranded_free_bytes": 0,
            "pending_free_bytes": 0,
            "segments": 0,
            "partially_active_segments": 0,
            "fully_free_segments": 0,
            "domains": defaultdict(list),
        }
    )

    for segment in snapshot.get("segments", []):
        device = int(segment.get("device", 0))
        target = per_device[device]
        target["segments"] += 1
        target["reserved_bytes"] += int(segment.get("total_size", 0))
        blocks = segment.get("blocks", [])
        active = [block for block in blocks if _is_active(str(block.get("state", "")))]
        inactive = [block for block in blocks if block.get("state") == "inactive"]
        if not active:
            target["fully_free_segments"] += 1
            target["releasable_cache_bytes"] += int(segment.get("total_size", 0))
        elif inactive:
            target["partially_active_segments"] += 1
            target["stranded_free_bytes"] += sum(int(block.get("size", 0)) for block in inactive)

        domain = (
            _pool_id(segment.get("segment_pool_id")),
            str(segment.get("segment_type", "unknown")),
            int(segment.get("stream", 0)),
        )
        for block in blocks:
            state = str(block.get("state", ""))
            size = int(block.get("size", 0))
            requested = int(block.get("requested_size", 0) or 0)
            if _is_active(state):
                target["active_block_bytes"] += size
                target["active_requested_bytes"] += requested
                target["internal_waste_bytes"] += max(0, size - requested)
                if _is_pending(state):
                    target["pending_free_bytes"] += size
            elif state == "inactive":
                target["domains"][domain].append(size)

    total: Dict[str, Any] = defaultdict(int)
    result_devices: Dict[str, Any] = {}
    for device, values in sorted(per_device.items()):
        domain_rows = []
        for (pool, segment_type, stream), sizes in sorted(values.pop("domains").items(), key=str):
            domain_rows.append(
                {
                    "pool_id": list(pool),
                    "segment_type": segment_type,
                    "stream": stream,
                    "free_bytes": sum(sizes),
                    "largest_free_block": max(sizes, default=0),
                    "free_blocks": len(sizes),
                }
            )
        values["free_domains"] = domain_rows
        result_devices[str(device)] = dict(values)
        for key, value in values.items():
            if isinstance(value, int):
                total[key] += value
    total["reserved_minus_active_bytes"] = total["reserved_bytes"] - total["active_block_bytes"]
    return {"total": dict(total), "devices": result_devices}


@dataclass
class AnnotationInterval:
    device: int
    name: str
    start_us: int
    end_us: int
    metadata: Optional[Dict[str, Any]] = None

    @property
    def duration_us(self) -> int:
        return max(0, self.end_us - self.start_us)


def build_annotation_intervals(snapshot: Dict[str, Any]) -> Dict[int, List[AnnotationInterval]]:
    stacks: DefaultDict[Tuple[int, str], List[int]] = defaultdict(list)
    intervals: DefaultDict[int, List[AnnotationInterval]] = defaultdict(list)
    for entry in sorted(snapshot.get("external_annotations", []), key=lambda item: int(item.get("time_us", 0))):
        device = int(entry.get("device", 0))
        name = str(entry.get("name", ""))
        stage = str(entry.get("stage", "")).upper()
        time_us = int(entry.get("time_us", 0))
        key = (device, name)
        if stage == "START":
            stacks[key].append(time_us)
        elif stage == "END" and stacks[key]:
            start = stacks[key].pop()
            intervals[device].append(
                AnnotationInterval(device, name, start, time_us, parse_phase_name(name))
            )
    for values in intervals.values():
        values.sort(key=lambda item: (item.start_us, item.end_us))
    return dict(intervals)


_STACK_RULES: Sequence[Tuple[Tuple[str, ...], str]] = (
    (("get_megatron_optimizer", "distributedoptimizer.__init__", "optimizer.__init__"), "init/optimizer"),
    (("model_provider", "initialize_affine_weight", "init_method_normal"), "init/model_parameters"),
    (("finalize_model_grads", "finish_grad_sync", "grad_sync"), "grad/sync"),
    (("all_gather_param", "start_param_sync", "finish_param_sync", "param_sync"), "param/sync"),
    (("fusedadam.step", "optimizer.step", "step_with_ready_grads"), "optimizer/state_update"),
    (("backward_step", "engine_run_backward", "autograd.backward"), "pipeline/unknown/backward"),
    (("forward_step", "forward_backward_step"), "pipeline/unknown/forward"),
)


def classify_stack(frames: Sequence[Dict[str, Any]]) -> Optional[str]:
    text = "\n".join(
        f"{frame.get('filename', '')}:{frame.get('name', '')}" for frame in frames
    ).lower()
    for patterns, phase in _STACK_RULES:
        if any(pattern in text for pattern in patterns):
            return phase
    return None


class PhaseResolver:
    def __init__(self, snapshot: Dict[str, Any]) -> None:
        self.intervals = build_annotation_intervals(snapshot)
        self.starts = {
            device: [interval.start_us for interval in values]
            for device, values in self.intervals.items()
        }

    def resolve(
        self, device: int, time_us: Optional[int], frames: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        active: List[AnnotationInterval] = []
        if time_us is not None:
            values = self.intervals.get(device, [])
            end = bisect.bisect_right(self.starts.get(device, []), time_us)
            active = [value for value in values[:end] if value.end_us >= time_us]

        explicit = [value for value in active if value.metadata]
        if explicit:
            chosen = min(explicit, key=lambda value: (value.duration_us, -value.start_us))
            metadata = dict(chosen.metadata or {})
            phase = str(metadata.pop("phase"))
            return {
                "primary_phase": phase,
                "phase_confidence": "explicit_marker",
                "phase_metadata": metadata,
                "annotations": [value.name for value in active if not value.metadata],
            }

        framework_phase = self._framework_phase(active)
        if framework_phase:
            return {
                "primary_phase": framework_phase,
                "phase_confidence": "framework_annotation",
                "phase_metadata": {},
                "annotations": [value.name for value in active],
            }

        stack_phase = classify_stack(frames)
        if stack_phase:
            return {
                "primary_phase": stack_phase,
                "phase_confidence": "stack_rule",
                "phase_metadata": {},
                "annotations": [value.name for value in active],
            }

        return {
            "primary_phase": "unknown",
            "phase_confidence": "unknown",
            "phase_metadata": {},
            "annotations": [value.name for value in active],
        }

    @staticmethod
    def _framework_phase(active: Sequence[AnnotationInterval]) -> Optional[str]:
        names = "\n".join(value.name.lower() for value in active)
        if "optimizer.step" in names or "fusedadam.step" in names:
            return "optimizer/state_update"
        if "param" in names and ("all_gather" in names or "sync" in names):
            return "param/sync"
        return None


@dataclass
class ReplayBlock:
    address: int
    size: int
    requested_size: int = 0
    state: str = "inactive"
    frames: List[Dict[str, Any]] = field(default_factory=list)
    alloc_time_us: Optional[int] = None
    allocation_id: Optional[int] = None


@dataclass
class ReplaySegment:
    device: int
    address: int
    total_size: int
    stream: int
    pool_id: Tuple[int, int] = (0, 0)
    segment_type: str = "unknown"
    blocks: List[ReplayBlock] = field(default_factory=list)

    def contains(self, address: int) -> bool:
        return self.address <= address < self.address + self.total_size

    def stranded_bytes(self) -> int:
        if not any(_is_active(block.state) for block in self.blocks):
            return 0
        return sum(block.size for block in self.blocks if block.state == "inactive")


class ReplayError(RuntimeError):
    pass


def _round_size(size: int, settings: Dict[str, Any]) -> int:
    divisions = settings.get("roundup_power2_divisions", {})
    if isinstance(divisions, dict) and any(int(value) != 0 for value in divisions.values()):
        raise ReplayError("roundup_power2_divisions is not supported by the v1 exact replay")
    return max(MIN_BLOCK, ((size + MIN_BLOCK - 1) // MIN_BLOCK) * MIN_BLOCK)


def _should_split(block_size: int, rounded: int, segment_type: str, settings: Dict[str, Any]) -> bool:
    remaining = block_size - rounded
    if remaining <= 0:
        return False
    if segment_type == "small":
        return remaining >= MIN_BLOCK
    max_split = int(settings.get("max_split_size", -1))
    under_limit = max_split < 0 or block_size < max_split
    return under_limit and remaining > SMALL_LIMIT


def _event_pool(event: Dict[str, Any]) -> Tuple[int, int]:
    return _pool_id(event.get("pool_id", event.get("segment_pool_id")))


def _next_alloc_for_segment(
    events: Sequence[Dict[str, Any]], index: int, address: int, size: int
) -> Optional[Dict[str, Any]]:
    end = address + size
    for event in events[index + 1 : index + 17]:
        action = event.get("action")
        if action == "alloc" and address <= int(event.get("addr", -1)) < end:
            return event
        if action in {"segment_alloc", "segment_free", "oom"}:
            break
    return None


class NativeReplay:
    def __init__(
        self,
        snapshot: Dict[str, Any],
        min_fragment_bytes: int = MIB,
        capture_indices: Optional[Sequence[Tuple[int, int]]] = None,
        record_timeline: bool = True,
    ) -> None:
        self.snapshot = snapshot
        self.settings = snapshot.get("allocator_settings", {})
        self.min_fragment_bytes = min_fragment_bytes
        self.capture_indices = set(capture_indices or ())
        self.record_timeline = record_timeline
        self.segments: DefaultDict[int, List[ReplaySegment]] = defaultdict(list)
        self.incidents: List[Dict[str, Any]] = []
        self.open_fragments: DefaultDict[Tuple[int, int], List[List[int]]] = defaultdict(list)
        self.metric_totals: DefaultDict[int, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.timeline: List[Dict[str, Any]] = []
        self.captured_states: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.pinner_watchers: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.next_allocation_id = 1
        self.device_end_time_us: Dict[int, int] = {}
        self.last_time_us = 0

    def run(self) -> Dict[str, Any]:
        if self.settings.get("expandable_segments"):
            return self._failure("expandable_segments=True is not supported by exact replay")
        if self.settings.get("trace_alloc_overflowed"):
            return self._failure("snapshot reports allocator trace ring overflow")
        try:
            for device, events in enumerate(self.snapshot.get("device_traces", [])):
                self._run_device(device, events)
            self._close_open_fragments(self.last_time_us)
            self._finalize_causal_chains()
            self._validate_final_snapshot()
        except ReplayError as exc:
            return self._failure(str(exc))
        incidents = [
            incident
            for incident in self.incidents
            if incident.get("type") != "fragment-created"
            or int(incident.get("stranded_delta_bytes", 0)) >= self.min_fragment_bytes
        ]
        return {
            "exact": True,
            "error": None,
            "incidents": incidents,
            "timeline": self.timeline,
            "captured_states": self.captured_states,
        }

    def _failure(self, reason: str) -> Dict[str, Any]:
        return {
            "exact": False,
            "error": reason,
            "incidents": [],
            "timeline": [],
            "captured_states": {},
        }

    def _run_device(self, device: int, events: Sequence[Dict[str, Any]]) -> None:
        for index, event in enumerate(events):
            action = str(event.get("action", ""))
            time_us = int(event.get("time_us", self.last_time_us))
            self.last_time_us = max(self.last_time_us, time_us)
            self.device_end_time_us[device] = max(self.device_end_time_us.get(device, 0), time_us)
            if action == "segment_alloc":
                self._segment_alloc(device, events, index, event, time_us)
            elif action == "segment_free":
                self._segment_free(device, event, time_us)
            elif action == "alloc":
                self._alloc(device, event, time_us)
            elif action == "free_requested":
                self._free_requested(device, event)
            elif action == "free_completed":
                self._free_completed(device, event, time_us)
            elif action in {"segment_map", "segment_unmap"}:
                raise ReplayError(f"{action} requires expandable-segment replay")
            if action in {
                "segment_alloc",
                "segment_free",
                "alloc",
                "free_requested",
                "free_completed",
            }:
                self._record_point(device, index, event, time_us)

    def _find_segment(self, device: int, address: int) -> ReplaySegment:
        for segment in self.segments[device]:
            if segment.contains(address):
                return segment
        raise ReplayError(f"address 0x{address:x} has no segment; trace probably starts mid-run")

    @staticmethod
    def _find_block(segment: ReplaySegment, address: int) -> Tuple[int, ReplayBlock]:
        for index, block in enumerate(segment.blocks):
            if block.address == address:
                return index, block
        raise ReplayError(f"address 0x{address:x} is not a block boundary")

    def _segment_alloc(
        self,
        device: int,
        events: Sequence[Dict[str, Any]],
        index: int,
        event: Dict[str, Any],
        time_us: int,
    ) -> None:
        address, total_size = int(event["addr"]), int(event["size"])
        if any(segment.contains(address) or address <= segment.address < address + total_size for segment in self.segments[device]):
            raise ReplayError(f"overlapping segment allocation at 0x{address:x}")

        next_alloc = _next_alloc_for_segment(events, index, address, total_size)
        stream = int(event.get("stream", next_alloc.get("stream", 0) if next_alloc else 0))
        pool = _event_pool(event)
        segment_type = "unknown"
        if next_alloc:
            rounded = _round_size(int(next_alloc["size"]), self.settings)
            segment_type = "small" if rounded <= SMALL_LIMIT else "large"
            domain_blocks = self._domain_free_blocks(device, pool, segment_type, stream)
            compatible_free = sum(block.size for block in domain_blocks)
            largest = max((block.size for block in domain_blocks), default=0)
            if compatible_free >= rounded and largest < rounded:
                pinners, blocking_segments = self._pinning_candidates(
                    device, pool, segment_type, stream
                )
                fragment_sources = self._fragment_sources(
                    device, blocking_segments, time_us
                )
                causal_chain = {
                    "confidence": "exact_replay_heuristic_attribution",
                    "reason": "compatible free bytes were sufficient but no free block fit the request",
                    "blocking_domain": {
                        "pool_id": list(pool),
                        "segment_type": segment_type,
                        "stream": stream,
                    },
                    "request": {
                        "requested_bytes": int(next_alloc["size"]),
                        "rounded_bytes": rounded,
                        "compatible_free_bytes": compatible_free,
                        "largest_free_block": largest,
                        "extra_segment_bytes": total_size,
                    },
                    "blocking_segments": blocking_segments,
                    "pinners": pinners,
                    "fragment_sources": fragment_sources,
                }
                self.incidents.append(
                    {
                        "type": "failed-fit",
                        "device": device,
                        "event_index": index,
                        "time_us": time_us,
                        "address": address,
                        "request_bytes": int(next_alloc["size"]),
                        "rounded_request_bytes": rounded,
                        "extra_segment_bytes": total_size,
                        "compatible_free_bytes": compatible_free,
                        "largest_free_block": largest,
                        "frames": next_alloc.get("frames", event.get("frames", [])),
                        "main_pinner": pinners[0] if pinners else None,
                        "causal_chain": causal_chain,
                    }
                )

        segment = ReplaySegment(
            device=device,
            address=address,
            total_size=total_size,
            stream=stream,
            pool_id=pool,
            segment_type=segment_type,
            blocks=[ReplayBlock(address, total_size)],
        )
        self.segments[device].append(segment)
        self.segments[device].sort(key=lambda value: value.address)
        self._update_metrics(device, None, segment)

    def _segment_free(self, device: int, event: Dict[str, Any], time_us: int) -> None:
        address = int(event["addr"])
        segment = self._find_segment(device, address)
        if segment.address != address or segment.total_size != int(event["size"]):
            raise ReplayError(f"segment_free mismatch at 0x{address:x}")
        if any(block.state != "inactive" for block in segment.blocks):
            raise ReplayError(f"segment_free removed active blocks at 0x{address:x}")
        self._apply_stranded_delta(segment, -segment.stranded_bytes(), time_us, event)
        self._update_metrics(device, segment, None)
        self.segments[device].remove(segment)

    def _alloc(self, device: int, event: Dict[str, Any], time_us: int) -> None:
        address, requested = int(event["addr"]), int(event["size"])
        segment = self._find_segment(device, address)
        before_contribution = self._segment_contribution(segment)
        before = segment.stranded_bytes()
        index, block = self._find_block(segment, address)
        if block.state != "inactive":
            raise ReplayError(f"alloc reused non-free block at 0x{address:x}")
        rounded = _round_size(requested, self.settings)
        inferred_type = "small" if rounded <= SMALL_LIMIT else "large"
        if segment.segment_type == "unknown":
            segment.segment_type = inferred_type
        if segment.segment_type != inferred_type:
            raise ReplayError(f"pool type mismatch at 0x{address:x}")
        if rounded > block.size:
            raise ReplayError(f"request does not fit block at 0x{address:x}")

        allocated_size = rounded if _should_split(block.size, rounded, segment.segment_type, self.settings) else block.size
        allocation_id = self.next_allocation_id
        self.next_allocation_id += 1
        allocated = ReplayBlock(
            address=address,
            size=allocated_size,
            requested_size=requested,
            state="active_allocated",
            frames=copy.deepcopy(event.get("frames", [])),
            alloc_time_us=time_us,
            allocation_id=allocation_id,
        )
        replacement = [allocated]
        if allocated_size < block.size:
            replacement.append(ReplayBlock(address + allocated_size, block.size - allocated_size))
        segment.blocks[index : index + 1] = replacement
        self._update_metrics(device, before_contribution, segment)
        self._record_stranded_change(segment, before, time_us, event)

    def _free_requested(self, device: int, event: Dict[str, Any]) -> None:
        segment = self._find_segment(device, int(event["addr"]))
        before_contribution = self._segment_contribution(segment)
        _, block = self._find_block(segment, int(event["addr"]))
        if block.state != "active_allocated":
            raise ReplayError(f"free_requested for non-active block at 0x{block.address:x}")
        block.state = "active_pending_free"
        self._update_metrics(device, before_contribution, segment)

    def _free_completed(self, device: int, event: Dict[str, Any], time_us: int) -> None:
        address = int(event["addr"])
        segment = self._find_segment(device, address)
        before_contribution = self._segment_contribution(segment)
        before = segment.stranded_bytes()
        index, block = self._find_block(segment, address)
        if not _is_active(block.state):
            raise ReplayError(f"free_completed for unknown allocation at 0x{address:x}")
        if block.allocation_id is not None:
            for pinner in self.pinner_watchers.pop(block.allocation_id, []):
                pinner["free_time_us"] = time_us
        block.state = "inactive"
        block.requested_size = 0
        block.frames = []
        block.alloc_time_us = None
        block.allocation_id = None
        if index > 0 and segment.blocks[index - 1].state == "inactive":
            previous = segment.blocks[index - 1]
            previous.size += block.size
            segment.blocks.pop(index)
            block = previous
            index -= 1
        if index + 1 < len(segment.blocks) and segment.blocks[index + 1].state == "inactive":
            block.size += segment.blocks[index + 1].size
            segment.blocks.pop(index + 1)
        self._update_metrics(device, before_contribution, segment)
        self._record_stranded_change(segment, before, time_us, event)

    @staticmethod
    def _segment_contribution(segment: ReplaySegment) -> Dict[str, int]:
        active = [block for block in segment.blocks if _is_active(block.state)]
        inactive = [block for block in segment.blocks if block.state == "inactive"]
        return {
            "reserved_bytes": segment.total_size,
            "active_block_bytes": sum(block.size for block in active),
            "active_requested_bytes": sum(block.requested_size for block in active),
            "internal_waste_bytes": sum(
                max(0, block.size - block.requested_size) for block in active
            ),
            "releasable_cache_bytes": segment.total_size if not active else 0,
            "stranded_free_bytes": sum(block.size for block in inactive) if active else 0,
            "pending_free_bytes": sum(block.size for block in active if _is_pending(block.state)),
        }

    def _update_metrics(
        self,
        device: int,
        before: Optional[Any],
        after: Optional[ReplaySegment],
    ) -> None:
        before_values = (
            self._segment_contribution(before)
            if isinstance(before, ReplaySegment)
            else before
            if isinstance(before, dict)
            else {}
        )
        after_values = self._segment_contribution(after) if after is not None else {}
        for key in set(before_values) | set(after_values):
            self.metric_totals[device][key] += int(after_values.get(key, 0)) - int(
                before_values.get(key, 0)
            )

    def _record_point(
        self, device: int, event_index: int, event: Dict[str, Any], time_us: int
    ) -> None:
        totals = self.metric_totals[device]
        reserved = int(totals.get("reserved_bytes", 0))
        row = {
            "device": device,
            "event_index": event_index,
            "time_us": time_us,
            "action": event.get("action"),
            "reserved_bytes": reserved,
            "active_block_bytes": int(totals.get("active_block_bytes", 0)),
            "releasable_cache_bytes": int(totals.get("releasable_cache_bytes", 0)),
            "stranded_free_bytes": int(totals.get("stranded_free_bytes", 0)),
            "pending_free_bytes": int(totals.get("pending_free_bytes", 0)),
            "fragmentation_ratio": (
                int(totals.get("stranded_free_bytes", 0)) / reserved if reserved else 0.0
            ),
            "frames": event.get("frames", []),
        }
        if self.record_timeline:
            self.timeline.append(row)
        if (device, event_index) in self.capture_indices:
            self.captured_states[(device, event_index)] = {
                "moment": dict(row),
                "segments": self._serialize_device_state(device),
            }

    def _serialize_device_state(self, device: int) -> List[Dict[str, Any]]:
        return [_serialize_replay_segment(segment) for segment in self.segments[device]]

    def _record_stranded_change(
        self, segment: ReplaySegment, before: int, time_us: int, event: Dict[str, Any]
    ) -> None:
        self._apply_stranded_delta(segment, segment.stranded_bytes() - before, time_us, event)

    def _apply_stranded_delta(
        self, segment: ReplaySegment, delta: int, time_us: int, event: Dict[str, Any]
    ) -> None:
        key = (segment.device, segment.address)
        if delta > 0:
            incident = {
                "type": "fragment-created",
                "device": segment.device,
                "time_us": time_us,
                "segment_address": segment.address,
                "stranded_delta_bytes": delta,
                "stranded_byte_us": 0,
                "trigger_action": event.get("action"),
                "trigger_address": event.get("addr"),
                "frames": event.get("frames", []),
            }
            incident_index = len(self.incidents)
            self.incidents.append(incident)
            self.open_fragments[key].append([incident_index, delta, time_us])
        elif delta < 0:
            remaining = -delta
            queue = self.open_fragments[key]
            while remaining and queue:
                incident_index, amount, start = queue[0]
                consumed = min(remaining, amount)
                self.incidents[incident_index]["stranded_byte_us"] += consumed * max(0, time_us - start)
                remaining -= consumed
                amount -= consumed
                if amount:
                    queue[0][1] = amount
                else:
                    queue.pop(0)

    def _close_open_fragments(self, end_time_us: int) -> None:
        for queue in self.open_fragments.values():
            for incident_index, amount, start in queue:
                self.incidents[incident_index]["stranded_byte_us"] += amount * max(0, end_time_us - start)

    def _domain_free_blocks(
        self, device: int, pool: Tuple[int, int], segment_type: str, stream: int
    ) -> List[ReplayBlock]:
        return [
            block
            for segment in self.segments[device]
            if segment.pool_id == pool and segment.segment_type == segment_type and segment.stream == stream
            for block in segment.blocks
            if block.state == "inactive"
        ]

    def _pinning_candidates(
        self, device: int, pool: Tuple[int, int], segment_type: str, stream: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        compatible_segments = [
            segment
            for segment in self.segments[device]
            if segment.pool_id == pool
            and segment.segment_type == segment_type
            and segment.stream == stream
        ]
        pinners: List[Dict[str, Any]] = []
        blocking_segments: List[Dict[str, Any]] = []
        for segment in compatible_segments:
            inactive = [block for block in segment.blocks if block.state == "inactive"]
            if not inactive:
                continue
            active = [block for block in segment.blocks if _is_active(block.state)]
            free_bytes = sum(block.size for block in inactive)
            blocking_segments.append(
                {
                    "segment_address": segment.address,
                    "segment_address_hex": f"0x{segment.address:x}",
                    "segment_bytes": segment.total_size,
                    "free_bytes": free_bytes,
                    "largest_free_block": max(block.size for block in inactive),
                    "active_bytes": sum(block.size for block in active),
                    "active_allocations": len(active),
                    "partially_active": bool(active),
                }
            )
            if not active:
                continue
            block = min(active, key=lambda value: (value.size, value.address))
            pinner = {
                "_allocation_id": block.allocation_id,
                "segment_address": segment.address,
                "segment_address_hex": f"0x{segment.address:x}",
                "segment_bytes": segment.total_size,
                "allocation_address": block.address,
                "allocation_address_hex": f"0x{block.address:x}",
                "allocation_bytes": block.size,
                "requested_bytes": block.requested_size,
                "state_at_exposure": block.state,
                "pinned_free_bytes": free_bytes,
                "pinning_score": free_bytes / max(1, block.size),
                "alloc_time_us": block.alloc_time_us,
                "frames": copy.deepcopy(block.frames),
                "co_pinners": len(active),
            }
            pinners.append(pinner)
            if block.allocation_id is not None:
                self.pinner_watchers[block.allocation_id].append(pinner)
        pinners.sort(
            key=lambda value: (
                float(value["pinning_score"]),
                int(value["pinned_free_bytes"]),
            ),
            reverse=True,
        )
        blocking_segments.sort(key=lambda value: int(value["free_bytes"]), reverse=True)
        return pinners, blocking_segments

    def _fragment_sources(
        self,
        device: int,
        blocking_segments: Sequence[Dict[str, Any]],
        exposure_time_us: int,
    ) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        for segment in blocking_segments:
            key = (device, int(segment["segment_address"]))
            for incident_index, amount, start in self.open_fragments.get(key, []):
                incident = self.incidents[incident_index]
                sources.append(
                    {
                        "segment_address": int(segment["segment_address"]),
                        "segment_address_hex": str(segment["segment_address_hex"]),
                        "created_time_us": int(start),
                        "age_at_exposure_us": max(0, exposure_time_us - int(start)),
                        "remaining_stranded_bytes": int(amount),
                        "original_stranded_delta_bytes": int(
                            incident.get("stranded_delta_bytes", amount)
                        ),
                        "trigger_action": incident.get("trigger_action"),
                        "trigger_address": incident.get("trigger_address"),
                        "frames": copy.deepcopy(incident.get("frames", [])),
                    }
                )
        sources.sort(key=lambda value: int(value["remaining_stranded_bytes"]), reverse=True)
        return sources

    def _finalize_causal_chains(self) -> None:
        for incident in self.incidents:
            chain = incident.get("causal_chain")
            if not isinstance(chain, dict):
                continue
            device = int(incident.get("device", 0))
            end_time_us = self.device_end_time_us.get(device, self.last_time_us)
            exposure_time = int(incident.get("time_us", end_time_us))
            for pinner in chain.get("pinners", []):
                pinner.pop("_allocation_id", None)
                free_time = pinner.get("free_time_us")
                alloc_time = pinner.get("alloc_time_us")
                lifetime_end = int(free_time) if free_time is not None else end_time_us
                pinner["free_time_us"] = free_time
                pinner["lifetime_complete"] = free_time is not None
                pinner["lifetime_is_lower_bound"] = free_time is None
                pinner["lifetime_us"] = (
                    max(0, lifetime_end - int(alloc_time)) if alloc_time is not None else None
                )
                pinner["age_at_exposure_us"] = (
                    max(0, exposure_time - int(alloc_time)) if alloc_time is not None else None
                )

    def _validate_final_snapshot(self) -> None:
        expected: Dict[int, List[Tuple[Any, ...]]] = defaultdict(list)
        for segment in self.snapshot.get("segments", []):
            device = int(segment.get("device", 0))
            blocks = tuple(
                (
                    int(block.get("address", 0)),
                    int(block.get("size", 0)),
                    int(block.get("requested_size", 0) or 0)
                    if _is_active(str(block.get("state", "")))
                    else 0,
                    "active_pending_free" if _is_pending(str(block.get("state", ""))) else str(block.get("state", "")),
                )
                for block in segment.get("blocks", [])
            )
            expected[device].append(
                (
                    int(segment.get("address", 0)),
                    int(segment.get("total_size", 0)),
                    int(segment.get("stream", 0)),
                    str(segment.get("segment_type", "unknown")),
                    blocks,
                )
            )
        actual: Dict[int, List[Tuple[Any, ...]]] = defaultdict(list)
        for device, segments in self.segments.items():
            for segment in segments:
                blocks = tuple(
                    (
                        block.address,
                        block.size,
                        block.requested_size,
                        "active_pending_free" if _is_pending(block.state) else block.state,
                    )
                    for block in segment.blocks
                )
                actual[device].append(
                    (segment.address, segment.total_size, segment.stream, segment.segment_type, blocks)
                )
        for values in expected.values():
            values.sort(key=lambda item: item[0])
        for values in actual.values():
            values.sort(key=lambda item: item[0])
        if dict(expected) != dict(actual):
            raise ReplayError("replayed allocator layout does not match final snapshot")


@dataclass
class ReverseBlock:
    address: int
    size: int
    requested_size: int
    state: str
    frames: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty_bytes: int = 0
    reconstructed: bool = False


@dataclass
class ReverseSegment:
    device: int
    address: int
    total_size: int
    stream: int
    pool_id: Tuple[int, int]
    segment_type: str
    blocks: Dict[int, ReverseBlock] = field(default_factory=dict)
    active_bytes: int = 0
    pending_bytes: int = 0
    uncertainty_bytes: int = 0

    def contains(self, address: int) -> bool:
        return self.address <= address < self.address + self.total_size

    def add_block(self, block: ReverseBlock) -> None:
        if block.address in self.blocks:
            raise ReplayError(f"reverse replay duplicated allocation at 0x{block.address:x}")
        if block.address < self.address or block.address + block.size > self.address + self.total_size:
            raise ReplayError(f"reverse replay allocation leaves segment at 0x{block.address:x}")
        for other in self.blocks.values():
            if block.address < other.address + other.size and other.address < block.address + block.size:
                raise ReplayError(f"reverse replay allocation overlaps at 0x{block.address:x}")
        self.blocks[block.address] = block
        self.active_bytes += block.size
        self.uncertainty_bytes += block.uncertainty_bytes
        if _is_pending(block.state):
            self.pending_bytes += block.size

    def remove_block(self, address: int) -> ReverseBlock:
        block = self.blocks.pop(address, None)
        if block is None:
            raise ReplayError(f"reverse replay cannot undo unknown allocation at 0x{address:x}")
        self.active_bytes -= block.size
        self.uncertainty_bytes -= block.uncertainty_bytes
        if _is_pending(block.state):
            self.pending_bytes -= block.size
        return block


class ReverseReplay:
    """Approximate retained history by undoing events from the final snapshot.

    Segment membership and allocation liveness are exact inside the retained
    trace window. Historical block sizes are conservative lower estimates:
    requests are rounded using the recorded/default native allocator policy,
    while a possible non-split tail is reported as an uncertainty interval.
    """

    def __init__(
        self,
        snapshot: Dict[str, Any],
        capture_indices: Optional[Sequence[Tuple[int, int]]] = None,
        record_timeline: bool = True,
    ) -> None:
        self.snapshot = snapshot
        self.settings = snapshot.get("allocator_settings", {})
        self.capture_indices = set(capture_indices or ())
        self.record_timeline = record_timeline
        self.segments: DefaultDict[int, Dict[int, ReverseSegment]] = defaultdict(dict)
        self.segment_bases: DefaultDict[int, List[int]] = defaultdict(list)
        self.metric_totals: DefaultDict[int, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.timeline: List[Dict[str, Any]] = []
        self.captured_states: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.assumptions = {
            "backend": "native",
            "rounding": "snapshot_or_default_512_bytes",
            "max_non_split_rounding_bytes": int(
                self.settings.get(
                    "max_non_split_rounding_size", DEFAULT_MAX_NON_SPLIT_ROUNDING
                )
            ),
            "missing_settings": "pytorch_native_defaults",
        }

    def run(self) -> Dict[str, Any]:
        try:
            self._check_supported()
            self._load_final_state()
            for device, events in enumerate(self.snapshot.get("device_traces", [])):
                self._run_device(device, events)
            self.timeline.sort(
                key=lambda row: (
                    int(row.get("time_us", 0)),
                    int(row.get("device", 0)),
                    int(row.get("event_index", 0)),
                )
            )
        except (ReplayError, ValueError, TypeError) as exc:
            return {
                "available": False,
                "error": str(exc),
                "timeline": [],
                "captured_states": {},
                "assumptions": self.assumptions,
            }
        available = bool(self.timeline or self.captured_states)
        return {
            "available": available,
            "error": None if available else "snapshot has no retained allocator events",
            "timeline": self.timeline,
            "captured_states": self.captured_states,
            "assumptions": self.assumptions,
        }

    def _check_supported(self) -> None:
        config = str(self.settings.get("PYTORCH_CUDA_ALLOC_CONF", ""))
        if "backend:cudaMallocAsync" in config.replace(" ", ""):
            raise ReplayError("reverse replay currently supports only backend:native")
        if self.settings.get("expandable_segments"):
            raise ReplayError("expandable_segments=True requires map/unmap-aware reverse replay")
        divisions = self.settings.get("roundup_power2_divisions", {})
        if isinstance(divisions, dict) and any(int(value) != 0 for value in divisions.values()):
            raise ReplayError("non-default roundup_power2_divisions is not supported")
        for events in self.snapshot.get("device_traces", []):
            if any(event.get("action") in {"segment_map", "segment_unmap"} for event in events):
                raise ReplayError("segment_map/segment_unmap requires expandable-segment replay")

    def _load_final_state(self) -> None:
        for raw_segment in self.snapshot.get("segments", []):
            device = int(raw_segment.get("device", 0))
            segment = ReverseSegment(
                device=device,
                address=int(raw_segment.get("address", 0)),
                total_size=int(raw_segment.get("total_size", 0)),
                stream=int(raw_segment.get("stream", 0)),
                pool_id=_pool_id(raw_segment.get("segment_pool_id")),
                segment_type=str(raw_segment.get("segment_type", "unknown")),
            )
            for raw_block in raw_segment.get("blocks", []):
                state = str(raw_block.get("state", ""))
                if not _is_active(state):
                    continue
                segment.add_block(
                    ReverseBlock(
                        address=int(raw_block.get("address", 0)),
                        size=int(raw_block.get("size", 0)),
                        requested_size=int(raw_block.get("requested_size", 0) or 0),
                        state=state,
                        frames=copy.deepcopy(raw_block.get("frames", [])),
                    )
                )
            self._insert_segment(segment)

    @staticmethod
    def _segment_contribution(segment: ReverseSegment) -> Dict[str, int]:
        active = len(segment.blocks)
        stranded = segment.total_size - segment.active_bytes if active else 0
        uncertainty = min(stranded, segment.uncertainty_bytes) if active else 0
        return {
            "reserved_bytes": segment.total_size,
            "active_block_bytes": segment.active_bytes,
            "releasable_cache_bytes": segment.total_size if not active else 0,
            "stranded_free_bytes": stranded,
            "fragmentation_uncertainty_bytes": uncertainty,
            "pending_free_bytes": segment.pending_bytes,
        }

    def _change_segment(
        self, device: int, segment: ReverseSegment, mutate: Any
    ) -> None:
        before = self._segment_contribution(segment)
        mutate()
        after = self._segment_contribution(segment)
        totals = self.metric_totals[device]
        for key in set(before) | set(after):
            totals[key] += int(after.get(key, 0)) - int(before.get(key, 0))

    def _insert_segment(self, segment: ReverseSegment) -> None:
        device = segment.device
        if segment.address in self.segments[device]:
            raise ReplayError(f"reverse replay duplicated segment at 0x{segment.address:x}")
        index = bisect.bisect_left(self.segment_bases[device], segment.address)
        if index and self.segments[device][self.segment_bases[device][index - 1]].contains(
            segment.address
        ):
            raise ReplayError(f"reverse replay overlapping segment at 0x{segment.address:x}")
        if index < len(self.segment_bases[device]):
            next_segment = self.segments[device][self.segment_bases[device][index]]
            if segment.address + segment.total_size > next_segment.address:
                raise ReplayError(f"reverse replay overlapping segment at 0x{segment.address:x}")
        self.segments[device][segment.address] = segment
        self.segment_bases[device].insert(index, segment.address)
        for key, value in self._segment_contribution(segment).items():
            self.metric_totals[device][key] += value

    def _remove_segment(self, device: int, address: int, size: int) -> None:
        segment = self.segments[device].get(address)
        if segment is None or segment.total_size != size:
            raise ReplayError(f"reverse replay cannot undo segment_alloc at 0x{address:x}")
        if segment.blocks:
            raise ReplayError(f"reverse replay segment still active at 0x{address:x}")
        for key, value in self._segment_contribution(segment).items():
            self.metric_totals[device][key] -= value
        del self.segments[device][address]
        self.segment_bases[device].remove(address)

    def _find_segment(self, device: int, address: int) -> ReverseSegment:
        bases = self.segment_bases[device]
        index = bisect.bisect_right(bases, address) - 1
        if index < 0:
            raise ReplayError(f"reverse replay address 0x{address:x} has no segment")
        segment = self.segments[device][bases[index]]
        if not segment.contains(address):
            raise ReplayError(f"reverse replay address 0x{address:x} has no segment")
        return segment

    def _uncertainty_limit(self, segment: ReverseSegment, rounded: int) -> int:
        if segment.segment_type == "small":
            return 0
        max_split = int(self.settings.get("max_split_size", -1))
        if max_split < 0 or rounded < max_split:
            return SMALL_LIMIT
        return int(
            self.settings.get(
                "max_non_split_rounding_size", DEFAULT_MAX_NON_SPLIT_ROUNDING
            )
        )

    def _run_device(self, device: int, events: Sequence[Dict[str, Any]]) -> None:
        rows = []
        for index in range(len(events) - 1, -1, -1):
            event = events[index]
            action = str(event.get("action", ""))
            if action in {
                "segment_alloc",
                "segment_free",
                "alloc",
                "free",
                "free_requested",
                "free_completed",
            }:
                row = self._record_point(device, index, event)
                if self.record_timeline:
                    rows.append(row)
                if (device, index) in self.capture_indices:
                    self.captured_states[(device, index)] = {
                        "moment": dict(row),
                        "segments": self._serialize_device_state(device),
                    }
            self._undo_event(device, event)
        rows.reverse()
        self.timeline.extend(rows)

    def _undo_event(self, device: int, event: Dict[str, Any]) -> None:
        action = str(event.get("action", ""))
        address = int(event.get("addr", 0))
        size = int(event.get("size", 0))
        if action == "alloc":
            segment = self._find_segment(device, address)
            self._change_segment(device, segment, lambda: segment.remove_block(address))
        elif action in {"free", "free_completed"}:
            segment = self._find_segment(device, address)
            rounded = _round_size(size, self.settings)
            block = ReverseBlock(
                address=address,
                size=rounded,
                requested_size=size,
                state="active_allocated" if action == "free" else "active_pending_free",
                frames=copy.deepcopy(event.get("frames", [])),
                uncertainty_bytes=self._uncertainty_limit(segment, rounded),
                reconstructed=True,
            )
            self._change_segment(device, segment, lambda: segment.add_block(block))
        elif action == "free_requested":
            segment = self._find_segment(device, address)
            block = segment.blocks.get(address)
            if block is not None and _is_pending(block.state):
                def clear_pending() -> None:
                    segment.pending_bytes -= block.size
                    block.state = "active_allocated"

                self._change_segment(device, segment, clear_pending)
        elif action == "segment_alloc":
            self._remove_segment(device, address, size)
        elif action == "segment_free":
            segment_type = "small" if size == 2 * MIB else "large"
            self._insert_segment(
                ReverseSegment(
                    device=device,
                    address=address,
                    total_size=size,
                    stream=int(event.get("stream", 0)),
                    pool_id=_event_pool(event),
                    segment_type=segment_type,
                )
            )

    def _record_point(
        self, device: int, event_index: int, event: Dict[str, Any]
    ) -> Dict[str, Any]:
        totals = self.metric_totals[device]
        reserved = int(totals.get("reserved_bytes", 0))
        stranded = int(totals.get("stranded_free_bytes", 0))
        uncertainty = min(
            stranded, int(totals.get("fragmentation_uncertainty_bytes", 0))
        )
        return {
            "device": device,
            "event_index": event_index,
            "time_us": int(event.get("time_us", 0)),
            "action": event.get("action"),
            "reserved_bytes": reserved,
            "active_block_bytes": int(totals.get("active_block_bytes", 0)),
            "releasable_cache_bytes": int(totals.get("releasable_cache_bytes", 0)),
            "stranded_free_bytes": stranded,
            "fragmentation_uncertainty_bytes": uncertainty,
            "fragmentation_ratio": stranded / reserved if reserved else 0.0,
            "fragmentation_ratio_lower": (
                (stranded - uncertainty) / reserved if reserved else 0.0
            ),
            "pending_free_bytes": int(totals.get("pending_free_bytes", 0)),
            "frames": event.get("frames", []),
            "approximate": True,
        }

    def _serialize_device_state(self, device: int) -> List[Dict[str, Any]]:
        result = []
        for address in self.segment_bases[device]:
            segment = self.segments[device][address]
            blocks = []
            cursor = segment.address
            for block in sorted(segment.blocks.values(), key=lambda value: value.address):
                if block.address > cursor:
                    blocks.append(
                        {
                            "address": f"0x{cursor:x}",
                            "size": block.address - cursor,
                            "requested_size": 0,
                            "state": "inactive",
                            "frames": [],
                        }
                    )
                blocks.append(
                    {
                        "address": f"0x{block.address:x}",
                        "size": block.size,
                        "requested_size": block.requested_size,
                        "state": block.state,
                        "frames": _compact_frames(block.frames),
                        "uncertainty_bytes": block.uncertainty_bytes,
                        "reconstructed": block.reconstructed,
                    }
                )
                cursor = block.address + block.size
            end = segment.address + segment.total_size
            if cursor < end:
                blocks.append(
                    {
                        "address": f"0x{cursor:x}",
                        "size": end - cursor,
                        "requested_size": 0,
                        "state": "inactive",
                        "frames": [],
                    }
                )
            result.append(
                {
                    "address": f"0x{segment.address:x}",
                    "total_size": segment.total_size,
                    "stream": segment.stream,
                    "segment_type": segment.segment_type,
                    "pool_id": list(segment.pool_id),
                    "partially_active": bool(segment.blocks)
                    and segment.active_bytes < segment.total_size,
                    "blocks": blocks,
                }
            )
        return result


def _compact_frames(frames: Sequence[Dict[str, Any]], limit: int = 12) -> List[str]:
    values = []
    for frame in frames[:limit]:
        filename = str(frame.get("filename", "")).replace("\\", "/").rsplit("/", 1)[-1]
        values.append(f"{filename}:{frame.get('line', 0)}:{frame.get('name', '')}")
    return values


def _serialize_replay_segment(segment: ReplaySegment) -> Dict[str, Any]:
    active = any(_is_active(block.state) for block in segment.blocks)
    return {
        "address": f"0x{segment.address:x}",
        "total_size": segment.total_size,
        "stream": segment.stream,
        "segment_type": segment.segment_type,
        "pool_id": list(segment.pool_id),
        "partially_active": active and any(block.state == "inactive" for block in segment.blocks),
        "blocks": [
            {
                "address": f"0x{block.address:x}",
                "size": block.size,
                "requested_size": block.requested_size,
                "state": block.state,
                "frames": _compact_frames(block.frames),
            }
            for block in segment.blocks
        ],
    }


def _serialize_final_state(snapshot: Dict[str, Any], device: int) -> List[Dict[str, Any]]:
    result = []
    for segment in snapshot.get("segments", []):
        if int(segment.get("device", 0)) != device:
            continue
        blocks = []
        cursor = int(segment.get("address", 0))
        active = False
        inactive = False
        for block in segment.get("blocks", []):
            address = int(block.get("address", cursor))
            state = str(block.get("state", "inactive"))
            active = active or _is_active(state)
            inactive = inactive or state == "inactive"
            blocks.append(
                {
                    "address": f"0x{address:x}",
                    "size": int(block.get("size", 0)),
                    "requested_size": int(block.get("requested_size", 0) or 0),
                    "state": state,
                    "frames": _compact_frames(block.get("frames", [])),
                }
            )
            cursor = address + int(block.get("size", 0))
        result.append(
            {
                "address": f"0x{int(segment.get('address', 0)):x}",
                "total_size": int(segment.get("total_size", 0)),
                "stream": int(segment.get("stream", 0)),
                "segment_type": str(segment.get("segment_type", "unknown")),
                "pool_id": list(_pool_id(segment.get("segment_pool_id"))),
                "partially_active": active and inactive,
                "blocks": blocks,
            }
        )
    return result


def _select_top_moments(
    timeline: Sequence[Dict[str, Any]],
    top_k: int,
    threshold: float,
    min_reserved_bytes: int = 0,
    min_stranded_bytes: int = 0,
) -> List[Dict[str, Any]]:
    eligible = [
        row
        for row in timeline
        if float(row["fragmentation_ratio"]) >= threshold
        and int(row["reserved_bytes"]) >= min_reserved_bytes
        and int(row["stranded_free_bytes"]) >= min_stranded_bytes
    ]
    ordered = sorted(
        eligible,
        key=lambda row: (float(row["fragmentation_ratio"]), int(row["stranded_free_bytes"])),
        reverse=True,
    )
    selected: List[Dict[str, Any]] = []
    for separation in (100, 10, 0):
        for row in ordered:
            key = (int(row["device"]), int(row["event_index"]))
            if any(
                int(other["device"]) == key[0]
                and abs(int(other["event_index"]) - key[1]) <= separation
                for other in selected
            ):
                continue
            selected.append(row)
            if len(selected) >= top_k:
                return selected
    return selected


def _downsample_timeline(
    timeline: Sequence[Dict[str, Any]], selected: Sequence[Dict[str, Any]], limit: int = 4000
) -> List[Dict[str, Any]]:
    keep_keys = {(int(row["device"]), int(row["event_index"])) for row in selected}
    if len(timeline) <= limit:
        rows = timeline
    else:
        stride = max(1, len(timeline) // limit)
        rows = [
            row
            for index, row in enumerate(timeline)
            if index % stride == 0
            or (int(row["device"]), int(row["event_index"])) in keep_keys
            or index == len(timeline) - 1
        ]
    required = (
        "device",
        "event_index",
        "time_us",
        "fragmentation_ratio",
        "stranded_free_bytes",
        "reserved_bytes",
    )
    optional = (
        "fragmentation_ratio_lower",
        "fragmentation_uncertainty_bytes",
        "approximate",
    )
    return [
        {
            **{key: row[key] for key in required},
            **{key: row[key] for key in optional if key in row},
        }
        for row in rows
    ]


def build_static_pinning_incidents(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find partially active segments even when exact event replay is unavailable."""
    live_allocs: DefaultDict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for device, events in enumerate(snapshot.get("device_traces", [])):
        for event in events:
            action = event.get("action")
            address = int(event.get("addr", -1))
            if action == "alloc":
                live_allocs[device][address] = event
            elif action == "free_completed":
                live_allocs[device].pop(address, None)

    incidents = []
    for segment in snapshot.get("segments", []):
        device = int(segment.get("device", 0))
        inactive = [block for block in segment.get("blocks", []) if block.get("state") == "inactive"]
        active = [block for block in segment.get("blocks", []) if _is_active(str(block.get("state", "")))]
        stranded = sum(int(block.get("size", 0)) for block in inactive)
        if not active or stranded < MIB:
            continue
        pinner = min(active, key=lambda block: int(block.get("size", 0)))
        address = int(pinner.get("address", 0))
        alloc_event = live_allocs[device].get(address, {})
        incidents.append(
            {
                "type": "segment-pinned",
                "device": device,
                "time_us": alloc_event.get("time_us"),
                "segment_address": int(segment.get("address", 0)),
                "segment_bytes": int(segment.get("total_size", 0)),
                "allocation_address": address,
                "allocation_bytes": int(pinner.get("size", 0)),
                "requested_bytes": int(pinner.get("requested_size", 0) or 0),
                "pinned_free_bytes": stranded,
                "co_pinners": len(active),
                "frames": alloc_event.get("frames", pinner.get("frames", [])),
            }
        )
    return incidents


def _phase_point(
    resolver: PhaseResolver,
    device: int,
    time_us: Optional[int],
    frames: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "time_us": time_us,
        **resolver.resolve(device, time_us, frames),
    }


def analyze_snapshot(
    snapshot: Dict[str, Any],
    min_fragment_bytes: int = MIB,
    top_k: int = 10,
    fragmentation_threshold: float = 0.0,
    min_reserved_bytes: int = 0,
    min_stranded_bytes: int = 0,
) -> Dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not 0.0 <= fragmentation_threshold <= 1.0:
        raise ValueError("fragmentation_threshold must be between 0 and 1")
    if min_reserved_bytes < 0 or min_stranded_bytes < 0:
        raise ValueError("minimum byte filters cannot be negative")
    summary = summarize_snapshot(snapshot)
    replay = NativeReplay(snapshot, min_fragment_bytes=min_fragment_bytes).run()
    reverse = (
        {"available": False, "error": None, "timeline": [], "captured_states": {}}
        if replay["exact"]
        else ReverseReplay(snapshot).run()
    )
    incidents = list(replay["incidents"] if replay["exact"] else [])
    incidents.extend(build_static_pinning_incidents(snapshot))
    resolver = PhaseResolver(snapshot)
    for incident in incidents:
        phase = resolver.resolve(
            int(incident.get("device", 0)),
            int(incident["time_us"]) if incident.get("time_us") is not None else None,
            incident.get("frames", []),
        )
        incident.update(phase)
        chain = incident.get("causal_chain")
        if isinstance(chain, dict):
            device = int(incident.get("device", 0))
            chain_id = (
                f"device{device}-event{int(incident.get('event_index', -1))}-"
                f"segment0x{int(incident.get('address', 0)):x}"
            )
            incident["chain_id"] = chain_id
            chain["chain_id"] = chain_id
            chain["exposed_at"] = _phase_point(
                resolver,
                device,
                int(incident["time_us"]) if incident.get("time_us") is not None else None,
                incident.get("frames", []),
            )
            for pinner in chain.get("pinners", []):
                alloc_time = (
                    int(pinner["alloc_time_us"])
                    if pinner.get("alloc_time_us") is not None
                    else None
                )
                pinner["created_at"] = _phase_point(
                    resolver, device, alloc_time, pinner.get("frames", [])
                )
            for source in chain.get("fragment_sources", []):
                created_time = (
                    int(source["created_time_us"])
                    if source.get("created_time_us") is not None
                    else None
                )
                source["created_at"] = _phase_point(
                    resolver, device, created_time, source.get("frames", [])
                )
            pinner = incident.get("main_pinner")
            if isinstance(pinner, dict) and isinstance(pinner.get("created_at"), dict):
                pinner.update(pinner["created_at"])

    incidents.sort(key=_incident_score, reverse=True)
    causal_chains = [
        incident["causal_chain"]
        for incident in incidents
        if isinstance(incident.get("causal_chain"), dict)
    ]
    dashboard = _build_dashboard_data(
        snapshot,
        summary,
        replay,
        reverse,
        resolver,
        incidents,
        top_k=top_k,
        fragmentation_threshold=fragmentation_threshold,
        min_fragment_bytes=min_fragment_bytes,
        min_reserved_bytes=min_reserved_bytes,
        min_stranded_bytes=min_stranded_bytes,
    )
    return {
        "summary": summary,
        "replay": {
            "exact": replay["exact"],
            "error": replay["error"],
            "mode": (
                "exact_forward"
                if replay["exact"]
                else "reverse_approximate"
                if reverse.get("available")
                else "final_snapshot_only"
            ),
            "reverse_available": bool(reverse.get("available")),
            "reverse_error": reverse.get("error"),
            "reverse_assumptions": reverse.get("assumptions", {}),
            "causal_chain_count": len(causal_chains),
            "causal_chains_available": bool(replay["exact"]),
        },
        "incidents": incidents,
        "causal_chains": causal_chains,
        "dashboard": dashboard,
    }


def _build_dashboard_data(
    snapshot: Dict[str, Any],
    summary: Dict[str, Any],
    replay: Dict[str, Any],
    reverse: Dict[str, Any],
    resolver: PhaseResolver,
    incidents: Sequence[Dict[str, Any]],
    *,
    top_k: int,
    fragmentation_threshold: float,
    min_fragment_bytes: int,
    min_reserved_bytes: int,
    min_stranded_bytes: int,
) -> Dict[str, Any]:
    moments: List[Dict[str, Any]] = []
    timeline: List[Dict[str, Any]] = []
    history = replay if replay["exact"] else reverse
    history_available = bool(replay["exact"] or reverse.get("available"))
    if history_available:
        selected = _select_top_moments(
            history["timeline"],
            top_k=top_k,
            threshold=fragmentation_threshold,
            min_reserved_bytes=min_reserved_bytes,
            min_stranded_bytes=min_stranded_bytes,
        )
        keys = [(int(row["device"]), int(row["event_index"])) for row in selected]
        if not selected:
            captured = {"captured_states": {}}
            captured_ok = True
        elif replay["exact"]:
            captured = NativeReplay(
                snapshot,
                min_fragment_bytes=min_fragment_bytes,
                capture_indices=keys,
                record_timeline=False,
            ).run()
            captured_ok = bool(captured["exact"])
        else:
            captured = ReverseReplay(
                snapshot,
                capture_indices=keys,
                record_timeline=False,
            ).run()
            captured_ok = bool(captured.get("available"))
        if captured_ok:
            for rank, row in enumerate(selected, 1):
                state = captured["captured_states"].get(
                    (int(row["device"]), int(row["event_index"]))
                )
                if not state:
                    continue
                phase = resolver.resolve(
                    int(row["device"]), int(row["time_us"]), row.get("frames", [])
                )
                moment = {key: value for key, value in row.items() if key != "frames"}
                moment.update(phase)
                moment.update(
                    {
                        "rank": rank,
                        "segments": state["segments"],
                        "history_mode": (
                            "exact_forward" if replay["exact"] else "reverse_approximate"
                        ),
                    }
                )
                moments.append(moment)
            timeline = _downsample_timeline(history["timeline"], selected)
    else:
        for rank, (device_text, values) in enumerate(
            sorted(
                summary["devices"].items(),
                key=lambda item: (
                    item[1].get("stranded_free_bytes", 0)
                    / max(1, item[1].get("reserved_bytes", 0))
                ),
                reverse=True,
            )[:top_k],
            1,
        ):
            device = int(device_text)
            events = snapshot.get("device_traces", [])
            device_events = events[device] if device < len(events) else []
            final_time = int(device_events[-1].get("time_us", 0)) if device_events else 0
            reserved = int(values.get("reserved_bytes", 0))
            stranded = int(values.get("stranded_free_bytes", 0))
            moment = {
                "rank": rank,
                "device": device,
                "event_index": len(device_events) - 1,
                "time_us": final_time,
                "action": "final_snapshot",
                "reserved_bytes": reserved,
                "active_block_bytes": int(values.get("active_block_bytes", 0)),
                "releasable_cache_bytes": int(values.get("releasable_cache_bytes", 0)),
                "stranded_free_bytes": stranded,
                "pending_free_bytes": int(values.get("pending_free_bytes", 0)),
                "fragmentation_ratio": stranded / reserved if reserved else 0.0,
                "fragmentation_ratio_lower": stranded / reserved if reserved else 0.0,
                "fragmentation_uncertainty_bytes": 0,
                "primary_phase": "final_snapshot",
                "phase_confidence": "approximate",
                "phase_metadata": {},
                "annotations": [],
                "history_mode": "final_snapshot_only",
                "segments": _serialize_final_state(snapshot, device),
            }
            moments.append(moment)
            timeline.append(
                {
                    key: moment[key]
                    for key in (
                        "device",
                        "event_index",
                        "time_us",
                        "fragmentation_ratio",
                        "stranded_free_bytes",
                        "reserved_bytes",
                    )
                }
            )

    phase_intervals = [
        {
            "device": interval.device,
            "start_us": interval.start_us,
            "end_us": interval.end_us,
            "phase": interval.metadata.get("phase"),
            "metadata": {
                key: value for key, value in interval.metadata.items() if key != "phase"
            },
        }
        for values in resolver.intervals.values()
        for interval in values
        if interval.metadata
    ]
    return {
        "exact": bool(replay["exact"]),
        "history_mode": (
            "exact_forward"
            if replay["exact"]
            else "reverse_approximate"
            if reverse.get("available")
            else "final_snapshot_only"
        ),
        "replay_error": replay.get("error"),
        "reverse_error": reverse.get("error"),
        "reverse_assumptions": reverse.get("assumptions", {}),
        "top_k": top_k,
        "initial_threshold": fragmentation_threshold,
        "min_reserved_bytes": min_reserved_bytes,
        "min_stranded_bytes": min_stranded_bytes,
        "timeline": timeline,
        "moments": moments,
        "phase_intervals": phase_intervals,
        "causal_chains": [
            incident["causal_chain"]
            for incident in incidents
            if isinstance(incident.get("causal_chain"), dict)
        ][:top_k],
        "incidents": [
            {
                key: incident.get(key)
                for key in (
                    "type",
                    "device",
                    "time_us",
                    "primary_phase",
                    "pinned_free_bytes",
                    "extra_segment_bytes",
                    "stranded_delta_bytes",
                    "chain_id",
                    "causal_chain",
                )
            }
            for incident in incidents[:top_k]
        ],
    }


def _incident_score(incident: Dict[str, Any]) -> int:
    return int(
        incident.get("extra_segment_bytes", 0)
        or incident.get("pinned_free_bytes", 0)
        or incident.get("stranded_delta_bytes", 0)
    )


def write_analysis(result: Dict[str, Any], output_dir: Path, top: int = 50) -> None:
    from .dashboard import write_dashboard

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result["summary"], handle, ensure_ascii=False, indent=2, sort_keys=True)
    with (output_dir / "replay.json").open("w", encoding="utf-8") as handle:
        json.dump(result["replay"], handle, ensure_ascii=False, indent=2, sort_keys=True)
    with (output_dir / "incidents.json").open("w", encoding="utf-8") as handle:
        json.dump(result["incidents"][:top], handle, ensure_ascii=False, indent=2, sort_keys=True)
    with (output_dir / "causal_chains.json").open("w", encoding="utf-8") as handle:
        json.dump(result["causal_chains"][:top], handle, ensure_ascii=False, indent=2, sort_keys=True)
    columns = (
        "type",
        "device",
        "time_us",
        "primary_phase",
        "phase_confidence",
        "stranded_delta_bytes",
        "stranded_byte_us",
        "pinned_free_bytes",
        "extra_segment_bytes",
        "request_bytes",
        "compatible_free_bytes",
        "largest_free_block",
        "segment_address",
        "allocation_address",
        "chain_id",
        "main_pinner_address",
        "main_pinner_pinning_score",
        "main_pinner_lifetime_us",
        "phase_metadata",
    )
    with (output_dir / "incidents.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for incident in result["incidents"][:top]:
            row = dict(incident)
            pinner = incident.get("main_pinner")
            if isinstance(pinner, dict):
                row["main_pinner_address"] = pinner.get("allocation_address")
                row["main_pinner_pinning_score"] = pinner.get("pinning_score")
                row["main_pinner_lifetime_us"] = pinner.get("lifetime_us")
            row["phase_metadata"] = json.dumps(row.get("phase_metadata", {}), ensure_ascii=False)
            writer.writerow(row)
    write_dashboard(result["dashboard"], output_dir / "dashboard.html")
