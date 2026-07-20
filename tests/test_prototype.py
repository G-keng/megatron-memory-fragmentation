from __future__ import annotations

import unittest
import tempfile
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from memory_fragmentation.analyzer import (
    MIB,
    PhaseResolver,
    analyze_snapshot,
    summarize_snapshot,
    write_analysis,
)
from memory_fragmentation.megatron_patch import (
    FILE_EDITS,
    SUPPORTED_COMMITS,
    SUPPORTED_VERSIONS,
    _apply_edits,
    format_supported_versions,
    main as patch_main,
)
from memory_fragmentation.phases import format_phase_name, memory_phase, parse_phase_name


def synthetic_snapshot():
    base = 0x10000000
    second = 0x20000000
    frame = [{"filename": "schedules.py", "line": 1, "name": "forward_step"}]

    def event(action, time_us, addr=0, size=0, frames=None):
        return {
            "action": action,
            "addr": addr,
            "size": size,
            "stream": 0,
            "time_us": time_us,
            "frames": frame if frames is None else frames,
        }

    events = [
        event("segment_alloc", 1, base, 20 * MIB),
        event("alloc", 2, base, 8 * MIB),
        event("alloc", 3, base + 8 * MIB, 8 * MIB),
        event("free_requested", 4, base, 8 * MIB),
        event("free_completed", 5, base, 8 * MIB),
        event("segment_alloc", 6, second, 10 * MIB),
        event("alloc", 7, second, 10 * MIB),
        event("snapshot", 8),
    ]
    annotations = [
        {
            "stage": "START",
            "name": "memtrace|phase=pipeline/steady/forward|iter=1|mb=2|chunk=0",
            "device": 0,
            "time_us": 5,
        },
        {
            "stage": "END",
            "name": "memtrace|phase=pipeline/steady/forward|iter=1|mb=2|chunk=0",
            "device": 0,
            "time_us": 8,
        },
    ]
    segments = [
        {
            "device": 0,
            "address": base,
            "total_size": 20 * MIB,
            "stream": 0,
            "segment_type": "large",
            "segment_pool_id": (0, 0),
            "blocks": [
                {
                    "address": base,
                    "size": 8 * MIB,
                    # Native snapshots retain the previous request on inactive blocks.
                    "requested_size": 8 * MIB,
                    "state": "inactive",
                },
                {
                    "address": base + 8 * MIB,
                    "size": 8 * MIB,
                    "requested_size": 8 * MIB,
                    "state": "active_allocated",
                    "frames": frame,
                },
                {
                    "address": base + 16 * MIB,
                    "size": 4 * MIB,
                    "requested_size": 0,
                    "state": "inactive",
                },
            ],
        },
        {
            "device": 0,
            "address": second,
            "total_size": 10 * MIB,
            "stream": 0,
            "segment_type": "large",
            "segment_pool_id": (0, 0),
            "blocks": [
                {
                    "address": second,
                    "size": 10 * MIB,
                    "requested_size": 10 * MIB,
                    "state": "active_allocated",
                    "frames": frame,
                }
            ],
        },
    ]
    return {
        "segments": segments,
        "device_traces": [events],
        "external_annotations": annotations,
        "allocator_settings": {
            "expandable_segments": False,
            "max_split_size": -1,
            "roundup_power2_divisions": {"1": 0, "2": 0},
        },
    }


def truncated_snapshot():
    base = 0x30000000

    def event(action, time_us, addr=0, size=0):
        return {
            "action": action,
            "addr": addr,
            "size": size,
            "stream": 0,
            "time_us": time_us,
            "frames": [{"filename": "schedules.py", "line": 2, "name": "forward_step"}],
        }

    # The segment and the two original allocations predate the retained trace.
    events = [
        event("free_requested", 10, base, 8 * MIB - 4),
        event("free_completed", 11, base, 8 * MIB - 4),
        event("alloc", 12, base, 4 * MIB),
        event("snapshot", 13),
    ]
    return {
        "segments": [
            {
                "device": 0,
                "address": base,
                "total_size": 20 * MIB,
                "stream": 0,
                "segment_type": "large",
                "segment_pool_id": (0, 0),
                "blocks": [
                    {
                        "address": base,
                        "size": 4 * MIB,
                        "requested_size": 4 * MIB,
                        "state": "active_allocated",
                    },
                    {
                        "address": base + 4 * MIB,
                        "size": 4 * MIB,
                        "requested_size": 0,
                        "state": "inactive",
                    },
                    {
                        "address": base + 8 * MIB,
                        "size": 8 * MIB,
                        "requested_size": 8 * MIB,
                        "state": "active_allocated",
                    },
                    {
                        "address": base + 16 * MIB,
                        "size": 4 * MIB,
                        "requested_size": 0,
                        "state": "inactive",
                    },
                ],
            }
        ],
        "device_traces": [events],
        "external_annotations": [],
        "allocator_settings": {
            "PYTORCH_CUDA_ALLOC_CONF": "",
            "expandable_segments": False,
            "max_split_size": -1,
            "roundup_power2_divisions": {"1": 0, "2": 0},
        },
    }


class PhaseTests(unittest.TestCase):
    def test_phase_name_round_trip(self):
        name = format_phase_name(
            "pipeline/steady/backward", iteration=7, microbatch=3, direction="backward", model_chunk=1
        )
        self.assertEqual(
            parse_phase_name(name),
            {
                "phase": "pipeline/steady/backward",
                "iteration": 7,
                "microbatch": 3,
                "direction": "backward",
                "model_chunk": 1,
            },
        )

    def test_nested_explicit_phase_keeps_framework_annotation(self):
        name = "memtrace|phase=pipeline/cooldown/backward|iter=4|mb=7|chunk=1"
        snapshot = {
            "segments": [],
            "device_traces": [[]],
            "external_annotations": [
                {"stage": "START", "name": "nccl:_reduce_scatter_base", "device": 0, "time_us": 10},
                {"stage": "START", "name": name, "device": 0, "time_us": 20},
                {"stage": "END", "name": name, "device": 0, "time_us": 30},
                {"stage": "END", "name": "nccl:_reduce_scatter_base", "device": 0, "time_us": 40},
            ],
        }
        resolved = PhaseResolver(snapshot).resolve(0, 25, [])
        self.assertEqual(resolved["primary_phase"], "pipeline/cooldown/backward")
        self.assertEqual(resolved["phase_metadata"]["microbatch"], 7)
        self.assertIn("nccl:_reduce_scatter_base", resolved["annotations"])

    def test_stack_fallback_is_coarse(self):
        snapshot = {"segments": [], "device_traces": [[]], "external_annotations": []}
        resolved = PhaseResolver(snapshot).resolve(
            0, 1, [{"filename": "schedules.py", "line": 2, "name": "backward_step"}]
        )
        self.assertEqual(resolved["primary_phase"], "pipeline/unknown/backward")
        self.assertEqual(resolved["phase_confidence"], "stack_rule")

    def test_nested_phase_inherits_iteration(self):
        fake_torch = SimpleNamespace(
            profiler=SimpleNamespace(record_function=lambda _name: nullcontext()),
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with memory_phase("train/iteration", iteration=9):
                with memory_phase("pipeline/steady/forward", microbatch=3) as name:
                    parsed = parse_phase_name(name)
        self.assertEqual(parsed["iteration"], 9)
        self.assertEqual(parsed["microbatch"], 3)


class MegatronPatchTests(unittest.TestCase):
    def test_supported_target_is_pinned(self):
        self.assertIn("c550cf6c41c31cd3ec72e05c25ea0c979f2b6631", SUPPORTED_COMMITS)
        self.assertEqual(len(SUPPORTED_VERSIONS), 1)

    def test_list_supported_does_not_require_target(self):
        with patch("builtins.print") as mocked_print:
            self.assertEqual(patch_main(["--list-supported"]), 0)
        output = mocked_print.call_args.args[0]
        self.assertEqual(output, format_supported_versions())
        self.assertIn("core_r0.13.0", output)
        self.assertIn("c550cf6c41c31cd3ec72e05c25ea0c979f2b6631", output)

    def test_every_edit_round_trips(self):
        for filename, edits in FILE_EDITS.items():
            for edit in edits:
                with self.subTest(filename=filename, edit=edit.name):
                    source = ("\n# repeated anchor boundary\n").join(
                        edit.before for _ in range(edit.count)
                    )
                    patched = _apply_edits(source, [edit])
                    self.assertNotEqual(patched, source)
                    self.assertEqual(_apply_edits(patched, [edit], reverse=True), source)


class AnalyzerTests(unittest.TestCase):
    def test_static_metrics(self):
        summary = summarize_snapshot(synthetic_snapshot())["total"]
        self.assertEqual(summary["reserved_bytes"], 30 * MIB)
        self.assertEqual(summary["active_block_bytes"], 18 * MIB)
        self.assertEqual(summary["stranded_free_bytes"], 12 * MIB)
        self.assertEqual(summary["releasable_cache_bytes"], 0)

    def test_exact_replay_and_failed_fit_phase(self):
        result = analyze_snapshot(synthetic_snapshot())
        self.assertTrue(result["replay"]["exact"], result["replay"]["error"])
        failed = [item for item in result["incidents"] if item["type"] == "failed-fit"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["compatible_free_bytes"], 12 * MIB)
        self.assertEqual(failed[0]["largest_free_block"], 8 * MIB)
        self.assertEqual(failed[0]["primary_phase"], "pipeline/steady/forward")
        self.assertEqual(failed[0]["phase_metadata"]["microbatch"], 2)
        self.assertTrue(result["dashboard"]["exact"])
        self.assertGreaterEqual(len(result["dashboard"]["moments"]), 1)
        self.assertTrue(result["dashboard"]["moments"][0]["segments"])

    def test_single_file_dashboard_output(self):
        result = analyze_snapshot(synthetic_snapshot(), top_k=10)
        with tempfile.TemporaryDirectory() as directory:
            write_analysis(result, Path(directory), top=10)
            html = (Path(directory) / "dashboard.html").read_text(encoding="utf-8")
            self.assertIn("GPU Memory Fragmentation Dashboard", html)
            self.assertIn('data-testid="moment-list"', html)
            self.assertNotIn("<img", html.lower())

    def test_truncated_trace_uses_reverse_top_k(self):
        result = analyze_snapshot(truncated_snapshot(), top_k=10)
        self.assertFalse(result["replay"]["exact"])
        self.assertEqual(result["replay"]["mode"], "reverse_approximate")
        self.assertTrue(result["replay"]["reverse_available"])
        dashboard = result["dashboard"]
        self.assertEqual(dashboard["history_mode"], "reverse_approximate")
        self.assertEqual(len(dashboard["moments"]), 3)
        self.assertGreater(len(dashboard["timeline"]), 1)
        self.assertEqual(dashboard["moments"][0]["event_index"], 1)
        for moment in dashboard["moments"]:
            self.assertLessEqual(
                moment["fragmentation_ratio_lower"], moment["fragmentation_ratio"]
            )
            self.assertTrue(moment["segments"])
        earliest = next(moment for moment in dashboard["moments"] if moment["event_index"] == 0)
        reconstructed = [
            block
            for segment in earliest["segments"]
            for block in segment["blocks"]
            if block.get("reconstructed")
        ]
        self.assertEqual(reconstructed[0]["requested_size"], 8 * MIB - 4)
        self.assertEqual(reconstructed[0]["size"], 8 * MIB)

    def test_top_k_filters_are_strict(self):
        result = analyze_snapshot(
            truncated_snapshot(),
            fragmentation_threshold=0.99,
            min_reserved_bytes=1024 * MIB,
        )
        self.assertEqual(result["dashboard"]["moments"], [])
        self.assertGreater(len(result["dashboard"]["timeline"]), 1)

    def test_existing_iter10_rank0_metrics_when_available(self):
        repository = Path(__file__).resolve().parents[1]
        snapshot = (
            repository
            / "data"
            / "memory snapshot"
            / "256gpus"
            / "iter10-rank0-snapshot.pickle"
        )
        if not snapshot.exists():
            self.skipTest("workspace golden snapshot is unavailable")
        from memory_fragmentation.analyzer import load_snapshot

        summary = summarize_snapshot(load_snapshot(snapshot))["total"]
        self.assertAlmostEqual(summary["releasable_cache_bytes"] / MIB, 24272.0, places=1)
        self.assertAlmostEqual(summary["stranded_free_bytes"] / MIB, 4601.5, delta=0.1)
        self.assertAlmostEqual(summary["internal_waste_bytes"] / MIB, 3.8, delta=0.1)


if __name__ == "__main__":
    unittest.main()
