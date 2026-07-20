"""Command line entry point for offline snapshot analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from . import __version__
from .analyzer import analyze_snapshot, format_bytes, load_snapshot, write_analysis


def _print_summary(result: Dict[str, Any], top: int) -> None:
    total = result["summary"]["total"]
    print("Allocator snapshot summary")
    for key in (
        "reserved_bytes",
        "active_block_bytes",
        "active_requested_bytes",
        "reserved_minus_active_bytes",
        "internal_waste_bytes",
        "releasable_cache_bytes",
        "stranded_free_bytes",
        "pending_free_bytes",
    ):
        print(f"  {key:30s} {format_bytes(int(total.get(key, 0)))}")
    replay = result["replay"]
    print(f"Exact replay: {replay['exact']}")
    print(f"History mode: {replay.get('mode', 'unknown')}")
    if replay.get("error"):
        print(f"  reason: {replay['error']}")
    if replay.get("reverse_error"):
        print(f"  reverse reason: {replay['reverse_error']}")

    print(f"Top {min(top, len(result['incidents']))} incidents")
    for incident in result["incidents"][:top]:
        magnitude = (
            incident.get("extra_segment_bytes")
            or incident.get("pinned_free_bytes")
            or incident.get("stranded_delta_bytes")
            or 0
        )
        metadata = incident.get("phase_metadata", {})
        suffix = " ".join(f"{key}={value}" for key, value in metadata.items())
        print(
            f"  {incident['type']:18s} {format_bytes(int(magnitude)):>12s} "
            f"phase={incident.get('primary_phase')} {suffix}".rstrip()
        )
        pinner = incident.get("main_pinner")
        if incident.get("type") == "failed-fit" and isinstance(pinner, dict):
            lifetime_suffix = "" if pinner.get("lifetime_complete") else "+"
            print(
                f"    cause: pinner={pinner.get('allocation_address_hex')} "
                f"allocation={format_bytes(int(pinner.get('allocation_bytes', 0)))} "
                f"pinned={format_bytes(int(pinner.get('pinned_free_bytes', 0)))} "
                f"score={float(pinner.get('pinning_score', 0.0)):.3f} "
                f"lifetime={int(pinner.get('lifetime_us', 0))}{lifetime_suffix} us"
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="megatron-memfrag",
        description="Analyze PyTorch CUDA allocator fragmentation",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("snapshot", type=Path, help="trusted snapshot pickle")
    parser.add_argument("-o", "--output-dir", type=Path, help="write JSON artifacts here")
    parser.add_argument("--top", type=int, default=50, help="number of incidents to retain/print")
    parser.add_argument("--top-k", type=int, default=10, help="number of dashboard moments")
    parser.add_argument(
        "--fragmentation-threshold",
        type=float,
        default=0.0,
        help="minimum fragmentation ratio used when selecting dashboard moments",
    )
    parser.add_argument(
        "--min-fragment-mib",
        type=float,
        default=1.0,
        help="minimum fragment-created delta to include",
    )
    parser.add_argument(
        "--min-reserved-mib",
        type=float,
        default=0.0,
        help="ignore Top-K moments below this reserved-memory size",
    )
    parser.add_argument(
        "--min-stranded-mib",
        type=float,
        default=0.0,
        help="ignore Top-K moments below this stranded-free size",
    )
    args = parser.parse_args(argv)

    snapshot = load_snapshot(args.snapshot)
    result = analyze_snapshot(
        snapshot,
        min_fragment_bytes=int(args.min_fragment_mib * 1024 * 1024),
        top_k=args.top_k,
        fragmentation_threshold=args.fragmentation_threshold,
        min_reserved_bytes=int(args.min_reserved_mib * 1024 * 1024),
        min_stranded_bytes=int(args.min_stranded_mib * 1024 * 1024),
    )
    _print_summary(result, args.top)
    if args.output_dir:
        write_analysis(result, args.output_dir, top=args.top)
        print(f"Wrote analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
