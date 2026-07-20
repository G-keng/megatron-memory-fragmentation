# Megatron Memory Fragmentation

A lightweight research toolkit for tracing, replaying, and visualizing GPU
memory fragmentation in large-scale Megatron-LM training.

The project targets PyTorch's native CUDA caching allocator. It combines
explicit training-phase annotations with allocator snapshots, exact forward
replay when the trace is complete, and conservative reverse replay when a
snapshot contains only a retained trace window.

> Research prototype: validate conclusions against the PyTorch version and
> allocator configuration used by the experiment before treating them as
> production diagnostics.

[中文说明](memory_fragmentation/README.md)

## Features

- Structured phase markers for initialization, pipeline warmup/steady/cooldown,
  gradient synchronization, optimizer updates, and parameter synchronization.
- Rank-selective `torch.cuda.memory` history collection with lightweight
  per-iteration statistics on all ranks.
- Native-allocator replay with segment, active block, inactive block, and
  pending-free state.
- Fragmentation metrics that separate internal waste, releasable cache, and
  non-releasable free space in partially active segments.
- Conservative reverse replay for truncated ring-buffer traces, including an
  uncertainty interval for possible non-split tails.
- A self-contained interactive HTML dashboard with a fragmentation timeline,
  logical phases, Top-K moments, and per-segment block layouts.
- Stack-based phase inference for snapshots collected without explicit markers.

## Install

The offline analyzer has no third-party runtime dependency. PyTorch is imported
only by the training-side collector and phase markers.

```bash
git clone https://github.com/G-keng/megatron-memory-fragmentation.git
cd megatron-memory-fragmentation
python -m pip install -e .
```

Use the PyTorch build already required by the training environment.

## Capture

Start history recording before the first CUDA allocation on selected ranks:

```python
from memory_fragmentation import get_collector

collector = get_collector()
collector.start()
```

```bash
export MEMORY_TRACE_RANKS=0,255
export MEMORY_TRACE_END_ITER=20
export MEMORY_TRACE_MAX_ENTRIES=1000000
export MEMORY_TRACE_DIR=memory-traces
```

Add markers where the Megatron scheduler already knows the pipeline phase and
microbatch identity:

```python
from memory_fragmentation import memory_phase

with memory_phase(
    "pipeline/steady/forward",
    iteration=iteration,
    microbatch=microbatch_id,
    direction="forward",
    model_chunk=model_chunk_id,
    pp_stage=pp_rank,
):
    forward_step(...)
```

Call `collector.sample(iteration, ...)` at iteration boundaries. See
[`megatron_instrumentation_example.py`](memory_fragmentation/megatron_instrumentation_example.py)
for all recommended marker locations.

## Analyze

```bash
megatron-memfrag snapshot.pickle \
  --output-dir analysis \
  --top-k 10 \
  --fragmentation-threshold 0.10 \
  --min-reserved-mib 1024 \
  --min-stranded-mib 64
```

The thresholds above retain moments with at least 10% fragmentation, 1 GiB
reserved memory, and 64 MiB stranded free space. All filters default to zero.

Generated artifacts:

- `summary.json`: final allocator metrics.
- `replay.json`: replay mode, failure reason, and reverse-replay assumptions.
- `incidents.json` / `incidents.csv`: exact replay incidents and final-state
  pinned-segment candidates.
- `dashboard.html`: standalone interactive report; no image generation or web
  server is required.

## Replay Modes

| Mode | Meaning |
| --- | --- |
| `exact_forward` | Trace starts from a known allocator state and final replay validation succeeds. |
| `reverse_approximate` | Final snapshot is the anchor and retained events are undone in reverse. |
| `final_snapshot_only` | The allocator configuration or trace actions are unsupported for replay. |

In reverse mode, segment lifetime and allocation addresses come from recorded
events. Historical block sizes are rounded using snapshot settings or PyTorch
native defaults. The dashboard reports a conservative fragmentation upper bound
and a lower bound that accounts for possible non-split tails. Approximate replay
does not emit `failed-fit` incidents.

## Safety And Data Handling

Only load snapshots that you trust. PyTorch memory snapshots are pickle files,
and loading an untrusted pickle can execute arbitrary code.

Snapshots may contain source paths, function names, annotations, and training
metadata. Do not publish raw traces without reviewing them. This repository's
ignore rules deliberately exclude snapshots, logs, generated dashboards, and
the local research corpus.

## Limitations

- Exact replay currently targets `backend:native`, non-expandable segments, and
  default 512-byte request rounding.
- Reverse replay does not reconstruct the allocator's historical free-list
  topology exactly.
- Old traces cannot reliably recover warmup/steady/cooldown or microbatch IDs
  from function names alone.
- PyTorch snapshots do not expose CUDA driver free lists or physical HBM page
  placement.

The original snapshot can also be inspected with
[PyTorch MemoryViz](https://docs.pytorch.org/memory_viz).

## Development

```bash
python -m unittest discover -s tests -v
```

## License

MIT
