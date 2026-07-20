"""Logical Megatron instrumentation examples; adapt names to the local fork.

This file is intentionally not imported by the tool.  It shows where the small
hooks belong without depending on a particular Megatron-LM revision.
"""

from memory_fragmentation import get_collector, memory_phase


def start_memory_trace():
    """Call at the training entry point before model construction."""
    collector = get_collector()
    collector.start()
    return collector


def initialize_distributed_and_checkpoint(init_distributed, load_checkpoint):
    with memory_phase("init/distributed"):
        init_distributed()
    with memory_phase("init/checkpoint"):
        load_checkpoint()


def initialize_model_and_optimizer(model_provider, optimizer_factory, args):
    with memory_phase("init/model_parameters", pp_stage=args.pipeline_model_parallel_rank):
        model = model_provider()
    with memory_phase("init/optimizer", pp_stage=args.pipeline_model_parallel_rank):
        optimizer = optimizer_factory(model)
    return model, optimizer


def pipeline_schedule_example(
    iteration,
    warmup_microbatches,
    steady_pairs,
    cooldown_microbatches,
    forward_step,
    backward_step,
    pp_stage,
):
    # Insert these contexts inside the schedule's existing loops.  Do not
    # recompute which microbatches are warmup/steady/cooldown in the tracer.
    for microbatch, model_chunk in warmup_microbatches:
        with memory_phase(
            "pipeline/warmup/forward",
            iteration=iteration,
            microbatch=microbatch,
            direction="forward",
            model_chunk=model_chunk,
            pp_stage=pp_stage,
        ):
            forward_step(microbatch, model_chunk)

    for forward_mb, backward_mb, model_chunk in steady_pairs:
        with memory_phase(
            "pipeline/steady/forward",
            iteration=iteration,
            microbatch=forward_mb,
            direction="forward",
            model_chunk=model_chunk,
            pp_stage=pp_stage,
        ):
            forward_step(forward_mb, model_chunk)
        with memory_phase(
            "pipeline/steady/backward",
            iteration=iteration,
            microbatch=backward_mb,
            direction="backward",
            model_chunk=model_chunk,
            pp_stage=pp_stage,
        ):
            backward_step(backward_mb, model_chunk)

    for microbatch, model_chunk in cooldown_microbatches:
        with memory_phase(
            "pipeline/cooldown/backward",
            iteration=iteration,
            microbatch=microbatch,
            direction="backward",
            model_chunk=model_chunk,
            pp_stage=pp_stage,
        ):
            backward_step(microbatch, model_chunk)


def finish_iteration(iteration, model, optimizer, args):
    with memory_phase("grad/finalize", iteration=iteration):
        model.finalize_model_grads()
    # In an overlap configuration put the start marker where communication is
    # launched and this finish marker where the handle is consumed.
    with memory_phase("grad/sync", iteration=iteration, direction="finish"):
        model.finish_grad_sync()
    with memory_phase("optimizer/prepare", iteration=iteration):
        optimizer.prepare_grads()
    with memory_phase("optimizer/state_update", iteration=iteration):
        optimizer.step()
    with memory_phase("param/sync", iteration=iteration, direction="finish"):
        optimizer.finish_param_sync()

    get_collector().sample(
        iteration,
        pp_stage=args.pipeline_model_parallel_rank,
        tp_rank=args.tensor_model_parallel_rank,
        ep_rank=args.expert_model_parallel_rank,
    )


def launch_overlapped_sync(iteration, model, optimizer):
    with memory_phase("grad/sync", iteration=iteration, direction="start"):
        model.start_grad_sync()
    with memory_phase("param/sync", iteration=iteration, direction="start"):
        optimizer.start_param_sync()
