"""Version-aware instrumentation patcher for a pinned Megatron-LM revision."""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SUPPORTED_COMMITS = {
    "c550cf6c41c31cd3ec72e05c25ea0c979f2b6631": "Megatron-LM core_r0.13.0 (2025-07-25)",
}
PATCH_ID = "memfrag-c550cf6c-v1"


class PatchError(RuntimeError):
    """Raised when a target does not match the supported source layout."""


@dataclass(frozen=True)
class Edit:
    name: str
    before: str
    after: str
    count: int = 1


def _edit(name: str, before: str, after: str, count: int = 1) -> Edit:
    return Edit(name, before, after, count)


TRAINING = "megatron/training/training.py"
SCHEDULES = "megatron/core/pipeline_parallel/schedules.py"
FINALIZE = "megatron/core/distributed/finalize_model_grads.py"


TRAINING_EDITS: Tuple[Edit, ...] = (
    _edit(
        "imports",
        "import torch\n\ntry:\n",
        f"import torch\n\nfrom memory_fragmentation import get_collector, memory_phase  # {PATCH_ID}\n\ntry:\n",
    ),
    _edit(
        "start collector and distributed initialization",
        """    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(
        extra_args_provider=extra_args_provider,
        args_defaults=args_defaults,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        store=store,
    )
""",
        """    # Start allocator history before Megatron creates its first CUDA tensor.
    get_collector().start()
    with memory_phase("init/distributed"):
        initialize_megatron(
            extra_args_provider=extra_args_provider,
            args_defaults=args_defaults,
            get_embedding_ranks=get_embedding_ranks,
            get_position_embedding_ranks=get_position_embedding_ranks,
            store=store,
        )
""",
    ),
    _edit(
        "model initialization",
        """    model = get_model(model_provider_func, model_type)
    unwrapped_model = unwrap_model(model)
""",
        """    with memory_phase("init/model_parameters"):
        model = get_model(model_provider_func, model_type)
        unwrapped_model = unwrap_model(model)
""",
    ),
    _edit(
        "optimizer initialization",
        """    optimizer = get_megatron_optimizer(
        config,
        model,
        no_wd_decay_cond,
        scale_lr_cond,
        lr_mult,
        use_gloo_process_groups=args.enable_gloo_process_groups,
    )
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
""",
        """    with memory_phase("init/optimizer"):
        optimizer = get_megatron_optimizer(
            config,
            model,
            no_wd_decay_cond,
            scale_lr_cond,
            lr_mult,
            use_gloo_process_groups=args.enable_gloo_process_groups,
        )
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
""",
    ),
    _edit(
        "checkpoint loading",
        """        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model,
            optimizer,
            opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2
            and getattr(args, "use_torch_fsdp2", False)
            and args.ckpt_format == "torch_dist",
        )
""",
        """        with memory_phase("init/checkpoint"):
            args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                model,
                optimizer,
                opt_param_scheduler,
                checkpointing_context=checkpointing_context,
                skip_load_to_model_and_opt=HAVE_FSDP2
                and getattr(args, "use_torch_fsdp2", False)
                and args.ckpt_format == "torch_dist",
            )
""",
    ),
    _edit(
        "optimizer update",
        """    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()
""",
        """    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    with memory_phase("optimizer/state_update", iteration=args.curr_iteration):
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()
""",
    ),
    _edit(
        "collector in train loop",
        """    energy_monitor = get_energy_monitor()
    one_logger = get_one_logger()
""",
        """    energy_monitor = get_energy_monitor()
    one_logger = get_one_logger()
    memory_trace_collector = get_collector()
""",
    ),
    _edit(
        "iteration phase and sampling",
        """        (
            loss_dict,
            skipped_iter,
            should_checkpoint,
            should_exit,
            exit_code,
            grad_norm,
            num_zeros_in_grad,
        ) = train_step(
            forward_step_func, train_data_iterator, model, optimizer, opt_param_scheduler, config
        )
        ft_integration.on_training_step_end()
""",
        """        with memory_phase(
            "train/iteration",
            iteration=iteration,
            pp_stage=mpu.get_pipeline_model_parallel_rank(),
        ):
            (
                loss_dict,
                skipped_iter,
                should_checkpoint,
                should_exit,
                exit_code,
                grad_norm,
                num_zeros_in_grad,
            ) = train_step(
                forward_step_func,
                train_data_iterator,
                model,
                optimizer,
                opt_param_scheduler,
                config,
            )
        memory_trace_collector.sample(iteration, num_microbatches=get_num_microbatches())
        ft_integration.on_training_step_end()
""",
    ),
)


SCHEDULE_EDITS: Tuple[Edit, ...] = (
    _edit(
        "imports and phase helper",
        """import torch
from torch.autograd.variable import Variable

from megatron.core import parallel_state
""",
        f"""import torch
from torch.autograd.variable import Variable

from memory_fragmentation import memory_phase  # {PATCH_ID}
from megatron.core import parallel_state
""",
    ),
    _edit(
        "phase helper",
        """def check_first_val_step(first_val_step, forward_only, cond):
""",
        """def _memory_traced_step(phase, microbatch, direction, model_chunk, step_func, *args, **kwargs):
    with memory_phase(
        phase,
        microbatch=microbatch,
        direction=direction,
        model_chunk=model_chunk,
        pp_stage=parallel_state.get_pipeline_model_parallel_rank(),
    ):
        return step_func(*args, **kwargs)


def check_first_val_step(first_val_step, forward_only, cond):
""",
    ),
    _edit(
        "no-pipeline forward loop",
        """            output_tensor, num_tokens = forward_step(
                forward_step_func,
""",
        """            output_tensor, num_tokens = _memory_traced_step(
                "pipeline/steady/forward", i, "forward", None, forward_step,
                forward_step_func,
""",
    ),
    _edit(
        "no-pipeline backward loop",
        """                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)
""",
        """                _memory_traced_step(
                    "pipeline/steady/backward", i, "backward", None, backward_step,
                    input_tensor, output_tensor, output_tensor_grad, model_type, config,
                )
""",
    ),
    _edit(
        "no-pipeline final forward",
        """    output_tensor, num_tokens = forward_step(
        forward_step_func,
""",
        """    output_tensor, num_tokens = _memory_traced_step(
        "pipeline/steady/forward", num_microbatches - 1, "forward", None, forward_step,
        forward_step_func,
""",
    ),
    _edit(
        "no-pipeline final backward",
        """        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    if config.finalize_model_grads_func is not None and not forward_only:
""",
        """        _memory_traced_step(
            "pipeline/steady/backward", num_microbatches - 1, "backward", None, backward_step,
            input_tensor, output_tensor, output_tensor_grad, model_type, config,
        )

    if config.finalize_model_grads_func is not None and not forward_only:
""",
    ),
    _edit(
        "interleaved initial parameter sync",
        """    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())
""",
        """    if config.param_sync_func is not None:
        for model_chunk_id in (0, 1):
            with memory_phase("param/sync", direction="start", model_chunk=model_chunk_id):
                config.param_sync_func[model_chunk_id](model[model_chunk_id].parameters())
""",
    ),
    _edit(
        "interleaved helper signatures",
        """    def forward_step_helper(
        virtual_microbatch_id, microbatch_id, checkpoint_activations_microbatch
    ):
""",
        """    def forward_step_helper(
        virtual_microbatch_id, microbatch_id, checkpoint_activations_microbatch, phase
    ):
""",
    ),
    _edit(
        "interleaved parameter sync",
        """                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func[param_sync_chunk_id](
                        model[param_sync_chunk_id].parameters()
                    )
""",
        """                if 1 < param_sync_chunk_id < num_model_chunks:
                    with memory_phase(
                        "param/sync", direction="start", model_chunk=param_sync_chunk_id
                    ):
                        config.param_sync_func[param_sync_chunk_id](
                            model[param_sync_chunk_id].parameters()
                        )
""",
    ),
    _edit(
        "interleaved forward",
        """        offset = num_released_microbatches(virtual_microbatch_id, model_chunk_id)
        input_tensor = input_tensors[model_chunk_id][microbatch_id - offset]

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step,
                forward_only,
                is_first_microbatch_for_model_chunk(virtual_microbatch_id),
            ),
            current_microbatch=microbatch_id,
            vp_stage=model_chunk_id,
        )
""",
        """        offset = num_released_microbatches(virtual_microbatch_id, model_chunk_id)
        input_tensor = input_tensors[model_chunk_id][microbatch_id - offset]

        output_tensor, num_tokens = _memory_traced_step(
            phase, microbatch_id, "forward", model_chunk_id, forward_step,
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step,
                forward_only,
                is_first_microbatch_for_model_chunk(virtual_microbatch_id),
            ),
            current_microbatch=microbatch_id,
            vp_stage=model_chunk_id,
        )
""",
    ),
    _edit(
        "interleaved backward helper",
        """    def backward_step_helper(virtual_microbatch_id):
        \"\"\"Helper method to run backward step with model split into chunks\"\"\"
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=False)
""",
        """    def backward_step_helper(virtual_microbatch_id, phase):
        \"\"\"Helper method to run backward step with model split into chunks\"\"\"
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=False)
        microbatch_id = microbatch_id_table[virtual_microbatch_id % total_num_microbatches]
""",
    ),
    _edit(
        "interleaved backward",
        """        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )
""",
        """        input_tensor_grad = _memory_traced_step(
            phase, microbatch_id, "backward", model_chunk_id, backward_step,
            input_tensor, output_tensor, output_tensor_grad, model_type, config,
        )
""",
    ),
    _edit(
        "interleaved gradient sync",
        """                enable_grad_sync()
                config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
""",
        """                enable_grad_sync()
                with memory_phase(
                    "grad/sync", direction="start", model_chunk=grad_sync_chunk_id
                ):
                    config.grad_sync_func[grad_sync_chunk_id](
                        model[grad_sync_chunk_id].parameters()
                    )
                synchronized_model_chunks.add(grad_sync_chunk_id)
""",
    ),
    _edit(
        "interleaved warmup call",
        """        output_tensor = forward_step_helper(k, microbatch_id, checkpoint_activations_microbatch)
""",
        """        output_tensor = forward_step_helper(
            k, microbatch_id, checkpoint_activations_microbatch, "pipeline/warmup/forward"
        )
""",
    ),
    _edit(
        "interleaved steady forward calls",
        """            output_tensor = forward_step_helper(
                forward_k, microbatch_id, checkpoint_activations_microbatch
            )
""",
        """            output_tensor = forward_step_helper(
                forward_k,
                microbatch_id,
                checkpoint_activations_microbatch,
                "pipeline/steady/forward",
            )
""",
        2,
    ),
    _edit(
        "interleaved steady backward calls",
        """            input_tensor_grad = backward_step_helper(backward_k)
""",
        """            input_tensor_grad = backward_step_helper(
                backward_k, "pipeline/steady/backward"
            )
""",
        2,
    ),
    _edit(
        "interleaved cooldown backward",
        """            input_tensor_grad = backward_step_helper(k)
""",
        """            input_tensor_grad = backward_step_helper(
                k, "pipeline/cooldown/backward"
            )
""",
    ),
    _edit(
        "interleaved remaining gradient sync",
        """                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)
""",
        """                if model_chunk_id not in synchronized_model_chunks:
                    with memory_phase(
                        "grad/sync", direction="start", model_chunk=model_chunk_id
                    ):
                        config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)
""",
    ),
    _edit(
        "non-interleaved warmup forward",
        """        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
""",
        """        output_tensor, num_tokens = _memory_traced_step(
            "pipeline/warmup/forward", i, "forward", None, forward_step,
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
""",
    ),
    _edit(
        "non-interleaved steady forward",
        """        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
""",
        """        output_tensor, num_tokens = _memory_traced_step(
            "pipeline/steady/forward",
            i + num_warmup_microbatches,
            "forward",
            None,
            forward_step,
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
""",
    ),
    _edit(
        "non-interleaved steady backward",
        """            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )
""",
        """            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor_grad = _memory_traced_step(
                "pipeline/steady/backward", i, "backward", None, backward_step,
                input_tensor, output_tensor, output_tensor_grad, model_type, config,
            )
""",
    ),
    _edit(
        "non-interleaved cooldown backward",
        """            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(
                send_tensor_shapes, config, parallel_state.is_pipeline_last_stage()
            )

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )
""",
        """            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(
                send_tensor_shapes, config, parallel_state.is_pipeline_last_stage()
            )

            input_tensor_grad = _memory_traced_step(
                "pipeline/cooldown/backward",
                i + num_microbatches_remaining,
                "backward",
                None,
                backward_step,
                input_tensor,
                output_tensor,
                output_tensor_grad,
                model_type,
                config,
            )
""",
    ),
    _edit(
        "non-interleaved remaining gradient sync",
        """            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())
""",
        """            if config.grad_sync_func is not None:
                with memory_phase("grad/sync", direction="start", model_chunk=0):
                    config.grad_sync_func(model.parameters())
""",
    ),
    _edit(
        "finalize gradients",
        """        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )
""",
        """        with memory_phase("grad/finalize"):
            config.finalize_model_grads_func(
                [model], total_num_tokens if config.calculate_per_token_loss else None
            )
""",
        2,
    ),
    _edit(
        "interleaved finalize gradients",
        """        config.finalize_model_grads_func(
            model, total_num_tokens if config.calculate_per_token_loss else None
        )
""",
        """        with memory_phase("grad/finalize"):
            config.finalize_model_grads_func(
                model, total_num_tokens if config.calculate_per_token_loss else None
            )
""",
    ),
)


FINALIZE_EDITS: Tuple[Edit, ...] = (
    _edit(
        "imports",
        """import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
""",
        f"""import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from memory_fragmentation import memory_phase  # {PATCH_ID}
""",
    ),
    _edit(
        "finish gradient sync",
        """    for model_chunk in model:
        model_chunk.finish_grad_sync()
""",
        """    for model_chunk_id, model_chunk in enumerate(model):
        with memory_phase("grad/sync", direction="finish", model_chunk=model_chunk_id):
            model_chunk.finish_grad_sync()
""",
    ),
)


FILE_EDITS: Dict[str, Tuple[Edit, ...]] = {
    TRAINING: TRAINING_EDITS,
    SCHEDULES: SCHEDULE_EDITS,
    FINALIZE: FINALIZE_EDITS,
}


def _git(root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode:
        raise PatchError(process.stderr.strip() or "git command failed")
    return process.stdout.strip()


def target_commit(root: Path) -> str:
    return _git(root, "rev-parse", "HEAD")


def verify_target(root: Path, force_version: bool = False) -> str:
    missing = [name for name in FILE_EDITS if not (root / name).is_file()]
    if missing:
        raise PatchError("not a compatible Megatron-LM tree; missing: " + ", ".join(missing))
    commit = target_commit(root)
    if commit not in SUPPORTED_COMMITS and not force_version:
        supported = ", ".join(value[:12] for value in SUPPORTED_COMMITS)
        raise PatchError(
            f"unsupported Megatron-LM commit {commit}; supported: {supported}. "
            "Use --force-version only after reviewing --dry-run."
        )
    return commit


def patch_state(root: Path) -> str:
    marked = []
    for relative in FILE_EDITS:
        text = (root / relative).read_text(encoding="utf-8")
        marked.append(PATCH_ID in text)
    if all(marked):
        return "applied"
    if not any(marked):
        return "not_applied"
    return "partial"


def _apply_edits(text: str, edits: Sequence[Edit], reverse: bool = False) -> str:
    ordered: Iterable[Edit] = reversed(edits) if reverse else edits
    for edit in ordered:
        source, replacement = (edit.after, edit.before) if reverse else (edit.before, edit.after)
        count = text.count(source)
        if count != edit.count:
            raise PatchError(
                f"anchor {edit.name!r} matched {count} times; expected {edit.count}"
            )
        text = text.replace(source, replacement, edit.count)
    return text


def render(root: Path, reverse: bool = False) -> Dict[str, Tuple[str, str]]:
    state = patch_state(root)
    expected = "applied" if reverse else "not_applied"
    if state == ("not_applied" if reverse else "applied"):
        return {}
    if state != expected:
        raise PatchError(f"target is in {state!r} state; refusing a partial transformation")
    rendered: Dict[str, Tuple[str, str]] = {}
    for relative, edits in FILE_EDITS.items():
        path = root / relative
        before = path.read_text(encoding="utf-8")
        after = _apply_edits(before, edits, reverse=reverse)
        rendered[relative] = (before, after)
    return rendered


def unified_diff(rendered: Dict[str, Tuple[str, str]]) -> str:
    chunks: List[str] = []
    for relative, (before, after) in rendered.items():
        chunks.extend(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    return "".join(chunks)


def write_rendered(root: Path, rendered: Dict[str, Tuple[str, str]]) -> None:
    for relative, (_, after) in rendered.items():
        path = root / relative
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            handle.write(after)
            temporary = Path(handle.name)
        temporary.chmod(path.stat().st_mode)
        temporary.replace(path)


def _ensure_clean_for_apply(root: Path) -> None:
    status = _git(root, "status", "--porcelain", "--", *FILE_EDITS.keys())
    if status:
        raise PatchError("target files already have uncommitted changes; review or stash them first")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Patch Megatron-LM c550cf6c with allocator collection and phase markers."
    )
    parser.add_argument("megatron_root", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="validate and report patch state (default)")
    action.add_argument("--dry-run", action="store_true", help="print the patch without writing files")
    action.add_argument("--apply", action="store_true", help="apply the instrumentation")
    action.add_argument("--revert", action="store_true", help="remove this instrumentation")
    parser.add_argument("--output", type=Path, help="write dry-run diff to this file")
    parser.add_argument("--force-version", action="store_true", help="allow a different commit if all anchors match")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.megatron_root.resolve()
    try:
        commit = verify_target(root, force_version=args.force_version)
        state = patch_state(root)
        if not (args.dry_run or args.apply or args.revert):
            print(json.dumps({"commit": commit, "description": SUPPORTED_COMMITS.get(commit), "state": state}))
            return 0 if state != "partial" else 2

        reverse = bool(args.revert)
        if args.apply and state == "not_applied":
            _ensure_clean_for_apply(root)
        rendered = render(root, reverse=reverse)
        diff = unified_diff(rendered)
        if args.dry_run:
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(diff, encoding="utf-8")
                print(args.output)
            else:
                sys.stdout.write(diff)
            return 0

        write_rendered(root, rendered)
        verb = "reverted" if reverse else "applied"
        print(f"{verb} {PATCH_ID}: {len(rendered)} files changed")
        return 0
    except (OSError, PatchError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
