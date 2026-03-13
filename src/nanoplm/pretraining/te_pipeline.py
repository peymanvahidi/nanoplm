"""Transformer Engine pretraining pipeline."""



import gc
import math
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
import wandb
from dion import Muon as DionMuon, NorMuon as DionNorMuon
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from nanoplm.data.manifest import read_manifest, validate_manifest_for_pipeline
from nanoplm.pretraining.collator import (
    DataCollatorWithFlattening,
    ProtDataCollatorForLM,
    build_power_of_two_buckets,
)
from nanoplm.pretraining.dataset import ShardedDataset, TokenPackingDataset
from nanoplm.pretraining.models.modern_bert.modeling import MHCLiteBlock
from nanoplm.pretraining.models.modern_bert.modelling_te import FP8_RECIPE
from nanoplm.pretraining.models.modern_bert.pure_model import TEProtModernBertMLM
from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig
from nanoplm.pretraining.utils import prepare_run_and_steps
from nanoplm.pretraining.pure_pipeline import (
    H100_PEAK_TFLOPS,
    N_PREFETCH_LAYERS_FSDP2,
    _create_optimizer,
    _create_scheduler,
    _ddp_sync_context,
    _dist_barrier,
    _estimate_model_flops_per_token,
    _evaluate,
    _first_nonfinite_grad,
    _first_nonfinite_param,
    _format_vram_for_log,
    _get_distributed_mode,
    _has_nonfinite_params,
    _load_checkpoint,
    _move_batch_to_device,
    _num_update_steps_per_epoch,
    _rebuild_scheduler_for_resume,
    _resolve_world_size,
    _save_checkpoint,
    _set_seed,
    _sync_train_loader_len,
)
from nanoplm.utils.common import create_dirs, get_device
from nanoplm.utils.logger import logger
from nanoplm.utils.wandb_artifacts import upload_run_source_snapshot
from nanoplm.pretraining.utils import get_num_workers


def _compile_te_mhc_blocks(
    model: TEProtModernBertMLM,
    *,
    compile_dynamic: bool,
    compile_mode: Optional[str],
) -> None:
    compiled_blocks = 0
    for layer in model.model.layers:
        if not isinstance(layer, MHCLiteBlock):
            continue
        if layer.triton_fused:
            if compile_mode is None:
                layer._compiled_mhc_pre_map_triton = torch.compile(
                    layer._mhc_pre_map_triton,
                    dynamic=compile_dynamic,
                    fullgraph=True,
                )
                layer._compiled_mhc_post_res_triton = torch.compile(
                    layer._mhc_post_res_triton,
                    dynamic=compile_dynamic,
                    fullgraph=True,
                )
            else:
                layer._compiled_mhc_pre_map_triton = torch.compile(
                    layer._mhc_pre_map_triton,
                    dynamic=compile_dynamic,
                    fullgraph=True,
                    mode=compile_mode,
                )
                layer._compiled_mhc_post_res_triton = torch.compile(
                    layer._mhc_post_res_triton,
                    dynamic=compile_dynamic,
                    fullgraph=True,
                    mode=compile_mode,
                )
        else:
            if compile_mode is None:
                layer._compiled_mhc_pre_map_pytorch = torch.compile(
                    layer._mhc_pre_map_pytorch,
                    dynamic=compile_dynamic,
                    fullgraph=True,
                )
                layer._compiled_mhc_post_res_pytorch = torch.compile(
                    layer._mhc_post_res_pytorch,
                    dynamic=compile_dynamic,
                    fullgraph=True,
                )
            else:
                layer._compiled_mhc_pre_map_pytorch = torch.compile(
                    layer._mhc_pre_map_pytorch,
                    dynamic=compile_dynamic,
                    fullgraph=True,
                    mode=compile_mode,
                )
                layer._compiled_mhc_post_res_pytorch = torch.compile(
                    layer._mhc_post_res_pytorch,
                    dynamic=compile_dynamic,
                    fullgraph=True,
                    mode=compile_mode,
                )
        compiled_blocks += 1
    if compiled_blocks:
        logger.info(
            "Locally compiled mHC-lite TE glue for %d blocks (dynamic=%s, mode=%s)",
            compiled_blocks,
            compile_dynamic,
            compile_mode,
        )


def _make_te_profiler(pretrain_config: PretrainingConfig, output_dir: str, is_main: bool):
    """Build profiler context and step callback for the TE training loop.

    When profiler_enabled and is_main:
    - If running under nsys (NSYS_PROFILING_SESSION_ID): uses CUDA Profiler API so
      nsys can capture; start/stop at profiler_start_step / profiler_end_step.
    - Otherwise: uses PyTorch profiler with a schedule and exports a Chrome trace
      to output_dir/profiler_traces/chrome_trace.json (view in chrome://tracing).

    Returns:
        (context_manager, step_callback): use as ``with context: ...; step_callback(global_step)``
        after each optimizer step.
    """
    if not getattr(pretrain_config, "profiler_enabled", False) or not is_main:
        return nullcontext(), lambda _: None

    start_step = getattr(pretrain_config, "profiler_start_step", 10)
    end_step = getattr(pretrain_config, "profiler_end_step", 15)
    running_under_nsys = "NSYS_PROFILING_SESSION_ID" in os.environ

    if running_under_nsys:
        logger.info(
            "Profiling enabled (Nsight): CUDA Profiler API will start at step %s, stop at %s. "
            "Run with: nsys profile -o <trace> --trace=cuda,nvtx,osrt,cudnn,cublas "
            "--capture-range=cudaProfilerApi --capture-range-end=stop ...",
            start_step,
            end_step,
        )

        class _NsightController:
            def __init__(self):
                self.started = False
                self.finished = False

            def step(self, gs: int) -> None:
                if self.finished:
                    return
                if gs == start_step and not self.started:
                    try:
                        torch.cuda.cudart().cudaProfilerStart()  # type: ignore[attr-defined]
                        self.started = True
                        logger.info("Nsight profiling started at step %s", gs)
                    except Exception as e:
                        logger.error("Failed to start CUDA profiler: %s", e)
                elif gs == end_step and self.started:
                    try:
                        torch.cuda.cudart().cudaProfilerStop()  # type: ignore[attr-defined]
                        self.started = False
                        self.finished = True
                        logger.info("Nsight profiling stopped at step %s", gs)
                    except Exception as e:
                        logger.error("Failed to stop CUDA profiler: %s", e)

        ctrl = _NsightController()
        return nullcontext(), ctrl.step
    else:
        trace_dir = Path(output_dir) / "profiler_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "chrome_trace.json"
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        wait = max(0, start_step)
        warmup = 1
        active = max(1, end_step - start_step - 1)

        def on_trace_ready(prof: torch.profiler.profile) -> None:
            prof.export_chrome_trace(str(trace_path))
            logger.info("Exported PyTorch profiler trace to %s", trace_path)

        prof = torch.profiler.profile(
            activities=activities,
            # Single profiling window; avoid repeating the cycle throughout training.
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=on_trace_ready,
            record_shapes=True,
        )
        logger.info(
            "Profiling enabled (PyTorch): trace will be written to %s (steps %s..%s). "
            "Open in chrome://tracing",
            trace_path,
            start_step,
            end_step,
        )
        return prof, lambda _: prof.step()


def run_te_pretraining(
    model: TEProtModernBertMLM,
    pretrain_config: PretrainingConfig,
    resume_config: Optional[ResumeConfig] = None,
) -> None:
    _set_seed(pretrain_config.seed)
    tokenizer = model.tokenizer
    device = torch.device(get_device())

    dataset_dir = Path(pretrain_config.dataset_dir)

    manifest = read_manifest(dataset_dir)
    validate_manifest_for_pipeline(manifest=manifest, expected_mode="pretrain")
    if manifest.max_seq_len <= 0:
        raise ValueError(f"Invalid manifest max_seq_len: {manifest.max_seq_len}")
    # Capture config/manifest for checkpoint serialization.
    _model_config = getattr(model, "model_config", None)
    _manifest = manifest
    model_max_pos = int(getattr(model.config, "max_position_embeddings", 0))
    if model_max_pos > 0 and manifest.max_seq_len > model_max_pos:
        raise ValueError(
            f"Dataset max_seq_len ({manifest.max_seq_len}) exceeds model "
            f"max_position_embeddings ({model_max_pos})."
        )

    train_shard_dir = dataset_dir / manifest.train_dir
    val_shard_dir = dataset_dir / manifest.val_dir
    train_sequences = manifest.train_sequences

    logger.info(f"Loaded config from manifest: {dataset_dir}")
    logger.info(f"  train_shards: {train_shard_dir}")
    logger.info(f"  val_shards: {val_shard_dir}")
    logger.info(f"  max_length: {manifest.max_seq_len}")
    logger.info(f"  train_sequences: {manifest.train_sequences}")
    logger.info(f"  val_sequences: {manifest.val_sequences}")

    if not train_shard_dir.exists():
        raise FileNotFoundError(f"Train shard directory not found: {train_shard_dir}")
    if not val_shard_dir.exists():
        raise FileNotFoundError(f"Validation shard directory not found: {val_shard_dir}")

    logger.info("Using ShardedDataset for pre-tokenized binary shards")
    try:
        train_ds = ShardedDataset(data_dir=str(train_shard_dir))
        val_ds = ShardedDataset(data_dir=str(val_shard_dir))
    except FileNotFoundError as e:
        logger.error(
            "Binary shards not found! You need to create them first.\n"
            "Run: nanoplm data from-yaml with pipeline_mode: 'pretrain'\n"
            f"Error: {e}"
        )
        raise

    use_packing = bool(pretrain_config.use_packing)
    use_static_inp_size = bool(pretrain_config.use_static_inp_size and use_packing)
    pack_tokens_per_micro = None
    min_tokens_per_seq = None
    seq_count_buckets = None
    max_seqlen_buckets = None
    if use_static_inp_size:
        pack_tokens_per_micro = pretrain_config.micro_batch_size * manifest.max_seq_len
        min_tokens_per_seq = max(
            1,
            int(manifest.min_seq_len) + int(tokenizer.num_special_tokens_to_add(pair=False)),
        )
        max_sequences_per_batch = max(1, pack_tokens_per_micro // min_tokens_per_seq + 1)
        seq_count_buckets = build_power_of_two_buckets(max_sequences_per_batch, min_power_of_two=32)
        max_seqlen_buckets = build_power_of_two_buckets(int(manifest.max_seq_len), min_power_of_two=32)

    create_dirs(pretrain_config.ckp_dir)

    effective_world_size = _resolve_world_size(pretrain_config)

    inferred_grad_accum_steps = pretrain_config.inferred_grad_accum_steps
    global_batch_size_samples = pretrain_config.global_batch_size_samples
    achieved_global_batch_tokens = pretrain_config.achieved_global_batch_tokens

    if (
        inferred_grad_accum_steps is None
        or global_batch_size_samples is None
        or achieved_global_batch_tokens is None
    ):
        raise ValueError(
            "Batch setup is missing on PretrainingConfig. "
            "Run pretraining through nanoplm CLI so inferred batch fields are populated."
        )

    world_tokens_per_micro_step = achieved_global_batch_tokens // max(
        1, inferred_grad_accum_steps
    )
    logger.info(
        "Batch setup: "
        f"target_global_batch_size={pretrain_config.global_batch_size:,} tokens, "
        f"micro_step_tokens={world_tokens_per_micro_step:,}, "
        f"grad_accum_steps={inferred_grad_accum_steps}, "
        f"effective_global_batch_size={achieved_global_batch_tokens:,} tokens"
    )

    (
        _run_name,
        wandb_run_name,
        output_dir,
        num_epochs,
        logging_steps,
        eval_steps,
        save_steps,
        _resume_step,
    ) = prepare_run_and_steps(
        pretrain_config=pretrain_config,
        resume_config=resume_config,
        train_samples=train_sequences,
        global_batch_size_samples=global_batch_size_samples,
    )

    num_workers = get_num_workers(pretrain_config.num_workers, effective_world_size)
    pin_memory = device.type == "cuda"
    # Disable persistent workers when packing: TokenPackingDataset + persistent_workers
    # can cause hangs near epoch boundaries (set_epoch doesn't reach workers).
    persistent_workers = num_workers > 0 and not use_packing
    prefetch_factor = pretrain_config.prefetch_factor if num_workers > 0 else None

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = bool(pretrain_config.multi_gpu)
    distributed_mode = _get_distributed_mode(pretrain_config, distributed=distributed)
    ddp_bucket_cap_mb = int(getattr(pretrain_config, "ddp_bucket_cap_mb", 25))

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        if not dist.is_initialized():
            if backend == "nccl":
                dist.init_process_group(backend=backend, device_id=device)
            else:
                dist.init_process_group(backend=backend)

    model.to(device)
    base_model = model
    compile_dynamic = not bool(use_packing and use_static_inp_size)
    compile_mode = (
        "max-autotune-no-cudagraphs"
        if getattr(pretrain_config, "use_compile_max_autotune", False)
        else None
    )
    if model.model.config.use_mhc_lite:
        _compile_te_mhc_blocks(
            model,
            compile_dynamic=compile_dynamic,
            compile_mode=compile_mode,
        )
    if model.model.resid_lambdas is not None and model.model.x0_lambdas is not None:
        model.model._blend_resid_x0 = torch.compile(
            model.model._blend_resid_x0,
            dynamic=True,
            fullgraph=True,
            mode='max-autotune-no-cudagraphs',
        )

    # -- Precision detection (needed before FSDP for MixedPrecisionPolicy) --
    use_bf16 = (
        pretrain_config.bf16
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = (
        pretrain_config.bf16
        and ((device.type == "cuda" and not torch.cuda.is_bf16_supported()) or device.type == "mps")
    )

    fsdp_mesh = None
    optimizer_dist = None
    if distributed and distributed_mode == "fsdp":
        # Apply FSDP2 per transformer layer, then at the root.
        fsdp_kwargs: dict = {}
        if use_bf16:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )
        fsdp_mesh = init_device_mesh("cuda", (effective_world_size,))
        for layer in base_model.model.layers:
            fully_shard(layer, mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_kwargs)
        fully_shard(base_model, mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_kwargs)
        optimizer_dist = fsdp_mesh

        # FSDP2 Explicit Prefetching
        if N_PREFETCH_LAYERS_FSDP2 > 1:
            layers = base_model.model.layers
            for i, layer in enumerate(layers):
                if i + 1 < len(layers):
                    layer.set_modules_to_forward_prefetch(
                        layers[i + 1 : i + 1 + N_PREFETCH_LAYERS_FSDP2]
                    )
                if i - 1 >= 0:
                    layer.set_modules_to_backward_prefetch(
                        list(reversed(layers[max(0, i - N_PREFETCH_LAYERS_FSDP2) : i]))
                    )

        eval_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        eval_sampler = SequentialSampler(val_ds)

    is_main = (not distributed) or dist.get_rank() == 0

    # Build train sampler.
    train_sampler = None
    if use_packing:
        tokens_per_micro = pretrain_config.micro_batch_size * manifest.max_seq_len
        # For packing, we use a sampler on the *underlying* dataset to handle shuffling and distributed sharding.
        if distributed:
            _inner_sampler = DistributedSampler(train_ds, shuffle=True, seed=pretrain_config.seed)
        else:
            _inner_sampler = RandomSampler(train_ds)
            
        # Wrap dataset for packing
        train_ds = TokenPackingDataset(
            train_ds,
            max_tokens_per_batch=tokens_per_micro,
            drop_last=True,
            split_samples=False,
            sampler=_inner_sampler,
        )
        
        inner_collator = ProtDataCollatorForLM(
            tokenizer=tokenizer,
            mlm_probability=pretrain_config.mlm_probability,
            mask_token_probability=pretrain_config.mask_replace_prob,
            random_token_probability=pretrain_config.random_token_prob,
            keep_probability=pretrain_config.keep_probability,
        )
        if use_static_inp_size:
            assert tokens_per_micro == pack_tokens_per_micro
            assert min_tokens_per_seq is not None
            assert seq_count_buckets is not None
            assert max_seqlen_buckets is not None
            collator = DataCollatorWithFlattening(
                collator=inner_collator,
                fixed_tokens_per_batch=tokens_per_micro,
                seq_count_buckets=seq_count_buckets,
                max_seqlen_buckets=max_seqlen_buckets,
            )
            logger.info(
                "Sequence packing ENABLED (static input size + bucketed metadata), "
                f"target={tokens_per_micro:,} tokens, "
                f"min_tokens_per_seq={min_tokens_per_seq}, "
                f"seq_count_buckets={seq_count_buckets}, "
                f"max_seqlen_buckets={max_seqlen_buckets}"
            )
        else:
            collator = DataCollatorWithFlattening(
                collator=inner_collator,
                pad_to_multiple_of=16,
            )
            logger.info(
                "Sequence packing ENABLED (dynamic metadata), "
                f"target={tokens_per_micro:,} tokens"
            )
        train_sampler = None
        
    elif distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=pretrain_config.seed)
        collator = ProtDataCollatorForLM(
            tokenizer=tokenizer,
            mlm_probability=pretrain_config.mlm_probability,
            mask_token_probability=pretrain_config.mask_replace_prob,
            random_token_probability=pretrain_config.random_token_prob,
            keep_probability=pretrain_config.keep_probability,
        )
    else:
        train_sampler = RandomSampler(train_ds)
        collator = ProtDataCollatorForLM(
            tokenizer=tokenizer,
            mlm_probability=pretrain_config.mlm_probability,
            mask_token_probability=pretrain_config.mask_replace_prob,
            random_token_probability=pretrain_config.random_token_prob,
            keep_probability=pretrain_config.keep_probability,
        )

    # Keep orig_model reference for checkpointing/eval.
    # Do not torch.compile TE modules: TE kernels and FP8 metadata cause graph breaks/recompiles.
    orig_model = base_model
    if distributed and distributed_mode == "ddp":
        ddp_kwargs = dict(
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=ddp_bucket_cap_mb,
        )
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
            ddp_kwargs["output_device"] = local_rank
        model = DistributedDataParallel(base_model, **ddp_kwargs)
        optimizer_dist = dist.group.WORLD

    eval_collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=pretrain_config.mlm_probability,
        mask_token_probability=pretrain_config.mask_replace_prob,
        random_token_probability=pretrain_config.random_token_prob,
        keep_probability=pretrain_config.keep_probability,
    )
    dl_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    if use_packing:
        train_loader = DataLoader(
            train_ds,
            batch_size=None,
            collate_fn=collator,
            **dl_kwargs,
        )
        # If using packing with IterableDataset, len() is an estimate.
        # Trainer expects exact length for steps per epoch.
        # We can relax this strictness or correct the estimate.
        # For now, let's just create the loader.
    else:
        train_loader = DataLoader(
            train_ds,
            sampler=train_sampler,
            batch_size=pretrain_config.micro_batch_size,
            collate_fn=collator,
            drop_last=False,
            **dl_kwargs,
        )
    eval_loader = DataLoader(
        val_ds,
        sampler=eval_sampler,
        batch_size=pretrain_config.micro_batch_size,
        collate_fn=eval_collator,
        drop_last=False,
        **dl_kwargs,
    )

    optimizer = _create_optimizer(orig_model, pretrain_config, distributed_mesh=optimizer_dist)

    synced_train_loader_len = _sync_train_loader_len(
        train_loader_len=len(train_loader),
        distributed=distributed,
        device=device,
    )
    steps_per_epoch = _num_update_steps_per_epoch(
        train_loader_len=synced_train_loader_len,
        grad_accum=inferred_grad_accum_steps,
    )
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = min(pretrain_config.warmup_steps, total_steps)
    scheduler = _create_scheduler(
        optimizer, warmup_steps, total_steps,
        pretrain_config.adam_learning_rate, pretrain_config.lr_decay_to_fraction,
        pretrain_config.lr_schedule,
    )

    start_step = 0
    start_epoch = 0
    if resume_config and resume_config.is_resume:
        logger.info(f"Resuming from checkpoint: {resume_config.checkpoint_dir}")
        start_step, start_epoch = _load_checkpoint(
            model=orig_model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dir=resume_config.checkpoint_dir,
            device=device,
            distributed=distributed,
            load_scheduler=False,
            distributed_mode=distributed_mode,
        )
        logger.info(f"Resumed at global_step={start_step}, epoch={start_epoch}")

        # Rebuild the LR scheduler from the checkpoint's saved schedule
        # params (with any explicit overrides from resume config).
        scheduler = _rebuild_scheduler_for_resume(
            optimizer=optimizer,
            resume_config=resume_config,
            pretrain_config=pretrain_config,
            checkpoint_dir=resume_config.checkpoint_dir,
            start_step=start_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            is_main=is_main,
        )

    resume_micro_step = 0
    resume_epoch = start_epoch
    if resume_config and resume_config.is_resume:
        steps_done = max(0, start_step - start_epoch * steps_per_epoch)
        resume_micro_step = steps_done * inferred_grad_accum_steps
        if resume_micro_step >= synced_train_loader_len:
            resume_micro_step = 0
            start_epoch = min(start_epoch + 1, num_epochs)
            resume_epoch = start_epoch
        if resume_micro_step > 0:
            logger.info(
                f"Skipping {resume_micro_step} micro-steps in resumed epoch {resume_epoch}"
            )

    wandb_enabled = False
    if is_main:
        try:
            wandb.init(
                project=pretrain_config.project_name,
                name=wandb_run_name,
                config={
                    "pretrain": pretrain_config.__dict__,
                    "total_steps": total_steps,
                    "warmup_steps": warmup_steps,
                },
            )
            wandb_enabled = wandb.run is not None
            if wandb_enabled:
                wandb.define_metric("train/global_step")
                wandb.define_metric("time_elapsed_sec", step_metric="train/global_step", step_sync=True)
                wandb.define_metric("train/time_elapsed_sec", step_metric="train/global_step", step_sync=True)
                wandb.define_metric("*", step_metric="train/global_step", step_sync=True)
                upload_run_source_snapshot()
        except Exception as exc:
            logger.warning(f"W&B init failed, continuing without logging. Error: {exc}")

    # When FSDP handles bf16 via MixedPrecisionPolicy, autocast is not needed.
    # Keep autocast only for non-distributed bf16 or fp16 paths.
    amp_dtype: Optional[torch.dtype] = None
    if use_bf16 and (not distributed or distributed_mode == "ddp"):
        amp_dtype = torch.bfloat16
    elif use_fp16:
        amp_dtype = torch.float16

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(pretrain_config.tf32)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = bool(pretrain_config.tf32)

    scaler = (
        torch.amp.GradScaler(enabled=(device.type == "cuda" and use_fp16))
        if torch.cuda.is_available()
        else None
    )

    _mb_cfg = orig_model.config
    _flops_per_token = _estimate_model_flops_per_token(_mb_cfg, manifest.max_seq_len)
    _peak_flops_per_gpu = H100_PEAK_TFLOPS * 1e12
    logger.info(
        f"MFU estimation: {_flops_per_token:,} training FLOPs/token, "
        f"H100 peak = {H100_PEAK_TFLOPS} TFLOPS"
    )
    _run_t0 = time.perf_counter()

    logger.info(
        "Starting Transformer Engine training: "
        f"epochs={num_epochs}, total_steps={total_steps}, warmup_steps={warmup_steps}, "
        f"grad_accum={inferred_grad_accum_steps}, "
        f"achieved_global_batch_size={achieved_global_batch_tokens:,} tokens"
    )
    logger.info(
        f"Precision config: bf16={use_bf16}, fp16={use_fp16}, "
        f"tf32={(pretrain_config.tf32 and device.type == 'cuda')}, fp8={pretrain_config.fp8}"
    )
    logger.info(
        "Distributed config: multi_gpu=%s mode=%s%s",
        distributed,
        distributed_mode,
        f" ddp_bucket_cap_mb={ddp_bucket_cap_mb}" if distributed_mode == "ddp" else "",
    )
    max_grad_norm = float(pretrain_config.max_grad_norm)
    logger.info(
        "Gradient clipping: "
        f"{'disabled (max_grad_norm=inf)' if math.isinf(max_grad_norm) else f'max_grad_norm={max_grad_norm}'}"
    )

    fp8_enabled = bool(pretrain_config.fp8 and device.type == "cuda")
    if pretrain_config.fp8 and device.type != "cuda":
        logger.warning("fp8=True but device is not CUDA. FP8 autocast will be disabled.")
    # For multi-GPU: pass the process group so TE reduces FP8 amax tensors across all ranks,
    # keeping scaling factors in sync (avoids divergent quantization between data-parallel replicas).
    fp8_group = dist.group.WORLD if (fp8_enabled and distributed and dist.is_initialized()) else None

    # ProRes: set up progressive residual warmup step tracking.
    _use_prores = getattr(orig_model.config, "use_prores", False) if hasattr(orig_model, "config") else False
    _prores_model = orig_model.model if hasattr(orig_model, "model") else None
    if _use_prores and _prores_model is not None and hasattr(_prores_model, "update_prores_alphas"):
        _prores_model.update_prores_alphas(start_step)
        logger.info(
            f"ProRes: T={orig_model.config.prores_T}, "
            f"last_layer_warmup_done_at_step={orig_model.config.prores_T * orig_model.config.num_hidden_layers}"
        )
    else:
        _prores_model = None

    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = start_step
    accum_loss = torch.zeros((), device=device)
    window_loss = torch.zeros((), device=device)
    window_steps = 0
    discard_accumulation = False

    token_count = torch.tensor(0, dtype=torch.long, device=device)
    raw_token_count = torch.zeros((), dtype=torch.long, device=device)
    log_window_t0 = time.perf_counter()
    first_step_of_run = True
    debug_non_finite_params = bool(getattr(pretrain_config, "debug_non_finite_params", True))
    # When packing, TokenPackingDataset holds the DistributedSampler; call set_epoch on it.
    _epoch_setter = train_ds if use_packing else train_sampler
    # [0]=loss, [1]=grad_norm (local)
    log_buf = torch.empty(2, device=device, dtype=torch.float32)

    profiler_ctx, profiler_step_cb = _make_te_profiler(pretrain_config, output_dir, is_main)

    with profiler_ctx:
        for epoch in range(start_epoch, num_epochs):
            if _epoch_setter is not None and hasattr(_epoch_setter, "set_epoch"):
                _epoch_setter.set_epoch(epoch)

            train_iter = iter(train_loader)
            # Reset timing window AFTER dataloader workers are ready so the
            # first logging window doesn't include iter(train_loader) overhead.
            if device.type == "cuda":
                torch.cuda.synchronize()
            log_window_t0 = time.perf_counter()

            epoch_ended_early = False
            for micro_step in range(synced_train_loader_len):
                has_batch = True
                try:
                    batch = next(train_iter)
                except StopIteration:
                    has_batch = False
                    batch = None

                # When packing + num_workers > 0, greedy bin-packing can produce
                # different batch counts per rank.  Coordinate so all ranks break
                # together to avoid FSDP / NCCL deadlock.
                if distributed and dist.is_initialized():
                    has_batch_t = torch.tensor(1 if has_batch else 0, device=device, dtype=torch.int32)
                    dist.all_reduce(has_batch_t, op=dist.ReduceOp.MIN)
                    if int(has_batch_t.item()) == 0:
                        epoch_ended_early = True
                        break
                elif not has_batch:
                    epoch_ended_early = True
                    break

                if resume_micro_step > 0 and epoch == resume_epoch and micro_step < resume_micro_step:
                    continue

                at_accum_boundary = (micro_step + 1) % inferred_grad_accum_steps == 0
                if discard_accumulation:
                    if at_accum_boundary:
                        discard_accumulation = False
                    continue

                batch = _move_batch_to_device(batch, device)

                # FSDP2: only reduce-scatter gradients at regular accumulation
                # boundaries.  (Do NOT use synced_train_loader_len as a fallback
                # — the loop may break early via all_reduce, making it
                # unreachable.  Partial accumulation at epoch end is discarded.)
                _is_sync_step = (micro_step + 1) % inferred_grad_accum_steps == 0
                if distributed and distributed_mode == "fsdp":
                    model.set_requires_gradient_sync(_is_sync_step)

                if "num_valid_tokens" in batch:
                    token_count += batch["num_valid_tokens"]
                    raw_token_count += batch["input_ids"].numel()
                elif "attention_mask" in batch:
                    token_count += batch["attention_mask"].sum()
                    raw_token_count += batch["attention_mask"].numel()
                else:
                    token_count += batch["input_ids"].numel()
                    raw_token_count += batch["input_ids"].numel()

                sync_ctx = _ddp_sync_context(model, distributed_mode=distributed_mode, sync=_is_sync_step)
                amp_ctx = (
                    torch.autocast(device_type=device.type, dtype=amp_dtype)
                    if amp_dtype is not None
                    else nullcontext()
                )
                fp8_ctx = te.autocast(enabled=fp8_enabled, recipe=FP8_RECIPE, amax_reduction_group=fp8_group) if fp8_enabled else nullcontext()
                with sync_ctx:
                    with amp_ctx, fp8_ctx:
                        fwd_kwargs = {
                            "input_ids": batch["input_ids"],
                            "labels": batch["labels"],
                        }
                        if "attention_mask" in batch:
                            fwd_kwargs["attention_mask"] = batch["attention_mask"]
                        if "cu_seqlens" in batch:
                            fwd_kwargs["cu_seqlens"] = batch["cu_seqlens"]
                            fwd_kwargs["max_seqlen"] = batch["max_seqlen"]
                        if fp8_enabled and inferred_grad_accum_steps > 1:
                            fwd_kwargs["is_first_microbatch"] = (micro_step % inferred_grad_accum_steps == 0)
                        out = model(**fwd_kwargs)

                    loss = out["loss"] if isinstance(out, dict) else out.loss
                    if not torch.isfinite(loss.detach()).all():
                        rank = dist.get_rank() if distributed and dist.is_initialized() else 0
                        logger.error(
                            "Skipping optimizer step %d due to non-finite loss on rank %d (epoch=%d micro_step=%d).",
                            global_step + 1,
                            rank,
                            epoch,
                            micro_step,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        accum_loss.zero_()
                        discard_accumulation = not at_accum_boundary
                        continue
                    loss = loss / inferred_grad_accum_steps

                    if scaler is not None and scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                accum_loss = accum_loss + loss.detach()

                # Skip to next micro-step if not at a regular accumulation
                # boundary.  We intentionally do NOT treat the last micro-step of
                # the epoch as a boundary: the loop can break early (via the
                # all_reduce exhaustion check above), making
                # synced_train_loader_len unreachable.  Any partial accumulation
                # at epoch end is discarded in the epoch-boundary cleanup below.
                if not at_accum_boundary:
                    continue

                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                try:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        orig_model.parameters(),
                        max_norm=max_grad_norm,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    if "non-finite" not in str(exc).lower():
                        raise
                    bad_grad = _first_nonfinite_grad(orig_model)
                    bad_name = bad_grad[0] if bad_grad is not None else "<unknown>"
                    rank = dist.get_rank() if distributed and dist.is_initialized() else 0
                    logger.error(
                        "Skipping optimizer step %d due to non-finite gradient in %s on rank %d (epoch=%d micro_step=%d).",
                        global_step + 1,
                        bad_name,
                        rank,
                        epoch,
                        micro_step,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss.zero_()
                    continue

                optimizer_step_skipped = False
                if scaler is not None and scaler.is_enabled():
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_step_skipped = scaler.get_scale() < old_scale
                else:
                    optimizer.step()

                if debug_non_finite_params and _has_nonfinite_params(orig_model):
                    bad_param = _first_nonfinite_param(orig_model)
                    bad_name = bad_param[0] if bad_param is not None else "<unknown>"
                    rank = dist.get_rank() if distributed and dist.is_initialized() else 0
                    raise RuntimeError(
                        f"Non-finite parameter detected after optimizer step {global_step + 1} "
                        f"in {bad_name} on rank {rank} (epoch={epoch} micro_step={micro_step})."
                    )

                if isinstance(optimizer, (DionMuon, DionNorMuon)):
                    muon_lr = optimizer.param_groups[0]["lr"]
                    learning_rate = optimizer.param_groups[1]["lr"]
                else:
                    learning_rate = optimizer.param_groups[0]["lr"]
                    muon_lr = None

                if not optimizer_step_skipped:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                profiler_step_cb(global_step)

                # ProRes: update alphas for next step (pure Python, no CUDA sync).
                if _prores_model is not None:
                    _prores_model.update_prores_alphas(global_step)

                window_loss += accum_loss.detach()
                window_steps += 1
                accum_loss.zero_()

                should_log = global_step % logging_steps == 0
                vram_log = ""
                tokens_per_sec = mfu = tok = raw_tok = 0.0
                step_tok = step_raw_tok = 0.0
                real_tokens_per_sec = real_tokens_per_sec_log = 0.0
                avg_step_ms = 0.0
                if should_log:
                    # Synchronize and measure only at logging boundaries to amortize sync cost.
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    window_dt = max(1e-9, t1 - log_window_t0)
                    log_window_t0 = t1
                    avg_step_ms = (window_dt * 1000.0) / max(1, window_steps)
                    tokens_per_sec = int((achieved_global_batch_tokens * window_steps) / window_dt)
                    log_buf[0] = window_loss / max(1, window_steps)
                    log_buf[1] = grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else (grad_norm if isinstance(grad_norm, torch.Tensor) else float(grad_norm))
                    log_vals = log_buf.cpu()
                    loss_to_log = float(log_vals[0])
                    grad_norm_val = float(log_vals[1])
                    tok = float(token_count.item())
                    raw_tok = float(raw_token_count.item())
                    if distributed and dist.is_initialized():
                        tok_buf = torch.tensor([tok, raw_tok], device=device)
                        dist.all_reduce(tok_buf, op=dist.ReduceOp.SUM)
                        tok, raw_tok = float(tok_buf[0].item()), float(tok_buf[1].item())
                    step_tok = tok / max(1, window_steps)
                    step_raw_tok = raw_tok / max(1, window_steps)
                    real_tokens_per_sec = tok / window_dt
                    real_tokens_per_sec_log = int(real_tokens_per_sec)
                    mfu = (
                        _flops_per_token * real_tokens_per_sec
                    ) / (_peak_flops_per_gpu * max(effective_world_size, 1))

                    token_count.zero_()
                    raw_token_count.zero_()

                    # VRAM logging only at eval steps (expensive: all_reduce + multiple .item())
                    if global_step % eval_steps == 0:
                        vram_log = _format_vram_for_log(
                            device=device,
                            distributed=distributed,
                            reset_peak=True,
                        )
                if should_log and is_main:
                    waste_pct = (1.0 - step_tok / max(step_raw_tok, 1)) * 100
                    wall_elapsed = time.perf_counter() - _run_t0
                    payload = {
                        "train/global_step": global_step,
                        "train/loss": loss_to_log,
                        "train/grad_norm": grad_norm_val,
                        "train/learning_rate": learning_rate,
                        "train/epoch": epoch + (micro_step + 1) / synced_train_loader_len,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/real_tokens_per_sec": real_tokens_per_sec,
                        "train/step_real_tokens": step_tok,
                        "train/step_raw_tokens": step_raw_tok,
                        "train/packing_waste_pct": waste_pct,
                        "time_elapsed_sec": wall_elapsed,
                        "train/time_elapsed_sec": wall_elapsed,
                    }
                    if muon_lr is not None:
                        payload["train/muon_lr"] = muon_lr
                    if wandb_enabled and wandb.run is not None:
                        try:
                            wandb.log(payload)
                        except Exception as exc:
                            wandb_enabled = False
                            logger.warning(f"W&B log failed; disabling logging. Error: {exc}")
                    muon_lr_str = f"muon_lr={muon_lr:.2e} " if muon_lr is not None else ""
                    logger.info(
                        f"[step {global_step}/{total_steps}] "
                        f"loss={loss_to_log:.4f} lr={learning_rate:.2e} {muon_lr_str}"
                        f"grad_norm={grad_norm_val:.4f} tok/s={tokens_per_sec:,} real_tok/s={real_tokens_per_sec_log:,} "
                        f"real_tok/step={step_tok:,.0f} "
                        f"dt={avg_step_ms:.2f}ms waste={waste_pct:.1f}% "
                        f"h100_mfu={mfu:.2%} {vram_log}"
                    )

                if first_step_of_run:
                    first_step_of_run = False
                    gc.collect()
                    gc.freeze()
                    gc.disable()
                elif global_step % 5000 == 0:
                    gc.collect()

                if should_log:
                    window_loss.zero_()
                    window_steps = 0

                if global_step % eval_steps == 0:
                    eval_loss = _evaluate(
                        model=orig_model,
                        eval_loader=eval_loader,
                        device=device,
                        distributed=distributed,
                        amp_dtype=amp_dtype,
                    )
                    if is_main:
                        if wandb_enabled and wandb.run is not None:
                            try:
                                wandb.log(
                                    {
                                        "train/global_step": global_step,
                                        "eval/loss": eval_loss,
                                    }
                                )
                            except Exception as exc:
                                wandb_enabled = False
                                logger.warning(f"W&B log failed; disabling logging. Error: {exc}")
                        logger.info(f"[step {global_step}] eval_loss={eval_loss:.4f}")
                    model.train()

                if global_step % save_steps == 0:
                    _save_checkpoint(
                        model=orig_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        epoch=epoch,
                        output_dir=output_dir,
                        logging_steps=logging_steps,
                        eval_steps=eval_steps,
                        save_steps=save_steps,
                        distributed=distributed,
                        is_main=is_main,
                        distributed_mode=distributed_mode,
                        model_config=_model_config,
                        manifest=_manifest,
                        pretrain_config=pretrain_config,
                        total_steps=total_steps,
                        warmup_steps=warmup_steps,
                    )

            # ---- Epoch boundary cleanup ----
            # Flush partial gradient-accumulation state left over from
            # micro-steps that ran forward+backward but never reached an
            # optimizer step.  Prevents gradient leak, loss contamination,
            # FSDP gradient desync, and token/loss counter bleed across the
            # boundary.  Timing window is reset at the top of the next epoch
            # iteration, after iter(train_loader).
            epoch_fwd_count = micro_step if epoch_ended_early else synced_train_loader_len
            partial_discarded = epoch_fwd_count % inferred_grad_accum_steps
            if is_main:
                logger.info(
                    "Epoch %d/%d complete: %d micro-steps, %d optimizer steps%s",
                    epoch + 1, num_epochs, epoch_fwd_count,
                    epoch_fwd_count // inferred_grad_accum_steps,
                    f" ({partial_discarded} trailing micro-step(s) discarded)"
                    if partial_discarded else "",
                )
            optimizer.zero_grad(set_to_none=True)
            accum_loss.zero_()
            discard_accumulation = False
            token_count.zero_()
            raw_token_count.zero_()
            window_loss.zero_()
            window_steps = 0

            if resume_micro_step > 0 and epoch == resume_epoch:
                resume_micro_step = 0

    if distributed and dist.is_initialized():
        _dist_barrier(local_rank)

    _save_checkpoint(
        model=orig_model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        epoch=num_epochs,
        output_dir=output_dir,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        distributed=distributed,
        is_main=is_main,
        distributed_mode=distributed_mode,
        model_config=_model_config,
        manifest=_manifest,
        pretrain_config=pretrain_config,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    if is_main:
        if wandb_enabled and wandb.run is not None:
            try:
                (Path(output_dir) / "wandb_run_id.txt").write_text(wandb.run.id, encoding="utf-8")
                wandb.finish()
            except Exception as exc:
                logger.warning(f"W&B finalize failed. Error: {exc}")

    if distributed and dist.is_initialized():
        _dist_barrier(local_rank)

    logger.info("Transformer Engine pretraining complete.")
