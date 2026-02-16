"""Transformer Engine pretraining pipeline."""



import gc
import os
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
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from nanoplm.data.manifest import read_manifest, validate_manifest_for_pipeline
from nanoplm.pretraining.collator import DataCollatorWithFlattening, ProtDataCollatorForLM
from nanoplm.pretraining.dataset import ShardedDataset, TokenPackingDataset
from nanoplm.pretraining.models.modern_bert.modelling_te import FP8_RECIPE
from nanoplm.pretraining.models.modern_bert.pure_model import TEProtModernBertMLM
from nanoplm.pretraining.pipeline import PretrainingConfig, ResumeConfig, _get_num_workers, _prepare_run_and_steps
from nanoplm.pretraining.pure_pipeline import (
    H100_PEAK_TFLOPS,
    N_PREFETCH_LAYERS_FSDP2,
    _create_optimizer,
    _create_scheduler,
    _dist_barrier,
    _estimate_model_flops_per_token,
    _evaluate,
    _get_warmup_steps,
    _load_checkpoint,
    _move_batch_to_device,
    _num_update_steps_per_epoch,
    _resolve_world_size,
    _save_checkpoint,
    _set_seed,
    _sync_train_loader_len,
)
from nanoplm.utils.common import create_dirs, get_device
from nanoplm.utils.logger import logger


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

    if use_packing:
        _pad_to = pretrain_config.micro_batch_size * manifest.max_seq_len
        
        # Sampler handled by train_ds (TokenPackingDataset) via wrapper later
        
        # We need this purely for collator instantiation here?
        # No, we can just instantiate collator like in pure_pipeline.py
        # But wait, te_pipeline follows pure_pipeline logic closely?
        
        inner_collator = ProtDataCollatorForLM(
            tokenizer=tokenizer,
            mlm_probability=pretrain_config.mlm_probability,
            mask_token_probability=pretrain_config.mask_replace_prob,
            random_token_probability=pretrain_config.random_token_prob,
            keep_probability=pretrain_config.keep_probability,
        )
        collator = DataCollatorWithFlattening(
            collator=inner_collator,
            pad_to_multiple_of=8,
        )
        logger.info(f"Sequence packing ENABLED (TokenPackingDataset + DataCollatorWithFlattening, target={_pad_to:,} tokens)")
    else:
        collator = ProtDataCollatorForLM(
            tokenizer=tokenizer,
            mlm_probability=pretrain_config.mlm_probability,
            mask_token_probability=pretrain_config.mask_replace_prob,
            random_token_probability=pretrain_config.random_token_prob,
            keep_probability=pretrain_config.keep_probability,
        )

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
    ) = _prepare_run_and_steps(
        pretrain_config=pretrain_config,
        resume_config=resume_config,
        train_samples=train_sequences,
        global_batch_size_samples=global_batch_size_samples,
    )

    num_workers = _get_num_workers(pretrain_config.num_workers, effective_world_size)
    pin_memory = device.type == "cuda"
    # Disable persistent workers when packing: TokenPackingDataset + persistent_workers
    # can cause hangs near epoch boundaries (set_epoch doesn't reach workers).
    persistent_workers = num_workers > 0 and not use_packing
    prefetch_factor = pretrain_config.prefetch_factor if num_workers > 0 else None

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = bool(pretrain_config.multi_gpu)

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        if not dist.is_initialized():
            if backend == "nccl":
                dist.init_process_group(backend=backend, device_id=local_rank)
            else:
                dist.init_process_group(backend=backend)

    model.to(device)

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
    if distributed:
        # Apply FSDP2 per transformer layer, then at the root.
        fsdp_kwargs: dict = {}
        if use_bf16:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )
        fsdp_mesh = init_device_mesh("cuda", (effective_world_size,))
        for layer in model.model.layers:
            fully_shard(layer, mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_kwargs)
        fully_shard(model, mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_kwargs)

        # FSDP2 Explicit Prefetching
        if N_PREFETCH_LAYERS_FSDP2 > 1:
            layers = model.model.layers
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
        is_main = dist.get_rank() == 0
    else:
        eval_sampler = SequentialSampler(val_ds)
        is_main = True

    # Build train sampler / batch_sampler.
    train_batch_sampler = None
    train_sampler = None
    if use_packing:
        _pad_to = pretrain_config.micro_batch_size * manifest.max_seq_len
         # For packing, we use a sampler on the *underlying* dataset to handle shuffling and distributed sharding.
        if distributed:
            _inner_sampler = DistributedSampler(train_ds, shuffle=True, seed=pretrain_config.seed)
        else:
            _inner_sampler = RandomSampler(train_ds)
            
        # Wrap dataset for packing
        train_ds = TokenPackingDataset(
            train_ds,
            max_tokens_per_batch=_pad_to,
            drop_last=False,
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
        # Configure collator to flatten and pad if necessary
        collator = DataCollatorWithFlattening(
            collator=inner_collator,
            pad_to_multiple_of=16, # Transformer Engine FP8 alignment
        )
        logger.info(f"Sequence packing ENABLED (TokenPackingDataset + DataCollatorWithFlattening, target={_pad_to:,} tokens)")
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

    # No torch.compile for the TE path: TE's fused kernels (MultiheadAttention,
    # LayerNormMLP) are already optimized, and FP8 metadata (amax history /
    # buffer_index) changes each step, causing excessive dynamo recompilation.
    orig_model = model

    eval_collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=pretrain_config.mlm_probability,
        mask_token_probability=pretrain_config.mask_replace_prob,
        random_token_probability=pretrain_config.random_token_prob,
        keep_probability=pretrain_config.keep_probability,
    )

    if use_packing:
         train_loader = DataLoader(
            train_ds,
            batch_size=None,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
         # If using packing with IterableDataset, len() is an estimate.
         # Trainer expects exact length for steps per epoch.
         # We can relax this strictness or correct the estimate.
         # For now, let's just create the loader.
    elif train_batch_sampler is not None:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            sampler=train_sampler,
            batch_size=pretrain_config.micro_batch_size,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=False,
        )
    eval_loader = DataLoader(
        val_ds,
        sampler=eval_sampler,
        batch_size=pretrain_config.micro_batch_size,
        collate_fn=eval_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False,
    )

    optimizer = _create_optimizer(model, pretrain_config, distributed_mesh=fsdp_mesh)

    synced_train_loader_len = _sync_train_loader_len(
        train_loader_len=len(train_loader),
        distributed=distributed,
        device=device,
    )
    # With TokenPackingDataset + DistributedSampler, greedy packing can yield different batch
    # counts per rank, causing desync and deadlock. Cap iteration at a safe minimum so all
    # ranks process the same number of batches.
    if use_packing and distributed:
        _total_tokens = train_ds.dataset.total_tokens
        _tokens_per_rank = _total_tokens // effective_world_size
        _min_safe_batches = max(1, _tokens_per_rank // train_ds.max_tokens_per_batch)
        if _min_safe_batches < synced_train_loader_len:
            logger.info(
                f"Capping micro-batches per epoch to {_min_safe_batches} (from {synced_train_loader_len}) "
                "to prevent distributed deadlock with variable packing"
            )
            synced_train_loader_len = _min_safe_batches
    steps_per_epoch = _num_update_steps_per_epoch(
        train_loader_len=synced_train_loader_len,
        grad_accum=inferred_grad_accum_steps,
    )
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = _get_warmup_steps(total_steps, float(pretrain_config.warmup_ratio))
    scheduler = _create_scheduler(optimizer, warmup_steps, total_steps)

    start_step = 0
    start_epoch = 0
    if resume_config and resume_config.is_resume:
        logger.info(f"Resuming from checkpoint: {resume_config.checkpoint_dir}")
        start_step, start_epoch = _load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dir=resume_config.checkpoint_dir,
            device=device,
            distributed=distributed,
        )
        logger.info(f"Resumed at global_step={start_step}, epoch={start_epoch}")

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
                wandb.define_metric("*", step_metric="train/global_step", step_sync=True)
        except Exception as exc:
            logger.warning(f"W&B init failed, continuing without logging. Error: {exc}")

    # When FSDP handles bf16 via MixedPrecisionPolicy, autocast is not needed.
    # Keep autocast only for non-distributed bf16 or fp16 paths.
    amp_dtype: Optional[torch.dtype] = None
    if use_bf16 and not distributed:
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
    _flops_per_token = _estimate_model_flops_per_token(
        num_layers=_mb_cfg.num_hidden_layers,
        hidden_size=_mb_cfg.hidden_size,
        intermediate_size=_mb_cfg.intermediate_size,
        seq_len=manifest.max_seq_len,
        vocab_size=_mb_cfg.vocab_size,
    )
    _peak_flops_per_gpu = H100_PEAK_TFLOPS * 1e12
    logger.info(
        f"MFU estimation: {_flops_per_token:,} training FLOPs/token, "
        f"H100 peak = {H100_PEAK_TFLOPS} TFLOPS"
    )

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

    fp8_enabled = bool(pretrain_config.fp8 and device.type == "cuda")
    if pretrain_config.fp8 and device.type != "cuda":
        logger.warning("fp8=True but device is not CUDA. FP8 autocast will be disabled.")
    # For multi-GPU: pass the process group so TE reduces FP8 amax tensors across all ranks,
    # keeping scaling factors in sync (avoids divergent quantization between data-parallel replicas).
    fp8_group = dist.group.WORLD if (fp8_enabled and distributed and dist.is_initialized()) else None

    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = start_step
    accum_loss = 0.0
    window_loss = 0.0
    window_steps = 0

    token_count = 0
    raw_token_count = 0
    token_t0: Optional[float] = None
    first_step_of_run = True

    # When packing, TokenPackingDataset holds the DistributedSampler; call set_epoch on it.
    # When not packing, the sampler is train_sampler or train_batch_sampler.
    _epoch_setter = (
        train_ds if use_packing else (train_batch_sampler if train_batch_sampler is not None else train_sampler)
    )

    for epoch in range(start_epoch, num_epochs):
        if _epoch_setter is not None and hasattr(_epoch_setter, "set_epoch"):
            _epoch_setter.set_epoch(epoch)

        for micro_step, batch in enumerate(train_loader):
            # Cap iteration to prevent desync when TokenPackingDataset yields different counts per rank
            if micro_step >= synced_train_loader_len:
                break
            if resume_micro_step > 0 and epoch == resume_epoch and micro_step < resume_micro_step:
                continue

            if distributed and N_PREFETCH_LAYERS_FSDP2 > 1:
                model.unshard()

            batch = _move_batch_to_device(batch, device)

            # FSDP2: only reduce-scatter gradients at accumulation boundaries.
            if distributed:
                _is_sync_step = (
                    (micro_step + 1) % inferred_grad_accum_steps == 0
                    or micro_step + 1 == synced_train_loader_len
                )
                model.set_requires_gradient_sync(_is_sync_step)

            if raw_token_count == 0:
                token_t0 = time.perf_counter()

            if "num_valid_tokens" in batch:
                token_count += int(batch["num_valid_tokens"])
                raw_token_count += int(batch["input_ids"].numel())
            elif "attention_mask" in batch:
                token_count += int(batch["attention_mask"].sum().item())
                raw_token_count += int(batch["attention_mask"].numel())
            else:
                token_count += int(batch["input_ids"].numel())
                raw_token_count += int(batch["input_ids"].numel())

            amp_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None
                else nullcontext()
            )
            fp8_ctx = te.autocast(enabled=fp8_enabled, recipe=FP8_RECIPE, amax_reduction_group=fp8_group) if fp8_enabled else nullcontext()
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
            loss = loss / inferred_grad_accum_steps

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()

            grad_accum = inferred_grad_accum_steps
            is_boundary = (micro_step + 1) % grad_accum == 0
            is_last_micro = micro_step + 1 == synced_train_loader_len
            if not (is_boundary or is_last_micro):
                continue

            # accum_loss = sum(Li/grad_accum). For partial last step, true mean = sum(Li)/n_micro.
            # Scale so reported loss = true mean: accum_loss * grad_accum / n_micro
            n_micro = (micro_step + 1) % grad_accum or grad_accum
            accum_loss *= grad_accum / n_micro

            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

            optimizer_step_skipped = False
            if scaler is not None and scaler.is_enabled():
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_skipped = scaler.get_scale() < old_scale
            else:
                optimizer.step()

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
            window_loss += accum_loss
            window_steps += 1
            accum_loss = 0.0

            t1 = time.perf_counter()
            if token_t0 is None:
                token_t0 = t1
            elapsed = t1 - token_t0
            tok = float(token_count)
            raw_tok = float(raw_token_count)

            if distributed and dist.is_initialized():
                tok_tensor = torch.tensor(tok, device=device)
                raw_tok_tensor = torch.tensor(raw_tok, device=device)
                elapsed_tensor = torch.tensor(float(elapsed), device=device)
                dist.all_reduce(tok_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(raw_tok_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
                tok = tok_tensor.item()
                raw_tok = raw_tok_tensor.item()
                elapsed = elapsed_tensor.item()

            tokens_per_sec = tok / max(elapsed, 1e-9)
            raw_tokens_per_sec = raw_tok / max(elapsed, 1e-9)

            achieved_flops_per_sec = raw_tokens_per_sec * _flops_per_token
            per_gpu_flops = achieved_flops_per_sec / max(effective_world_size, 1)
            mfu = per_gpu_flops / _peak_flops_per_gpu

            token_count = 0
            raw_token_count = 0
            token_t0 = None

            should_log = global_step % logging_steps == 0
            if should_log and is_main:
                loss_to_log = window_loss / max(1, window_steps)
                payload = {
                    "train/global_step": global_step,
                    "train/loss": loss_to_log,
                    "train/grad_norm": grad_norm,
                    "train/learning_rate": learning_rate,
                    "train/epoch": epoch + (micro_step + 1) / synced_train_loader_len,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/raw_tokens_per_sec": raw_tokens_per_sec,
                    "train/step_real_tokens": int(tok),
                    "train/step_raw_tokens": int(raw_tok),
                    "train/packing_waste_pct": (1.0 - tok / max(raw_tok, 1)) * 100,
                }
                if muon_lr is not None:
                    payload["train/muon_lr"] = muon_lr
                if wandb_enabled and wandb.run is not None:
                    try:
                        wandb.log(payload)
                    except Exception as exc:
                        wandb_enabled = False
                        logger.warning(f"W&B log failed; disabling logging. Error: {exc}")
                waste_pct = (1.0 - tok / max(raw_tok, 1)) * 100
                muon_lr_str = f"muon_lr={muon_lr:.2e} " if muon_lr is not None else ""
                logger.info(
                    f"[step {global_step}/{total_steps}] "
                    f"loss={loss_to_log:.4f} lr={learning_rate:.2e} {muon_lr_str}"
                    f"grad_norm={grad_norm:.4f} tok/s={tokens_per_sec:,.0f} "
                    f"raw_tok/s={raw_tokens_per_sec:,.0f} "
                    f"step_tokens={int(tok):,} waste={waste_pct:.1f}% "
                    f"h100_mfu={mfu:.2%}"
                )

            if first_step_of_run:
                first_step_of_run = False
                gc.collect()
                gc.freeze()
                gc.disable()
            elif global_step % 5000 == 0:
                gc.collect()

            if should_log:
                window_loss = 0.0
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
                    model=model,
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
                )

        if resume_micro_step > 0 and epoch == resume_epoch:
            resume_micro_step = 0

    if distributed and dist.is_initialized():
        _dist_barrier(local_rank)

    _save_checkpoint(
        model=model,
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
