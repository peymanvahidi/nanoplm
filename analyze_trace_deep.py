#!/usr/bin/env python3
"""Deep analysis of optimizer step: per-parameter-group breakdown, kernel launch cadence,
CPU-GPU gaps, and aten::item (scalar tensor -> CPU) calls."""

import json
from collections import defaultdict, Counter
from pathlib import Path

def load_trace(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return data

def analyze_opt_step_deep(trace_path):
    print(f"\n{'='*80}")
    print(f"DEEP OPTIMIZER ANALYSIS: {Path(trace_path).parent.parent.name}")
    print(f"{'='*80}")

    events = load_trace(trace_path)

    # Filter to duration events
    dur_events = [e for e in events if e.get("ph") == "X" and e.get("dur", 0) > 0]

    # Find all optimizer step events
    step_events = [e for e in dur_events if "Optimizer.step#NorMuon.step" == e.get("name", "")]

    if not step_events:
        print("No NorMuon optimizer step events found")
        return

    print(f"\nFound {len(step_events)} optimizer steps")

    # Analyze one representative step (the longest one)
    step = max(step_events, key=lambda e: e["dur"])
    step_start = step["ts"]
    step_end = step_start + step["dur"]
    step_dur_ms = step["dur"] / 1000

    print(f"Representative step: {step_dur_ms:.1f}ms")

    # === aten::item analysis ===
    print(f"\n{'='*60}")
    print("aten::item / _local_scalar_dense ANALYSIS")
    print("(These force CPU-GPU sync for scalar value reads)")
    print(f"{'='*60}")

    item_events = [e for e in dur_events if e["name"] in ("aten::item", "aten::_local_scalar_dense")]
    item_in_step = [e for e in item_events if e["ts"] >= step_start and e["ts"] <= step_end]

    total_item_time = sum(e["dur"] for e in item_events) / 1000
    item_in_step_time = sum(e["dur"] for e in item_in_step) / 1000

    print(f"\nTotal aten::item calls across trace: {len(item_events)//2}")  # item + _local_scalar_dense pair
    print(f"Total time in aten::item: {total_item_time:.1f}ms")
    print(f"aten::item calls inside optimizer step: {len(item_in_step)//2}")
    print(f"Time in aten::item during opt step: {item_in_step_time:.1f}ms")

    # Find cudaStreamSynchronize events (the actual blocking calls)
    sync_events = [e for e in dur_events if e["name"] == "cudaStreamSynchronize"]
    sync_in_step = [e for e in sync_events if e["ts"] >= step_start and e["ts"] <= step_end]

    total_sync_time = sum(e["dur"] for e in sync_events) / 1000
    sync_in_step_time = sum(e["dur"] for e in sync_in_step) / 1000

    print(f"\nTotal cudaStreamSynchronize calls: {len(sync_events)}")
    print(f"Total sync time: {total_sync_time:.1f}ms")
    print(f"Sync calls inside optimizer step: {len(sync_in_step)}")
    print(f"Sync time during opt step: {sync_in_step_time:.1f}ms")

    if sync_events:
        print(f"\nSync durations distribution:")
        for s in sorted(sync_events, key=lambda e: e["dur"], reverse=True)[:20]:
            print(f"  ts={s['ts']:.0f}, dur={s['dur']/1000:.2f}ms")

    # === Kernel launch pattern inside optimizer step ===
    print(f"\n{'='*60}")
    print("KERNEL LAUNCH PATTERN IN OPTIMIZER STEP")
    print(f"{'='*60}")

    kernels_in_step = [e for e in dur_events
                       if e.get("cat") == "kernel"
                       and e["ts"] >= step_start and e["ts"] <= step_end]

    if kernels_in_step:
        kernels_sorted = sorted(kernels_in_step, key=lambda e: e["ts"])

        # Compute inter-kernel gaps
        gaps = []
        for i in range(1, len(kernels_sorted)):
            prev_end = kernels_sorted[i-1]["ts"] + kernels_sorted[i-1]["dur"]
            curr_start = kernels_sorted[i]["ts"]
            gap = curr_start - prev_end
            if gap > 0:
                gaps.append(gap)

        total_kernel_time = sum(e["dur"] for e in kernels_in_step)
        total_gap_time = sum(gaps)

        print(f"\nKernels in optimizer step: {len(kernels_in_step)}")
        print(f"Total kernel compute time: {total_kernel_time/1000:.1f}ms")
        print(f"Total inter-kernel gaps: {total_gap_time/1000:.1f}ms")
        print(f"Kernel occupancy ratio: {total_kernel_time/(total_kernel_time+total_gap_time)*100:.1f}%")

        # Gap distribution
        if gaps:
            gaps_sorted = sorted(gaps, reverse=True)
            print(f"\nInter-kernel gap distribution:")
            print(f"  Max gap: {gaps_sorted[0]/1000:.2f}ms")
            print(f"  Median gap: {gaps_sorted[len(gaps)//2]:.1f}us")
            print(f"  Avg gap: {sum(gaps)/len(gaps):.1f}us")
            print(f"  Gaps > 100us: {sum(1 for g in gaps if g > 100)}")
            print(f"  Gaps > 10us: {sum(1 for g in gaps if g > 10)}")
            print(f"  Gaps > 1us: {sum(1 for g in gaps if g > 1)}")

        # Kernel duration distribution inside opt step
        print(f"\nKernel durations inside opt step:")
        micro = [e for e in kernels_in_step if e["dur"] < 10]
        small = [e for e in kernels_in_step if 10 <= e["dur"] < 50]
        medium = [e for e in kernels_in_step if 50 <= e["dur"] < 500]
        large = [e for e in kernels_in_step if e["dur"] >= 500]

        print(f"  Micro (<10us): {len(micro)} kernels, {sum(e['dur'] for e in micro)/1000:.2f}ms total")
        print(f"  Small (10-50us): {len(small)} kernels, {sum(e['dur'] for e in small)/1000:.2f}ms total")
        print(f"  Medium (50-500us): {len(medium)} kernels, {sum(e['dur'] for e in medium)/1000:.2f}ms total")
        print(f"  Large (>500us): {len(large)} kernels, {sum(e['dur'] for e in large)/1000:.2f}ms total")

        # What are the micro kernels?
        print(f"\nMicro kernel names inside opt step:")
        micro_names = Counter(e["name"] for e in micro)
        for name, count in micro_names.most_common(15):
            avg_dur = sum(e["dur"] for e in micro if e["name"] == name) / count
            print(f"  {name[:100]}: count={count}, avg={avg_dur:.1f}us")

        # What are the small kernels?
        print(f"\nSmall kernel names inside opt step:")
        small_names = Counter(e["name"] for e in small)
        for name, count in small_names.most_common(15):
            avg_dur = sum(e["dur"] for e in small if e["name"] == name) / count
            print(f"  {name[:100]}: count={count}, avg={avg_dur:.1f}us")

    # === NCCL comms inside opt step ===
    print(f"\n{'='*60}")
    print("NCCL / COMMUNICATION IN OPTIMIZER STEP")
    print(f"{'='*60}")

    nccl_in_step = [e for e in dur_events
                    if ("nccl" in e["name"].lower() or "alltoall" in e["name"].lower()
                        or "allreduce" in e["name"].lower() or "allgather" in e["name"].lower()
                        or "record_param_comms" in e["name"].lower())
                    and e["ts"] >= step_start and e["ts"] <= step_end]

    if nccl_in_step:
        nccl_names = Counter(e["name"] for e in nccl_in_step)
        total_nccl_time = sum(e["dur"] for e in nccl_in_step) / 1000
        print(f"\nComm ops inside optimizer step: {len(nccl_in_step)}")
        print(f"Total comm time: {total_nccl_time:.1f}ms")
        for name, count in nccl_names.most_common():
            durs = [e["dur"] for e in nccl_in_step if e["name"] == name]
            print(f"  {name}: count={count}, total={sum(durs)/1000:.1f}ms, avg={sum(durs)/len(durs)/1000:.2f}ms")

    # === Per-profiler-step time breakdown ===
    print(f"\n{'='*60}")
    print("PER-STEP TIME BREAKDOWN")
    print(f"{'='*60}")

    profiler_steps = sorted(
        [e for e in dur_events if "ProfilerStep" in e.get("name", "") and e.get("cat") == "user_annotation"],
        key=lambda e: e["ts"]
    )

    for ps in profiler_steps:
        ps_start = ps["ts"]
        ps_end = ps_start + ps["dur"]
        ps_dur = ps["dur"] / 1000

        # Find opt step time within this profiler step
        opt_in_ps = [e for e in step_events if e["ts"] >= ps_start and e["ts"] <= ps_end]
        opt_time = sum(e["dur"] for e in opt_in_ps) / 1000

        # Find fwd/bwd time (kernels not in opt step)
        fwd_bwd_kernels = [e for e in dur_events
                           if e.get("cat") == "kernel"
                           and e["ts"] >= ps_start and e["ts"] <= ps_end
                           and not any(e["ts"] >= o["ts"] and e["ts"] <= o["ts"]+o["dur"] for o in opt_in_ps)]
        fwd_bwd_time = sum(e["dur"] for e in fwd_bwd_kernels) / 1000

        print(f"\n{ps['name']}: {ps_dur:.1f}ms total")
        print(f"  Optimizer step: {opt_time:.1f}ms ({opt_time/ps_dur*100:.1f}%)")
        print(f"  Fwd/Bwd GPU kernels: {fwd_bwd_time:.1f}ms")

    # === aten::item caller context ===
    print(f"\n{'='*60}")
    print("aten::item CALLER PATTERNS (what triggers scalar reads)")
    print(f"{'='*60}")

    # Look at events that enclose aten::item events
    item_only = [e for e in dur_events if e["name"] == "aten::item"]

    # For each item call, find the nearest enclosing parent op
    for item_ev in sorted(item_only, key=lambda e: e["ts"])[:10]:  # First 10
        parents = [e for e in dur_events
                   if e["ts"] <= item_ev["ts"]
                   and (e["ts"] + e["dur"]) >= (item_ev["ts"] + item_ev["dur"])
                   and e is not item_ev
                   and e["name"] != "aten::_local_scalar_dense"
                   and e.get("cat") in ("cpu_op", "user_annotation")]

        if parents:
            # Get the most specific (shortest duration) parent
            closest = min(parents, key=lambda e: e["dur"])
            print(f"  item() at ts={item_ev['ts']:.0f} dur={item_ev['dur']}us -> caller: {closest['name']} ({closest['cat']})")

    # === NorMuon-specific patterns ===
    print(f"\n{'='*60}")
    print("NorMuon OPTIMIZER OPERATION SEQUENCE")
    print(f"{'='*60}")

    # Get user annotations inside opt step
    annots_in_step = sorted(
        [e for e in dur_events
         if e.get("cat") in ("user_annotation", "gpu_user_annotation")
         and e["ts"] >= step_start and e["ts"] <= step_end],
        key=lambda e: e["ts"]
    )

    if annots_in_step:
        annot_names = Counter(e["name"] for e in annots_in_step)
        print(f"\nAnnotations inside opt step:")
        for name, count in annot_names.most_common():
            durs = [e["dur"] for e in annots_in_step if e["name"] == name]
            print(f"  {name}: count={count}, total={sum(durs)/1000:.1f}ms, avg={sum(durs)/len(durs)/1000:.2f}ms")

    # === Compare across all traces ===
    print(f"\n{'='*60}")
    print("cudaMemcpyAsync INSIDE OPT STEP")
    print(f"{'='*60}")

    memcpy_in_step = [e for e in dur_events
                      if "Memcpy" in e.get("name", "") or "memcpy" in e.get("name", "").lower()
                      and e["ts"] >= step_start and e["ts"] <= step_end]

    if memcpy_in_step:
        memcpy_names = Counter(e["name"] for e in memcpy_in_step)
        print(f"Memcpy ops in opt step: {len(memcpy_in_step)}")
        for name, count in memcpy_names.most_common():
            durs = [e["dur"] for e in memcpy_in_step if e["name"] == name]
            print(f"  {name}: count={count}, total={sum(durs)/1000:.2f}ms")

traces = sorted(Path("/workspace/nanoplm/output/pretraining_checkpoints").glob("*/profiler_traces/chrome_trace.json"))
analyze_opt_step_deep(str(traces[-1]))
