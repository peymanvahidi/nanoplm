#!/usr/bin/env python3
"""Analyze Chrome trace JSON files from PyTorch profiler to identify optimizer step bottlenecks."""

import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

def load_trace(path):
    with open(path) as f:
        data = json.load(f)
    # Chrome trace format: {"traceEvents": [...]}
    if isinstance(data, dict):
        events = data.get("traceEvents", [])
    else:
        events = data
    return events

def analyze_trace(trace_path):
    print(f"\n{'='*80}")
    print(f"ANALYZING: {trace_path}")
    print(f"{'='*80}")

    events = load_trace(trace_path)

    # Separate CPU and GPU events
    cpu_events = []
    gpu_events = []
    all_events = []

    for e in events:
        if e.get("ph") not in ("X", "B", "E"):
            continue
        cat = e.get("cat", "")
        name = e.get("name", "")
        dur = e.get("dur", 0)
        ts = e.get("ts", 0)
        tid = e.get("tid", "")
        pid = e.get("pid", "")

        entry = {
            "name": name,
            "cat": cat,
            "dur": dur,
            "ts": ts,
            "tid": tid,
            "pid": pid,
            "ph": e.get("ph"),
            "args": e.get("args", {}),
        }
        all_events.append(entry)

        if "kernel" in cat.lower() or "cuda" in cat.lower() or "gpu" in str(tid).lower():
            gpu_events.append(entry)
        elif "cpu" in cat.lower() or cat in ("cpu_op", "user_annotation", "Trace", "python_function"):
            cpu_events.append(entry)

    print(f"\nTotal events: {len(all_events)}")
    print(f"CPU-side events: {len(cpu_events)}")
    print(f"GPU-side events: {len(gpu_events)}")

    # --- Categorize all events ---
    categories = Counter()
    for e in all_events:
        categories[e["cat"]] += 1
    print(f"\nEvent categories:")
    for cat, count in categories.most_common(20):
        print(f"  {cat}: {count}")

    # --- Find optimizer-related events ---
    print(f"\n{'='*60}")
    print("OPTIMIZER STEP ANALYSIS")
    print(f"{'='*60}")

    opt_events = [e for e in all_events if any(kw in e["name"].lower() for kw in
        ["optimizer", "adam", "step", "zero_grad", "clip_grad", "unscale", "grad_scal"])]

    if opt_events:
        print(f"\nOptimizer-related events: {len(opt_events)}")
        opt_names = Counter(e["name"] for e in opt_events)
        for name, count in opt_names.most_common(30):
            durations = [e["dur"] for e in opt_events if e["name"] == name and e["dur"] > 0]
            if durations:
                print(f"  {name}: count={count}, total={sum(durations)/1000:.1f}ms, avg={sum(durations)/len(durations)/1000:.2f}ms")
            else:
                print(f"  {name}: count={count}")

    # --- Find all kernel launches (short GPU kernels indicating micro-kernels) ---
    print(f"\n{'='*60}")
    print("MICRO-KERNEL ANALYSIS (GPU kernels < 10us)")
    print(f"{'='*60}")

    kernel_events = [e for e in all_events if e["cat"] == "kernel" or "kernel" in e["cat"].lower()]
    if not kernel_events:
        # Try to find GPU events by other means
        kernel_events = [e for e in all_events if "cuda" in e["cat"].lower() and e.get("dur", 0) > 0]

    if kernel_events:
        micro_kernels = [e for e in kernel_events if 0 < e["dur"] < 10]  # < 10 microseconds
        small_kernels = [e for e in kernel_events if 10 <= e["dur"] < 50]
        medium_kernels = [e for e in kernel_events if 50 <= e["dur"] < 500]
        large_kernels = [e for e in kernel_events if e["dur"] >= 500]

        print(f"\nKernel size distribution:")
        print(f"  Micro (<10us):   {len(micro_kernels)} kernels")
        print(f"  Small (10-50us): {len(small_kernels)} kernels")
        print(f"  Medium (50-500us): {len(medium_kernels)} kernels")
        print(f"  Large (>500us):  {len(large_kernels)} kernels")

        if micro_kernels:
            micro_names = Counter(e["name"] for e in micro_kernels)
            print(f"\nTop micro-kernels (<10us) by frequency:")
            for name, count in micro_names.most_common(20):
                durs = [e["dur"] for e in micro_kernels if e["name"] == name]
                print(f"  {name}: count={count}, avg={sum(durs)/len(durs):.1f}us, total={sum(durs)/1000:.2f}ms")

        if small_kernels:
            small_names = Counter(e["name"] for e in small_kernels)
            print(f"\nTop small kernels (10-50us) by frequency:")
            for name, count in small_names.most_common(20):
                durs = [e["dur"] for e in small_kernels if e["name"] == name]
                print(f"  {name}: count={count}, avg={sum(durs)/len(durs):.1f}us, total={sum(durs)/1000:.2f}ms")
    else:
        print("No kernel events found in trace.")

    # --- CPU ops analysis (looking for CPU-side dispatch overhead) ---
    print(f"\n{'='*60}")
    print("CPU-SIDE DISPATCH ANALYSIS")
    print(f"{'='*60}")

    cpu_ops = [e for e in all_events if e["cat"] in ("cpu_op",) and e.get("dur", 0) > 0]
    if cpu_ops:
        # Find CPU ops that happen during optimizer step region
        cpu_op_names = Counter(e["name"] for e in cpu_ops)
        print(f"\nTop CPU ops by frequency (total: {len(cpu_ops)}):")
        for name, count in cpu_op_names.most_common(30):
            durs = [e["dur"] for e in cpu_ops if e["name"] == name]
            print(f"  {name}: count={count}, total={sum(durs)/1000:.1f}ms, avg={sum(durs)/len(durs):.1f}us")

    # --- Look for cudaLaunchKernel / cuda runtime calls ---
    print(f"\n{'='*60}")
    print("CUDA RUNTIME ANALYSIS")
    print(f"{'='*60}")

    cuda_runtime = [e for e in all_events if e["cat"] in ("cuda_runtime", "Runtime") and e.get("dur", 0) > 0]
    if cuda_runtime:
        cuda_rt_names = Counter(e["name"] for e in cuda_runtime)
        print(f"\nCUDA runtime calls (total: {len(cuda_runtime)}):")
        for name, count in cuda_rt_names.most_common(20):
            durs = [e["dur"] for e in cuda_runtime if e["name"] == name]
            print(f"  {name}: count={count}, total={sum(durs)/1000:.1f}ms, avg={sum(durs)/len(durs):.1f}us")

    # --- Find synchronization points ---
    print(f"\n{'='*60}")
    print("SYNCHRONIZATION / BLOCKING ANALYSIS")
    print(f"{'='*60}")

    sync_events = [e for e in all_events if any(kw in e["name"].lower() for kw in
        ["synchronize", "cudadevicesync", "cudastreamsync", "cudamemcpy", "wait", "barrier",
         "cudaeventsynch", "cudastreamwait"])]

    if sync_events:
        sync_names = Counter(e["name"] for e in sync_events)
        print(f"\nSync/blocking events (total: {len(sync_events)}):")
        for name, count in sync_names.most_common(20):
            durs = [e["dur"] for e in sync_events if e["name"] == name and e["dur"] > 0]
            if durs:
                print(f"  {name}: count={count}, total={sum(durs)/1000:.1f}ms, avg={sum(durs)/len(durs):.1f}us, max={max(durs)/1000:.2f}ms")
            else:
                print(f"  {name}: count={count}")

    # --- FP8 specific operations ---
    print(f"\n{'='*60}")
    print("FP8 / AMAX / SCALING ANALYSIS")
    print(f"{'='*60}")

    fp8_events = [e for e in all_events if any(kw in e["name"].lower() for kw in
        ["fp8", "amax", "scale", "cast", "quantiz", "dequantiz", "e4m3", "e5m2", "float8"])]

    if fp8_events:
        fp8_names = Counter(e["name"] for e in fp8_events)
        print(f"\nFP8-related events (total: {len(fp8_events)}):")
        for name, count in fp8_names.most_common(30):
            durs = [e["dur"] for e in fp8_events if e["name"] == name and e["dur"] > 0]
            if durs:
                print(f"  {name}: count={count}, total={sum(durs)/1000:.1f}ms, avg={sum(durs)/len(durs):.1f}us")
            else:
                print(f"  {name}: count={count}")
    else:
        print("No FP8-specific events found.")

    # --- Identify the optimizer step time window and what happens inside ---
    print(f"\n{'='*60}")
    print("OPTIMIZER STEP TIME WINDOW DEEP DIVE")
    print(f"{'='*60}")

    # Find the top-level optimizer step event
    step_events = [e for e in all_events if "optimizer" in e["name"].lower() and "step" in e["name"].lower()]
    if not step_events:
        step_events = [e for e in all_events if e["name"].lower().strip() == "optimizer.step"]
    if not step_events:
        step_events = [e for e in all_events if "step" in e["name"].lower() and e.get("dur", 0) > 100000]  # > 100ms

    if step_events:
        # Take the longest step event as the representative
        step_event = max(step_events, key=lambda e: e.get("dur", 0))
        step_start = step_event["ts"]
        step_end = step_start + step_event["dur"]
        print(f"\nOptimizer step duration: {step_event['dur']/1000:.1f}ms")
        print(f"  Name: {step_event['name']}")
        print(f"  Time window: {step_start} - {step_end}")

        # Find all events within this time window
        events_in_step = [e for e in all_events
                         if e["ts"] >= step_start and e["ts"] <= step_end
                         and e is not step_event]

        # Categorize events inside optimizer step
        step_cats = Counter(e["cat"] for e in events_in_step)
        print(f"\n  Events inside optimizer step: {len(events_in_step)}")
        print(f"  Categories:")
        for cat, count in step_cats.most_common():
            total_dur = sum(e["dur"] for e in events_in_step if e["cat"] == cat and e["dur"] > 0)
            print(f"    {cat}: {count} events, total {total_dur/1000:.1f}ms")

        # Top ops inside step
        step_op_names = Counter(e["name"] for e in events_in_step if e.get("dur", 0) > 0)
        print(f"\n  Top operations inside optimizer step:")
        for name, count in step_op_names.most_common(30):
            durs = [e["dur"] for e in events_in_step if e["name"] == name and e["dur"] > 0]
            if durs:
                print(f"    {name}: count={count}, total={sum(durs)/1000:.1f}ms, avg={sum(durs)/len(durs):.1f}us")
    else:
        print("Could not identify optimizer step time window.")
        # Try user annotations
        annotations = [e for e in all_events if e["cat"] in ("user_annotation", "Trace")]
        if annotations:
            ann_names = Counter(e["name"] for e in annotations)
            print("\nUser annotations found:")
            for name, count in ann_names.most_common(20):
                durs = [e["dur"] for e in annotations if e["name"] == name and e["dur"] > 0]
                if durs:
                    print(f"  {name}: count={count}, total={sum(durs)/1000:.1f}ms")
                else:
                    print(f"  {name}: count={count}")

    # --- Overall time breakdown ---
    print(f"\n{'='*60}")
    print("OVERALL TIME BREAKDOWN (top-level annotations)")
    print(f"{'='*60}")

    annotations = [e for e in all_events if e["cat"] in ("user_annotation", "Trace", "python_function")
                   and e.get("dur", 0) > 10000]  # > 10ms
    if annotations:
        ann_sorted = sorted(annotations, key=lambda e: e["dur"], reverse=True)
        for e in ann_sorted[:30]:
            print(f"  {e['name']}: {e['dur']/1000:.1f}ms (cat={e['cat']})")

    # --- elementwise kernel analysis ---
    print(f"\n{'='*60}")
    print("ELEMENTWISE / SMALL KERNEL PATTERN")
    print(f"{'='*60}")

    elementwise = [e for e in all_events if any(kw in e["name"].lower() for kw in
        ["elementwise", "vectorized", "fill", "copy", "mul_", "add_", "div_", "where",
         "clamp", "abs", "sqrt", "rsqrt", "lerp", "addcmul", "addcdiv",
         "aten::mul", "aten::add", "aten::div", "aten::copy", "aten::fill",
         "foreach", "_foreach"]) and e.get("dur", 0) > 0]

    if elementwise:
        ew_names = Counter(e["name"] for e in elementwise)
        total_ew_time = sum(e["dur"] for e in elementwise) / 1000
        print(f"\nElementwise/small ops (total: {len(elementwise)}, total time: {total_ew_time:.1f}ms):")
        for name, count in ew_names.most_common(30):
            durs = [e["dur"] for e in elementwise if e["name"] == name]
            print(f"  {name}: count={count}, total={sum(durs)/1000:.1f}ms, avg={sum(durs)/len(durs):.1f}us")


if __name__ == "__main__":
    traces = sorted(Path("/workspace/nanoplm/output/pretraining_checkpoints").glob("*/profiler_traces/chrome_trace.json"))

    if not traces:
        print("No trace files found!")
        sys.exit(1)

    # Analyze the most recent/largest trace (likely most complete)
    # Use the last one (most recent run)
    print(f"Found {len(traces)} trace files:")
    for t in traces:
        print(f"  {t} ({t.stat().st_size / 1024 / 1024:.1f} MB)")

    # Analyze the latest trace
    analyze_trace(str(traces[-1]))
