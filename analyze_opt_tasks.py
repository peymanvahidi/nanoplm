#!/usr/bin/env python3
"""Count unique compiled graph calls inside one optimizer step to understand task structure."""

import json
from collections import Counter
from pathlib import Path

traces = sorted(Path("/workspace/nanoplm/output/pretraining_checkpoints").glob("*/profiler_traces/chrome_trace.json"))
trace_path = str(traces[-1])

with open(trace_path) as f:
    data = json.load(f)
events = data.get("traceEvents", [])

dur_events = [e for e in events if e.get("ph") == "X" and e.get("dur", 0) > 0]

# Find one optimizer step
step_events = [e for e in dur_events if e.get("name") == "Optimizer.step#NorMuon.step"]
step = max(step_events, key=lambda e: e["dur"])
step_start = step["ts"]
step_end = step_start + step["dur"]

# Get annotations inside step - these correspond to compiled graph calls (= task invocations)
annots = [e for e in dur_events
          if e.get("cat") == "user_annotation"
          and e["ts"] >= step_start and e["ts"] <= step_end
          and "CompiledFxGraph" in e.get("name", "")]

annots_sorted = sorted(annots, key=lambda e: e["ts"])

# Count unique graph hashes
graph_counter = Counter(e["name"] for e in annots)
print(f"Compiled graph invocations inside one opt step: {len(annots)}")
print(f"Unique compiled graphs: {len(graph_counter)}")
print()
for name, count in graph_counter.most_common():
    durs = [e["dur"] for e in annots if e["name"] == name]
    print(f"  {name}: {count}x, total={sum(durs)/1000:.1f}ms, avg={sum(durs)/count/1000:.2f}ms")

# Also count all_to_all inside step
a2a = [e for e in dur_events
       if "all_to_all" in e.get("name", "").lower()
       and e["ts"] >= step_start and e["ts"] <= step_end]
print(f"\nall_to_all calls inside step: {len(a2a)}")

# Count addmm (Newton-Schulz GEMMs) inside step
addmm = [e for e in dur_events
         if "addmm" in e.get("name", "").lower()
         and e["ts"] >= step_start and e["ts"] <= step_end]
print(f"addmm calls inside step: {len(addmm)}")

# Count NS kernels
ns = [e for e in dur_events
      if e.get("cat") == "kernel"
      and ("ns_line" in e.get("name", ""))
      and e["ts"] >= step_start and e["ts"] <= step_end]
print(f"ns_line kernels inside step: {len(ns)}")

# What's the sequential structure? Count tasks by looking at yield points (all_to_all waits)
print(f"\n--- Task sequential structure (first 20 annotations by time) ---")
for a in annots_sorted[:20]:
    rel_ts = (a["ts"] - step_start) / 1000
    print(f"  t={rel_ts:.2f}ms dur={a['dur']/1000:.2f}ms {a['name']}")
