"""
SharedMemory Loading Benchmark
Compares the old approach (pickle full arrays across process boundary)
vs the new approach (create SharedMemory in worker, return metadata only).

Usage:
    python tests/shm_loading_benchmark.py
    python tests/shm_loading_benchmark.py --num-shards 10 --seqs-per-shard 50000 --max-len 512
"""

import os
import time
import argparse
import tempfile
import tracemalloc
import numpy as np
import h5py
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory


# ---------------------------------------------------------------------------
# Synthetic shard creation
# ---------------------------------------------------------------------------

def create_synthetic_shards(tmp_dir, num_shards, seqs_per_shard, max_len):
    """Create synthetic HDF5 shards with random uint8 data."""
    shard_paths = []
    for i in range(num_shards):
        path = tmp_dir / f"shard_{i:04d}.h5"
        with h5py.File(path, "w") as f:
            ds = f.create_dataset(
                "input_ids", (seqs_per_shard,),
                dtype=h5py.special_dtype(vlen=np.uint8),
            )
            for j in range(seqs_per_shard):
                seq_len = np.random.randint(50, max_len + 1)
                ds[j] = np.random.randint(1, 30, size=seq_len, dtype=np.uint8)
        shard_paths.append(path)
    return shard_paths


# ---------------------------------------------------------------------------
# OLD approach: worker returns full arrays (pickled across process boundary)
# ---------------------------------------------------------------------------

def _worker_old(path_str: str):
    """Old approach: read HDF5, return full numpy arrays."""
    path = Path(path_str)
    with h5py.File(path, "r") as f:
        n = len(f["input_ids"])
        shard_arrays = [np.array(f["input_ids"][i], dtype=np.uint8) for i in range(n)]
        max_len = max(a.shape[0] for a in shard_arrays)
        inputs = np.zeros((n, max_len), dtype=np.uint8)
        masks = np.zeros((n, max_len), dtype=np.uint8)
        for i, arr in enumerate(shard_arrays):
            inputs[i, : len(arr)] = arr
            masks[i, : len(arr)] = 1
    return inputs, masks


def load_old(shard_paths, max_workers):
    """Old approach: workers return arrays, parent copies into SharedMemory."""
    num_shards = len(shard_paths)
    futures_data = [None] * num_shards

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(_worker_old, str(p)): idx
            for idx, p in enumerate(shard_paths)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            futures_data[idx] = fut.result()

    shm_blocks = []
    for inputs_arr, masks_arr in futures_data:
        shm_in = shared_memory.SharedMemory(create=True, size=inputs_arr.nbytes)
        shm_mask = shared_memory.SharedMemory(create=True, size=masks_arr.nbytes)
        np.copyto(
            np.ndarray(inputs_arr.shape, dtype=np.uint8, buffer=shm_in.buf),
            inputs_arr,
        )
        np.copyto(
            np.ndarray(masks_arr.shape, dtype=np.uint8, buffer=shm_mask.buf),
            masks_arr,
        )
        shm_blocks.append((shm_in, shm_mask, inputs_arr.shape))

    return shm_blocks


# ---------------------------------------------------------------------------
# NEW approach: worker creates SharedMemory, returns metadata only
# ---------------------------------------------------------------------------

def _worker_new(path_str: str):
    """New approach: read HDF5 directly into SharedMemory, return metadata."""
    path = Path(path_str)
    with h5py.File(path, "r") as f:
        n = len(f["input_ids"])
        shard_arrays = [np.array(f["input_ids"][i], dtype=np.uint8) for i in range(n)]
        max_len = max(a.shape[0] for a in shard_arrays)

    shape = (n, max_len)
    nbytes = n * max_len

    shm_in = shared_memory.SharedMemory(create=True, size=nbytes)
    shm_mask = shared_memory.SharedMemory(create=True, size=nbytes)

    inputs = np.ndarray(shape, dtype=np.uint8, buffer=shm_in.buf)
    masks = np.ndarray(shape, dtype=np.uint8, buffer=shm_mask.buf)
    inputs[:] = 0
    masks[:] = 0

    for i, arr in enumerate(shard_arrays):
        inputs[i, : len(arr)] = arr
        masks[i, : len(arr)] = 1

    shm_in_name, shm_mask_name = shm_in.name, shm_mask.name
    shm_in.close()
    shm_mask.close()

    return shm_in_name, shm_mask_name, shape


def load_new(shard_paths, max_workers):
    """New approach: workers create SharedMemory, parent attaches by name."""
    num_shards = len(shard_paths)
    futures_data = [None] * num_shards

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(_worker_new, str(p)): idx
            for idx, p in enumerate(shard_paths)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            futures_data[idx] = fut.result()

    shm_blocks = []
    for shm_in_name, shm_mask_name, shape in futures_data:
        shm_in = shared_memory.SharedMemory(name=shm_in_name, create=False)
        shm_mask = shared_memory.SharedMemory(name=shm_mask_name, create=False)
        shm_blocks.append((shm_in, shm_mask, shape))

    return shm_blocks


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_shm_blocks(shm_blocks):
    for shm_in, shm_mask, _ in shm_blocks:
        shm_in.close()
        shm_in.unlink()
        shm_mask.close()
        shm_mask.unlink()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(label, load_fn, shard_paths, max_workers):
    """Run a single benchmark: measure wall time and peak parent memory."""
    tracemalloc.start()

    t0 = time.perf_counter()
    shm_blocks = load_fn(shard_paths, max_workers)
    elapsed = time.perf_counter() - t0

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Compute total data in SharedMemory
    total_shm_bytes = sum(
        shm_in.size + shm_mask.size for shm_in, shm_mask, _ in shm_blocks
    )

    cleanup_shm_blocks(shm_blocks)

    peak_mb = peak_bytes / (1024 * 1024)
    shm_mb = total_shm_bytes / (1024 * 1024)

    print(f"  {label}")
    print(f"    Wall time:            {elapsed:.2f}s")
    print(f"    Parent peak memory:   {peak_mb:.1f} MB")
    print(f"    SharedMemory total:   {shm_mb:.1f} MB")
    print(f"    Overhead (peak-shm):  {peak_mb - shm_mb:.1f} MB")
    print()

    return {
        "label": label,
        "wall_time": elapsed,
        "peak_mb": peak_mb,
        "shm_mb": shm_mb,
        "overhead_mb": peak_mb - shm_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="SharedMemory loading benchmark")
    parser.add_argument("--num-shards", type=int, default=5,
                        help="Number of HDF5 shards (default: 5)")
    parser.add_argument("--seqs-per-shard", type=int, default=10000,
                        help="Sequences per shard (default: 10000)")
    parser.add_argument("--max-len", type=int, default=512,
                        help="Max sequence length (default: 512)")
    parser.add_argument("--max-workers", type=int, default=0,
                        help="Max parallel workers (default: auto)")
    args = parser.parse_args()

    num_shards = args.num_shards
    seqs_per_shard = args.seqs_per_shard
    max_len = args.max_len

    if args.max_workers > 0:
        max_workers = args.max_workers
    else:
        max_workers = max(1, min((os.cpu_count() or 1) // 2, num_shards))

    estimated_mb = num_shards * seqs_per_shard * max_len * 2 / (1024 * 1024)

    print("=" * 60)
    print("SharedMemory Loading Benchmark")
    print("=" * 60)
    print(f"  Shards:           {num_shards}")
    print(f"  Seqs per shard:   {seqs_per_shard:,}")
    print(f"  Max seq length:   {max_len}")
    print(f"  Workers:          {max_workers}")
    print(f"  Est. data size:   ~{estimated_mb:.0f} MB "
          f"({num_shards} x {seqs_per_shard} x {max_len} x 2 arrays x uint8)")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        print("Creating synthetic shards...")
        shard_paths = create_synthetic_shards(
            tmp_dir, num_shards, seqs_per_shard, max_len,
        )
        print(f"  Created {len(shard_paths)} shards in {tmp_dir}")
        print()

        print("-" * 60)
        old = run_benchmark(
            "OLD (pickle arrays -> copy to SharedMemory)",
            load_old, shard_paths, max_workers,
        )

        print("-" * 60)
        new = run_benchmark(
            "NEW (create SharedMemory in worker)",
            load_new, shard_paths, max_workers,
        )

        # Summary
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Wall time:   {old['wall_time']:.2f}s -> {new['wall_time']:.2f}s "
              f"({new['wall_time'] / old['wall_time']:.2f}x)")
        print(f"  Peak mem:    {old['peak_mb']:.1f} MB -> {new['peak_mb']:.1f} MB "
              f"(saved {old['peak_mb'] - new['peak_mb']:.1f} MB)")
        print(f"  Overhead:    {old['overhead_mb']:.1f} MB -> {new['overhead_mb']:.1f} MB")
        print()


if __name__ == "__main__":
    main()
