"""MixtureDataset: several pretraining train sets drawn by weight (share of sequences).

One mixture epoch has ``N = sum(n_s)`` positions, n_s being each source's length
(sequences, or train clusters for a cluster source). Source s owns ``k_s`` of them,
``k_s = w_s * N`` rounded so the k_s sum to N (largest remainder): positions
``[start_s, start_s + k_s)``. The shares are exact in every mixture epoch; the packing
sampler shuffles the positions, so they hold in expectation within a part of one.

Each source reads its own stream. Mixture epoch E gives source s the stream items
``q = E * k_s + r`` for r < k_s, so pass ``p = q // n_s`` of that source covers every
item once before the next pass starts, across mixture-epoch boundaries. A source with
``k_s > n_s`` repeats within one mixture epoch, one with ``k_s < n_s`` takes part of a
pass. Inside a pass, item ``u = q % n_s`` is ``(a * u + b) mod n_s``, an affine
permutation whose a, b come from a splitmix64 hash of (seed, source name, pass): each
pass reads the source in a new order, and a partial pass a spread-out subset.

A cluster source draws its member with that source's own pass number ``p`` as the
epoch (ClusterSampledDataset.item_at), so every pass, repeats included, draws new
members.

Nothing here holds RNG state: an item is a pure function of (seed, mixture epoch,
position), so it does not depend on num_workers, world size, fetch order or resume.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

from nanoplm.data.datasets.cluster_sampled import ClusterSampledDataset
from nanoplm.data.datasets.pretrain import ShardedDataset
from nanoplm.data.datasets.splitmix import MASK64, splitmix64
from nanoplm.utils import logger

# b"MIXPRM": separates the per-pass permutation keys from the member stream
PERM_TAG = 0x4D_49_58_50_52_4D
_FINGERPRINT_TAG = b"mixture-v1"


def allocate(weights: Sequence[float], total: int) -> list:
    """Integer counts with ``sum == total`` closest to ``weights * total`` (largest
    remainder, ties to the earlier source). Every source gets at least one."""
    raw = [w * total for w in weights]
    counts = [int(math.floor(x)) for x in raw]
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in order[: total - sum(counts)]:
        counts[i] += 1
    for i, c in enumerate(counts):
        if c == 0:
            j = max(range(len(counts)), key=lambda t: counts[t])
            counts[j] -= 1
            counts[i] = 1
    return counts


def _name_key(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")


def source_key(seed: int, name: str) -> int:
    """The 64-bit key every pass permutation of source ``name`` starts from."""
    return splitmix64(((int(seed) ^ PERM_TAG) & MASK64) ^ _name_key(name))


def _affine(key: int, pass_: int, n: int) -> tuple:
    if n <= 1:
        return 1, 0
    k = splitmix64(key ^ (int(pass_) & MASK64))
    a = 1 + (k % (n - 1))
    while math.gcd(a, n) != 1:
        a = a % (n - 1) + 1
    b = splitmix64(k) % n
    return a, b


def pass_permutation(seed: int, name: str, pass_: int, n: int) -> tuple:
    """(a, b) of the affine permutation ``u -> (a*u + b) % n`` of one source's pass."""
    return _affine(source_key(seed, name), pass_, n)


class MixtureDataset(torch.utils.data.Dataset):
    """Map-style dataset over several train sets. See the module docstring."""

    def __init__(self, names: Sequence[str], datasets: Sequence, weights: Sequence[float],
                 seed: int, epoch: int = 0) -> None:
        if not (len(names) == len(datasets) == len(weights)) or len(names) < 2:
            raise ValueError("a mixture needs two or more (name, dataset, weight)")
        self.names = list(names)
        self.datasets = list(datasets)
        self.weights = [float(w) for w in weights]
        self.seed = int(seed)
        self._epoch = int(epoch)
        self.sizes = [len(d) for d in self.datasets]
        if min(self.sizes) == 0:
            empty = self.names[self.sizes.index(0)]
            raise ValueError(f"mixture source {empty} has no train sequences")
        self._n = sum(self.sizes)
        self.counts = allocate(self.weights, self._n)
        self.starts = np.concatenate([[0], np.cumsum(self.counts)]).astype(np.int64)
        self._keys = [source_key(self.seed, n) for n in self.names]
        self._perm_cache: dict = {}
        self._lengths_cache: Optional[tuple] = None

    # ---- the contract -----------------------------------------------------------
    def __len__(self) -> int:
        return self._n

    @property
    def epoch(self) -> int:
        return self._epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def is_cluster(self, s: int) -> bool:
        return isinstance(self.datasets[s], ClusterSampledDataset)

    def _perm(self, s: int, p: int) -> tuple:
        key = (s, p)
        got = self._perm_cache.get(key)
        if got is None:
            if len(self._perm_cache) > 4096:
                self._perm_cache.clear()
            got = _affine(self._keys[s], p, self.sizes[s])
            self._perm_cache[key] = got
        return got

    def locate(self, idx: int, epoch: Optional[int] = None) -> tuple:
        """(source, item, source pass) position ``idx`` reads in mixture epoch ``epoch``."""
        if idx < 0 or idx >= self._n:
            raise IndexError(f"Index {idx} out of bounds for a mixture of {self._n}")
        e = self._epoch if epoch is None else int(epoch)
        s = int(np.searchsorted(self.starts, idx, side="right")) - 1
        q = e * self.counts[s] + (idx - int(self.starts[s]))
        n = self.sizes[s]
        p, u = divmod(q, n)
        a, b = self._perm(s, p)
        return s, (a * u + b) % n, p

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s, j, p = self.locate(int(idx))
        ds = self.datasets[s]
        return ds.item_at(j, p) if self.is_cluster(s) else ds[j]

    def source_slots(self, s: int, epoch: Optional[int] = None) -> tuple:
        """(items, passes) of source s's k_s positions in mixture epoch ``epoch``."""
        e = self._epoch if epoch is None else int(epoch)
        n, k = self.sizes[s], self.counts[s]
        q = e * k + np.arange(k, dtype=np.int64)
        p, u = np.divmod(q, n)
        # one (a, b) per pass, then all slots at once: a small source can repeat
        # thousands of times in one mixture epoch. q is contiguous, so are the passes.
        which = p - p[0]
        ab = np.array([_affine(self._keys[s], int(pv), n)
                       for pv in range(int(p[0]), int(p[-1]) + 1)],
                      dtype=np.uint64).reshape(-1, 2)
        with np.errstate(over="ignore"):
            items = ((ab[which, 0] * u.astype(np.uint64) + ab[which, 1])
                     % np.uint64(n)).astype(np.int64)
        return items, p

    def get_all_sequence_lengths(self) -> np.ndarray:
        """Length of the sequence every position gives in the current mixture epoch."""
        key = (self._epoch, self.seed)
        if self._lengths_cache is None or self._lengths_cache[0] != key:
            parts = []
            for s, ds in enumerate(self.datasets):
                items, passes = self.source_slots(s)
                if self.is_cluster(s):
                    parts.append(ds.lengths_at(items, passes))
                else:
                    parts.append(np.asarray(ds.get_all_sequence_lengths())[items])
            self._lengths_cache = (key, np.concatenate(parts).astype(np.int32))
        return self._lengths_cache[1]

    # ---- pickling: drop the caches, keep the sources ----------------------------
    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_lengths_cache"] = None
        state["_perm_cache"] = {}
        return state

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(_FINGERPRINT_TAG)
        h.update((self.seed & MASK64).to_bytes(8, "little"))
        for name, ds, k in zip(self.names, self.datasets, self.counts):
            h.update(name.encode() + b"\0" + ds.fingerprint().encode()
                     + int(k).to_bytes(8, "little"))
        return h.hexdigest()[:16]


def open_mixture(dataset_dir: Union[str, Path], manifest, seed: int) -> MixtureDataset:
    """The MixtureDataset a mixture manifest describes; each source's manifest is
    checked against the hash the mixture recorded."""
    from nanoplm.data.datasets.cluster_sampled import open_train_dataset
    from nanoplm.data.infra.mixture import load_sources

    sources = load_sources(dataset_dir, manifest)
    datasets = [open_train_dataset(s.dir, s.manifest, seed=seed) for s in sources]
    mix = MixtureDataset([s.name for s in sources], datasets, [s.weight for s in sources],
                         seed=seed)
    logger.info("Mixture train set: " + ", ".join(
        f"{s.name} {s.weight:g} ({len(d):,} {'clusters' if s.kind == 'cluster' else 'seqs'})"
        for s, d in zip(sources, datasets)))
    return mix


def open_val_datasets(dataset_dir: Union[str, Path], manifest) -> dict:
    """{source name: validation ShardedDataset} of a mixture, in mixture order."""
    from nanoplm.data.infra.mixture import load_sources
    return {s.name: ShardedDataset(data_dir=str(s.dir / s.manifest.val_dir))
            for s in load_sources(dataset_dir, manifest)}
