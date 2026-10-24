"""splitmix64, the stateless 64-bit hash the cluster member draw and the mixture's
per-pass permutations are keyed on. Scalar and numpy versions give the same values."""
from __future__ import annotations

import numpy as np

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
C1 = 0xBF58476D1CE4E5B9
C2 = 0x94D049BB133111EB


def splitmix64(z: int) -> int:
    """splitmix64 output function applied to z + gamma (one full avalanche)."""
    z = (z + GAMMA) & MASK64
    z = ((z ^ (z >> 30)) * C1) & MASK64
    z = ((z ^ (z >> 27)) * C2) & MASK64
    return z ^ (z >> 31)


def splitmix64_vec(z: np.ndarray) -> np.ndarray:
    """``splitmix64`` over a uint64 array (wraps mod 2**64; call under
    ``np.errstate(over="ignore")``)."""
    z = z + np.uint64(GAMMA)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(C1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(C2)
    return z ^ (z >> np.uint64(31))
