"""
Optimizer factory for MyPLM.

Currently supports AdamW by default. Designed to be easily extended to other
optimizers (e.g., Muon) without touching training pipeline code.
"""

from typing import Iterable, Mapping, Any

import torch


def get_optimizer(
    name: str,
    params: Iterable[Mapping[str, Any]] | Iterable[torch.nn.Parameter],
    *,
    lr: float,
    weight_decay: float | None = None,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """
    Create and return a torch optimizer by name.

    Args:
        name: Optimizer name (case-insensitive), e.g. "adamw", "muon".
        params: Iterable of parameters or param groups.
        lr: Learning rate.
        weight_decay: Optional weight decay; if None, uses optimizer default.
        **kwargs: Extra keyword args forwarded to the optimizer ctor (e.g., betas, eps).

    Returns:
        torch.optim.Optimizer instance.

    Raises:
        ValueError: If the optimizer name is unknown.
        ImportError: If a 3rd-party optimizer (e.g., Muon) is requested but not installed.
    """

    key = (name or "").strip().lower()

    if key in {"adamw", "adam_w"}:
        # Use PyTorch's AdamW
        kwargs_local = dict(kwargs)
        if weight_decay is not None:
            kwargs_local.setdefault("weight_decay", weight_decay)
        return torch.optim.AdamW(params, lr=lr, **kwargs_local)

    if key == "muon":
        # Optional: support Muon if available. We don't add it as a hard dependency.
        try:
            # Example import paths; adjust based on the actual package when integrating.
            # from muon import Muon  # noqa: F401
            from muon import Muon  # type: ignore
        except Exception as e:  # ImportError or other
            raise ImportError(
                "Muon optimizer is not installed. Install the appropriate package and try again."
            ) from e

        kwargs_local = dict(kwargs)
        if weight_decay is not None:
            kwargs_local.setdefault("weight_decay", weight_decay)
        return Muon(params, lr=lr, **kwargs_local)  # type: ignore[name-defined]

    raise ValueError(f"Unknown optimizer: {name}")


__all__ = ["get_optimizer"]
