from __future__ import annotations

import os
from typing import IO, BinaryIO

import torch


CheckpointTarget = str | os.PathLike[str] | BinaryIO | IO[bytes]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: CheckpointTarget,
) -> None:
    """Serialize model, optimizer, and iteration state."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: CheckpointTarget,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model and optimizer state and return the saved iteration."""
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    checkpoint = torch.load(src, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])


def load_model_checkpoint(src: CheckpointTarget, model: torch.nn.Module) -> int:
    """Restore only model weights for evaluation or text generation."""
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    checkpoint = torch.load(src, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return int(checkpoint["iteration"])
