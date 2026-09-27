from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample random next-token prediction sequences from a 1D token array."""
    if dataset.ndim != 1:
        raise ValueError("dataset must be a one-dimensional token array")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive")
    if len(dataset) <= context_length:
        raise ValueError("dataset must contain more than context_length tokens")

    starts = torch.randint(0, len(dataset) - context_length, (batch_size,)).tolist()
    sequences = np.stack([np.asarray(dataset[start : start + context_length + 1]) for start in starts])
    batch = torch.as_tensor(sequences, dtype=torch.long, device=device)
    return batch[:, :-1], batch[:, 1:]


def load_token_dataset(
    path: str | os.PathLike[str],
    dtype: str | np.dtype = np.uint16,
) -> npt.NDArray:
    """Open a .npy or raw token file without eagerly loading it into memory."""
    path_string = os.fspath(path)
    if path_string.endswith(".npy"):
        dataset = np.load(path_string, mmap_mode="r")
    else:
        dataset = np.memmap(path_string, mode="r", dtype=dtype)
    if dataset.ndim != 1:
        raise ValueError("token dataset must be one-dimensional")
    return dataset
