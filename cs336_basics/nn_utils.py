from __future__ import annotations

import torch
from torch import Tensor


def softmax(in_features: Tensor, dim: int) -> Tensor:
    """Apply a numerically stable softmax along ``dim``."""
    shifted = in_features - in_features.max(dim=dim, keepdim=True).values
    exponentials = torch.exp(shifted)
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """Return mean cross-entropy over all leading batch dimensions."""
    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")
    if inputs.shape[:-1] != targets.shape:
        raise ValueError("targets must match every input dimension except the vocabulary dimension")

    shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    log_normalizer = torch.log(torch.exp(shifted).sum(dim=-1))
    target_logits = torch.gather(shifted, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()
