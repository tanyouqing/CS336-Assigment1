from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor


class AdamW(torch.optim.Optimizer):
    """Adam with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            learning_rate = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                parameter.mul_(1.0 - learning_rate * weight_decay)
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)

                adjusted_learning_rate = learning_rate * math.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)
                denominator = exp_avg_sq.sqrt().add_(eps)
                parameter.addcdiv_(exp_avg, denominator, value=-adjusted_learning_rate)

        return loss


def get_lr_cosine_schedule(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine learning-rate decay with linear warmup and a final floor."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    if warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    if cosine_cycle_iters <= warmup_iters:
        raise ValueError("cosine_cycle_iters must be greater than warmup_iters")
    if max_learning_rate < 0 or min_learning_rate < 0:
        raise ValueError("learning rates must be non-negative")
    if min_learning_rate > max_learning_rate:
        raise ValueError("min_learning_rate cannot exceed max_learning_rate")

    if iteration < warmup_iters:
        return iteration / warmup_iters * max_learning_rate
    if iteration <= cosine_cycle_iters:
        progress = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)
    return min_learning_rate


@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip the global L2 norm of all available parameter gradients in place."""
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return

    norm_device = gradients[0].device
    squared_norms = [
        gradient.detach().to(device=norm_device, dtype=torch.float32).square().sum() for gradient in gradients
    ]
    global_norm = torch.sqrt(torch.stack(squared_norms).sum())
    scaling_factor = max_l2_norm / (global_norm + 1e-6)

    if scaling_factor < 1.0:
        for gradient in gradients:
            gradient.mul_(scaling_factor.to(device=gradient.device, dtype=gradient.dtype))
