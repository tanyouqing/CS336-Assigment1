from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch, load_token_dataset
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping


@dataclass
class TrainingConfig:
    """Serializable configuration for a complete language-model training run."""

    train_data: str
    validation_data: str
    output_dir: str
    experiment_name: str = "baseline"
    dataset_dtype: str = "uint16"
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1_344
    rope_theta: float = 10_000.0
    normalization: Literal["pre", "post", "none"] = "pre"
    use_rope: bool = True
    ffn_type: Literal["swiglu", "silu"] = "swiglu"
    batch_size: int = 32
    max_steps: int = 5_000
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 100
    cosine_cycle_iters: int | None = None
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 10
    checkpoint_interval: int = 500
    seed: int = 42
    device: str = "auto"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    resume_from: str | None = None

    def validate(self) -> None:
        positive_integer_fields = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "eval_interval": self.eval_interval,
            "eval_batches": self.eval_batches,
            "log_interval": self.log_interval,
            "checkpoint_interval": self.checkpoint_interval,
        }
        invalid = [name for name, value in positive_integer_fields.items() if value <= 0]
        if invalid:
            raise ValueError(f"The following fields must be positive: {', '.join(invalid)}")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.normalization not in {"pre", "post", "none"}:
            raise ValueError(f"Unsupported normalization: {self.normalization}")
        if self.ffn_type not in {"swiglu", "silu"}:
            raise ValueError(f"Unsupported ffn_type: {self.ffn_type}")

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> TrainingConfig:
        with open(path, encoding="utf-8") as config_file:
            return cls(**json.load(config_file))

    def to_json(self, path: str | os.PathLike[str]) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as config_file:
            json.dump(asdict(self), config_file, indent=2, ensure_ascii=False)
            config_file.write("\n")


class ExperimentLogger:
    """Write console summaries and machine-readable JSON Lines records."""

    def __init__(self, log_path: str | os.PathLike[str]) -> None:
        self.start_time = time.perf_counter()
        self.log_file = open(log_path, "a", encoding="utf-8")

    def log(self, event: str, step: int, **metrics: float | int | str) -> None:
        record = {
            "event": event,
            "step": step,
            "wall_time_seconds": time.perf_counter() - self.start_time,
            **metrics,
        }
        line = json.dumps(record, ensure_ascii=False)
        print(line, flush=True)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()

    def __enter__(self) -> ExperimentLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str) -> torch.dtype:
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtypes[dtype]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {dtype}") from error


def build_model(
    config: TrainingConfig,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> TransformerLM:
    """Construct the configured baseline or ablation model."""
    return TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        normalization=config.normalization,
        use_rope=config.use_rope,
        ffn_type=config.ffn_type,
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
    num_batches: int,
) -> float:
    """Estimate mean per-token validation loss over sampled batches."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        total_loss += cross_entropy(model(inputs), targets).item()
    model.train(was_training)
    return total_loss / num_batches


def train(config: TrainingConfig) -> TransformerLM:
    """Run a configurable training job and return the final model."""
    config.validate()
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    run_dir = Path(config.output_dir) / config.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(run_dir / "config.json")

    train_data = load_token_dataset(config.train_data, config.dataset_dtype)
    validation_data = load_token_dataset(config.validation_data, config.dataset_dtype)
    model = build_model(config, device=device, dtype=dtype)
    optimizer = AdamW(
        model.parameters(),
        lr=config.max_learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )

    start_iteration = 0
    if config.resume_from is not None:
        start_iteration = load_checkpoint(config.resume_from, model, optimizer)

    cosine_cycle_iters = config.cosine_cycle_iters or config.max_steps
    tokens_per_step = config.batch_size * config.context_length
    model.train()

    with ExperimentLogger(run_dir / "metrics.jsonl") as logger:
        logger.log("start", start_iteration, device=str(device), dtype=config.dtype)
        for iteration in range(start_iteration, config.max_steps):
            learning_rate = get_lr_cosine_schedule(
                iteration,
                config.max_learning_rate,
                config.min_learning_rate,
                config.warmup_iters,
                cosine_cycle_iters,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            inputs, targets = get_batch(
                train_data,
                config.batch_size,
                config.context_length,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy(model(inputs), targets)
            loss.backward()
            gradient_clipping(model.parameters(), config.max_grad_norm)
            optimizer.step()

            completed_steps = iteration + 1
            if completed_steps % config.log_interval == 0:
                logger.log(
                    "train",
                    completed_steps,
                    train_loss=loss.item(),
                    learning_rate=learning_rate,
                    tokens_processed=completed_steps * tokens_per_step,
                )

            if completed_steps % config.eval_interval == 0:
                validation_loss = evaluate(
                    model,
                    validation_data,
                    config.batch_size,
                    config.context_length,
                    device,
                    config.eval_batches,
                )
                logger.log(
                    "validation",
                    completed_steps,
                    validation_loss=validation_loss,
                    perplexity=math_exp(validation_loss),
                    tokens_processed=completed_steps * tokens_per_step,
                )

            if completed_steps % config.checkpoint_interval == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    completed_steps,
                    run_dir / f"checkpoint_{completed_steps:07d}.pt",
                )

        save_checkpoint(model, optimizer, config.max_steps, run_dir / "checkpoint_final.pt")
        logger.log("complete", config.max_steps, tokens_processed=config.max_steps * tokens_per_step)
    return model


def math_exp(value: float) -> float:
    """Exponentiate a scalar validation loss without introducing NumPy scalars."""
    return float(np.exp(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CS336 Transformer language model")
    parser.add_argument("--config", required=True, help="Path to a TrainingConfig JSON file")
    args = parser.parse_args()
    train(TrainingConfig.from_json(args.config))


if __name__ == "__main__":
    main()
