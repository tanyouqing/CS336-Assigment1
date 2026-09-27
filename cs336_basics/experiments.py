from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from cs336_basics.train import TrainingConfig


def tinystories_baseline(
    train_data: str,
    validation_data: str,
    output_dir: str,
) -> TrainingConfig:
    """Return the 17M non-embedding-parameter TinyStories configuration."""
    return TrainingConfig(
        train_data=train_data,
        validation_data=validation_data,
        output_dir=output_dir,
        experiment_name="tinystories_baseline",
        vocab_size=10_000,
        context_length=256,
        d_model=512,
        d_ff=1_344,
        num_layers=4,
        num_heads=16,
        rope_theta=10_000.0,
        batch_size=256,
        max_steps=5_000,
        cosine_cycle_iters=5_000,
    )


def openwebtext_baseline(
    train_data: str,
    validation_data: str,
    output_dir: str,
) -> TrainingConfig:
    """Return the matching-compute OpenWebText baseline configuration."""
    return replace(
        tinystories_baseline(train_data, validation_data, output_dir),
        experiment_name="owt_baseline",
        vocab_size=32_000,
    )


def ablation_configs(base: TrainingConfig) -> dict[str, TrainingConfig]:
    """Create the normalization, positional, and FFN ablations from one baseline."""
    return {
        "no_norm": replace(base, experiment_name=f"{base.experiment_name}_no_norm", normalization="none"),
        "post_norm": replace(base, experiment_name=f"{base.experiment_name}_post_norm", normalization="post"),
        "nope": replace(base, experiment_name=f"{base.experiment_name}_nope", use_rope=False),
        "silu_ffn": replace(
            base,
            experiment_name=f"{base.experiment_name}_silu_ffn",
            ffn_type="silu",
            d_ff=4 * base.d_model,
        ),
    }


def learning_rate_sweep(
    base: TrainingConfig,
    learning_rates: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3),
) -> dict[str, TrainingConfig]:
    return {
        f"lr_{learning_rate:g}": replace(
            base,
            experiment_name=f"{base.experiment_name}_lr_{learning_rate:g}",
            max_learning_rate=learning_rate,
            min_learning_rate=learning_rate / 10,
        )
        for learning_rate in learning_rates
    }


def batch_size_sweep(
    base: TrainingConfig,
    batch_sizes: tuple[int, ...] = (1, 16, 64, 128, 256),
) -> dict[str, TrainingConfig]:
    """Vary batch size while holding the approximate processed-token budget fixed."""
    token_budget = base.batch_size * base.max_steps * base.context_length
    configs: dict[str, TrainingConfig] = {}
    for batch_size in batch_sizes:
        max_steps = max(1, round(token_budget / (batch_size * base.context_length)))
        configs[f"batch_{batch_size}"] = replace(
            base,
            experiment_name=f"{base.experiment_name}_batch_{batch_size}",
            batch_size=batch_size,
            max_steps=max_steps,
            cosine_cycle_iters=max_steps,
        )
    return configs


def write_experiment_configs(
    configs: dict[str, TrainingConfig],
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    for name, config in configs.items():
        config_path = output_path / f"{name}.json"
        config.to_json(config_path)
        manifest[name] = str(config_path)
    with open(output_path / "manifest.json", "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write CS336 experiment configurations without running them")
    parser.add_argument("--dataset", choices=("tinystories", "owt"), required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--run-output-dir", required=True)
    parser.add_argument("--config-output-dir", required=True)
    parser.add_argument(
        "--suite",
        choices=("baseline", "ablations", "learning-rates", "batch-sizes", "all"),
        default="all",
    )
    args = parser.parse_args()

    factory = tinystories_baseline if args.dataset == "tinystories" else openwebtext_baseline
    baseline = factory(args.train_data, args.validation_data, args.run_output_dir)
    configs = {"baseline": baseline}
    if args.suite in {"ablations", "all"}:
        configs.update(ablation_configs(baseline))
    if args.suite in {"learning-rates", "all"}:
        configs.update(learning_rate_sweep(baseline))
    if args.suite in {"batch-sizes", "all"}:
        configs.update(batch_size_sweep(baseline))
    if args.suite != "all" and args.suite != "baseline":
        configs.pop("baseline")
    write_experiment_configs(configs, args.config_output_dir)


if __name__ == "__main__":
    main()
