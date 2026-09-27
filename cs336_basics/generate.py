from __future__ import annotations

import argparse

import torch
from torch import Tensor

from cs336_basics.checkpoint import load_model_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import softmax
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train import TrainingConfig, build_model, resolve_device, resolve_dtype


def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample token IDs from batched logits using temperature and nucleus sampling."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must lie in (0, 1]")

    probabilities = softmax(logits / temperature, dim=-1)
    if top_p == 1.0:
        return torch.multinomial(probabilities, num_samples=1, generator=generator)

    sorted_probabilities, sorted_indices = torch.sort(probabilities, dim=-1, descending=True)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
    remove = cumulative_probabilities - sorted_probabilities >= top_p
    sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
    sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(dim=-1, keepdim=True)
    sampled_sorted_indices = torch.multinomial(
        sorted_probabilities,
        num_samples=1,
        generator=generator,
    )
    return torch.gather(sorted_indices, dim=-1, index=sampled_sorted_indices)


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int] | Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Autoregressively extend one prompt and return prompt plus generated IDs."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    device = next(model.parameters()).device
    token_ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=device)
    if token_ids.ndim != 1 or token_ids.numel() == 0:
        raise ValueError("prompt_ids must be a non-empty one-dimensional sequence")
    token_ids = token_ids.unsqueeze(0)

    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        model_input = token_ids[:, -model.context_length :]
        next_token = sample_next_token(
            model(model_input)[:, -1, :],
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        token_ids = torch.cat((token_ids, next_token), dim=-1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    model.train(was_training)
    return token_ids.squeeze(0).tolist()


def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    eos_token: str | None = "<|endoftext|>",
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> str:
    """Encode a prompt, generate a completion, and decode it to text."""
    eos_token_id = None
    if eos_token is not None:
        eos_token_id = tokenizer.token_to_id.get(eos_token.encode("utf-8"))
    generated_ids = generate(
        model,
        tokenizer.encode(prompt),
        max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_p=top_p,
        generator=generator,
    )
    return tokenizer.decode(generated_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a CS336 checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    config = TrainingConfig.from_json(args.config)
    device = resolve_device(config.device)
    model = build_model(config, device=device, dtype=resolve_dtype(config.dtype))
    load_model_checkpoint(args.checkpoint, model)
    tokenizer = Tokenizer.from_files(
        args.vocab,
        args.merges,
        special_tokens=["<|endoftext|>"],
    )
    print(
        generate_text(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    )


if __name__ == "__main__":
    main()
