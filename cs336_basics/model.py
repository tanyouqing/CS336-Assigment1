from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

from cs336_basics.nn_utils import softmax


class Linear(nn.Module):
    """A bias-free linear transformation with weights stored as (out, in)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("... i, o i -> ... o", x, self.weight)


class Embedding(nn.Module):
    """Map integer token IDs to learned embedding vectors."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("num_embeddings and embedding_dim must be positive")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root mean square normalization over the final tensor dimension."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if eps < 0:
            raise ValueError("eps must be non-negative")

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_float.square(), dim=-1, keepdim=True) + self.eps)
        normalized = x_float / rms
        return (normalized * self.weight.to(torch.float32)).to(input_dtype)


def silu(in_features: Tensor) -> Tensor:
    """Apply the SiLU/Swish activation elementwise."""
    return in_features * torch.sigmoid(in_features)


class SwiGLU(nn.Module):
    """The gated position-wise feed-forward network used by the Transformer."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_ff is None:
            approximate_d_ff = math.ceil(8 * d_model / 3)
            d_ff = 64 * math.ceil(approximate_d_ff / 64)
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")

        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFeedForward(nn.Module):
    """Two-layer SiLU feed-forward network used for the gating ablation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


class Identity(nn.Module):
    """Identity module used when normalization is ablated."""

    def forward(self, x: Tensor) -> Tensor:
        return x


class RotaryPositionalEmbedding(nn.Module):
    """Apply pairwise rotary position embeddings to query or key vectors."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if theta <= 0:
            raise ValueError("theta must be positive")
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k must be a positive even integer")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        dimension_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inverse_frequencies = theta ** (-dimension_indices / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.einsum("s, d -> s d", positions, inverse_frequencies)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError(f"Expected final dimension {self.d_k}, received {x.shape[-1]}")

        token_positions = token_positions.to(device=self.cos.device, dtype=torch.long)
        # Insert singleton batch dimensions when one position sequence is shared
        # across batches or attention heads.
        while token_positions.ndim < x.ndim - 1:
            token_positions = token_positions.unsqueeze(-2)

        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute scaled dot-product attention over arbitrary batch dimensions."""
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Query and key dimensions must match")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("Key and value sequence lengths must match")

    d_k = query.shape[-1]
    scores = torch.einsum("... q d, ... k d -> ... q k", query, key) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attention_weights = softmax(scores, dim=-1)
    return torch.einsum("... q k, ... k v -> ... q v", attention_weights, value)


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional rotary embeddings."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if (theta is None) != (max_seq_len is None):
            raise ValueError("theta and max_seq_len must either both be set or both be None")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = (
            RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device)
            if theta is not None and max_seq_len is not None
            else None
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        return x.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        return x.transpose(-3, -2).contiguous().flatten(-2)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        sequence_length = x.shape[-2]
        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)

        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=x.device,
        ).tril()
        attended = scaled_dot_product_attention(query, key, value, mask=causal_mask)
        return self.output_proj(self._merge_heads(attended))


class TransformerBlock(nn.Module):
    """A pre-norm Transformer block with attention and SwiGLU sublayers."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        normalization: Literal["pre", "post", "none"] = "pre",
        use_rope: bool = True,
        ffn_type: Literal["swiglu", "silu"] = "swiglu",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if normalization not in {"pre", "post", "none"}:
            raise ValueError(f"Unsupported normalization mode: {normalization}")
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError(f"Unsupported feed-forward type: {ffn_type}")

        self.normalization = normalization
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            theta=theta if use_rope else None,
            max_seq_len=max_seq_len if use_rope else None,
            device=device,
            dtype=dtype,
        )
        if normalization == "none":
            self.ln1 = Identity()
            self.ln2 = Identity()
        else:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = (
            SwiGLU(d_model, d_ff, device=device, dtype=dtype)
            if ffn_type == "swiglu"
            else SiLUFeedForward(d_model, d_ff, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.normalization == "pre":
            x = x + self.attn(self.ln1(x), token_positions)
            return x + self.ffn(self.ln2(x))
        if self.normalization == "post":
            x = self.ln1(x + self.attn(x, token_positions))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(x, token_positions)
        return x + self.ffn(x)


class TransformerLM(nn.Module):
    """A decoder-only Transformer language model that returns token logits."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        normalization: Literal["pre", "post", "none"] = "pre",
        use_rope: bool = True,
        ffn_type: Literal["swiglu", "silu"] = "swiglu",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    normalization=normalization,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if normalization == "pre" else Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(f"Input sequence length {sequence_length} exceeds context length {self.context_length}")

        token_positions = torch.arange(sequence_length, device=token_ids.device)
        hidden_states = self.token_embeddings(token_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, token_positions)
        return self.lm_head(self.ln_final(hidden_states))
