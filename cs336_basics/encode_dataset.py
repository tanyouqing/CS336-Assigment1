from __future__ import annotations

import argparse
import os

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def encode_dataset(
    tokenizer: Tokenizer,
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    dtype: str | np.dtype = np.uint16,
    buffer_size: int = 1_000_000,
) -> int:
    """Stream a text corpus into a raw, memory-mappable token-ID file."""
    if buffer_size <= 0:
        raise ValueError("buffer_size must be positive")
    output_dtype = np.dtype(dtype)
    if not np.issubdtype(output_dtype, np.integer):
        raise ValueError("token output dtype must be an integer dtype")
    max_token_id = max(tokenizer.vocab, default=-1)
    if max_token_id > np.iinfo(output_dtype).max:
        raise ValueError(f"Token ID {max_token_id} does not fit in {output_dtype}")

    total_tokens = 0
    buffer: list[int] = []
    with open(input_path, encoding="utf-8") as input_file, open(output_path, "wb") as output_file:
        for token_id in tokenizer.encode_iterable(input_file):
            buffer.append(token_id)
            if len(buffer) >= buffer_size:
                token_array = np.asarray(buffer, dtype=output_dtype)
                output_file.write(token_array.tobytes())
                total_tokens += len(buffer)
                buffer.clear()
        if buffer:
            token_array = np.asarray(buffer, dtype=output_dtype)
            output_file.write(token_array.tobytes())
            total_tokens += len(buffer)
    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a corpus into a raw token-ID file")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--dtype", default="uint16")
    parser.add_argument("--special-token", action="append", default=[])
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_token)
    total_tokens = encode_dataset(tokenizer, args.input, args.output, dtype=args.dtype)
    print(f"Encoded {total_tokens} tokens to {args.output}")


if __name__ == "__main__":
    main()
