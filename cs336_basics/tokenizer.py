from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator

import regex


PRETOKEN_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _count_pretokens(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    """Split text at special-token boundaries and count UTF-8 byte pre-tokens."""
    if special_tokens:
        # Prefer a longer special token when one token is a prefix of another.
        delimiter = regex.compile(
            "|".join(regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
        )
        segments = delimiter.split(text)
    else:
        segments = (text,)

    counts: Counter[tuple[bytes, ...]] = Counter()
    for segment in segments:
        for match in PRETOKEN_PATTERN.finditer(segment):
            pretoken_bytes = match.group().encode("utf-8")
            counts[tuple(bytes([byte]) for byte in pretoken_bytes)] += 1
    return counts


def _merge_pair(symbols: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Merge every non-overlapping occurrence of pair, scanning left-to-right."""
    merged_symbol = pair[0] + pair[1]
    result: list[bytes] = []
    index = 0

    while index < len(symbols):
        if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
            result.append(merged_symbol)
            index += 2
        else:
            result.append(symbols[index])
            index += 1

    return tuple(result)


def _gpt2_byte_decoder() -> dict[str, int]:
    """Return the inverse of GPT-2's printable byte-to-Unicode mapping."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    code_points = byte_values.copy()
    next_code_point_offset = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            code_points.append(256 + next_code_point_offset)
            next_code_point_offset += 1
    return {chr(code_point): byte_value for byte_value, code_point in zip(byte_values, code_points, strict=True)}


class Tokenizer:
    """A byte-level BPE tokenizer with optional indivisible special tokens."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])

        if any(token == "" for token in self.special_tokens):
            raise ValueError("Special tokens must be non-empty strings")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("Special tokens must be unique")

        token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        next_token_id = max(self.vocab, default=-1) + 1
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in token_to_id:
                self.vocab[next_token_id] = token_bytes
                token_to_id[token_bytes] = next_token_id
                next_token_id += 1

        self.token_to_id = token_to_id
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_token_to_id = {
            token: self.token_to_id[token.encode("utf-8")] for token in self.special_tokens
        }

        if self.special_tokens:
            alternatives = "|".join(
                regex.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)
            )
            self._special_token_pattern: regex.Pattern[str] | None = regex.compile(alternatives)
        else:
            self._special_token_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike[str],
        merges_filepath: str | os.PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Load GPT-2-style JSON vocabulary and text merge files."""
        byte_decoder = _gpt2_byte_decoder()

        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            serialized_vocab: dict[str, int] = json.load(vocab_file)
        vocab = {
            token_id: bytes(byte_decoder[character] for character in serialized_token)
            for serialized_token, token_id in serialized_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as merges_file:
            for line in merges_file:
                parts = line.rstrip().split(" ")
                if len(parts) != 2:
                    continue
                left, right = parts
                merges.append(
                    (
                        bytes(byte_decoder[character] for character in left),
                        bytes(byte_decoder[character] for character in right),
                    )
                )

        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken: str) -> Iterator[int]:
        symbols = tuple(bytes([byte]) for byte in pretoken.encode("utf-8"))

        while len(symbols) > 1:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank = len(self.merge_ranks)
            for pair in zip(symbols, symbols[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break
            symbols = _merge_pair(symbols, best_pair)

        for symbol in symbols:
            yield self.token_to_id[symbol]

    def _encode_ordinary_text(self, text: str) -> Iterator[int]:
        for match in PRETOKEN_PATTERN.finditer(text):
            yield from self._encode_pretoken(match.group())

    def _encode_text(self, text: str) -> Iterator[int]:
        if self._special_token_pattern is None:
            yield from self._encode_ordinary_text(text)
            return

        previous_end = 0
        for match in self._special_token_pattern.finditer(text):
            yield from self._encode_ordinary_text(text[previous_end : match.start()])
            yield self.special_token_to_id[match.group()]
            previous_end = match.end()
        yield from self._encode_ordinary_text(text[previous_end:])

    def encode(self, text: str) -> list[int]:
        """Encode text into BPE token IDs."""
        return list(self._encode_text(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode strings from an iterable without collecting all IDs."""
        for text in iterable:
            yield from self._encode_text(text)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs, replacing any malformed UTF-8 byte sequences."""
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer on a UTF-8 text corpus.

    Special tokens are included in the vocabulary, but act as hard boundaries and
    do not contribute to pre-token or pair statistics.
    """
    if any(token == "" for token in special_tokens):
        raise ValueError("Special tokens must be non-empty strings")
    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("Special tokens must be unique")

    initial_vocab_size = len(special_tokens) + 256
    if vocab_size < initial_vocab_size:
        raise ValueError(f"vocab_size must be at least {initial_vocab_size}")

    vocab: dict[int, bytes] = {}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    for byte_value in range(256):
        vocab[len(vocab)] = bytes([byte_value])

    with open(input_path, encoding="utf-8") as corpus_file:
        text = corpus_file.read()
    pretoken_counts = _count_pretokens(text, special_tokens)

    # Give each distinct pre-token a stable integer ID. The current symbolization
    # changes after every merge, while its corpus frequency remains constant.
    words = list(pretoken_counts)
    word_frequencies = [pretoken_counts[word] for word in words]

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_word_ids: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for word_id, (word, frequency) in enumerate(zip(words, word_frequencies, strict=True)):
        local_pair_counts = Counter(zip(word, word[1:]))
        for pair, occurrence_count in local_pair_counts.items():
            pair_counts[pair] += occurrence_count * frequency
            pair_to_word_ids[pair].add(word_id)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and pair_counts:
        # The pair itself is the secondary key, implementing the required
        # lexicographically-greatest tie break.
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        affected_word_ids = tuple(pair_to_word_ids[best_pair])

        for word_id in affected_word_ids:
            old_word = words[word_id]
            frequency = word_frequencies[word_id]

            old_local_counts = Counter(zip(old_word, old_word[1:]))
            for pair, occurrence_count in old_local_counts.items():
                updated_count = pair_counts[pair] - occurrence_count * frequency
                if updated_count:
                    pair_counts[pair] = updated_count
                else:
                    del pair_counts[pair]

                word_ids = pair_to_word_ids[pair]
                word_ids.discard(word_id)
                if not word_ids:
                    del pair_to_word_ids[pair]

            new_word = _merge_pair(old_word, best_pair)
            words[word_id] = new_word

            new_local_counts = Counter(zip(new_word, new_word[1:]))
            for pair, occurrence_count in new_local_counts.items():
                pair_counts[pair] += occurrence_count * frequency
                pair_to_word_ids[pair].add(word_id)

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

    return vocab, merges
