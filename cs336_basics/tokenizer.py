from __future__ import annotations

import os
from collections import Counter, defaultdict

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
