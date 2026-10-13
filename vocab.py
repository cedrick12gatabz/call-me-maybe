"""Vocabulary handling: token ids to real text, plus reusable masks."""
from typing import Any

import numpy as np
from pydantic import BaseModel

NUMBER_CHARS = set("-0123456789.")


def bytes_to_unicode() -> dict[int, str]:
    """Return the GPT-2 table mapping bytes to printable characters."""
    keep = (list(range(ord("!"), ord("~") + 1))
            + list(range(0xA1, 0xAC + 1))
            + list(range(0xAE, 0xFF + 1)))
    chars = keep[:]
    extra = 0
    for byte in range(256):
        if byte not in keep:
            keep.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(keep, chars)}


def decode_token(token: str, decoder: dict[str, int]) -> str:
    """Turn a vocabulary entry into its real text ('' if unusable)."""
    try:
        return bytes(decoder[ch] for ch in token).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return ""


def find_token(strings: list[str], text: str) -> int:
    """Return the id of the token whose text is exactly `text`."""
    try:
        return strings.index(text)
    except ValueError:
        raise ValueError(f"token {text!r} not in vocabulary") from None


class TokenTable(BaseModel):
    """Real text of every token id and precomputed token groups."""

    strings: list[str]
    plain: Any
    escape_ids: list[int]
    nocontrol_ids: list[int]
    number_ids: list[int]
    quote_id: int
    comma_id: int
    close_id: int

    @classmethod
    def build(cls, vocab: dict[str, int], size: int) -> "TokenTable":
        """Build the table.

        Args:
            vocab: Mapping token text -> id from the vocabulary file.
            size: Number of logits produced by the model.

        Returns:
            The ready-to-use token table.
        """
        decoder = {v: k for k, v in bytes_to_unicode().items()}
        strings = [""] * size
        for token, idx in vocab.items():
            if 0 <= idx < size:
                strings[idx] = decode_token(token, decoder)
        plain = np.zeros(size, dtype=bool)
        escape_ids: list[int] = []
        nocontrol_ids: list[int] = []
        number_ids: list[int] = []
        for idx, text in enumerate(strings):
            if not text or any(ord(c) < 0x20 for c in text):
                continue
            nocontrol_ids.append(idx)
            if "\\" in text:
                escape_ids.append(idx)
            elif '"' not in text:
                plain[idx] = True
            if set(text) <= NUMBER_CHARS:
                number_ids.append(idx)
        return cls(
            strings=strings,
            plain=plain,
            escape_ids=escape_ids,
            nocontrol_ids=nocontrol_ids,
            number_ids=number_ids,
            quote_id=find_token(strings, '"'),
            comma_id=find_token(strings, ","),
            close_id=find_token(strings, "}"),
        )