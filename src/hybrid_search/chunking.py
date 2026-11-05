"""Structure-aware chunking.

Text is split into paragraphs first. Paragraphs that fit the token budget are kept whole;
longer ones are split into sentences, and sentences longer than the budget are split by
words. Units are then packed greedily into chunks, and each new chunk starts with up to
`overlap_tokens` of trailing context from the previous one.

Token counts are an approximation (words and punctuation marks), which is sufficient for
staying within embedding model context limits without depending on a specific tokenizer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from itertools import pairwise

_TOKEN = re.compile(r"\w+|[^\w\s]")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Chunk:
    ordinal: int
    text: str
    token_count: int

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _Unit:
    text: str
    tokens: int
    paragraph: int


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def _split_words(text: str, max_tokens: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in text.split(" "):
        tokens = count_tokens(word)
        if current and current_tokens + tokens > max_tokens:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def _units(text: str, max_tokens: int) -> list[_Unit]:
    units: list[_Unit] = []
    paragraphs = (_WHITESPACE.sub(" ", p).strip() for p in _PARAGRAPH_BREAK.split(text))
    for index, paragraph in enumerate(p for p in paragraphs if p):
        if count_tokens(paragraph) <= max_tokens:
            units.append(_Unit(paragraph, count_tokens(paragraph), index))
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            pieces = (
                [sentence] if count_tokens(sentence) <= max_tokens else _split_words(sentence, max_tokens)
            )
            units.extend(_Unit(piece, count_tokens(piece), index) for piece in pieces)
    return units


def _join(units: list[_Unit]) -> str:
    text = units[0].text
    for previous, unit in pairwise(units):
        text += (" " if unit.paragraph == previous.paragraph else "\n\n") + unit.text
    return text


def chunk_text(text: str, max_tokens: int = 384, overlap_tokens: int = 48) -> list[Chunk]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be at least 0 and smaller than max_tokens")

    chunks: list[list[_Unit]] = []
    current: list[_Unit] = []
    current_tokens = 0

    for unit in _units(text, max_tokens):
        if current and current_tokens + unit.tokens > max_tokens:
            chunks.append(current)
            overlap: list[_Unit] = []
            overlap_size = 0
            for previous in reversed(current):
                if overlap_size + previous.tokens > overlap_tokens:
                    break
                overlap.insert(0, previous)
                overlap_size += previous.tokens
            if overlap_size + unit.tokens > max_tokens:
                overlap, overlap_size = [], 0
            current, current_tokens = overlap, overlap_size
        current.append(unit)
        current_tokens += unit.tokens

    if current:
        chunks.append(current)
    return [
        Chunk(ordinal, _join(units), sum(u.tokens for u in units)) for ordinal, units in enumerate(chunks)
    ]
