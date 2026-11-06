"""Reciprocal Rank Fusion (Cormack, Clarke and Buettcher, 2009).

Each list contributes weight / (k + rank) for every item it contains, with 1-based ranks.
RRF uses ranks only, so it needs no score normalization between retrievers whose scores
are not comparable (full-text rank versus cosine similarity).
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence


def reciprocal_rank_fusion[T: Hashable](
    rankings: Mapping[str, Sequence[T]],
    weights: Mapping[str, float] | None = None,
    k: int = 60,
) -> list[tuple[T, float]]:
    """Returns items ordered by fused score, highest first. Ties keep first-seen order."""
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[T, float] = {}
    for name, ranking in rankings.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
