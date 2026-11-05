"""Standard ranked-retrieval metrics. Gains follow the trec_eval convention used by BEIR."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    positives = {doc for doc, grade in relevant.items() if grade > 0}
    if not positives:
        return 0.0
    return len(positives.intersection(ranked[:k])) / len(positives)


def mrr_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    for rank, doc in enumerate(ranked[:k], start=1):
        if relevant.get(doc, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    def dcg(gains: Sequence[int]) -> float:
        return sum(gain / math.log2(position + 2) for position, gain in enumerate(gains))

    actual = dcg([relevant.get(doc, 0) for doc in ranked[:k]])
    ideal = dcg(sorted((g for g in relevant.values() if g > 0), reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile, p in [0, 100]."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]
