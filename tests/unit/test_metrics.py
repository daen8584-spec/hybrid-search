import math

import pytest

from hybrid_search.evaluation.metrics import mrr_at_k, ndcg_at_k, percentile, recall_at_k

QRELS = {"d1": 1, "d4": 1}


def test_recall_counts_relevant_documents_within_k() -> None:
    assert recall_at_k(["d1", "d2", "d3", "d4"], QRELS, k=3) == 0.5
    assert recall_at_k(["d1", "d2", "d3", "d4"], QRELS, k=4) == 1.0


def test_recall_ignores_zero_grade_judgements() -> None:
    assert recall_at_k(["d1"], {"d1": 1, "d2": 0}, k=10) == 1.0


def test_mrr_uses_the_first_relevant_rank() -> None:
    assert mrr_at_k(["d2", "d4", "d1"], QRELS, k=10) == 0.5
    assert mrr_at_k(["d2", "d3"], QRELS, k=10) == 0.0


def test_ndcg_is_one_for_an_ideal_ranking() -> None:
    assert ndcg_at_k(["d1", "d4", "d2"], QRELS, k=10) == pytest.approx(1.0)


def test_ndcg_matches_a_hand_computed_value() -> None:
    # relevant documents at ranks 2 and 4
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(["d2", "d1", "d3", "d4"], QRELS, k=10) == pytest.approx(expected)


def test_ndcg_supports_graded_relevance() -> None:
    graded = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], graded, k=2) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], graded, k=2) < 1.0


def test_metrics_are_zero_without_relevant_documents() -> None:
    assert recall_at_k(["a"], {}, k=10) == ndcg_at_k(["a"], {}, k=10) == 0.0


def test_percentile_uses_nearest_rank() -> None:
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0
    assert percentile(values, 99) == 99.0
    assert percentile([5.0], 99) == 5.0


@pytest.mark.parametrize("p", [-1, 101])
def test_percentile_rejects_out_of_range_p(p: float) -> None:
    with pytest.raises(ValueError):
        percentile([1.0], p)
