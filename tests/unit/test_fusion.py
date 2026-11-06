import pytest

from hybrid_search.fusion import reciprocal_rank_fusion


def test_items_found_by_both_retrievers_rank_first() -> None:
    fused = reciprocal_rank_fusion({"keyword": ["a", "b", "c"], "vector": ["d", "b", "a"]})
    assert [item for item, _ in fused][:2] == ["a", "b"]


def test_scores_follow_the_rrf_formula() -> None:
    fused = dict(reciprocal_rank_fusion({"keyword": ["a"], "vector": ["b", "a"]}, k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 61)


def test_weights_shift_the_balance_between_retrievers() -> None:
    rankings = {"keyword": ["a", "b"], "vector": ["b", "a"]}
    keyword_heavy = reciprocal_rank_fusion(rankings, weights={"keyword": 2.0, "vector": 1.0})
    vector_heavy = reciprocal_rank_fusion(rankings, weights={"keyword": 1.0, "vector": 2.0})
    assert keyword_heavy[0][0] == "a"
    assert vector_heavy[0][0] == "b"


def test_empty_rankings_produce_no_results() -> None:
    assert reciprocal_rank_fusion({"keyword": [], "vector": []}) == []


def test_negative_k_is_rejected() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"keyword": ["a"]}, k=-1)
