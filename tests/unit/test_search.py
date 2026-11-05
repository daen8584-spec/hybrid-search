import math

from hybrid_search.search import Candidate, RankedDocument, apply_rerank, fuse, snippet


def test_fusion_records_the_rank_from_each_retriever() -> None:
    keyword = [Candidate("a", 1, 2.0), Candidate("b", 2, 1.0)]
    vector = [Candidate("b", 3, 0.9), Candidate("c", 4, 0.8)]
    fused = {doc.document_id: doc for doc in fuse(keyword, vector, limit=10)}

    assert (fused["a"].keyword_rank, fused["a"].vector_rank) == (1, None)
    assert (fused["b"].keyword_rank, fused["b"].vector_rank) == (2, 1)
    assert (fused["c"].keyword_rank, fused["c"].vector_rank) == (None, 2)


def test_fusion_ranks_agreement_first_and_respects_the_limit() -> None:
    keyword = [Candidate("a", 1, 2.0), Candidate("b", 2, 1.0)]
    vector = [Candidate("b", 3, 0.9), Candidate("c", 4, 0.8)]
    fused = fuse(keyword, vector, limit=2)
    assert [doc.document_id for doc in fused] == ["b", "a"]


def test_fusion_prefers_the_chunk_matched_by_vector_search() -> None:
    fused = fuse([Candidate("a", 1, 2.0)], [Candidate("a", 7, 0.9)], limit=1)
    assert fused[0].chunk_id == 7


def test_snippet_collapses_whitespace_and_cuts_at_a_word_boundary() -> None:
    assert snippet("one\n\ntwo   three") == "one two three"
    text = "word " * 100
    cut = snippet(text, max_chars=22)
    assert cut == "word word word word..."


def ranked(*ids: str) -> list[RankedDocument]:
    return [RankedDocument(doc_id, i, 1.0 / (i + 1), i + 1, None) for i, doc_id in enumerate(ids)]


def test_rerank_with_infinite_weight_uses_the_reranker_order_for_the_top() -> None:
    first_stage = ranked("a", "b", "c", "d")
    result = apply_rerank(first_stage, [0.1, 0.9, 0.5], math.inf)
    assert [doc.document_id for doc in result] == ["b", "c", "a", "d"]
    assert [doc.rerank_rank for doc in result] == [1, 2, 3, None]
    assert result[0].keyword_rank == 2  # first-stage ranks are kept


def test_rerank_with_zero_weight_keeps_the_first_stage_order() -> None:
    result = apply_rerank(ranked("a", "b", "c"), [0.1, 0.9, 0.5], 0.0)
    assert [doc.document_id for doc in result] == ["a", "b", "c"]
    assert [doc.rerank_rank for doc in result] == [3, 1, 2]


def test_rerank_fusion_balances_both_rankings() -> None:
    # "c" is third in the first stage but first for the reranker; with equal weights it
    # overtakes "b" (second and last) but not "a" (first and second).
    result = apply_rerank(ranked("a", "b", "c"), [0.5, 0.1, 0.9], 1.0)
    assert [doc.document_id for doc in result] == ["a", "c", "b"]


def test_reranked_documents_stay_ahead_of_unreranked_ones() -> None:
    result = apply_rerank(ranked("a", "b", "c", "d"), [0.2, 0.1], 0.5)
    assert [doc.document_id for doc in result][2:] == ["c", "d"]
