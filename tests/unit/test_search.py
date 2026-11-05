from hybrid_search.search import Candidate, fuse, snippet


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
