import pytest

from hybrid_search.chunking import chunk_text, count_tokens


def test_short_text_is_a_single_chunk() -> None:
    chunks = chunk_text("Vitamin D supports bone health.\n\nIt is made in the skin.")
    assert len(chunks) == 1
    assert chunks[0].text == "Vitamin D supports bone health.\n\nIt is made in the skin."
    assert chunks[0].ordinal == 0


def test_empty_text_has_no_chunks() -> None:
    assert chunk_text("   \n\n  ") == []


def test_whitespace_inside_paragraphs_is_normalized() -> None:
    assert chunk_text("Line one\ncontinues   here.")[0].text == "Line one continues here."


def test_chunks_respect_the_token_budget() -> None:
    sentences = " ".join(f"Sentence number {i} describes a measured result." for i in range(200))
    chunks = chunk_text(sentences, max_tokens=60, overlap_tokens=10)
    assert len(chunks) > 1
    assert all(chunk.token_count <= 60 for chunk in chunks)
    assert all(count_tokens(chunk.text) == chunk.token_count for chunk in chunks)


def test_consecutive_chunks_overlap() -> None:
    sentences = " ".join(f"Finding {i} was replicated." for i in range(60))
    first, second = chunk_text(sentences, max_tokens=40, overlap_tokens=12)[:2]
    first_sentence_of_second = second.text.split(". ")[0] + "."
    last_sentence_of_first = first.text.split(". ")[-1]
    assert first_sentence_of_second in first.text
    assert last_sentence_of_first in second.text
    assert first.text.endswith(
        second.text[: second.text.index(last_sentence_of_first) + len(last_sentence_of_first)]
    )


def test_paragraphs_are_kept_whole_when_they_fit() -> None:
    paragraphs = [f"Paragraph {i}. " + "It has several words of content. " * 4 for i in range(6)]
    chunks = chunk_text("\n\n".join(paragraphs), max_tokens=80, overlap_tokens=0)
    for chunk in chunks:
        for paragraph in chunk.text.split("\n\n"):
            assert paragraph.startswith("Paragraph ")


def test_a_sentence_longer_than_the_budget_is_split_by_words() -> None:
    long_sentence = " ".join(["word"] * 250)
    chunks = chunk_text(long_sentence, max_tokens=100, overlap_tokens=0)
    assert [c.token_count for c in chunks] == [100, 100, 50]


@pytest.mark.parametrize(("max_tokens", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_budgets_are_rejected(max_tokens: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("text", max_tokens=max_tokens, overlap_tokens=overlap)
