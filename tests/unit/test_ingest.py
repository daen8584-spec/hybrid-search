from hybrid_search.ingest import document_hash, embedding_input, embedding_key


def test_embedding_input_prepends_the_title() -> None:
    assert embedding_input("Aspirin", "Reduces fever.") == "Aspirin\n\nReduces fever."
    assert embedding_input("", "Reduces fever.") == "Reduces fever."


def test_embedding_keys_depend_on_model_and_text() -> None:
    assert embedding_key("m1", "text") == embedding_key("m1", "text")
    assert embedding_key("m1", "text") != embedding_key("m2", "text")
    assert embedding_key("m1", "a") != embedding_key("m1", "b")


def test_document_hash_changes_with_anything_that_changes_the_chunks() -> None:
    base = document_hash("m", 384, 48, "t", "body", {"a": 1})
    assert base == document_hash("m", 384, 48, "t", "body", {"a": 1})
    assert base != document_hash("other-model", 384, 48, "t", "body", {"a": 1})
    assert base != document_hash("m", 256, 48, "t", "body", {"a": 1})
    assert base != document_hash("m", 384, 48, "t", "body", {"a": 2})
