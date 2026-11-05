import json
import zipfile
from pathlib import Path

import pytest

from hybrid_search.evaluation.beir import iter_corpus, load_dataset, safe_extract


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    root = tmp_path / "tiny"
    (root / "qrels").mkdir(parents=True)
    corpus = [
        {"_id": "d1", "title": "Aspirin", "text": "Aspirin reduces fever.", "metadata": {}},
        {"_id": "d2", "title": "", "text": "Vitamin D and bone density."},
    ]
    queries = [{"_id": "q1", "text": "does aspirin reduce fever"}, {"_id": "q2", "text": "unjudged query"}]
    (root / "corpus.jsonl").write_text("\n".join(json.dumps(row) for row in corpus) + "\n")
    (root / "queries.jsonl").write_text("\n".join(json.dumps(row) for row in queries) + "\n")
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    return root


def test_corpus_rows_become_documents(dataset_dir: Path) -> None:
    documents = list(iter_corpus(dataset_dir))
    assert [d.id for d in documents] == ["d1", "d2"]
    assert documents[1].title == ""
    assert documents[1].metadata == {}


def test_only_judged_queries_are_loaded(dataset_dir: Path) -> None:
    dataset = load_dataset(dataset_dir)
    assert [q.id for q in dataset.queries] == ["q1"]
    assert dataset.qrels == {"q1": {"d1": 1}}
    assert dataset.name == "tiny"


def test_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="escapes"):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_extraction_writes_regular_entries(tmp_path: Path) -> None:
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("scifact/corpus.jsonl", "{}")
    safe_extract(archive, tmp_path / "out")
    assert (tmp_path / "out" / "scifact" / "corpus.jsonl").read_text() == "{}"
