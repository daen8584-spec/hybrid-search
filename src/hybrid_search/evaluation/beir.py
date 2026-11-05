"""Loader for datasets in the BEIR format (corpus.jsonl, queries.jsonl, qrels/<split>.tsv)."""

from __future__ import annotations

import csv
import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    id: str
    title: str
    text: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Query:
    id: str
    text: str


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    queries: list[Query]
    qrels: dict[str, dict[str, int]]


def iter_corpus(directory: Path) -> Iterator[CorpusDocument]:
    with (directory / "corpus.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                row = json.loads(line)
                yield CorpusDocument(
                    id=str(row["_id"]),
                    title=row.get("title") or "",
                    text=row.get("text") or "",
                    metadata=row.get("metadata") or {},
                )


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or row[0] == "query-id":
                continue
            query_id, document_id, grade = row[0], row[1], int(row[2])
            qrels.setdefault(query_id, {})[document_id] = grade
    return qrels


def load_dataset(directory: Path, split: str = "test") -> Dataset:
    """Loads the queries that have judgements in the given split, in file order."""
    qrels = load_qrels(directory / "qrels" / f"{split}.tsv")
    queries = []
    with (directory / "queries.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                row = json.loads(line)
                if str(row["_id"]) in qrels:
                    queries.append(Query(id=str(row["_id"]), text=row["text"]))
    return Dataset(name=directory.name, queries=queries, qrels=qrels)


def safe_extract(archive: Path, destination: Path) -> None:
    """Extracts a zip archive, refusing entries that would be written outside the destination."""
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            target = (root / member).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"archive entry escapes the destination: {member}")
        bundle.extractall(root)
