#!/usr/bin/env python3
"""Check that the active FDA and EMA context files are complete and readable."""

import csv
import json
import sys
from pathlib import Path

try:
    import faiss
except ModuleNotFoundError:
    faiss = None


ROOT = Path(__file__).resolve().parents[1]
DATABASES = ("fda", "ema")


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def count_csv_rows(path):
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def validate_database(database, version):
    if faiss is None:
        raise RuntimeError(
            "FAISS is not installed. Activate the RAG-LLM conda environment first."
        )

    base = ROOT / "data/latest_db"
    paths = {
        "statements": base / f"{database}_statements__{version}.json",
        "core": base / f"moalmanac_{database}_core__{version}.csv",
        "context": base / f"moalmanac_{database}_context__{version}.json",
        "index": base
        / "indexes"
        / f"text-embedding-3-small_{database}_structured_context__{version}.faiss",
        "index_context": base
        / "indexes"
        / f"text-embedding-3-small_{database}_structured_context__{version}.json",
        "entities": ROOT
        / "context_retriever/entities"
        / f"moalmanac_{database}_ner_entities__{version}.json",
    }

    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        names = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        raise RuntimeError(f"{database.upper()}: missing files: {names}")

    statements = load_json(paths["statements"])
    context = load_json(paths["context"])
    index_context = load_json(paths["index_context"])
    entities = load_json(paths["entities"])
    core_rows = count_csv_rows(paths["core"])
    index_rows = faiss.read_index(str(paths["index"])).ntotal

    counts = {
        "statements": len(statements),
        "core CSV": core_rows,
        "context": len(context),
        "index context": len(index_context),
        "FAISS index": index_rows,
        "entities": len(entities),
    }

    if not counts["context"]:
        raise RuntimeError(f"{database.upper()}: the context database is empty")

    if len(set(counts.values())) != 1:
        details = ", ".join(f"{name}={count}" for name, count in counts.items())
        raise RuntimeError(f"{database.upper()}: row counts do not match: {details}")

    print(f"OK: {database.upper()} has {counts['context']} records")


def main():
    cache_path = ROOT / "db_version_cache.json"
    try:
        version = load_json(cache_path)["version"]
        if not isinstance(version, str) or not version.strip():
            raise ValueError("version is empty")

        print(f"Checking context database version {version}")
        for database in DATABASES:
            validate_database(database, version)
    except (KeyError, OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("All context database checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
