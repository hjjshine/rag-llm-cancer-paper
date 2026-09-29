#!/usr/bin/env python3
"""Delete FDA and EMA context database files that are not currently in use."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PATTERNS = [
    "data/latest_db/fda_statements__*.json",
    "data/latest_db/ema_statements__*.json",
    "data/latest_db/moalmanac_fda_core__*.csv",
    "data/latest_db/moalmanac_ema_core__*.csv",
    "data/latest_db/moalmanac_fda_context__*.json",
    "data/latest_db/moalmanac_ema_context__*.json",
    "data/latest_db/indexes/text-embedding-3-small_fda_structured_context__*",
    "data/latest_db/indexes/text-embedding-3-small_ema_structured_context__*",
    "context_retriever/entities/moalmanac_fda_ner_entities__*.json",
    "context_retriever/entities/moalmanac_ema_ner_entities__*.json",
]


def main():
    with (ROOT / "db_version_cache.json").open() as handle:
        current_version = json.load(handle)["version"]

    for pattern in PATTERNS:
        for path in ROOT.glob(pattern):
            if current_version not in path.name:
                print(f"Deleting {path.relative_to(ROOT)}")
                path.unlink()

    print(f"Kept context database version {current_version}.")


if __name__ == "__main__":
    main()
