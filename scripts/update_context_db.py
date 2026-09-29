#!/usr/bin/env python3
"""Update the FDA and EMA context databases."""

import json
import os
import sys
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

ORGANIZATIONS = ("fda", "ema")
AGENTS_URL = "https://api.moalmanac.org/agents"


def get_remote_version():
    response = requests.get(AGENTS_URL)
    response.raise_for_status()
    return response.json()["service"]["last_updated"]


def get_local_version():
    with (ROOT / "db_version_cache.json").open() as handle:
        return json.load(handle)["version"]


def save_local_version(version):
    with (ROOT / "db_version_cache.json").open("w") as handle:
        json.dump({"version": version}, handle)


def expected_files(version):
    files = []
    for organization in ORGANIZATIONS:
        files.extend(
            [
                ROOT / f"data/latest_db/{organization}_statements__{version}.json",
                ROOT / f"data/latest_db/moalmanac_{organization}_core__{version}.csv",
                ROOT / f"data/latest_db/moalmanac_{organization}_context__{version}.json",
                ROOT / f"data/latest_db/indexes/text-embedding-3-small_{organization}_structured_context__{version}.faiss",
                ROOT / f"data/latest_db/indexes/text-embedding-3-small_{organization}_structured_context__{version}.json",
                ROOT / f"context_retriever/entities/moalmanac_{organization}_ner_entities__{version}.json",
            ]
        )
    return files


def main():
    current_version = get_local_version()
    new_version = get_remote_version()

    print(f"Current version: {current_version}")
    print(f"Available version: {new_version}")

    if current_version == new_version:
        print("The context database is already up to date.")
        return

    from utils.context_db import update_db_files

    update_db_files(new_version, list(ORGANIZATIONS), force_rebuild=True)

    new_files = set(expected_files(new_version))
    missing_files = [path for path in new_files if not path.exists()]
    if missing_files:
        missing = "\n".join(str(path.relative_to(ROOT)) for path in missing_files)
        raise RuntimeError(f"The update did not create all expected files:\n{missing}")

    save_local_version(new_version)
    print(f"Context database updated to {new_version}.")


if __name__ == "__main__":
    main()
