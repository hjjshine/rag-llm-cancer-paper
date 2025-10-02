#utils/check_db_version.py
import requests
import json
import os
from datetime import datetime
import warnings
from utils.context_db import update_db_files

#local db version
CACHE_FILE = "db_version_cache.json"

def get_remote_version():
    agents = requests.get('https://api.moalmanac.org/agents').json()
    return agents['service']['last_updated']

def get_local_version():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f).get("version")
    return None

def save_local_version(version):
    with open(CACHE_FILE, "w") as f:
        json.dump({"version": version}, f)

def sync_db(org):
    remote = get_remote_version()
    local = get_local_version()
    remote_dt = datetime.strptime(remote, "%Y-%m-%d")
    local_dt = datetime.strptime(local, "%Y-%m-%d")
    if remote_dt > local_dt:
        update_db_files(version=remote, organizations=[org])
        save_local_version(remote)
    elif remote_dt == local_dt:
        print(f"DB is already up to date (version={local})")
    else:
        warnings.warn(
            f"Local DB version ({local}) is newer than remote ({remote}). "
            "This is unexpected and may indicate a problem."
        )

