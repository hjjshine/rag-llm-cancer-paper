#utils/check_db_version.py
import requests
import json
import os
from datetime import datetime


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