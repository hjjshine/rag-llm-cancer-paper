#!/usr/bin/env bash
# Validate the context files and restart Streamlit. Run this on the VM as root.
set -euo pipefail

APP_DIR=/srv/ragllm/rag-llm-cancer-paper
PYTHON=/srv/ragllm/miniforge3/envs/rag/bin/python
SERVICE=rag-llm

cd "$APP_DIR"

echo "Checking the context database"
sudo -u ragllm "$PYTHON" scripts/validate_context_db.py

echo "Restarting $SERVICE"
systemctl restart "$SERVICE"

for attempt in {1..60}; do
  if curl --fail --silent http://127.0.0.1:8501/_stcore/health >/dev/null; then
    systemctl is-active --quiet "$SERVICE"
    echo "Website is healthy."
    echo "Git commit: $(sudo -u ragllm git rev-parse --short HEAD)"
    echo "Database version: $(sudo -u ragllm "$PYTHON" -c 'import json; print(json.load(open("db_version_cache.json"))["version"])')"
    exit 0
  fi
  sleep 1
done

echo "Website did not become healthy within 60 seconds." >&2
journalctl -u "$SERVICE" -n 30 --no-pager >&2
exit 1
