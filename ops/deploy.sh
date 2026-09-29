#!/usr/bin/env bash
# Pull main on the VM, restart the website, and check the public URL.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash ops/deploy.sh PROJECT_ID ZONE VM_NAME" >&2
  exit 2
fi

PROJECT_ID=$1
ZONE=$2
VM_NAME=$3
SITE_URL=${SITE_URL:-https://llm.moalmanac.org}
APP_DIR=/srv/ragllm/rag-llm-cancer-paper

echo "Deploying main to $VM_NAME in project $PROJECT_ID ($ZONE)"

gcloud compute ssh "$VM_NAME" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --command "sudo -u ragllm git -C $APP_DIR switch main && \
sudo -u ragllm git -C $APP_DIR pull --ff-only origin main && \
sudo bash $APP_DIR/ops/refresh_site.sh"

echo "Checking $SITE_URL"
curl --fail --silent --show-error "$SITE_URL/_stcore/health"
echo
echo "Deployment finished successfully."
