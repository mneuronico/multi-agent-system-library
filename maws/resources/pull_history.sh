#!/usr/bin/env bash
set -euo pipefail

# Defaults
CONF="params.json"
DEST="./history"        # local destination folder
PREFIX="history"        # S3 prefix (logical folder)

# Args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config) CONF="$2"; shift 2 ;;
    -d|--dest)   DEST="$2"; shift 2 ;;
    -p|--prefix) PREFIX="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Checks
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing '$1' in PATH"; exit 1; }; }
need aws
need jq

[[ -f "${CONF}" ]] || { echo "${CONF} not found"; exit 1; }

# Read params + defaults like bootstrap.sh
PROJECT=$(jq -r '.project' "${CONF}")
REGION=$(jq -r '.region' "${CONF}")

HISTORY_BUCKET=$(jq -r '.history_bucket // empty' "${CONF}")
[[ -z "${HISTORY_BUCKET}" || "${HISTORY_BUCKET}" == "null" ]] && HISTORY_BUCKET="${PROJECT}-history-bucket"

# Show plan
echo ">> Region: ${REGION}"
echo ">> Bucket: ${HISTORY_BUCKET}"
echo ">> S3 prefix: ${PREFIX}/"
echo ">> Local destination: ${DEST}/"

# Create destination folder
mkdir -p "${DEST}"

# Sync from S3 to local (download only)
aws s3 sync "s3://${HISTORY_BUCKET}/${PREFIX}/" "${DEST}/" --region "${REGION}"

echo "✅ Done: contents of s3://${HISTORY_BUCKET}/${PREFIX}/ synced to ${DEST}/"
