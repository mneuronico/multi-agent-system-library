#!/usr/bin/env bash
set -euo pipefail

# Defaults
CONF="params.json"
DEST="./history"        # local destination folder
PREFIX="history"        # S3 prefix (logical folder)
USER_IDS=()

# Args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config) CONF="$2"; shift 2 ;;
    -d|--dest)   DEST="$2"; shift 2 ;;
    -p|--prefix) PREFIX="$2"; shift 2 ;;
    --user-id)   USER_IDS+=("$2"); shift 2 ;;
    --user-ids)
      IFS=',' read -r -a _ids <<< "$2"
      for _id in "${_ids[@]}"; do USER_IDS+=("${_id}"); done
      shift 2
      ;;
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

safe_key_part() {
  local raw="$1"
  if [[ -n "${raw}" && "${raw}" != "." && "${raw}" != ".." && "${raw}" != *".."* && "${raw}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    printf '%s' "${raw}"
    return 0
  fi

  local py
  py="$(command -v python3 || command -v python || true)"
  [[ -n "${py}" ]] || { echo "python3 or python is required to encode unsafe user IDs" >&2; return 1; }
  "${py}" - "$raw" <<'PY'
import base64
import sys

raw = sys.argv[1]
encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")
print(f"u_{encoded or 'empty'}", end="")
PY
}

if [[ ${#USER_IDS[@]} -eq 0 ]]; then
  # Sync from S3 to local (download only)
  aws s3 sync "s3://${HISTORY_BUCKET}/${PREFIX}/" "${DEST}/" --region "${REGION}"

  echo "[OK] Done: contents of s3://${HISTORY_BUCKET}/${PREFIX}/ synced to ${DEST}/"
  exit 0
fi

failed=0
downloaded=0
for raw_user_id in "${USER_IDS[@]}"; do
  user_id="$(echo "${raw_user_id}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [[ -n "${user_id}" ]] || continue
  key_part="$(safe_key_part "${user_id}")" || exit 1
  src="s3://${HISTORY_BUCKET}/${PREFIX}/${key_part}.sqlite"
  dst="${DEST}/${key_part}.sqlite"
  echo ">> Pull user ${user_id} -> ${dst}"
  if aws s3 cp "${src}" "${dst}" --region "${REGION}"; then
    downloaded=$((downloaded + 1))
  else
    echo "[WARN] Failed to download ${src}" >&2
    failed=1
  fi
done

if [[ ${downloaded} -eq 0 && ${failed} -ne 0 ]]; then
  echo "[ERR] No requested history files were downloaded." >&2
  exit 1
fi

if [[ ${failed} -ne 0 ]]; then
  echo "[WARN] Downloaded ${downloaded} requested history file(s), but at least one failed." >&2
  exit 1
fi

echo "[OK] Downloaded ${downloaded} requested history file(s) from s3://${HISTORY_BUCKET}/${PREFIX}/ to ${DEST}/"
