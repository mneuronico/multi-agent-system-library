#!/usr/bin/env bash
set -euo pipefail

# Defaults
CONF="params.json"
DEST="./history"        # carpeta local destino
PREFIX="history"        # prefijo en S3 (carpeta lógica)

# Args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config) CONF="$2"; shift 2 ;;
    -d|--dest)   DEST="$2"; shift 2 ;;
    -p|--prefix) PREFIX="$2"; shift 2 ;;
    *) echo "Arg desconocido: $1"; exit 1 ;;
  esac
done

# Checks
need() { command -v "$1" >/dev/null 2>&1 || { echo "Falta '$1' en PATH"; exit 1; }; }
need aws
need jq

[[ -f "${CONF}" ]] || { echo "No existe ${CONF}"; exit 1; }

# Leer params + defaults como en bootstrap.sh
PROJECT=$(jq -r '.project' "${CONF}")
REGION=$(jq -r '.region' "${CONF}")

HISTORY_BUCKET=$(jq -r '.history_bucket // empty' "${CONF}")
[[ -z "${HISTORY_BUCKET}" || "${HISTORY_BUCKET}" == "null" ]] && HISTORY_BUCKET="${PROJECT}-history-bucket"

# Mostrar plan
echo ">> Región: ${REGION}"
echo ">> Bucket: ${HISTORY_BUCKET}"
echo ">> Prefijo S3: ${PREFIX}/"
echo ">> Destino local: ${DEST}/"

# Crear carpeta destino
mkdir -p "${DEST}"

# Sincronizar de S3 a local (solo descarga)
aws s3 sync "s3://${HISTORY_BUCKET}/${PREFIX}/" "${DEST}/" --region "${REGION}"

echo "✅ Listo: contenido de s3://${HISTORY_BUCKET}/${PREFIX}/ sincronizado en ${DEST}/"
