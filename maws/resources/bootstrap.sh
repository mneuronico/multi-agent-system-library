#!/usr/bin/env bash
set -euo pipefail

# --- Args ---
CONF="params.json"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config) CONF="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

die() { printf '%s\n' "$@" >&2; exit 1; }

[[ -f "${CONF}" ]] || { echo "${CONF} not found"; exit 1; }

# --- Checks ---
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing '$1' in PATH"; exit 1; }; }
need aws
need sam
need jq
need sed

[[ -f ".env.prod" ]] || { echo ".env.prod is missing"; exit 1; }
[[ -f "${CONF}" ]] || { echo "${CONF} not found"; exit 1; }

# --- Read config ---
echo ">> Reading config: ${CONF}"

# Booleans use has(): `.x // default` would turn an explicit false into the default.
# Usage: param_bool <parent object path, "." for top level> <key> <default: true|false>
param_bool() {
  local parent="$1" key="$2" default="$3" value label
  label="${key}"
  [[ "${parent}" != "." ]] && label="${parent#.}.${key}"
  value=$(jq -r --arg k "${key}" \
    "if ((${parent}) | type) == \"object\" and ((${parent}) | has(\$k)) and ((${parent})[\$k] != null) then (${parent})[\$k] else ${default} end" \
    "${CONF}")
  case "$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes|on) printf 'true' ;;
    false|0|no|off) printf 'false' ;;
    *) die "[ERR] ${label} in ${CONF} must be true or false (got: ${value})." ;;
  esac
}

PROJECT=$(jq -r '.project // empty' "${CONF}")
REGION=$(jq -r '.region // empty' "${CONF}")
BOT=$(jq -r '.bot // empty' "${CONF}" | tr '[:upper:]' '[:lower:]') # "telegram" | "whatsapp"
[[ -n "${PROJECT}" ]] || die "[ERR] ${CONF} has no \"project\"."
[[ -n "${REGION}" ]] || die "[ERR] ${CONF} has no \"region\"."
if [[ -z "${BOT}" ]]; then
  echo "[WARN] ${CONF} has no \"bot\"; deploying a WhatsApp bot (the runtime default). Set \"bot\" explicitly."
  BOT="whatsapp"
fi
[[ "${BOT}" == "telegram" || "${BOT}" == "whatsapp" ]] || die "[ERR] \"bot\" in ${CONF} must be \"telegram\" or \"whatsapp\" (got: ${BOT})."

# Default names below are used by existing deployments: never change them (a different history
# bucket name means a new, empty history bucket). New projects get explicit names in params.json.
STACK=$(jq -r '.stack_name // empty' "${CONF}")
[[ -z "${STACK}" || "${STACK}" == "null" ]] && STACK="${PROJECT}-stack"

HISTORY_BUCKET=$(jq -r '.history_bucket // empty' "${CONF}")
[[ -z "${HISTORY_BUCKET}" || "${HISTORY_BUCKET}" == "null" ]] && HISTORY_BUCKET="${PROJECT}-history-bucket"

DEPLOY_BUCKET=$(jq -r '.deployment_bucket // empty' "${CONF}")
[[ -z "${DEPLOY_BUCKET}" || "${DEPLOY_BUCKET}" == "null" ]] && DEPLOY_BUCKET="${PROJECT}-deployment-bucket"

# Lambda code is staged here so only project code ends up in the deployment zip.
PACKAGE_DIR=".maws_package"
# Generated deploy files (template.yaml) are build artifacts in BUILD_DIR. Files in the project
# root (template.yaml, samconfig.toml, requirements.txt, lambda_function.py) are never written.
BUILD_DIR=".maws_build"
TEMPLATE_FILE="${BUILD_DIR}/template.yaml"
SAM_BUILD_DIR=".aws-sam/build"

ENV_PARAM=$(jq -r '.env_param_name // empty' "${CONF}")
[[ -z "${ENV_PARAM}" || "${ENV_PARAM}" == "null" ]] && ENV_PARAM="/${PROJECT}/prod/env"

API_PATH=$(jq -r '.api_path // "/webhook"' "${CONF}")

SYNC_TOKENS_S3=$(param_bool . sync_tokens_s3 true)
TOKENS_PREFIX=$(jq -r '.tokens_s3_prefix // "secrets"' "${CONF}")
VERBOSE=$(param_bool . verbose false)
LAMBDA_TIMEOUT=$(jq -r '.lambda_timeout // 120' "${CONF}")
[[ "${LAMBDA_TIMEOUT}" =~ ^[0-9]+$ ]] || die "[ERR] lambda_timeout in ${CONF} must be a whole number of seconds (got: ${LAMBDA_TIMEOUT})."
# SQS makes a message visible again VISIBILITY_TIMEOUT seconds after a worker received it. The user
# lock must outlive a running invocation (lambda_timeout) but expire before that redelivery, or the
# retry of a crashed worker's message finds the user still busy. The runtime ties the lock to the
# invocation deadline + 15 s; LOCK_TTL_SECONDS is its fallback when no Lambda context is available.
VISIBILITY_TIMEOUT=$((LAMBDA_TIMEOUT + 30))
LOCK_TTL_SECONDS=$((LAMBDA_TIMEOUT + 15))

BUSY_POLICY=$(jq -r '.busy_policy // "drop"' "${CONF}" | tr '[:upper:]' '[:lower:]')
[[ "${BUSY_POLICY}" == "drop" || "${BUSY_POLICY}" == "fifo" ]] || die "[ERR] busy_policy in ${CONF} must be \"drop\" or \"fifo\" (got: ${BUSY_POLICY})."
PERSIST_FILES_S3=$(param_bool . persist_files_s3 false)
FILES_PREFIX=$(jq -r '.files_s3_prefix // "files"' "${CONF}")

HISTORY_MODE=$(jq -r '.history_mode // empty' "${CONF}")
[[ -z "${HISTORY_MODE}" || "${HISTORY_MODE}" == "null" ]] && HISTORY_MODE="per_user"
HISTORY_ROTATION=$(jq -r '.history_rotation // "message_count"' "${CONF}")
HISTORY_MAX_MESSAGES=$(jq -r '.history_max_messages // 1000' "${CONF}")
HISTORY_PERIOD=$(jq -r '.history_period // "1w"' "${CONF}")

MANAGER_KWARGS_JSON=$(jq -c '.runtime.manager_kwargs // {}' "${CONF}")
BOT_KWARGS_JSON=$(jq -c '.runtime.bot_kwargs // {}' "${CONF}")
ENSURE_DELIVERY=$(param_bool .runtime ensure_delivery true)
DELIVERY_TIMEOUT=$(jq -r '.runtime.delivery_timeout // 60.0' "${CONF}")
MAX_ALLOWED_MESSAGE_DELAY=$(jq -r '.runtime.max_allowed_message_delay // empty' "${CONF}")
# busy_policy=drop: max seconds a worker waits for a busy user lock before dropping (runtime default 30).
BUSY_WAIT_SECONDS=$(jq -r '.runtime.busy_wait_seconds // empty' "${CONF}")
[[ -z "${BUSY_WAIT_SECONDS}" || "${BUSY_WAIT_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "[ERR] runtime.busy_wait_seconds in ${CONF} must be a number of seconds (got: ${BUSY_WAIT_SECONDS})."
# How long a processed message is remembered so a repeated delivery is skipped (runtime default: 8 days).
DEDUP_RETENTION_DAYS=$(jq -r '.runtime.dedup_retention_days // empty' "${CONF}")
[[ -z "${DEDUP_RETENTION_DAYS}" || "${DEDUP_RETENTION_DAYS}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "[ERR] runtime.dedup_retention_days in ${CONF} must be a number of days, 0 to turn processed-message records off (got: ${DEDUP_RETENTION_DAYS})."

TELEGRAM_SECRET_ENV_KEY=$(jq -r '.webhook_security.telegram_secret_token_env_key // empty' "${CONF}")
[[ -z "${TELEGRAM_SECRET_ENV_KEY}" || "${TELEGRAM_SECRET_ENV_KEY}" == "null" ]] && TELEGRAM_SECRET_ENV_KEY="TELEGRAM_WEBHOOK_SECRET_TOKEN"
WHATSAPP_APP_SECRET_ENV_KEY=$(jq -r '.webhook_security.whatsapp_app_secret_env_key // empty' "${CONF}")
[[ -z "${WHATSAPP_APP_SECRET_ENV_KEY}" || "${WHATSAPP_APP_SECRET_ENV_KEY}" == "null" ]] && WHATSAPP_APP_SECRET_ENV_KEY="WHATSAPP_APP_SECRET"

CAPTURE_FAILED_EVENTS=$(param_bool .failure_handling capture_failed_events false)
# FIFO: a DLQ is the default. Without one, SQS retries a failing message until it expires (4 days)
# and every later message of that chat waits behind it.
WORKER_DLQ_DEFAULT="false"
[[ "${BUSY_POLICY}" == "fifo" ]] && WORKER_DLQ_DEFAULT="true"
WORKER_DLQ=$(param_bool .failure_handling worker_dlq "${WORKER_DLQ_DEFAULT}")
MAX_RECEIVE_COUNT=$(jq -r '.failure_handling.max_receive_count // 3' "${CONF}")
[[ "${MAX_RECEIVE_COUNT}" =~ ^[0-9]+$ && "${MAX_RECEIVE_COUNT}" -ge 1 && "${MAX_RECEIVE_COUNT}" -le 1000 ]] || die "[ERR] failure_handling.max_receive_count in ${CONF} must be 1-1000 (got: ${MAX_RECEIVE_COUNT})."
# busy_policy=drop: async self-invocations that still fail after Lambda's 2 retries go to an SQS queue.
ASYNC_FAILURE_QUEUE=$(param_bool .failure_handling async_failure_queue true)
[[ "${BUSY_POLICY}" == "drop" ]] || ASYNC_FAILURE_QUEUE="false"

# Optional external history append endpoint (/history/outgoing). Off by default.
HISTORY_APPEND_ENABLED=$(param_bool .history_append enabled false)
HISTORY_APPEND_ROLE=$(jq -r '.history_append.role // empty' "${CONF}")

LAMBDA_RUNTIME=$(jq -r '.infra.runtime // "python3.12"' "${CONF}")
ARCHITECTURE=$(jq -r '.infra.architecture // "x86_64"' "${CONF}")
MEMORY_SIZE=$(jq -r '.infra.memory_size // 256' "${CONF}")
# CloudWatch log retention. Default 30 days: enough to investigate problems reported weeks later,
# while the logs (chat ids, and message content with verbose=true) are not kept forever.
# 0 keeps logs forever.
LOG_RETENTION_DAYS=$(jq -r '.infra.log_retention_days // empty' "${CONF}")
[[ -z "${LOG_RETENTION_DAYS}" ]] && LOG_RETENTION_DAYS=30
case " 0 1 3 5 7 14 30 60 90 120 150 180 365 400 545 731 1096 1827 2192 2557 2922 3288 3653 " in
  *" ${LOG_RETENTION_DAYS} "*) ;;
  *) die "[ERR] infra.log_retention_days in ${CONF} must be 0 (keep forever) or a CloudWatch Logs retention:" \
       "      1 3 5 7 14 30 60 90 120 150 180 365 400 545 731 1096 1827 2192 2557 2922 3288 3653 (got: ${LOG_RETENTION_DAYS})." ;;
esac
RESERVED_CONCURRENCY=$(jq -r '.infra.reserved_concurrency // empty' "${CONF}")
S3_VERSIONING=$(param_bool .infra s3_versioning false)
S3_LIFECYCLE_DAYS=$(jq -r '.infra.s3_lifecycle_days // empty' "${CONF}")
[[ -z "${S3_LIFECYCLE_DAYS}" || "${S3_LIFECYCLE_DAYS}" =~ ^[1-9][0-9]*$ ]] || die "[ERR] infra.s3_lifecycle_days in ${CONF} must be a positive whole number of days (got: ${S3_LIFECYCLE_DAYS})."
# With versioning on, every history upload keeps the previous copy as a noncurrent version.
S3_NONCURRENT_DAYS=$(jq -r '.infra.s3_noncurrent_version_days // empty' "${CONF}")
[[ -z "${S3_NONCURRENT_DAYS}" ]] && S3_NONCURRENT_DAYS=30
[[ "${S3_NONCURRENT_DAYS}" =~ ^[0-9]+$ ]] || die "[ERR] infra.s3_noncurrent_version_days in ${CONF} must be a whole number of days, 0 to keep them (got: ${S3_NONCURRENT_DAYS})."
S3_ENCRYPTION=$(jq -r '.infra.s3_encryption // empty' "${CONF}")
TRACING_ENABLED=$(param_bool .infra tracing true)
FUNCTION_TRACING="Active"
[[ "${TRACING_ENABLED}" == "false" ]] && FUNCTION_TRACING="PassThrough"

REQUIREMENTS_SOURCE=$(jq -r '.requirements_source // "git"' "${CONF}")
REQUIREMENTS_REF=$(jq -r '.requirements_ref // empty' "${CONF}")
case "${REQUIREMENTS_SOURCE}" in
  git|local) ;;
  pypi) die "[ERR] requirements_source \"pypi\" is not supported: the PyPI package named 'mas' is a different"             "      project, not this library. Use \"git\" (the default, pin requirements_ref) or \"local\"." ;;
  *) die "[ERR] requirements_source in ${CONF} must be \"git\" or \"local\" (got: ${REQUIREMENTS_SOURCE})." ;;
esac

TELEGRAM_ENV_KEY=$(jq -r '.telegram.env_token_key // empty' "${CONF}")
[[ -z "${TELEGRAM_ENV_KEY}" || "${TELEGRAM_ENV_KEY}" == "null" ]] && TELEGRAM_ENV_KEY="TELEGRAM_TOKEN"

WHATSAPP_VERIFY_ENV_KEY=$(jq -r '.whatsapp.verify_env_key // empty' "${CONF}")
[[ -z "${WHATSAPP_VERIFY_ENV_KEY}" || "${WHATSAPP_VERIFY_ENV_KEY}" == "null" ]] && WHATSAPP_VERIFY_ENV_KEY="WHATSAPP_VERIFY_TOKEN"


SPECIAL_JSON=$(jq -c '.special_token_files // []' "${CONF}")
TOKEN_ENV_MAP=$(jq -c '.token_env_map // {}' "${CONF}")
mapfile -t EXTRA_REQ < <(jq -r '.extra_requirements[]?' "${CONF}" 2>/dev/null || true)

# Read KEY from .env.prod the way the Lambda runtime does (maws/runtime.py,
# _load_env_from_ssm_if_needed): the first KEY=value line wins, surrounding whitespace (including a
# Windows CR) is stripped, and quotes are kept. Checking that value is what proves the deployed bot
# will authenticate webhooks.
env_file_value() {
  local wanted="$1" line key value
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" || "${line}" == \#* || "${line}" != *=* ]] && continue
    key="${line%%=*}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${line#*=}"
    value="${value#"${value%%[![:space:]]*}"}"
    if [[ "${key}" == "${wanted}" ]]; then
      printf '%s' "${value}"
      return 0
    fi
  done < .env.prod
  return 0
}

# The keys of .env.prod, in the same format env_file_value reads.
env_file_keys() {
  local line key
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" || "${line}" == \#* || "${line}" != *=* ]] && continue
    key="${line%%=*}"
    key="${key%"${key##*[![:space:]]}"}"
    printf '%s\n' "${key}"
  done < .env.prod
  return 0
}

# Like the runtime: the configured key first, then the default key.
env_secret_value() {
  local value
  value="$(env_file_value "$1")"
  if [[ -z "${value}" && "$1" != "$2" ]]; then
    value="$(env_file_value "$2")"
  fi
  printf '%s' "${value}"
}

unauthenticated_warning() {
  echo ""
  echo "##############################################################################"
  echo "# [WARN] DEPLOYING AN UNAUTHENTICATED WEBHOOK: $1"
  echo "# webhook_security.allow_unauthenticated=true. Anyone who finds the webhook URL"
  echo "# can send fake updates as any user of this bot. Do not use this in production."
  echo "##############################################################################"
  echo ""
}

# --- Webhook authentication (checked before anything is uploaded or deployed) ---
# Booleans use has(): `.x // default` would turn an explicit false into the default.
WEBHOOK_ALLOW_UNAUTHENTICATED=$(jq -r 'if (.webhook_security | type) == "object" and (.webhook_security | has("allow_unauthenticated")) then .webhook_security.allow_unauthenticated else false end' "${CONF}")
[[ "${WEBHOOK_ALLOW_UNAUTHENTICATED}" == "true" ]] || WEBHOOK_ALLOW_UNAUTHENTICATED="false"
WEBHOOK_WHATSAPP_VERIFY=$(jq -r 'if (.webhook_security | type) == "object" and (.webhook_security | has("whatsapp_verify_signature")) then .webhook_security.whatsapp_verify_signature else true end' "${CONF}")
[[ "${WEBHOOK_WHATSAPP_VERIFY}" == "false" ]] || WEBHOOK_WHATSAPP_VERIFY="true"

# Telegram accepts secret_token values of 1-256 characters from A-Z, a-z, 0-9, _ and -.
# The class is spelled out so it does not depend on the locale's collation, and the length is
# checked separately because {1,256} exceeds RE_DUP_MAX (255) and would not compile.
TELEGRAM_SECRET_RE='^[ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-]+$'
ENV_FORMAT_HINT="Use plain KEY=value lines in .env.prod (no quotes, no 'export'): Lambda reads the file that way."
GENERATE_SECRET_HINT="Generate one with:  python3 -c \"import secrets; print(secrets.token_urlsafe(32))\""

echo ">> Checking webhook authentication"
TELEGRAM_TOKEN_VALUE=""
TELEGRAM_SECRET_VALUE=""
if [[ "${BOT}" == "telegram" ]]; then
  TELEGRAM_TOKEN_VALUE="$(env_file_value "${TELEGRAM_ENV_KEY}")"
  [[ -n "${TELEGRAM_TOKEN_VALUE}" ]] || die \
    "[ERR] .env.prod has no ${TELEGRAM_ENV_KEY}; the bot token is needed to run the bot and set its webhook." \
    "      ${ENV_FORMAT_HINT}"
  TELEGRAM_SECRET_VALUE="$(env_secret_value "${TELEGRAM_SECRET_ENV_KEY}" "TELEGRAM_WEBHOOK_SECRET_TOKEN")"
  if [[ -n "${TELEGRAM_SECRET_VALUE}" ]]; then
    [[ "${TELEGRAM_SECRET_VALUE}" =~ ${TELEGRAM_SECRET_RE} && ${#TELEGRAM_SECRET_VALUE} -le 256 ]] || die \
      "[ERR] ${TELEGRAM_SECRET_ENV_KEY} in .env.prod is not a valid Telegram secret token." \
      "      Telegram only accepts 1-256 characters from A-Z, a-z, 0-9, _ and -." \
      "      ${ENV_FORMAT_HINT}" \
      "      ${GENERATE_SECRET_HINT}"
  elif [[ "${WEBHOOK_ALLOW_UNAUTHENTICATED}" == "true" ]]; then
    unauthenticated_warning "Telegram webhook without ${TELEGRAM_SECRET_ENV_KEY}."
  else
    die \
      "[ERR] .env.prod has no ${TELEGRAM_SECRET_ENV_KEY}. Without it anyone can post fake Telegram updates to the webhook." \
      "      Add a line ${TELEGRAM_SECRET_ENV_KEY}=<secret> to .env.prod and re-run maws update." \
      "      ${GENERATE_SECRET_HINT}" \
      "      To deploy without webhook authentication (NOT recommended), set" \
      "      \"webhook_security\": {\"allow_unauthenticated\": true} in params.json."
  fi
else
  WHATSAPP_APP_SECRET_VALUE="$(env_secret_value "${WHATSAPP_APP_SECRET_ENV_KEY}" "WHATSAPP_APP_SECRET")"
  if [[ "${WEBHOOK_WHATSAPP_VERIFY}" == "true" && -n "${WHATSAPP_APP_SECRET_VALUE}" ]]; then
    :
  elif [[ "${WEBHOOK_ALLOW_UNAUTHENTICATED}" == "true" ]]; then
    WEBHOOK_WHATSAPP_VERIFY="false"
    unauthenticated_warning "WhatsApp webhook POSTs are not signature-verified."
  elif [[ "${WEBHOOK_WHATSAPP_VERIFY}" != "true" ]]; then
    die \
      "[ERR] webhook_security.whatsapp_verify_signature=false turns off WhatsApp webhook authentication." \
      "      Remove it from params.json (verification is on by default) and set ${WHATSAPP_APP_SECRET_ENV_KEY} in .env.prod." \
      "      To deploy without webhook authentication (NOT recommended), set" \
      "      \"webhook_security\": {\"allow_unauthenticated\": true} in params.json."
  else
    die \
      "[ERR] .env.prod has no ${WHATSAPP_APP_SECRET_ENV_KEY}. Without it anyone can post fake WhatsApp messages to the webhook." \
      "      Copy the App secret from Meta App Dashboard > App settings > Basic and add" \
      "      ${WHATSAPP_APP_SECRET_ENV_KEY}=<app secret> to .env.prod, then re-run maws update." \
      "      To deploy without webhook authentication (NOT recommended), set" \
      "      \"webhook_security\": {\"allow_unauthenticated\": true} in params.json."
  fi
fi

# --- Deploy-tool settings from .env.prod ---
# .env.prod is read, never sourced: its lines are not shell code (values with spaces, quotes, $ or
# ; would break or run). Only settings for aws/sam/pip (AWS_*, SAM_CLI_*, PIP_*, proxies) are
# exported to this deploy, as they were when the file used to be sourced; every value reaches Lambda
# through SSM.
strip_matching_quotes() {
  local v="$1"
  if [[ ${#v} -ge 2 ]]; then
    if [[ "${v:0:1}" == '"' && "${v: -1}" == '"' ]] || [[ "${v:0:1}" == "'" && "${v: -1}" == "'" ]]; then
      v="${v:1:${#v}-2}"
    fi
  fi
  printf '%s' "${v}"
}
DEPLOY_ENV_KEY_RE='^(AWS_[A-Za-z0-9_]+|SAM_CLI_[A-Za-z0-9_]+|PIP_[A-Za-z0-9_]+|HTTPS?_PROXY|NO_PROXY|https?_proxy|no_proxy)$'
while IFS= read -r env_key; do
  [[ "${env_key}" =~ ${DEPLOY_ENV_KEY_RE} ]] || continue
  export "${env_key}=$(strip_matching_quotes "$(env_file_value "${env_key}")")"
  echo "[INFO] Using ${env_key} from .env.prod for this deploy."
done < <(env_file_keys | sort -u)

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
[[ "${ACCOUNT_ID}" =~ ^[0-9]{12}$ ]] || die "[ERR] Could not determine the AWS account id (aws sts get-caller-identity returned '${ACCOUNT_ID}')."
echo ">> AWS account: ${ACCOUNT_ID}  Region: ${REGION}"

# --- Helpers for logical IDs (letters/numbers only and starting with a letter) ---
to_camel_case() {
  # "casancrem-2" -> "Casancrem2"
  local s="$1"
  s="${s//[^[:alnum:]]/ }"
  echo "$s" | awk '{for(i=1;i<=NF;i++){ $i=toupper(substr($i,1,1)) substr($i,2) }}1' | tr -d ' '
}
logical_id() {
  local base
  base="$(to_camel_case "$1")"
  [[ "$base" =~ ^[0-9] ]] && base="R${base}"
  echo "$base"
}

PROJECT_LOGICAL="$(logical_id "$PROJECT")"
FUNC_ID="${PROJECT_LOGICAL}Function"
BUCKET_ID="${PROJECT_LOGICAL}HistoryBucket"

# --- MAS requirement ---
MAS_EXTRA="maws,telegram,whatsapp,env"
case "${REQUIREMENTS_SOURCE}" in
  local)
    if [[ -z "${REQUIREMENTS_REF}" || "${REQUIREMENTS_REF}" == "null" ]]; then
      echo "requirements_source=local requires requirements_ref to point at a local package path"
      exit 1
    fi
    MAS_REQUIREMENT="${REQUIREMENTS_REF}"
    # A bare path to a wheel, sdist or source folder installs no extras, and the Lambda needs them
    # (python-telegram-bot for Telegram). A full requirement line with its own extras is kept as is.
    if [[ "${REQUIREMENTS_REF}" != *"["* && ( "${REQUIREMENTS_REF}" =~ ^[./] || "${REQUIREMENTS_REF}" =~ \.(whl|tar\.gz|zip)$ ) ]]; then
      MAS_REQUIREMENT="${REQUIREMENTS_REF}[${MAS_EXTRA}]"
    fi
    ;;
  git)
    MAS_REQUIREMENT="mas[${MAS_EXTRA}] @ git+https://github.com/mneuronico/multi-agent-system-library.git"
    if [[ -n "${REQUIREMENTS_REF}" && "${REQUIREMENTS_REF}" != "null" ]]; then
      MAS_REQUIREMENT="${MAS_REQUIREMENT}@${REQUIREMENTS_REF}"
    else
      echo "[WARN] requirements_ref is not set: Lambda gets whatever is on the default branch of"
      echo "       github.com/mneuronico/multi-agent-system-library at deploy time, so two deploys of the"
      echo "       same project can ship different MAS/MAWS code. Pin it in params.json to a commit SHA"
      echo "       (most reproducible) or a release tag, e.g. \"requirements_ref\": \"<commit sha>\"."
    fi
    ;;
esac

# --- Logs: an existing stack may already own the legacy log group (/aws/lambda/<function>) ---
LEGACY_LOG_GROUP_ID="${PROJECT_LOGICAL}FunctionLogGroup"
LEGACY_LOG_GROUP=$(aws cloudformation describe-stack-resource \
  --stack-name "${STACK}" --logical-resource-id "${LEGACY_LOG_GROUP_ID}" --region "${REGION}" \
  --query StackResourceDetail.PhysicalResourceId --output text 2>/dev/null || true)
[[ "${LEGACY_LOG_GROUP}" == "None" ]] && LEGACY_LOG_GROUP=""
LOG_GROUP_ID="${PROJECT_LOGICAL}Logs"
LOG_GROUP_NAME="/aws/lambda/${STACK}-maws"
if [[ -n "${LEGACY_LOG_GROUP}" ]]; then
  LOG_GROUP_ID="${LEGACY_LOG_GROUP_ID}"
  LOG_GROUP_NAME="${LEGACY_LOG_GROUP}"
fi

# --- template.yaml (build artifact) ---
echo ">> Generating ${TEMPLATE_FILE}"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cat > "${TEMPLATE_FILE}" <<EOF
# Generated by maws update (bootstrap.sh) from ${CONF} on every deploy: do not edit, change params.json.
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: ${PROJECT}

Globals:
  Function:
    Timeout: ${LAMBDA_TIMEOUT}
    Runtime: ${LAMBDA_RUNTIME}
    MemorySize: ${MEMORY_SIZE}
    Tracing: ${FUNCTION_TRACING}
  Api:
    TracingEnabled: ${TRACING_ENABLED}

Parameters:
  EnvParamName:
    Type: String
    Default: "${ENV_PARAM}"
  EnvParamVersion:
    Type: String
    Default: ""
    Description: SSM parameter version the function reads (set by maws update).

Resources:
  ProcessingLocksTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub "\${AWS::StackName}-locks"
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: user_id
          AttributeType: S
      KeySchema:
        - AttributeName: user_id
          KeyType: HASH
      TimeToLiveSpecification:
        AttributeName: expiresAt
        Enabled: true
  ${BUCKET_ID}:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: ${HISTORY_BUCKET}
EOF

if [[ "${S3_VERSIONING}" == "true" ]]; then
  cat >> "${TEMPLATE_FILE}" <<'EOF'
      VersioningConfiguration:
        Status: Enabled
EOF
fi

if [[ -n "${S3_ENCRYPTION}" && "${S3_ENCRYPTION}" != "null" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: ${S3_ENCRYPTION}
EOF
fi

# Lifecycle: expire user data (histories, files, failed-event captures) only, never the token
# files under tokens_s3_prefix; with versioning, also expire noncurrent versions.
LIFECYCLE_RULES=""
if [[ -n "${S3_LIFECYCLE_DAYS}" ]]; then
  for rule in "ExpireHistory:history" "ExpireFiles:${FILES_PREFIX%/}" "ExpireFailedEvents:failed-events"; do
    rule_id="${rule%%:*}"
    rule_prefix="${rule#*:}"
    [[ -n "${rule_prefix}" ]] || continue
    if [[ "${rule_prefix}" == "${TOKENS_PREFIX%/}" ]]; then
      echo "[WARN] files_s3_prefix equals tokens_s3_prefix (${rule_prefix}); not expiring it so token files are kept."
      continue
    fi
    # Every shared-history file holds many users' history and MAS reads all of them, but a file
    # is never written again once it is closed: expiring it would delete live history.
    if [[ "${rule_id}" == "ExpireHistory" && "${HISTORY_MODE}" == "shared" ]]; then
      echo "[INFO] history_mode=shared: history/ does not expire; infra.s3_lifecycle_days applies to files and failed events only."
      continue
    fi
    LIFECYCLE_RULES+="          - Id: ${rule_id}
            Status: Enabled
            Prefix: ${rule_prefix}/
            ExpirationInDays: ${S3_LIFECYCLE_DAYS}
"
  done
fi
if [[ "${S3_VERSIONING}" == "true" && "${S3_NONCURRENT_DAYS}" -gt 0 ]]; then
  LIFECYCLE_RULES+="          - Id: ExpireNoncurrentVersions
            Status: Enabled
            NoncurrentVersionExpiration:
              NoncurrentDays: ${S3_NONCURRENT_DAYS}
          - Id: RemoveExpiredDeleteMarkers
            Status: Enabled
            ExpiredObjectDeleteMarker: true
"
elif [[ "${S3_VERSIONING}" == "true" ]]; then
  echo "[WARN] infra.s3_noncurrent_version_days=0: every old version of every history upload is kept (and billed)."
fi
if [[ -n "${LIFECYCLE_RULES}" ]]; then
  printf '      LifecycleConfiguration:\n        Rules:\n%s' "${LIFECYCLE_RULES}" >> "${TEMPLATE_FILE}"
fi

# Lambda writes to this stack-managed log group (LoggingConfig.LogGroup). A new name is used because
# Lambda already created /aws/lambda/<function> for deployed functions, and CloudFormation cannot
# create a log group that exists. Stacks that already own that group keep it.
if [[ -n "${LEGACY_LOG_GROUP}" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
  ${LOG_GROUP_ID}:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub "/aws/lambda/\${${FUNC_ID}}"
EOF
else
  cat >> "${TEMPLATE_FILE}" <<EOF
  ${LOG_GROUP_ID}:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub "/aws/lambda/\${AWS::StackName}-maws"
EOF
fi
if [[ "${LOG_RETENTION_DAYS}" -gt 0 ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
      RetentionInDays: ${LOG_RETENTION_DAYS}
EOF
fi

if [[ "${BUSY_POLICY}" == "fifo" && "${WORKER_DLQ}" == "true" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
  ${PROJECT_LOGICAL}WorkerDlq:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub "\${AWS::StackName}-worker-dlq.fifo"
      FifoQueue: true
      MessageRetentionPeriod: 1209600

EOF
fi

if [[ "${BUSY_POLICY}" == "fifo" ]]; then
  if [[ "${WORKER_DLQ}" != "true" ]]; then
    echo "[WARN] failure_handling.worker_dlq=false: a job that keeps failing is retried until SQS drops it"
    echo "       (4 days) and blocks every later message of that chat meanwhile. The DLQ is the default."
  fi
  cat >> "${TEMPLATE_FILE}" <<EOF
  ${PROJECT_LOGICAL}WorkerQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub "\${AWS::StackName}-worker.fifo"
      FifoQueue: true
      ContentBasedDeduplication: false
      VisibilityTimeout: ${VISIBILITY_TIMEOUT}
EOF
  if [[ "${WORKER_DLQ}" == "true" ]]; then
    cat >> "${TEMPLATE_FILE}" <<EOF
      RedrivePolicy:
        deadLetterTargetArn: !GetAtt ${PROJECT_LOGICAL}WorkerDlq.Arn
        maxReceiveCount: ${MAX_RECEIVE_COUNT}
EOF
  fi
  echo "" >> "${TEMPLATE_FILE}"
fi

if [[ "${ASYNC_FAILURE_QUEUE}" == "true" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
  ${PROJECT_LOGICAL}AsyncFailureQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub "\${AWS::StackName}-async-failures"
      MessageRetentionPeriod: 1209600
      SqsManagedSseEnabled: true

EOF
fi

cat >> "${TEMPLATE_FILE}" <<EOF
  ${FUNC_ID}:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: ${PACKAGE_DIR}
      Handler: lambda_function.lambda_handler
      Runtime: ${LAMBDA_RUNTIME}
      Description: "MAS Bot Lambda"
      Architectures: [ ${ARCHITECTURE} ]
      LoggingConfig:
        LogFormat: JSON
EOF

if [[ -z "${LEGACY_LOG_GROUP}" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
        LogGroup: !Ref ${LOG_GROUP_ID}
EOF
fi

if [[ -n "${RESERVED_CONCURRENCY}" && "${RESERVED_CONCURRENCY}" != "null" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
      ReservedConcurrentExecutions: ${RESERVED_CONCURRENCY}
EOF
fi

if [[ "${ASYNC_FAILURE_QUEUE}" == "true" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
      EventInvokeConfig:
        MaximumRetryAttempts: 2
        DestinationConfig:
          OnFailure:
            Type: SQS
            Destination: !GetAtt ${PROJECT_LOGICAL}AsyncFailureQueue.Arn
EOF
fi

# No lambda:InvokeFunction here: the self-invoke permission is a separate AWS::IAM::Policy scoped
# to this function (see SelfInvokePolicy below). Referencing the function from its own role's
# policies would be a circular dependency.
cat >> "${TEMPLATE_FILE}" <<EOF
      Policies:
        - AWSLambdaBasicExecutionRole
        - AWSXRayDaemonWriteAccess
        - Statement:
            - Effect: Allow
              Action: [ "ssm:GetParameter" ]
              Resource: !Sub arn:aws:ssm:\${AWS::Region}:\${AWS::AccountId}:parameter\${EnvParamName}
        - Statement:
            - Effect: Allow
              Action: [ "s3:ListBucket" ]
              Resource: "arn:aws:s3:::${HISTORY_BUCKET}"
            - Effect: Allow
              Action: [ "s3:GetObject", "s3:PutObject" ]
              Resource: "arn:aws:s3:::${HISTORY_BUCKET}/*"
EOF

if [[ "${PERSIST_FILES_S3}" == "true" ]]; then
  # Files deleted during a job (for example by clearing a user's history) are deleted in S3 too.
  FILES_PREFIX_ARN="${FILES_PREFIX#/}"
  FILES_PREFIX_ARN="${FILES_PREFIX_ARN%/}"
  cat >> "${TEMPLATE_FILE}" <<EOF
        - Statement:
            - Effect: Allow
              Action: [ "s3:DeleteObject" ]
              Resource: "arn:aws:s3:::${HISTORY_BUCKET}/${FILES_PREFIX_ARN}/*"
EOF
fi

if [[ "${BUSY_POLICY}" == "fifo" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
        - Statement:
            - Effect: Allow
              Action: [ "sqs:SendMessage" ]
              Resource: !GetAtt ${PROJECT_LOGICAL}WorkerQueue.Arn
EOF
fi

if [[ "${ASYNC_FAILURE_QUEUE}" == "true" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
        - Statement:
            - Effect: Allow
              Action: [ "sqs:SendMessage" ]
              Resource: !GetAtt ${PROJECT_LOGICAL}AsyncFailureQueue.Arn
EOF
fi

cat >> "${TEMPLATE_FILE}" <<EOF
        - Statement:
            - Effect: Allow
              Action:
                - dynamodb:PutItem
                - dynamodb:DeleteItem
              Resource: !GetAtt ProcessingLocksTable.Arn
      Environment:
        Variables:
          # Core
          BOT_TYPE: "${BOT}"
          VERBOSE: "${VERBOSE}"
          BUCKET_NAME: "${HISTORY_BUCKET}"
          ENV_PARAMETER_NAME: "${ENV_PARAM}"
          # The function reads exactly this SSM version, so a failed deploy never leaves it
          # running with secrets written for a different deploy.
          MAWS_ENV_PARAMETER_VERSION: !Ref EnvParamVersion

          # Runtime behavior
          MAWS_BUSY_POLICY: "${BUSY_POLICY}"
          PERSIST_FILES_S3: "${PERSIST_FILES_S3}"
          FILES_S3_PREFIX: "${FILES_PREFIX}"
          HISTORY_MODE: "${HISTORY_MODE}"
          HISTORY_ROTATION: "${HISTORY_ROTATION}"
          HISTORY_MAX_MESSAGES: "${HISTORY_MAX_MESSAGES}"
          HISTORY_PERIOD: "${HISTORY_PERIOD}"
          MAWS_MANAGER_KWARGS_JSON: '${MANAGER_KWARGS_JSON}'
          MAWS_BOT_KWARGS_JSON: '${BOT_KWARGS_JSON}'
          ENSURE_DELIVERY: "${ENSURE_DELIVERY}"
          DELIVERY_TIMEOUT: "${DELIVERY_TIMEOUT}"
          MAX_ALLOWED_MESSAGE_DELAY: "${MAX_ALLOWED_MESSAGE_DELAY}"

          # Tokens / files
          SYNC_TOKENS_S3: "${SYNC_TOKENS_S3}"
          TOKENS_S3_PREFIX: "${TOKENS_PREFIX}"
          SPECIAL_TOKEN_FILES_JSON: '${SPECIAL_JSON}'
          TOKEN_ENV_MAP_JSON: '${TOKEN_ENV_MAP}'

          # Provider-supported webhook security (mandatory unless explicitly allowed)
          TELEGRAM_WEBHOOK_SECRET_TOKEN_ENV_KEY: "${TELEGRAM_SECRET_ENV_KEY}"
          WHATSAPP_VERIFY_SIGNATURE: "${WEBHOOK_WHATSAPP_VERIFY}"
          WHATSAPP_APP_SECRET_ENV_KEY: "${WHATSAPP_APP_SECRET_ENV_KEY}"
          MAWS_ALLOW_UNAUTHENTICATED_WEBHOOKS: "${WEBHOOK_ALLOW_UNAUTHENTICATED}"

          # Failure capture
          CAPTURE_FAILED_EVENTS: "${CAPTURE_FAILED_EVENTS}"
EOF

if [[ -n "${BUSY_WAIT_SECONDS}" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
          MAWS_BUSY_WAIT_SECONDS: "${BUSY_WAIT_SECONDS}"
EOF
fi

if [[ -n "${DEDUP_RETENTION_DAYS}" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
          MAWS_DEDUP_RETENTION_DAYS: "${DEDUP_RETENTION_DAYS}"
EOF
fi

if [[ "${BUSY_POLICY}" == "fifo" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
          MAWS_QUEUE_URL: !Ref ${PROJECT_LOGICAL}WorkerQueue
EOF
fi

if [[ "${HISTORY_APPEND_ENABLED}" == "true" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
          MAWS_HISTORY_APPEND_ENABLED: "true"
EOF
  if [[ -n "${HISTORY_APPEND_ROLE}" ]]; then
    cat >> "${TEMPLATE_FILE}" <<EOF
          MAWS_HISTORY_APPEND_ROLE: "${HISTORY_APPEND_ROLE}"
EOF
  fi
fi

cat >> "${TEMPLATE_FILE}" <<EOF

          # Locks
          LOCKS_TABLE_NAME: !Ref ProcessingLocksTable
          LOCK_TTL_SECONDS: "${LOCK_TTL_SECONDS}"
      Events:
EOF

if [[ "${BOT}" == "telegram" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
        TelegramWebhook:
          Type: Api
          Properties:
            Path: "${API_PATH}"
            Method: post
EOF
else
  cat >> "${TEMPLATE_FILE}" <<EOF
        WhatsAppWebhookGET:
          Type: Api
          Properties: { Path: "${API_PATH}", Method: get }
        WhatsAppWebhookPOST:
          Type: Api
          Properties: { Path: "${API_PATH}", Method: post }
EOF
fi

if [[ "${HISTORY_APPEND_ENABLED}" == "true" ]]; then
  cat >> "${TEMPLATE_FILE}" <<'EOF'
        HistoryAppendPOST:
          Type: Api
          Properties: { Path: "/history/outgoing", Method: post }
EOF
fi

if [[ "${BUSY_POLICY}" == "fifo" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
        WorkerQueueEvent:
          Type: SQS
          Properties:
            Queue: !GetAtt ${PROJECT_LOGICAL}WorkerQueue.Arn
            BatchSize: 1
EOF
fi

# busy_policy=drop self-invokes the function: allow exactly this function (and its versions/aliases).
if [[ "${BUSY_POLICY}" == "drop" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF

  ${PROJECT_LOGICAL}SelfInvokePolicy:
    Type: AWS::IAM::Policy
    Properties:
      PolicyName: maws-self-invoke
      Roles:
        - !Ref ${FUNC_ID}Role
      PolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Action: lambda:InvokeFunction
            Resource:
              - !GetAtt ${FUNC_ID}.Arn
              - !Sub "\${${FUNC_ID}.Arn}:*"
EOF
fi

cat >> "${TEMPLATE_FILE}" <<EOF

Outputs:
  ApiUrl:
    Description: Base URL for the API
    Value: !Sub "https://\${ServerlessRestApi}.execute-api.\${AWS::Region}.amazonaws.com/Prod"
  FunctionLogGroup:
    Description: CloudWatch log group of the bot function
    Value: !Ref ${LOG_GROUP_ID}
EOF
if [[ "${BUSY_POLICY}" == "fifo" && "${WORKER_DLQ}" == "true" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
  WorkerDeadLetterQueueUrl:
    Description: SQS FIFO dead-letter queue of failed worker jobs
    Value: !Ref ${PROJECT_LOGICAL}WorkerDlq
EOF
fi
if [[ "${ASYNC_FAILURE_QUEUE}" == "true" ]]; then
  cat >> "${TEMPLATE_FILE}" <<EOF
  AsyncFailureQueueUrl:
    Description: SQS queue receiving async worker invocations that failed after Lambda's retries
    Value: !Ref ${PROJECT_LOGICAL}AsyncFailureQueue
EOF
fi

# --- Buckets: never use a bucket owned by another account ---
# Bucket names are global; a predictable name may have been created by someone else first.
head_bucket_owned() {
  aws s3api head-bucket --bucket "$1" --expected-bucket-owner "${ACCOUNT_ID}" 2>&1
}
is_not_found() { [[ "$1" == *"(404)"* || "$1" == *"Not Found"* || "$1" == *"NoSuchBucket"* ]]; }

echo ">> Ensuring deployment bucket: ${DEPLOY_BUCKET}"
if ! HEAD_OUT=$(head_bucket_owned "${DEPLOY_BUCKET}"); then
  if is_not_found "${HEAD_OUT}"; then
    aws s3 mb "s3://${DEPLOY_BUCKET}" --region "${REGION}" || die \
      "[ERR] Could not create deployment bucket ${DEPLOY_BUCKET} (the name may be taken by another account)." \
      "      Set \"deployment_bucket\" in params.json to a unique name."
    HEAD_OUT=$(head_bucket_owned "${DEPLOY_BUCKET}") || die \
      "[ERR] Deployment bucket ${DEPLOY_BUCKET} is not owned by account ${ACCOUNT_ID}: ${HEAD_OUT}"
  else
    die "[ERR] Deployment bucket ${DEPLOY_BUCKET} exists but is not owned by account ${ACCOUNT_ID} (or is not accessible):" \
      "      ${HEAD_OUT}" \
      "      Set \"deployment_bucket\" in params.json to a bucket name your account owns, or a new unique name."
  fi
fi

echo ">> Checking history bucket: ${HISTORY_BUCKET}"
if ! HEAD_OUT=$(head_bucket_owned "${HISTORY_BUCKET}"); then
  is_not_found "${HEAD_OUT}" || die \
    "[ERR] History bucket ${HISTORY_BUCKET} exists but is not owned by account ${ACCOUNT_ID} (or is not accessible):" \
    "      ${HEAD_OUT}" \
    "      If this is a new project, set \"history_bucket\" in params.json to a new unique name." \
    "      Never change it for a deployed project: a new bucket means starting with empty histories."
  echo "   (does not exist yet; the stack creates it)"
fi

# --- Ensure modern build toolchain before sam build (best effort) ---
# An externally managed system Python (PEP 668, e.g. Debian/Ubuntu) refuses this; that must not
# abort the deploy under set -e.
echo ">> Upgrading pip / setuptools / wheel (best effort)"
if ! python3 -m pip install --quiet --upgrade pip "setuptools>=77" wheel; then
  echo "[WARN] Could not upgrade pip/setuptools/wheel (for example an externally managed system Python, PEP 668)."
  echo "       Continuing with the installed versions. If sam build then fails building packages, run maws update from a virtualenv."
fi

# sam build needs a local interpreter matching the Lambda runtime (unless it builds in a container).
if [[ "${LAMBDA_RUNTIME}" == python3.* ]] && ! command -v "${LAMBDA_RUNTIME}" >/dev/null 2>&1; then
  LOCAL_PYTHON=$(python3 -c 'import sys; print("python%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "unknown")
  if [[ "${LOCAL_PYTHON}" != "${LAMBDA_RUNTIME}" ]]; then
    echo "[WARN] The Lambda runtime is ${LAMBDA_RUNTIME} but ${LAMBDA_RUNTIME} is not on PATH (python3 is ${LOCAL_PYTHON})."
    echo "       The build will likely fail: install ${LAMBDA_RUNTIME}, or set \"infra\": {\"runtime\": \"${LOCAL_PYTHON}\"} in params.json."
  fi
fi

# --- Stage the Lambda package ---
# SAM CLI does not read .samignore: everything under CodeUri ships in the zip and stays in the
# deployment bucket. Copy only the project code, never secrets (.env*, api_keys.json), user
# histories or deploy files. Secrets reach Lambda through SSM instead.
echo ">> Staging Lambda package in ${PACKAGE_DIR}/"
rm -rf "${PACKAGE_DIR}"
python3 - "${PACKAGE_DIR}" <<'PY'
import fnmatch
import os
import shutil
import sys

dest = sys.argv[1]
top_level = {
    dest, ".maws_build", ".aws-sam", ".git", "history", "files", "logs", "tmp", "local_s3_bucket", ".venv", "venv",
    "params.json", "params.tmp.json", "samconfig.toml", "template.yaml", ".samignore", ".bootstrap_state.json",
    ".gitignore", "bootstrap.sh", "deploy.sh", "pull_history.sh", "README.md",
}
anywhere = [
    ".env", ".env.*", "*.env", "api_keys.json",
    "*.sqlite", "*.sqlite-wal", "*.sqlite-shm", "__pycache__", "*.pyc",
]
root = os.path.abspath(".")

def ignore(directory, names):
    skipped = {n for n in names if any(fnmatch.fnmatch(n, p) for p in anywhere)}
    if os.path.abspath(directory) == root:
        skipped |= top_level & set(names)
    return skipped

shutil.copytree(".", dest, ignore=ignore)
PY
if [[ -f "api_keys.json" ]]; then
  echo "[WARN] api_keys.json is not packaged. Move those keys to .env.prod so they reach Lambda via SSM."
fi

# --- lambda_function.py and requirements.txt (generated into the package only) ---
# A project file is used as is when you wrote it. MAWS writes its own version into the package
# (never into the project) when the project has none, or when the project file was generated by
# maws: it starts with GENERATED_MARKER, or it is exactly what older maws versions wrote there.
GENERATED_MARKER="# Generated by maws update"
LEGACY_HANDLER=$'import os\nfrom maws import build_lambda_handler\n\nBOT_TYPE = os.environ.get("BOT_TYPE", "whatsapp")\nlambda_handler = build_lambda_handler(BOT_TYPE)'
LEGACY_MAS_RE='^mas(\[[A-Za-z0-9_,]*\])?( @ git[+]https://github[.]com/mneuronico/multi-agent-system-library[.]git(@[^[:space:]]+)?|==[^[:space:]]+)?$'
# What older maws wrote for requirements_source "local" pointing at a mas wheel or sdist.
LEGACY_LOCAL_MAS_RE='^([^[:space:]]*/)?mas-[0-9][^/[:space:]]*[.](whl|tar[.]gz|zip)(\[[A-Za-z0-9_,]*\])?$'

handler_is_generated() {
  [[ -f lambda_function.py ]] || return 0
  local content
  content="$(sed 's/\r$//' lambda_function.py)"
  [[ "${content}" == "${GENERATED_MARKER}"* || "${content}" == "${LEGACY_HANDLER}" ]]
}

is_extra_requirement() {
  local wanted="$1" r
  for r in "${EXTRA_REQ[@]}"; do
    [[ "${r}" == "${wanted}" ]] && return 0
  done
  return 1
}

requirement_matches_params() {
  local line="$1"
  [[ "${line}" == "requests" || "${line}" == "boto3" || "${line}" == "${MAS_REQUIREMENT}" ]] && return 0
  [[ "${line}" =~ ${LEGACY_MAS_RE} ]] && return 0
  # Older maws wrote a local requirements_ref without the extras.
  [[ "${REQUIREMENTS_SOURCE}" == "local" && "${line}" == "${REQUIREMENTS_REF}" ]] && return 0
  is_extra_requirement "${line}"
}

requirements_are_generated() {
  [[ -f requirements.txt ]] || return 0
  local line first=1 matches_params=1 has_requests=0 has_boto3=0 has_old_mas=0 has_comment_or_option=0
  local -a lines=()
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    if [[ ${first} -eq 1 && "${line}" == "${GENERATED_MARKER}"* ]]; then
      return 0
    fi
    first=0
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" ]] && continue
    lines+=("${line}")
    requirement_matches_params "${line}" || matches_params=0
    case "${line}" in
      requests) has_requests=1 ;;
      boto3) has_boto3=1 ;;
      "#"* | -*) has_comment_or_option=1 ;;
    esac
    [[ "${line}" =~ ${LEGACY_MAS_RE} || "${line}" =~ ${LEGACY_LOCAL_MAS_RE} ]] && has_old_mas=1
  done < requirements.txt
  [[ ${matches_params} -eq 1 ]] && return 0
  # MAWS 0.3.x rewrote this file from params.json on every deploy, so after params.json changes
  # (another requirements_ref, other extra_requirements) it no longer matches line by line. Its
  # shape still tells it apart: requests, boto3 and a mas line, sorted, with no comments or options.
  [[ ${has_requests} -eq 1 && ${has_boto3} -eq 1 && ${has_old_mas} -eq 1 && ${has_comment_or_option} -eq 0 ]] || return 1
  # 0.3.x sorted in the deploy machine's locale.
  printf '%s\n' "${lines[@]}" | LC_ALL=C sort -cu 2>/dev/null || printf '%s\n' "${lines[@]}" | sort -cu 2>/dev/null
}

if handler_is_generated; then
  echo ">> Generating ${PACKAGE_DIR}/lambda_function.py (${BOT})"
  cat > "${PACKAGE_DIR}/lambda_function.py" <<'PY'
# Generated by maws update into the Lambda package. To customise the handler, create your own
# lambda_function.py in the project folder: maws then deploys yours unchanged.
import os
from maws import build_lambda_handler

BOT_TYPE = os.environ.get("BOT_TYPE", "whatsapp")
lambda_handler = build_lambda_handler(BOT_TYPE)
PY
else
  echo "[INFO] Using your lambda_function.py (it was not generated by maws)."
fi

if requirements_are_generated; then
  echo ">> Generating ${PACKAGE_DIR}/requirements.txt"
  if [[ -f requirements.txt ]]; then
    echo "[INFO] requirements.txt in the project folder was written by an older maws, so it is not deployed:"
    echo "       the package gets one generated from params.json. You can delete it. If you wrote it"
    echo "       yourself, add a comment line to it and maws deploys it unchanged."
  fi
  {
    echo "${GENERATED_MARKER} from params.json (requirements_source/requirements_ref/extra_requirements)."
    {
      echo "requests"
      echo "boto3"
      echo "${MAS_REQUIREMENT}"
      for r in "${EXTRA_REQ[@]}"; do echo "$r"; done
    } | awk 'NF' | sort -u
  } > "${PACKAGE_DIR}/requirements.txt"
else
  echo "[INFO] Using your requirements.txt (it was not generated by maws); requirements_source,"
  echo "       requirements_ref and extra_requirements in params.json do not apply to it."
  if ! grep -Eq '^[[:space:]]*mas([[:space:]]|\[|=|@|$)' requirements.txt; then
    echo "[WARN] Your requirements.txt has no 'mas' line: the Lambda needs ${MAS_REQUIREMENT}"
  fi
fi

if [[ -f template.yaml || -f samconfig.toml ]]; then
  echo "[INFO] template.yaml / samconfig.toml in the project folder are not used: maws generates"
  echo "       ${TEMPLATE_FILE} from params.json on every deploy. Delete them if you did not write them."
fi

# --- Build (generated template, staged package): sam build
# --base-dir . resolves CodeUri (${PACKAGE_DIR}) from the project folder, not from ${BUILD_DIR}.
echo ">> sam build"
sam build --template-file "${TEMPLATE_FILE}" --base-dir . --build-dir "${SAM_BUILD_DIR}"

# --- Upload .env.prod to SSM ---
# Every upload creates a new parameter version and the function reads the exact version it was
# deployed with (MAWS_ENV_PARAMETER_VERSION). Uploading before the deploy is therefore safe: if
# the deploy fails, CloudFormation rolls back and the running function keeps reading its old
# version. (Uploading after the deploy instead would let the new function start with old secrets.)
echo ">> Uploading .env.prod to SSM: ${ENV_PARAM}"
ENV_PARAM_VERSION=$(aws ssm put-parameter \
  --name "${ENV_PARAM}" \
  --type "SecureString" \
  --value "file://.env.prod" \
  --overwrite \
  --region "${REGION}" \
  --query Version --output text)
[[ "${ENV_PARAM_VERSION}" =~ ^[0-9]+$ ]] || die "[ERR] Could not read the new version of SSM parameter ${ENV_PARAM} (got: '${ENV_PARAM_VERSION}')."
echo "   SSM parameter version ${ENV_PARAM_VERSION}"

# --- Deploy ---
# All deploy settings are passed here; no samconfig.toml is generated.
echo ">> sam deploy"
sam deploy \
  --template-file "${SAM_BUILD_DIR}/template.yaml" \
  --stack-name "${STACK}" \
  --s3-bucket "${DEPLOY_BUCKET}" \
  --s3-prefix "${STACK}" \
  --region "${REGION}" \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides "EnvParamVersion=${ENV_PARAM_VERSION}" \
  --no-confirm-changeset \
  --no-fail-on-empty-changeset

# --- Retrieve API URL ---
API_URL=$(aws cloudformation describe-stacks \
  --stack-name "${STACK}" \
  --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" \
  --output text --region "${REGION}")

if [[ -z "${API_URL}" || "${API_URL}" == "None" ]]; then
  echo "Could not get ApiUrl from outputs. Check CloudFormation."
  exit 1
fi

WEBHOOK_URL="${API_URL}$(printf "%s" "${API_PATH}")"
echo ">> API URL: ${API_URL}"
echo ">> Webhook full URL: ${WEBHOOK_URL}"

if [[ "${BOT}" == "telegram" ]]; then
  # setWebhook is idempotent and always called, so a changed URL or secret always reaches Telegram.
  # secret_token is only omitted on allow_unauthenticated deploys without a configured secret.
  echo ">> Setting Telegram webhook..."
  CURL_ARGS=(--data-urlencode "url=${WEBHOOK_URL}")
  if [[ -n "${TELEGRAM_SECRET_VALUE}" ]]; then
    CURL_ARGS+=(--data-urlencode "secret_token=${TELEGRAM_SECRET_VALUE}")
  fi
  if ! TG_RESPONSE=$(curl -sS --max-time 30 "https://api.telegram.org/bot${TELEGRAM_TOKEN_VALUE}/setWebhook" "${CURL_ARGS[@]}"); then
    die "[ERR] Could not reach the Telegram API to set the webhook." \
      "      The stack is deployed, but Telegram will not deliver updates to it. Re-run maws update to retry."
  fi
  TG_OK=$(printf '%s' "${TG_RESPONSE}" | jq -r 'if type == "object" then (.ok == true) else false end' 2>/dev/null || echo "false")
  if [[ "${TG_OK}" != "true" ]]; then
    TG_DESCRIPTION=$(printf '%s' "${TG_RESPONSE}" | jq -r 'if type == "object" then (.description // "no description") else "unexpected response" end' 2>/dev/null || echo "unexpected (non-JSON) response")
    die "[ERR] Telegram rejected setWebhook: ${TG_DESCRIPTION}" \
      "      The stack is deployed, but Telegram will not deliver updates to it." \
      "      Check ${TELEGRAM_ENV_KEY} and ${TELEGRAM_SECRET_ENV_KEY} in .env.prod, then re-run maws update."
  fi
  if [[ -n "${TELEGRAM_SECRET_VALUE}" ]]; then
    echo "[OK] Telegram webhook set to: ${WEBHOOK_URL} (secret token enabled)"
  else
    echo "[OK] Telegram webhook set to: ${WEBHOOK_URL} (UNAUTHENTICATED: no secret token)"
  fi
else
  echo "[INFO] WhatsApp webhooks:"
  echo "    - Configure the callback URL in Meta Developers:"
  echo "      ${WEBHOOK_URL}"
  echo "    - Verify token (from .env.prod) => key: ${WHATSAPP_VERIFY_ENV_KEY:-WHATSAPP_VERIFY_TOKEN}"
  echo "    - Remember to add the same verify token in your Meta app."
  if [[ "${WEBHOOK_WHATSAPP_VERIFY}" == "true" ]]; then
    echo "    - POST signature verification (X-Hub-Signature-256) is enabled with ${WHATSAPP_APP_SECRET_ENV_KEY}."
  else
    echo "    - [WARN] POST signature verification is OFF (webhook_security.allow_unauthenticated=true)."
  fi
fi

if [[ "${HISTORY_APPEND_ENABLED}" == "true" ]]; then
  echo "[INFO] History append endpoint: ${API_URL}/history/outgoing"
  if [[ -z "$(env_file_value MAWS_HISTORY_WRITE_SECRET)" ]]; then
    echo "[WARN] history_append.enabled=true but .env.prod has no MAWS_HISTORY_WRITE_SECRET; the endpoint will answer 503."
  fi
fi

echo "[INFO] Logs: CloudWatch log group ${LOG_GROUP_NAME} (retention: $([[ "${LOG_RETENTION_DAYS}" -gt 0 ]] && echo "${LOG_RETENTION_DAYS} days" || echo "never expire"))."
echo "       Dropped/failed updates: CloudWatch metrics MAWS/DroppedMessages and MAWS/FailedMessages."
if [[ "${BUSY_POLICY}" == "fifo" && "${WORKER_DLQ}" == "true" ]]; then
  echo "[INFO] Failed worker jobs go to SQS queue ${STACK}-worker-dlq.fifo after ${MAX_RECEIVE_COUNT} receives."
fi
if [[ "${ASYNC_FAILURE_QUEUE}" == "true" ]]; then
  echo "[INFO] Worker invocations that fail after Lambda's 2 retries go to SQS queue ${STACK}-async-failures."
fi

echo "[OK] Done."
