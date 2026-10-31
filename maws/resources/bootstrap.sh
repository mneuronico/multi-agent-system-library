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

PROJECT=$(jq -r '.project' "${CONF}")
REGION=$(jq -r '.region' "${CONF}")
BOT=$(jq -r '.bot' "${CONF}") # "telegram" | "whatsapp"

STACK=$(jq -r '.stack_name // empty' "${CONF}")
[[ -z "${STACK}" || "${STACK}" == "null" ]] && STACK="${PROJECT}-stack"

HISTORY_BUCKET=$(jq -r '.history_bucket // empty' "${CONF}")
[[ -z "${HISTORY_BUCKET}" || "${HISTORY_BUCKET}" == "null" ]] && HISTORY_BUCKET="${PROJECT}-history-bucket"

DEPLOY_BUCKET=$(jq -r '.deployment_bucket // empty' "${CONF}")
[[ -z "${DEPLOY_BUCKET}" || "${DEPLOY_BUCKET}" == "null" ]] && DEPLOY_BUCKET="${PROJECT}-deployment-bucket"

# Lambda code is staged here so only project code ends up in the deployment zip.
PACKAGE_DIR=".maws_package"

ENV_PARAM=$(jq -r '.env_param_name // empty' "${CONF}")
[[ -z "${ENV_PARAM}" || "${ENV_PARAM}" == "null" ]] && ENV_PARAM="/${PROJECT}/prod/env"

API_PATH=$(jq -r '.api_path // "/webhook"' "${CONF}")

SYNC_TOKENS_S3=$(jq -r '.sync_tokens_s3 // true' "${CONF}")
TOKENS_PREFIX=$(jq -r '.tokens_s3_prefix // "secrets"' "${CONF}")
VERBOSE=$(jq -r '.verbose // false' "${CONF}")
LAMBDA_TIMEOUT=$(jq -r '.lambda_timeout // 120' "${CONF}")
# SQS makes a message visible again VISIBILITY_TIMEOUT seconds after a worker received it. The user
# lock must outlive a running invocation (lambda_timeout) but expire before that redelivery, or the
# retry of a crashed worker's message finds the user still busy. The runtime ties the lock to the
# invocation deadline + 15 s; LOCK_TTL_SECONDS is its fallback when no Lambda context is available.
VISIBILITY_TIMEOUT=$((LAMBDA_TIMEOUT + 30))
LOCK_TTL_SECONDS=$((LAMBDA_TIMEOUT + 15))

BUSY_POLICY=$(jq -r '.busy_policy // "drop"' "${CONF}")
PERSIST_FILES_S3=$(jq -r '.persist_files_s3 // false' "${CONF}")
FILES_PREFIX=$(jq -r '.files_s3_prefix // "files"' "${CONF}")

HISTORY_MODE=$(jq -r '.history_mode // empty' "${CONF}")
[[ -z "${HISTORY_MODE}" || "${HISTORY_MODE}" == "null" ]] && HISTORY_MODE="per_user"
HISTORY_ROTATION=$(jq -r '.history_rotation // "message_count"' "${CONF}")
HISTORY_MAX_MESSAGES=$(jq -r '.history_max_messages // 1000' "${CONF}")
HISTORY_PERIOD=$(jq -r '.history_period // "1w"' "${CONF}")

MANAGER_KWARGS_JSON=$(jq -c '.runtime.manager_kwargs // {}' "${CONF}")
BOT_KWARGS_JSON=$(jq -c '.runtime.bot_kwargs // {}' "${CONF}")
ENSURE_DELIVERY=$(jq -r '.runtime.ensure_delivery // true' "${CONF}")
DELIVERY_TIMEOUT=$(jq -r '.runtime.delivery_timeout // 60.0' "${CONF}")
MAX_ALLOWED_MESSAGE_DELAY=$(jq -r '.runtime.max_allowed_message_delay // empty' "${CONF}")

TELEGRAM_SECRET_ENV_KEY=$(jq -r '.webhook_security.telegram_secret_token_env_key // empty' "${CONF}")
[[ -z "${TELEGRAM_SECRET_ENV_KEY}" || "${TELEGRAM_SECRET_ENV_KEY}" == "null" ]] && TELEGRAM_SECRET_ENV_KEY="TELEGRAM_WEBHOOK_SECRET_TOKEN"
WHATSAPP_APP_SECRET_ENV_KEY=$(jq -r '.webhook_security.whatsapp_app_secret_env_key // empty' "${CONF}")
[[ -z "${WHATSAPP_APP_SECRET_ENV_KEY}" || "${WHATSAPP_APP_SECRET_ENV_KEY}" == "null" ]] && WHATSAPP_APP_SECRET_ENV_KEY="WHATSAPP_APP_SECRET"

CAPTURE_FAILED_EVENTS=$(jq -r '.failure_handling.capture_failed_events // false' "${CONF}")
WORKER_DLQ=$(jq -r '.failure_handling.worker_dlq // false' "${CONF}")

# Optional external history append endpoint (/history/outgoing). Off by default.
HISTORY_APPEND_ENABLED=$(jq -r '.history_append.enabled // false' "${CONF}")
HISTORY_APPEND_ROLE=$(jq -r '.history_append.role // empty' "${CONF}")

LAMBDA_RUNTIME=$(jq -r '.infra.runtime // "python3.12"' "${CONF}")
ARCHITECTURE=$(jq -r '.infra.architecture // "x86_64"' "${CONF}")
MEMORY_SIZE=$(jq -r '.infra.memory_size // 256' "${CONF}")
LOG_RETENTION_DAYS=$(jq -r '.infra.log_retention_days // empty' "${CONF}")
RESERVED_CONCURRENCY=$(jq -r '.infra.reserved_concurrency // empty' "${CONF}")
S3_VERSIONING=$(jq -r '.infra.s3_versioning // false' "${CONF}")
S3_LIFECYCLE_DAYS=$(jq -r '.infra.s3_lifecycle_days // empty' "${CONF}")
S3_ENCRYPTION=$(jq -r '.infra.s3_encryption // empty' "${CONF}")
TRACING_ENABLED=$(jq -r '.infra.tracing // true' "${CONF}")
FUNCTION_TRACING="Active"
[[ "${TRACING_ENABLED}" == "false" ]] && FUNCTION_TRACING="PassThrough"

REQUIREMENTS_SOURCE=$(jq -r '.requirements_source // "git"' "${CONF}")
REQUIREMENTS_REF=$(jq -r '.requirements_ref // empty' "${CONF}")

TELEGRAM_ENV_KEY=$(jq -r '.telegram.env_token_key // empty' "${CONF}")
[[ -z "${TELEGRAM_ENV_KEY}" || "${TELEGRAM_ENV_KEY}" == "null" ]] && TELEGRAM_ENV_KEY="TELEGRAM_TOKEN"

WHATSAPP_VERIFY_ENV_KEY=$(jq -r '.whatsapp.verify_env_key // empty' "${CONF}")
[[ -z "${WHATSAPP_VERIFY_ENV_KEY}" || "${WHATSAPP_VERIFY_ENV_KEY}" == "null" ]] && WHATSAPP_VERIFY_ENV_KEY="WHATSAPP_VERIFY_TOKEN"


SPECIAL_JSON=$(jq -c '.special_token_files // []' "${CONF}")
TOKEN_ENV_MAP=$(jq -c '.token_env_map // {}' "${CONF}")
mapfile -t EXTRA_REQ < <(jq -r '.extra_requirements[]?' "${CONF}" 2>/dev/null || true)

die() { printf '%s\n' "$@" >&2; exit 1; }

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

# --- Load .env.prod into the environment (aws/sam see it too) ---
# After the checks above, which read the file without executing it, so a malformed secret gets a
# clear error. Trailing CRs are stripped so a .env.prod saved on Windows does not break bash.
set -a
source <(sed 's/\r$//' ./.env.prod)
set +a

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
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

# --- requirements.txt ---
echo ">> Generating requirements.txt"
MAS_EXTRA="maws,telegram,whatsapp,env"
case "${REQUIREMENTS_SOURCE}" in
  pypi)
    if [[ -n "${REQUIREMENTS_REF}" && "${REQUIREMENTS_REF}" != "null" ]]; then
      MAS_REQUIREMENT="mas[${MAS_EXTRA}]==${REQUIREMENTS_REF}"
    else
      MAS_REQUIREMENT="mas[${MAS_EXTRA}]"
    fi
    ;;
  local)
    if [[ -z "${REQUIREMENTS_REF}" || "${REQUIREMENTS_REF}" == "null" ]]; then
      echo "requirements_source=local requires requirements_ref to point at a local package path"
      exit 1
    fi
    MAS_REQUIREMENT="${REQUIREMENTS_REF}"
    ;;
  git|*)
    MAS_REQUIREMENT="mas[${MAS_EXTRA}] @ git+https://github.com/mneuronico/multi-agent-system-library.git"
    if [[ -n "${REQUIREMENTS_REF}" && "${REQUIREMENTS_REF}" != "null" ]]; then
      MAS_REQUIREMENT="${MAS_REQUIREMENT}@${REQUIREMENTS_REF}"
    fi
    ;;
esac
{
  echo "requests"
  echo "boto3"
  echo "${MAS_REQUIREMENT}"
  for r in "${EXTRA_REQ[@]}"; do echo "$r"; done
} | awk 'NF' | sort -u > requirements.txt

# --- samconfig.toml ---
echo ">> Generating samconfig.toml"
cat > samconfig.toml <<EOF
version = 0.1
[default]
[default.deploy]
[default.deploy.parameters]
stack_name = "${STACK}"
s3_bucket = "${DEPLOY_BUCKET}"
s3_prefix = "${STACK}"
region = "${REGION}"
confirm_changeset = true
capabilities = "CAPABILITY_IAM"
image_repositories = []
EOF

# --- Common Lambda (templates) ---

# --- Write minimal lambda_function.py ---
echo ">> Generating lambda_function.py (${BOT})"
cat > lambda_function.py <<'PY'
import os
from maws import build_lambda_handler

BOT_TYPE = os.environ.get("BOT_TYPE", "whatsapp")
lambda_handler = build_lambda_handler(BOT_TYPE)
PY

# --- template.yaml ---
echo ">> Generating template.yaml"
cat > template.yaml <<EOF
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: ${PROJECT}

Globals:
  Function:
    Timeout: ${LAMBDA_TIMEOUT}
    Runtime: ${LAMBDA_RUNTIME}
    MemorySize: ${MEMORY_SIZE}
    Tracing: ${FUNCTION_TRACING}
    LoggingConfig:
      LogFormat: JSON
  Api:
    TracingEnabled: ${TRACING_ENABLED}

Parameters:
  EnvParamName:
    Type: String
    Default: "${ENV_PARAM}"

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
  cat >> template.yaml <<'EOF'
      VersioningConfiguration:
        Status: Enabled
EOF
fi

if [[ -n "${S3_ENCRYPTION}" && "${S3_ENCRYPTION}" != "null" ]]; then
  cat >> template.yaml <<EOF
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: ${S3_ENCRYPTION}
EOF
fi

if [[ -n "${S3_LIFECYCLE_DAYS}" && "${S3_LIFECYCLE_DAYS}" != "null" ]]; then
  cat >> template.yaml <<EOF
      LifecycleConfiguration:
        Rules:
          - Id: ExpireMawsObjects
            Status: Enabled
            ExpirationInDays: ${S3_LIFECYCLE_DAYS}
EOF
fi

if [[ "${BUSY_POLICY}" == "fifo" && "${WORKER_DLQ}" == "true" ]]; then
  cat >> template.yaml <<EOF
  ${PROJECT_LOGICAL}WorkerDlq:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub "\${AWS::StackName}-worker-dlq.fifo"
      FifoQueue: true
      MessageRetentionPeriod: 1209600

EOF
fi

if [[ "${BUSY_POLICY}" == "fifo" ]]; then
  cat >> template.yaml <<EOF
  ${PROJECT_LOGICAL}WorkerQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub "\${AWS::StackName}-worker.fifo"
      FifoQueue: true
      ContentBasedDeduplication: false
      VisibilityTimeout: ${VISIBILITY_TIMEOUT}
EOF
  if [[ "${WORKER_DLQ}" == "true" ]]; then
    cat >> template.yaml <<EOF
      RedrivePolicy:
        deadLetterTargetArn: !GetAtt ${PROJECT_LOGICAL}WorkerDlq.Arn
        maxReceiveCount: 3
EOF
  fi
  echo "" >> template.yaml
fi

cat >> template.yaml <<EOF
  ${FUNC_ID}:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: ${PACKAGE_DIR}
      Handler: lambda_function.lambda_handler
      Runtime: ${LAMBDA_RUNTIME}
      Description: "MAS Bot Lambda"
      Architectures: [ ${ARCHITECTURE} ]
EOF

if [[ -n "${RESERVED_CONCURRENCY}" && "${RESERVED_CONCURRENCY}" != "null" ]]; then
  cat >> template.yaml <<EOF
      ReservedConcurrentExecutions: ${RESERVED_CONCURRENCY}
EOF
fi

cat >> template.yaml <<EOF
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
        - Statement:
            - Effect: Allow
              Action: [ "lambda:InvokeFunction" ]
              Resource: "*"
EOF

if [[ "${BUSY_POLICY}" == "fifo" ]]; then
  cat >> template.yaml <<EOF
        - Statement:
            - Effect: Allow
              Action: [ "sqs:SendMessage" ]
              Resource: !GetAtt ${PROJECT_LOGICAL}WorkerQueue.Arn
EOF
fi

cat >> template.yaml <<EOF
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

if [[ "${BUSY_POLICY}" == "fifo" ]]; then
  cat >> template.yaml <<EOF
          MAWS_QUEUE_URL: !Ref ${PROJECT_LOGICAL}WorkerQueue
EOF
fi

if [[ "${HISTORY_APPEND_ENABLED}" == "true" ]]; then
  cat >> template.yaml <<EOF
          MAWS_HISTORY_APPEND_ENABLED: "true"
EOF
  if [[ -n "${HISTORY_APPEND_ROLE}" ]]; then
    cat >> template.yaml <<EOF
          MAWS_HISTORY_APPEND_ROLE: "${HISTORY_APPEND_ROLE}"
EOF
  fi
fi

cat >> template.yaml <<EOF

          # Locks
          LOCKS_TABLE_NAME: !Ref ProcessingLocksTable
          LOCK_TTL_SECONDS: "${LOCK_TTL_SECONDS}"
      Events:
EOF

if [[ "${BOT}" == "telegram" ]]; then
  cat >> template.yaml <<EOF
        TelegramWebhook:
          Type: Api
          Properties:
            Path: "${API_PATH}"
            Method: post
EOF
else
  cat >> template.yaml <<EOF
        WhatsAppWebhookGET:
          Type: Api
          Properties: { Path: "${API_PATH}", Method: get }
        WhatsAppWebhookPOST:
          Type: Api
          Properties: { Path: "${API_PATH}", Method: post }
EOF
fi

if [[ "${HISTORY_APPEND_ENABLED}" == "true" ]]; then
  cat >> template.yaml <<'EOF'
        HistoryAppendPOST:
          Type: Api
          Properties: { Path: "/history/outgoing", Method: post }
EOF
fi

if [[ "${BUSY_POLICY}" == "fifo" ]]; then
  cat >> template.yaml <<EOF
        WorkerQueueEvent:
          Type: SQS
          Properties:
            Queue: !GetAtt ${PROJECT_LOGICAL}WorkerQueue.Arn
            BatchSize: 1
EOF
fi

if [[ -n "${LOG_RETENTION_DAYS}" && "${LOG_RETENTION_DAYS}" != "null" ]]; then
  cat >> template.yaml <<EOF

  ${PROJECT_LOGICAL}FunctionLogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub "/aws/lambda/\${${FUNC_ID}}"
      RetentionInDays: ${LOG_RETENTION_DAYS}
EOF
fi

cat >> template.yaml <<'EOF'

Outputs:
  ApiUrl:
    Description: Base URL for the API
    Value: !Sub "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod"
EOF

# --- Create deployment bucket if it doesn't exist ---
echo ">> Ensuring deployment bucket: ${DEPLOY_BUCKET}"
if ! aws s3api head-bucket --bucket "${DEPLOY_BUCKET}" 2>/dev/null; then
  aws s3 mb "s3://${DEPLOY_BUCKET}" --region "${REGION}"
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
    echo "       sam build will likely fail: install ${LAMBDA_RUNTIME}, or set \"infra\": {\"runtime\": \"${LOCAL_PYTHON}\"} in params.json."
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
    dest, ".aws-sam", ".git", "history", "files", "logs", "tmp", "local_s3_bucket", ".venv", "venv",
    "params.json", "samconfig.toml", "template.yaml", ".samignore", ".bootstrap_state.json",
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

# --- sam build ---
echo ">> sam build"
sam build

# --- Upload .env.prod to SSM ---
echo ">> Uploading .env.prod to SSM: ${ENV_PARAM}"
aws ssm put-parameter \
  --name "${ENV_PARAM}" \
  --type "SecureString" \
  --value "file://.env.prod" \
  --overwrite \
  --region "${REGION}" >/dev/null

# --- Deploy ---
echo ">> sam deploy"
sam deploy --no-confirm-changeset

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
  if [[ -z "${MAWS_HISTORY_WRITE_SECRET:-}" ]]; then
    echo "[WARN] history_append.enabled=true but .env.prod has no MAWS_HISTORY_WRITE_SECRET; the endpoint will answer 503."
  fi
fi

echo "[OK] Done."
