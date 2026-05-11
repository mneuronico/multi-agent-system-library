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

ENV_PARAM=$(jq -r '.env_param_name // empty' "${CONF}")
[[ -z "${ENV_PARAM}" || "${ENV_PARAM}" == "null" ]] && ENV_PARAM="/${PROJECT}/prod/env"

API_PATH=$(jq -r '.api_path // "/webhook"' "${CONF}")

SYNC_TOKENS_S3=$(jq -r '.sync_tokens_s3 // true' "${CONF}")
TOKENS_PREFIX=$(jq -r '.tokens_s3_prefix // "secrets"' "${CONF}")
VERBOSE=$(jq -r '.verbose // false' "${CONF}")
LAMBDA_TIMEOUT=$(jq -r '.lambda_timeout // 120' "${CONF}")
VISIBILITY_TIMEOUT=$((LAMBDA_TIMEOUT + 30))

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
WHATSAPP_VERIFY_SIGNATURE=$(jq -r '.webhook_security.whatsapp_verify_signature // false' "${CONF}")
WHATSAPP_APP_SECRET_ENV_KEY=$(jq -r '.webhook_security.whatsapp_app_secret_env_key // empty' "${CONF}")
[[ -z "${WHATSAPP_APP_SECRET_ENV_KEY}" || "${WHATSAPP_APP_SECRET_ENV_KEY}" == "null" ]] && WHATSAPP_APP_SECRET_ENV_KEY="WHATSAPP_APP_SECRET"

CAPTURE_FAILED_EVENTS=$(jq -r '.failure_handling.capture_failed_events // false' "${CONF}")
WORKER_DLQ=$(jq -r '.failure_handling.worker_dlq // false' "${CONF}")

LAMBDA_RUNTIME=$(jq -r '.infra.runtime // "python3.9"' "${CONF}")
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

# --- Load .env.prod into the environment (for setWebhook, etc.) ---
set -a
source ./.env.prod
set +a

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

# --- .samignore ---
echo ">> Generating .samignore"
cat > .samignore <<'EOF'
.aws-sam/
__pycache__/
tmp/
local_s3_bucket/
history/
files/
.gitignore
samconfig.toml
README.md
.env.prod
deploy.sh
bootstrap.sh
params.json
EOF

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
      CodeUri: .
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

          # Provider-supported webhook security
          TELEGRAM_WEBHOOK_SECRET_TOKEN_ENV_KEY: "${TELEGRAM_SECRET_ENV_KEY}"
          WHATSAPP_VERIFY_SIGNATURE: "${WHATSAPP_VERIFY_SIGNATURE}"
          WHATSAPP_APP_SECRET_ENV_KEY: "${WHATSAPP_APP_SECRET_ENV_KEY}"

          # Failure capture
          CAPTURE_FAILED_EVENTS: "${CAPTURE_FAILED_EVENTS}"
EOF

if [[ "${BUSY_POLICY}" == "fifo" ]]; then
  cat >> template.yaml <<EOF
          MAWS_QUEUE_URL: !Ref ${PROJECT_LOGICAL}WorkerQueue
EOF
fi

cat >> template.yaml <<EOF

          # Locks
          LOCKS_TABLE_NAME: !Ref ProcessingLocksTable
          LOCK_TTL_SECONDS: "180"
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

# --- Local state to avoid repeating setWebhook ---
STATE_FILE=".bootstrap_state.json"
touch "${STATE_FILE}"

if [[ "${BOT}" == "telegram" ]]; then
  KEY="${TELEGRAM_ENV_KEY:-TELEGRAM_TOKEN}"
  TOKEN="${!KEY:-}"
  SECRET_KEY="${TELEGRAM_SECRET_ENV_KEY:-TELEGRAM_WEBHOOK_SECRET_TOKEN}"
  SECRET_TOKEN="${!SECRET_KEY:-}"
  if [[ -z "${TOKEN}" ]]; then
    echo "[WARN] Could not find ${KEY} in .env.prod; cannot set the Telegram webhook."
  else
    PREV=$(jq -r '."telegram_webhook" // empty' "${STATE_FILE}" 2>/dev/null || true)
    if [[ "${PREV}" != "${WEBHOOK_URL}" ]]; then
      echo ">> Setting Telegram webhook..."
      CURL_ARGS=(-d "url=${WEBHOOK_URL}")
      if [[ -n "${SECRET_TOKEN}" ]]; then
        CURL_ARGS+=(-d "secret_token=${SECRET_TOKEN}")
      fi
      curl -s "https://api.telegram.org/bot${TOKEN}/setWebhook" "${CURL_ARGS[@]}" >/dev/null
      tmp=$(mktemp)
      jq --arg url "${WEBHOOK_URL}" '.telegram_webhook=$url' "${STATE_FILE}" 2>/dev/null > "$tmp" || echo "{\"telegram_webhook\":\"${WEBHOOK_URL}\"}" > "$tmp"
      mv "$tmp" "${STATE_FILE}"
      echo "[OK] Telegram webhook set to: ${WEBHOOK_URL}"
    else
      echo "Telegram webhook was already set; skipping."
    fi
  fi
else
  echo "[INFO] WhatsApp webhooks:"
  echo "    - Configure the callback URL in Meta Developers:"
  echo "      ${WEBHOOK_URL}"
  echo "    - Verify token (from .env.prod) => key: ${WHATSAPP_VERIFY_ENV_KEY:-WHATSAPP_VERIFY_TOKEN}"
  echo "    - Remember to add the same verify token in your Meta app."
  if [[ "${WHATSAPP_VERIFY_SIGNATURE}" == "true" ]]; then
    echo "    - POST signature verification is enabled; .env.prod must contain ${WHATSAPP_APP_SECRET_ENV_KEY}."
  else
    echo "    - For production, consider setting webhook_security.whatsapp_verify_signature=true and ${WHATSAPP_APP_SECRET_ENV_KEY}."
  fi
fi

echo "[OK] Done."
