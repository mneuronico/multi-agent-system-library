#!/usr/bin/env bash
set -euo pipefail

# --- Args ---
CONF="params.json"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config) CONF="$2"; shift 2 ;;
    *) echo "Arg desconocido: $1"; exit 1 ;;
  esac
done

[[ -f "${CONF}" ]] || { echo "No existe ${CONF}"; exit 1; }

# --- Checks ---
need() { command -v "$1" >/dev/null 2>&1 || { echo "Falta '$1' en PATH"; exit 1; }; }
need aws
need sam
need jq
need sed

[[ -f ".env.prod" ]] || { echo "Falta .env.prod"; exit 1; }
[[ -f "${CONF}" ]] || { echo "No existe ${CONF}"; exit 1; }

# --- Leer config ---
echo ">> Leyendo config: ${CONF}"

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


TELEGRAM_ENV_KEY=$(jq -r '.telegram.env_token_key // empty' "${CONF}")
[[ -z "${TELEGRAM_ENV_KEY}" || "${TELEGRAM_ENV_KEY}" == "null" ]] && TELEGRAM_ENV_KEY="TELEGRAM_TOKEN"

WHATSAPP_VERIFY_ENV_KEY=$(jq -r '.whatsapp.verify_env_key // empty' "${CONF}")
[[ -z "${WHATSAPP_VERIFY_ENV_KEY}" || "${WHATSAPP_VERIFY_ENV_KEY}" == "null" ]] && WHATSAPP_VERIFY_ENV_KEY="WHATSAPP_VERIFY_TOKEN"


SPECIAL_JSON=$(jq -c '.special_token_files // []' "${CONF}")
TOKEN_ENV_MAP=$(jq -c '.token_env_map // {}' "${CONF}")
mapfile -t EXTRA_REQ < <(jq -r '.extra_requirements[]?' "${CONF}" 2>/dev/null || true)

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo ">> Cuenta AWS: ${ACCOUNT_ID}  Región: ${REGION}"

# --- Helpers para logical IDs (solo letras/números y empezando con letra) ---
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

# --- Cargar .env.prod en ambiente (para setWebhook, etc.) ---
set -a
source ./.env.prod
set +a

# --- requirements.txt ---
echo ">> Generando requirements.txt"
{
  echo "requests"
  echo "boto3"
  echo "mas @ git+https://github.com/mneuronico/multi-agent-system-library.git"
  for r in "${EXTRA_REQ[@]}"; do echo "$r"; done
} | awk 'NF' | sort -u > requirements.txt

# --- .samignore ---
echo ">> Generando .samignore"
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
echo ">> Generando samconfig.toml"
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

# --- Lambda común (plantillas) ---

# --- Escribir lambda_function.py mínimo ---
echo ">> Generando lambda_function.py (${BOT})"
cat > lambda_function.py <<'PY'
import os
from maws import build_lambda_handler

BOT_TYPE = os.environ.get("BOT_TYPE", "whatsapp")
lambda_handler = build_lambda_handler(BOT_TYPE)
PY

# --- template.yaml ---
echo ">> Generando template.yaml"
cat > template.yaml <<EOF
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: ${PROJECT}

Globals:
  Function:
    Timeout: 120
    Runtime: python3.9
    MemorySize: 256
    Tracing: Active
    LoggingConfig:
      LogFormat: JSON
  Api:
    TracingEnabled: true

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

  ${FUNC_ID}:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: .
      Handler: lambda_function.lambda_handler
      Runtime: python3.9
      Description: "MAS Bot Lambda"
      Architectures: [ x86_64 ]
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

          # Tokens / archivos
          SYNC_TOKENS_S3: "${SYNC_TOKENS_S3}"
          TOKENS_S3_PREFIX: "${TOKENS_PREFIX}"
          SPECIAL_TOKEN_FILES_JSON: '${SPECIAL_JSON}'
          TOKEN_ENV_MAP_JSON: '${TOKEN_ENV_MAP}'

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

cat >> template.yaml <<'EOF'

Outputs:
  ApiUrl:
    Description: Base URL for the API
    Value: !Sub "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod"
EOF

# --- Crear deployment bucket si no existe ---
echo ">> Asegurando bucket de deployment: ${DEPLOY_BUCKET}"
if ! aws s3api head-bucket --bucket "${DEPLOY_BUCKET}" 2>/dev/null; then
  aws s3 mb "s3://${DEPLOY_BUCKET}" --region "${REGION}"
fi

# --- sam build ---
echo ">> sam build"
sam build

# --- Subir .env.prod a SSM ---
echo ">> Subiendo .env.prod a SSM: ${ENV_PARAM}"
aws ssm put-parameter \
  --name "${ENV_PARAM}" \
  --type "SecureString" \
  --value "file://.env.prod" \
  --overwrite \
  --region "${REGION}" >/dev/null

# --- Deploy ---
echo ">> sam deploy"
sam deploy --no-confirm-changeset

# --- Obtener API URL ---
API_URL=$(aws cloudformation describe-stacks \
  --stack-name "${STACK}" \
  --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" \
  --output text --region "${REGION}")

if [[ -z "${API_URL}" || "${API_URL}" == "None" ]]; then
  echo "No pude obtener ApiUrl desde outputs. Revisá CloudFormation."
  exit 1
fi

WEBHOOK_URL="${API_URL}$(printf "%s" "${API_PATH}")"
echo ">> API URL: ${API_URL}"
echo ">> Webhook full URL: ${WEBHOOK_URL}"

# --- Estado local para no repetir setWebhook ---
STATE_FILE=".bootstrap_state.json"
touch "${STATE_FILE}"

if [[ "${BOT}" == "telegram" ]]; then
  KEY="${TELEGRAM_ENV_KEY:-TELEGRAM_TOKEN}"
  TOKEN="${!KEY:-}"
  if [[ -z "${TOKEN}" ]]; then
    echo "⚠️  No encontré ${KEY} en .env.prod; no puedo setear Telegram webhook."
  else
    PREV=$(jq -r '."telegram_webhook" // empty' "${STATE_FILE}" 2>/dev/null || true)
    if [[ "${PREV}" != "${WEBHOOK_URL}" ]]; then
      echo ">> Seteando webhook de Telegram…"
      curl -s "https://api.telegram.org/bot${TOKEN}/setWebhook" \
        -d "url=${WEBHOOK_URL}" >/dev/null
      tmp=$(mktemp)
      jq --arg url "${WEBHOOK_URL}" '.telegram_webhook=$url' "${STATE_FILE}" 2>/dev/null > "$tmp" || echo "{\"telegram_webhook\":\"${WEBHOOK_URL}\"}" > "$tmp"
      mv "$tmp" "${STATE_FILE}"
      echo "✅ Telegram webhook seteado a: ${WEBHOOK_URL}"
    else
      echo "Webhook Telegram ya estaba seteado; omito."
    fi
  fi
else
  echo "ℹ️  WhatsApp Webhooks:"
  echo "    - Configurá en Meta Developers el callback URL:"
  echo "      ${WEBHOOK_URL}"
  echo "    - Verify token (de .env.prod) => clave: ${WHATSAPP_VERIFY_ENV_KEY:-WHATSAPP_VERIFY_TOKEN}"
  echo "    - Recordá agregar el mismo verify token en tu app de Meta."
fi

echo "✅ Listo."
