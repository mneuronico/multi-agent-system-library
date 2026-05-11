# maws - MAS on AWS (Lambda + API Gateway)

**maws** is a helper library for deploying **MAS** bots on **AWS Lambda** behind **API Gateway**, featuring:

* lazy initialization of `AgentSystemManager`
* webhook handling for **WhatsApp** and **Telegram**
* webhook verification (WhatsApp)
* **self-invocation** to process work in the background (avoids API Gateway timeouts)
* optional **SQS FIFO** processing for production bots that must not drop messages
* per-user **locking** with **DynamoDB TTL** to avoid concurrent processing
* optional S3-backed user media/file persistence
* synchronization of **tokens/keys** between **S3** and the ephemeral `/tmp` filesystem
* configuration loading from **SSM Parameter Store** (SecureString)

> If you already use MAS, maws lets you shrink your `lambda_function.py` to roughly three lines.

---

## Installation

### pip (from the repository root)

```bash
pip install \
  "mas[maws,telegram,whatsapp,env] @ git+https://github.com/mneuronico/multi-agent-system-library.git"
```

The repository root package includes both **MAS** and **maws**.
* You can also **pin to a commit/tag**:

```bash
pip install \
  "mas[maws,telegram,whatsapp,env] @ git+https://github.com/mneuronico/multi-agent-system-library.git@<commit>"
```

### `requirements.txt` example

```txt
boto3
requests
mas[maws,telegram,whatsapp,env] @ git+https://github.com/mneuronico/multi-agent-system-library.git
```

> **Notes**
>
> * `boto3` and `botocore` are Lambda runtime dependencies; the `maws` extra includes `boto3` for local tooling even though Lambda provides it.
> * MAS is a dependency of maws (imported as `from mas import AgentSystemManager, WhatsappBot, TelegramBot`).

---

## Quick usage

Your `lambda_function.py` can be this simple:

```python
# lambda_function.py
import os
from maws import build_lambda_handler

BOT_TYPE = os.environ.get("BOT_TYPE", "whatsapp")  # "whatsapp" | "telegram"
lambda_handler = build_lambda_handler(BOT_TYPE)
```

That's it. `maws` wires everything to MAS, handles webhook GET/POST, background processing via self-invocation, history import/export, locks, and more.

---

## Environment variables (contract)

| Variable                   | Required | Default      | Description                                                         |
| -------------------------- | -------- | ------------ | ------------------------------------------------------------------- |
| `BOT_TYPE`                 | No       | `"whatsapp"` | `"whatsapp"` or `"telegram"`.                                        |
| `VERBOSE`                  | No       | `"false"`    | `true/false` for detailed logging in MAS and the bots.               |
| `BUCKET_NAME`              | **Yes**  | none         | S3 bucket with histories and (optionally) tokens/keys.               |
| `ENV_PARAMETER_NAME`       | No       | none         | **SSM** (SecureString) parameter containing the `.env` to inject.    |
| `MAWS_BUSY_POLICY`         | No       | `"drop"`     | `"drop"` keeps current behavior; `"fifo"` queues work in SQS FIFO.   |
| `MAWS_QUEUE_URL`           | FIFO     | none         | SQS FIFO queue URL used when `MAWS_BUSY_POLICY=fifo`.                |
| `HISTORY_MODE`             | No       | `"per_user"` | `"per_user"` or `"shared"`; mirrors MAS history storage.             |
| `HISTORY_ROTATION`         | No       | `"message_count"` | Shared history rotation: `message_count`, `time_period`, or `both`. |
| `HISTORY_MAX_MESSAGES`     | No       | `"1000"`     | Shared history message-count rotation threshold.                     |
| `HISTORY_PERIOD`           | No       | `"1w"`       | Shared history time-period rotation window.                          |
| `PERSIST_FILES_S3`         | No       | `"false"`    | When `true`, syncs per-user files/media to S3.                       |
| `FILES_S3_PREFIX`          | No       | `"files"`    | S3 prefix for persisted user files/media.                            |
| `SYNC_TOKENS_S3`           | No       | `"true"`     | When `true`, fetch tokens from S3 before falling back to the package.|
| `TOKENS_S3_PREFIX`         | No       | `"secrets"`  | S3 prefix for tokens/keys (e.g., `secrets/<file>`).                  |
| `SPECIAL_TOKEN_FILES_JSON` | No       | `"[]"`       | JSON list of token file names to sync into `/tmp`.                   |
| `TOKEN_ENV_MAP_JSON`       | No       | `"{}"`       | JSON `{ "file": "ENV_VAR" }` to expose file paths via env vars.      |
| `LOCKS_TABLE_NAME`         | No       | none         | DynamoDB table name for per-user **locking**.                        |
| `LOCK_TTL_SECONDS`         | No       | `"180"`      | Lock TTL (seconds).                                                  |
| `TELEGRAM_WEBHOOK_SECRET_TOKEN` | No  | none         | Provider-supported Telegram webhook secret token.                    |
| `WHATSAPP_VERIFY_SIGNATURE` | No      | `"false"`   | When `true`, verify WhatsApp `X-Hub-Signature-256`.                  |
| `WHATSAPP_APP_SECRET`      | If verifying | none    | Meta app secret used for WhatsApp signature verification.            |
| `MAWS_MANAGER_KWARGS_JSON` | No       | `"{}"`       | Extra kwargs for `AgentSystemManager`; MAWS-managed paths still win. |
| `MAWS_BOT_KWARGS_JSON`     | No       | `"{}"`       | Extra kwargs for the selected MAS bot constructor.                   |
| `CAPTURE_FAILED_EVENTS`    | No       | `"false"`    | Write failed jobs/events to S3 under `failed-events/`.               |

### How do tokens work?

* **SPECIAL_TOKEN_FILES_JSON**: list of file names (e.g., `["openai.key", "facebook.json"]`).
  maws will attempt to:

  1. Download `s3://BUCKET_NAME/TOKENS_S3_PREFIX/<file>` to `/tmp/<file>`, or
  2. Copy `<file>` from the **package** (read-only) to `/tmp/<file>` as fallback.

* **TOKEN_ENV_MAP_JSON**: map to **expose** the path via environment variables, e.g.:

  ```json
  { "facebook.json": "FACEBOOK_CREDENTIALS_PATH", "openai.key": "OPENAI_API_KEY_PATH" }
  ```

  This lets your code read `os.environ["FACEBOOK_CREDENTIALS_PATH"]` and open the file.

---

## maws API

### `build_lambda_handler(bot_type: str) -> Callable`

Returns a **Lambda handler** already configured for the chosen bot.

* Lazily initializes `AgentSystemManager`.
* Instantiates the MAS bot (`WhatsappBot` / `TelegramBot`) with `verbose` coming from `VERBOSE`.
* GET (WhatsApp): calls `handle_webhook_verification` on the bot.
* POST: by default **self-invokes** the same Lambda with a normalized "worker" payload (background mode).
* Optional FIFO mode: enqueues normalized jobs to SQS FIFO instead of self-invoking.
* In the "second hop":

  * Takes a **lock** per `user_id` (or one global lock in shared-history mode) if `LOCKS_TABLE_NAME` is set; if the lock exists in default `"drop"` mode, it **ignores** the update.
  * **Imports** history from S3 if present.
  * Optionally syncs per-user files/media from S3.
  * Executes `process_webhook_update`.
  * **Exports** history and optionally files/media to S3.
  * **Releases** the lock (or TTL cleans it up).

You do not need to change anything inside your MAS bot/base class.

---

## IAM / CloudFormation (SAM example)

Add something like this to your `template.yaml` (summary):

```yaml
Resources:
  ProcessingLocksTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub "${AWS::StackName}-locks"
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions: [{ AttributeName: user_id, AttributeType: S }]
      KeySchema: [{ AttributeName: user_id, KeyType: HASH }]
      TimeToLiveSpecification: { AttributeName: expiresAt, Enabled: true }

  MyFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: lambda_function.lambda_handler
      Runtime: python3.9
      CodeUri: .
      Policies:
        - AWSLambdaBasicExecutionRole
        - AWSXRayDaemonWriteAccess
        - Statement:
            - Effect: Allow
              Action: [ "ssm:GetParameter" ]
              Resource: !Sub arn:aws:ssm:${AWS::Region}:${AWS::AccountId}:parameter/my/param/name
        - Statement:
            - Effect: Allow
              Action: [ "s3:ListBucket" ]
              Resource: "arn:aws:s3:::<BUCKET_NAME>"
            - Effect: Allow
              Action: [ "s3:GetObject", "s3:PutObject" ]
              Resource: "arn:aws:s3:::<BUCKET_NAME>/*"
        - Statement:
            - Effect: Allow
              Action: [ "lambda:InvokeFunction" ]
              Resource: "*"
        - Statement:
            - Effect: Allow
              Action: [ "dynamodb:PutItem", "dynamodb:DeleteItem" ]
              Resource: !GetAtt ProcessingLocksTable.Arn
      Environment:
        Variables:
          BOT_TYPE: "whatsapp"
          VERBOSE: "false"
          BUCKET_NAME: "<BUCKET_NAME>"
          ENV_PARAMETER_NAME: "/my/app/prod/env"
          SYNC_TOKENS_S3: "true"
          TOKENS_S3_PREFIX: "secrets"
          SPECIAL_TOKEN_FILES_JSON: '[]'
          TOKEN_ENV_MAP_JSON: '{}'
          LOCKS_TABLE_NAME: !Ref ProcessingLocksTable
          LOCK_TTL_SECONDS: "180"
      Events:
        WebhookGET:
          Type: Api
          Properties: { Path: "/webhook", Method: get }
        WebhookPOST:
          Type: Api
          Properties: { Path: "/webhook", Method: post }
```

> Switch GET/POST according to **WhatsApp** (GET+POST) or **Telegram** (POST only).

---

## Self-invocation pattern

To avoid API Gateway timeouts, maws:

1. Responds **quickly** to the POST (200 OK).
2. **Self-invokes** with the actual payload to process in the background.
3. Uses DynamoDB TTL as a per-user lock (`user_id`) to avoid duplicates when messages arrive in rapid succession.

If you do not configure `LOCKS_TABLE_NAME`, locking is **disabled** (best-effort behavior).

### Busy-message policy

The default `MAWS_BUSY_POLICY=drop` preserves the original MAWS behavior: if a user's lock is already held, the new update is acknowledged and ignored. This can be desirable when processing messages sent while the bot was still thinking would make the conversation order confusing.

Set `busy_policy` to `"fifo"` in `params.json` to generate an SQS FIFO queue. In FIFO mode MAWS enqueues one normalized job per provider message, uses the chat id as the FIFO message group in per-user history mode, and uses one global group in shared-history mode to avoid SQLite history races. Failed SQS jobs can retry and, when configured, move to a DLQ.

```json
{
  "busy_policy": "fifo",
  "failure_handling": {
    "worker_dlq": true,
    "capture_failed_events": true
  }
}
```

### Webhook security

MAWS only uses provider-supported webhook security mechanisms:

* **Telegram**: set `TELEGRAM_WEBHOOK_SECRET_TOKEN` in `.env.prod`. During deploy, MAWS passes it to Telegram `setWebhook` as `secret_token`. At runtime, MAWS requires incoming POSTs to include the matching `X-Telegram-Bot-Api-Secret-Token` header.
* **WhatsApp**: keep the existing GET verify-token flow. For POST authenticity, set `webhook_security.whatsapp_verify_signature=true` in `params.json` and set `WHATSAPP_APP_SECRET` in `.env.prod`. MAWS verifies `X-Hub-Signature-256` against the raw request body.

Both checks are opt-in so existing bots do not break. Production deployments should enable the matching provider mechanism.

---

## Loading environment from SSM

If `ENV_PARAMETER_NAME` is set, maws reads that **SecureString** parameter (in `.env` format) and exports **all** keys to `os.environ` **without** overriding already-present variables.

Accepted format:

```
# comment
KEY=VALUE
ANOTHER=VAL
```

---

## User histories and files

* User histories are stored as SQLite files in S3 under `history/<chat_id>.sqlite`.
* maws automatically imports/exports around processing.
* With `history_mode="shared"`, MAWS syncs MAS shared-history SQLite files under `history/shared/` and serializes workers globally so time-period/message-count rotation does not race.
* `maws pull-history` syncs all history files by default. Use `--user-id <id>` repeatedly or `--user-ids <id,id>` to download only selected user histories.
* By default, user media and generated files remain in Lambda `/tmp/files` only. Set `persist_files_s3=true` to sync per-user files under `files/<chat_id>/` in the history bucket.
* For **tokens** defined in `SPECIAL_TOKEN_FILES_JSON`, it syncs S3 to `/tmp` (or package to `/tmp` if missing in S3) and can optionally expose paths via `TOKEN_ENV_MAP_JSON`.

---

## Logging

* `VERBOSE=true` enables detailed traces in bots/MAS.
* CloudWatch Logs centralizes Lambda logs (low cost; remember to rotate).

---

## Requirements

* Python 3.9+ (compatible with the Lambda `python3.9` runtime).
* AWS: S3, API Gateway, (optional) DynamoDB, (optional) SSM Parameter Store.
* Libraries: `mas`, `boto3`, `requests` (plus their transitive dependencies).

---

## Minimal end-to-end example

**lambda_function.py**

```python
import os
from maws import build_lambda_handler

lambda_handler = build_lambda_handler(os.environ.get("BOT_TYPE", "whatsapp"))
```

**requirements.txt**

```txt
boto3
requests
mas[maws,telegram,whatsapp,env] @ git+https://github.com/mneuronico/multi-agent-system-library.git
```

**template.yaml**: see the IAM / CloudFormation section above.

---

## Frequently asked questions

**Can I use maws without DynamoDB?**
Yes. Omit `LOCKS_TABLE_NAME` to disable locking.

**What if I don't have tokens in S3?**
maws will try to copy them from the **package** (zip root) to `/tmp`. If they do not exist, it logs a warning and continues.

**Do I need MAS to use maws?**
Yes. maws acts as the "glue" between AWS and **MAS**.

---

## Versioning and licensing

* Versioning lives in `maws/__init__.py` (e.g., `__version__ = "0.1.0"`).
* See the main repository **LICENSE** for licensing details.
