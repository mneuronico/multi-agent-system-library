# maws - MAS on AWS (Lambda + API Gateway)

**maws** is a helper library for deploying **MAS** bots on **AWS Lambda** behind **API Gateway**, featuring:

* lazy initialization of `AgentSystemManager`
* webhook handling for **WhatsApp** and **Telegram**
* mandatory webhook authentication (Telegram secret token, WhatsApp `X-Hub-Signature-256`) with an explicit opt-out
* webhook verification (WhatsApp GET)
* **self-invocation** to process work in the background (avoids API Gateway timeouts), one worker per chat and webhook
* optional **SQS FIFO** processing for production bots that must not drop messages, with a dead-letter queue by default
* per-user **locking** with **DynamoDB** conditional writes to avoid concurrent processing
* each message processed once: repeated deliveries (provider retries, duplicate async invocations, SQS redeliveries) are skipped and counted (`MAWS/DuplicateMessages`)
* dropped and failed updates logged at WARNING and counted as CloudWatch metrics (`MAWS/DroppedMessages`, `MAWS/FailedMessages`)
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
* You can also **pin to a commit/tag** (recommended for deploys, so every deploy ships the same code):

```bash
pip install \
  "mas[maws,telegram,whatsapp,env] @ git+https://github.com/mneuronico/multi-agent-system-library.git@<commit>"
```

`maws update` builds the Lambda requirement from `params.json`:

* `requirements_source: "git"` (default) installs from this GitHub repository. Without `requirements_ref` it installs **whatever is on the default branch at deploy time**, so two deploys of the same project can ship different code; the deploy prints a warning. Set `requirements_ref` to a commit SHA (most reproducible) or a release tag. Check which tags exist (`git ls-remote --tags https://github.com/mneuronico/multi-agent-system-library`) before pinning one: not every released version is tagged.
* `requirements_source: "local"` uses `requirements_ref` as a local path or requirement line.

Any other value stops the deploy. In particular there is no `pypi` source: the package named `mas` on PyPI is a different project, not this library, so never install MAS as a bare `mas` / `mas[...]` requirement.

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

With `maws update` you do not need this file: the deploy writes it (and `requirements.txt`) into the staged package `.maws_package/` only. If your project folder has its own `lambda_function.py` or `requirements.txt`, it is deployed unchanged; files that older maws versions generated in the project folder are recognised and replaced in the package (never in your folder). `template.yaml` is generated in `.maws_build/` on every deploy; no `samconfig.toml` is written. Project-folder `template.yaml` / `samconfig.toml` files are never read or overwritten by maws.

---

## Environment variables (contract)

| Variable                   | Required | Default      | Description                                                         |
| -------------------------- | -------- | ------------ | ------------------------------------------------------------------- |
| `BOT_TYPE`                 | No       | `"whatsapp"` | `"whatsapp"` or `"telegram"`.                                        |
| `VERBOSE`                  | No       | `"false"`    | `true/false` for detailed logging in MAS and the bots.               |
| `BUCKET_NAME`              | **Yes**  | none         | S3 bucket with histories and (optionally) tokens/keys.               |
| `ENV_PARAMETER_NAME`       | No       | none         | **SSM** (SecureString) parameter containing the `.env` to inject.    |
| `MAWS_ENV_PARAMETER_VERSION` | No     | none (latest) | SSM parameter version to read. `maws update` sets it to the version it uploaded. |
| `MAWS_BUSY_POLICY`         | No       | `"drop"`     | `"drop"` keeps current behavior; `"fifo"` queues work in SQS FIFO.   |
| `MAWS_BUSY_WAIT_SECONDS`   | No       | `"30"`       | `drop` mode: max seconds a worker waits for a busy user lock before dropping (never more than half of the remaining invocation time; `0` = drop at once). |
| `MAWS_DEDUP_RETENTION_DAYS` | No      | `"8"`        | Days a processed message is remembered, so a repeated delivery of it is skipped (needs `LOCKS_TABLE_NAME`; `0` = off). See "Repeated deliveries". |
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
| `LOCK_TTL_SECONDS`         | No       | `"180"` (deploys: `lambda_timeout + 15`) | Lock lifetime when no Lambda context is available. Inside Lambda a lock expires 15 s after the invocation's deadline. |
| `TELEGRAM_WEBHOOK_SECRET_TOKEN` | Telegram | none    | Telegram webhook secret token. Without it Telegram POSTs are refused (500) unless unauthenticated webhooks are allowed. |
| `WHATSAPP_VERIFY_SIGNATURE` | No      | `"true"`     | Verify WhatsApp `X-Hub-Signature-256`. `false` is only honored with `MAWS_ALLOW_UNAUTHENTICATED_WEBHOOKS=true`. |
| `WHATSAPP_APP_SECRET`      | WhatsApp | none        | Meta app secret used for WhatsApp signature verification.            |
| `MAWS_ALLOW_UNAUTHENTICATED_WEBHOOKS` | No | `"false"` | Explicit opt-out: accept webhook POSTs when no provider secret is configured. Not for production. |
| `MAWS_HISTORY_APPEND_ENABLED` | No    | `"false"`    | Enables the `/history/outgoing` endpoint. When `false`, MAWS behaves exactly as before. |
| `MAWS_HISTORY_WRITE_SECRET` | For history writes | none | Shared secret used to authenticate external history append requests. |
| `MAWS_HISTORY_APPEND_ROLE` | No       | inferred     | Agent name that appended messages are attributed to.                 |
| `MAWS_HISTORY_APPEND_LOCK_WAIT_SECONDS` | No | `"60"` | How long an append waits for a busy user lock before retrying.       |
| `MAWS_MANAGER_KWARGS_JSON` | No       | `"{}"`       | Extra kwargs for `AgentSystemManager`; MAWS-managed paths still win. |
| `MAWS_BOT_KWARGS_JSON`     | No       | `"{}"`       | Extra kwargs for the selected MAS bot constructor.                   |
| `CAPTURE_FAILED_EVENTS`    | No       | `"false"`    | Write failed jobs/events to S3 under `failed-events/`. Secrets are redacted first (see Failure handling). |

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

* **Writing tokens back**: after a job, a token file is uploaded to S3 only if its content changed (SHA-256 compared with what the container read or last wrote), using an S3 conditional write: `If-Match` the ETag the container read, or `If-None-Match: *` when S3 had no copy. If another container changed the token in the meantime, the write is refused, a WARNING is logged, this container's copy is discarded and the S3 version is fetched again before the next job. A file that only came from the package is not copied to S3 unless your code changes it. (boto3 versions older than S3 conditional writes, late 2024, fall back to a plain write with a warning.)

---

## maws API

### `build_lambda_handler(bot_type: str) -> Callable`

Returns a **Lambda handler** already configured for the chosen bot.

* Lazily initializes `AgentSystemManager`.
* Instantiates the MAS bot (`WhatsappBot` / `TelegramBot`) with `verbose` coming from `VERBOSE`.
* GET (WhatsApp): calls `handle_webhook_verification` on the bot.
* POST: by default **self-invokes** the same Lambda asynchronously with normalized "worker" jobs (background mode): one invocation per chat (per lock) and webhook, carrying that chat's messages in provider order.
* Optional FIFO mode: enqueues normalized jobs to SQS FIFO instead of self-invoking.
* If the jobs cannot be handed off (SQS `SendMessage` or the async invoke fails), the POST answers `5xx` so Telegram/WhatsApp deliver the update again.
* In the "second hop":

  * **Claims** each message in the locks table; a message that was already processed, or that another worker is processing, is skipped (`MAWS/DuplicateMessages`). See "Repeated deliveries".
  * Takes a **lock** per `user_id` (or one global lock in shared-history mode) if `LOCKS_TABLE_NAME` is set, once for all the jobs of the invocation. If the lock is held in default `"drop"` mode, it **waits** up to `MAWS_BUSY_WAIT_SECONDS` (default 30, at most half of the remaining invocation time) and then **drops** the update (WARNING log + `MAWS/DroppedMessages` metric). In `"fifo"` mode the SQS invocation fails instead, so SQS redelivers the message after the visibility timeout.
  * **Imports** history from S3 if present. The container's local copy is deleted first, so a history deleted in S3 is never brought back from a warm container.
  * Optionally syncs per-user files/media from S3.
  * Executes `process_webhook_update` for each job, in order, and records each message as processed.
  * **Exports** history and optionally files/media to S3.
  * **Releases** the lock, also when saving fails (a crashed worker's lock simply expires).

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
      Runtime: python3.12
      Timeout: 120
      CodeUri: package/   # only your code: never .env files, histories or other secrets
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
          LOCK_TTL_SECONDS: "135"   # Timeout + 15
          # .env (via SSM) must contain WHATSAPP_APP_SECRET (or TELEGRAM_WEBHOOK_SECRET_TOKEN for Telegram)
      Events:
        WebhookGET:
          Type: Api
          Properties: { Path: "/webhook", Method: get }
        WebhookPOST:
          Type: Api
          Properties: { Path: "/webhook", Method: post }

  # The self-invocation permission, scoped to this function. It is a separate policy attached to
  # the role SAM generates (<FunctionLogicalId>Role): putting !GetAtt MyFunction.Arn in the
  # function's own Policies would be a circular dependency.
  MyFunctionSelfInvokePolicy:
    Type: AWS::IAM::Policy
    Properties:
      PolicyName: maws-self-invoke
      Roles: [ !Ref MyFunctionRole ]
      PolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Action: lambda:InvokeFunction
            Resource: [ !GetAtt MyFunction.Arn, !Sub "${MyFunction.Arn}:*" ]
```

> Switch GET/POST according to **WhatsApp** (GET+POST) or **Telegram** (POST only).
>
> `maws update` generates a complete template (in `.maws_build/template.yaml`) with all of this plus the log group, the dead-letter / failure queues and the S3 lifecycle described below.

---

## Self-invocation pattern

To avoid API Gateway timeouts, maws:

1. Responds **quickly** to the POST (200 OK once the work is handed off; `502` if the asynchronous self-invocation fails, so the provider retries).
2. **Self-invokes** asynchronously with the actual payload to process in the background: one invocation per chat (all of that chat's messages from the webhook, in order).
3. Uses a DynamoDB item as a per-user lock (`user_id`) to avoid duplicates when messages arrive in rapid succession.

If one chat's hand-off fails after another chat's succeeded, the provider's retry hands off the successful one again, and its worker skips the messages that were already processed (see "Repeated deliveries"). Asynchronous invocations that fail are retried twice by Lambda and then sent to the `<stack>-async-failures` SQS queue (see Failure handling).

If you do not configure `LOCKS_TABLE_NAME`, locking is **disabled** (best-effort behavior).

Lock expiry is enforced by the conditional write (`expiresAt < now`), not by DynamoDB TTL, which only cleans up old items and can lag for days. Inside Lambda a lock expires 15 seconds after the invocation's deadline: it outlives any running worker, and a crashed worker's lock is gone before SQS redelivers its message (the queue's visibility timeout is `lambda_timeout + 30`). Each lock carries a random token, so a worker only deletes its own lock.

### Repeated deliveries

The same message can reach a worker more than once: the provider retries a whole webhook when one chat's hand-off failed after another chat's succeeded (or when the webhook did not answer in time), Lambda occasionally delivers an asynchronous invocation twice, and SQS redelivers a message whose worker failed after processing it. MAWS processes each message once:

* Before taking the user lock, the worker **claims** each message in the locks table with the same conditional write as the lock. The key is `maws-event#<provider>#<chat id>#<event id>`, where the event id is Telegram's `update_id` or WhatsApp's `wamid`.
* Once the bot has processed the message, the claim becomes a **processed** record that is kept for `runtime.dedup_retention_days` days (`MAWS_DEDUP_RETENTION_DAYS`, default 8; WhatsApp retries a webhook for up to 7 days). It is written before the history is saved, because the replies have already been sent.
* A delivery of a message that has a claim or a record is skipped at once, without waiting for the user lock. It is logged (`[maws][dedup] Skipping ...`) and counted in `MAWS/DuplicateMessages`, not in `MAWS/DroppedMessages`.

A message that was not processed keeps no record, so the usual retries still process it: when it was dropped because the user stayed busy, when the user lock was busy in FIFO mode, when its history could not be loaded, or when processing raised (FIFO retries it). Two cases are worth knowing:

* If a worker crashes or times out while processing a message, its claim expires 15 s after that invocation's deadline, like its lock. Lambda's retry or the SQS redelivery then processes the message again, so replies sent before the crash can be sent twice.
* If saving the history fails after the message was processed, the message is not processed again: the error is logged (and captured with `capture_failed_events`) and that turn is missing from the history.

The records live in the stack's locks table, so they need no extra resource or IAM permission: they cost two DynamoDB writes per message, and DynamoDB TTL deletes them after they expire. Without `LOCKS_TABLE_NAME` there are no records. `"runtime": {"dedup_retention_days": 0}` turns them off. History appends (`/history/outgoing`) keep their own deduplication by `event_id`.

### Busy-message policy

With the default `MAWS_BUSY_POLICY=drop`, a worker that finds the user's lock held **waits** for it, then processes the update. Messages that belong together are not lost:

* a WhatsApp webhook that carries several messages of one chat is processed by **one** worker, in order, under one lock;
* Telegram albums and long texts split into several messages arrive as separate webhooks; each worker waits for the previous one to finish.

The wait is bounded: at most `runtime.busy_wait_seconds` in `params.json` (`MAWS_BUSY_WAIT_SECONDS`, default 30 s) and never more than half of the invocation's remaining time (so with `lambda_timeout: 30` at most about 15 s), leaving time to process the update afterwards. The worker polls the lock every 0.5 s and is billed while it waits.

**When drop mode still drops** an update (it is acknowledged to the provider and not processed):

* the user's lock is still held when the wait ends: a long model run, several updates queued behind each other, or a worker that crashed or timed out (its lock only expires 15 s after that invocation's deadline);
* with `history_mode: "shared"` the lock is global, so any other chat's work can make an update wait and be dropped;
* `busy_wait_seconds: 0` drops as soon as the lock is held (the behaviour before this wait existed).

Every drop is logged at WARNING (`[maws][drop] Dropped ... update <event id> for chat <chat id>`) and counted in the CloudWatch metric `MAWS/DroppedMessages` (dimension `FunctionName`), emitted as an Embedded Metric Format log line (no extra IAM permission). Updates that fail while being processed are not retried in drop mode either; they are logged with a traceback and counted in `MAWS/FailedMessages`. Set an alarm on these metrics, e.g. `Sum > 0` over 5 minutes.

Without `LOCKS_TABLE_NAME` there is no lock: nothing is dropped, but concurrent workers of one chat overwrite each other's history (last writer wins). Order between separate webhooks is not guaranteed in drop mode. Use FIFO when every message must be processed, in order.

Set `busy_policy` to `"fifo"` in `params.json` to generate an SQS FIFO queue. In FIFO mode MAWS enqueues one normalized job per provider message, uses the chat id as the FIFO message group in per-user history mode, and uses one global group in shared-history mode to avoid SQLite history races. If enqueueing fails, the webhook answers `503` so the provider retries; jobs that were already enqueued are deduplicated by SQS (content-based `MessageDeduplicationId`, 5-minute window), and later repeats are skipped by the worker (see "Repeated deliveries"). A job whose user lock is still held (for example by a crashed worker) fails its invocation instead of being acknowledged, so SQS redelivers it after the visibility timeout and the group keeps its order; each such retry counts toward the DLQ's `maxReceiveCount`. A worker without `BUCKET_NAME` or whose MAS initialization fails also fails the invocation, so the job is retried and ends in the DLQ instead of being deleted.

The worker DLQ is **on by default** for FIFO (`<stack>-worker-dlq.fifo`, `maxReceiveCount` 3, 14-day retention). Without it, SQS retries a failing message until it expires (4 days) and every later message of that chat waits behind it.

```json
{
  "busy_policy": "fifo",
  "failure_handling": {
    "worker_dlq": true,
    "max_receive_count": 3,
    "capture_failed_events": true
  }
}
```

### Failure handling

| Where failed work goes | Mode | Default |
| --- | --- | --- |
| `<stack>-worker-dlq.fifo` (SQS FIFO DLQ of the worker queue) | `fifo` | on (`failure_handling.worker_dlq`, `failure_handling.max_receive_count`) |
| `<stack>-async-failures` (SQS on-failure destination of async self-invocations, after Lambda's 2 retries) | `drop` | on (`failure_handling.async_failure_queue`) |
| `failed-events/` in the history bucket (diagnostic copies, secrets redacted) | both | off (`failure_handling.capture_failed_events`) |

The queue URLs are stack outputs (`WorkerDeadLetterQueueUrl`, `AsyncFailureQueueUrl`). Messages stay 14 days.

Inspect (without removing them; they become visible again after the visibility timeout):

```bash
aws sqs get-queue-attributes --queue-url "$DLQ_URL" --attribute-names ApproximateNumberOfMessages
aws sqs receive-message --queue-url "$DLQ_URL" --max-number-of-messages 10 --visibility-timeout 30 \
  --attribute-names All --message-attribute-names All
```

An async-failures message is an invocation record: `requestPayload` is the worker event (`{"maws_job": ...}` or `{"maws_jobs": [...]}`) and `responsePayload` / `requestContext.condition` say why it failed.

Redrive, once the cause is fixed:

* Worker DLQ (FIFO): use "Start DLQ redrive" on the DLQ in the SQS console (or `aws sqs start-message-move-task --source-arn <dlq arn>` where your CLI/region supports it for FIFO queues). The messages go back to `<stack>-worker.fifo` in their groups.
* Async failures: invoke the function again with each record's `requestPayload`, then delete the message:

  ```bash
  aws sqs receive-message --queue-url "$ASYNC_URL" --max-number-of-messages 10 --output json > batch.json
  jq -c '.Messages[]' batch.json | while read -r msg; do
    printf '%s' "$msg" | jq -c '.Body | fromjson | .requestPayload' > payload.json
    aws lambda invoke --function-name "$FUNCTION" --invocation-type Event \
      --cli-binary-format raw-in-base64-out --payload file://payload.json /dev/null &&
    aws sqs delete-message --queue-url "$ASYNC_URL" \
      --receipt-handle "$(printf '%s' "$msg" | jq -r .ReceiptHandle)"
  done
  ```

A redriven message that was already processed is skipped. One whose worker failed or crashed half-way is processed again, so replies it had already sent can be sent twice.

Failed-event captures never store webhook secrets: header, query and field values whose names look secret (`X-Telegram-Bot-Api-Secret-Token`, `X-Hub-Signature-256`, `Authorization`, cookies, API keys, `hub.verify_token`, ...) are replaced by `[REDACTED]`, and so are Telegram bot tokens and the values of secret-looking keys loaded from SSM wherever they appear (for example in an error message).

The Lambda logs record INFO, where httpx logs every request URL. The Telegram Bot API puts the bot token in that URL, so the runtime redacts it from httpx's log lines.

### Webhook security

MAWS only uses provider-supported webhook security mechanisms, and webhook authentication is **mandatory**:

* **Telegram**: `TELEGRAM_WEBHOOK_SECRET_TOKEN` in `.env.prod` (`maws start` generates one). During deploy, MAWS passes it to Telegram `setWebhook` as `secret_token`. At runtime, MAWS requires incoming POSTs to include the matching `X-Telegram-Bot-Api-Secret-Token` header. Telegram only accepts 1-256 characters from `A-Z a-z 0-9 _ -`; generate one with `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`.
* **WhatsApp**: keep the existing GET verify-token flow. Meta gives up on the verification GET after about 6 seconds, so MAWS answers it from `webhook_verify_token` in `runtime.bot_kwargs` or `WEBHOOK_VERIFY_TOKEN` / `WHATSAPP_VERIFY_TOKEN` in `.env.prod` without building the manager, even on a cold start. Only when none of them is set does it build the bot to check the token. For POST authenticity, set `WHATSAPP_APP_SECRET` in `.env.prod` (Meta App Dashboard > App settings > Basic > App secret). MAWS verifies `X-Hub-Signature-256` against the raw request body; `webhook_security.whatsapp_verify_signature` defaults to `true`.

`maws update` fails before uploading anything when the secret is missing (or, for Telegram, has invalid characters). Write `.env.prod` as plain `KEY=value` lines without quotes or `export`: Lambda reads the file that way. At runtime a POST without a configured secret is refused with `500`, so the provider retries once the configuration is fixed.

Telegram `setWebhook` runs on every deploy; if Telegram rejects it, `maws update` exits non-zero and prints Telegram's `description` (the stack is already deployed at that point; fix `.env.prod` and re-run).

To deploy without webhook authentication (local experiments only), opt out explicitly:

```json
{
  "webhook_security": {
    "allow_unauthenticated": true
  }
}
```

The deploy then prints a loud warning and sets `MAWS_ALLOW_UNAUTHENTICATED_WEBHOOKS=true`. A secret that is configured is still enforced (for WhatsApp, unless `whatsapp_verify_signature` is `false`). `whatsapp_verify_signature: false` is only accepted together with this opt-out.

### Appending externally sent messages

Use this when another service sends WhatsApp/Telegram messages from the same number (bulk templates, reminders, auto-replies) and the bot must know about them when the user answers.

The endpoint is **off by default**: the route is not deployed and the runtime does not look at it, so bots that do not opt in behave exactly as before. Enable it in `params.json` and add the secret to `.env.prod`:

```json
{
  "history_append": {
    "enabled": true,
    "role": "my_agent"
  }
}
```

```
MAWS_HISTORY_WRITE_SECRET=<long random string>
```

`history_append.enabled` deploys `POST /history/outgoing` and sets `MAWS_HISTORY_APPEND_ENABLED=true`. `history_append.role` sets `MAWS_HISTORY_APPEND_ROLE`: the agent the message is attributed to. MAS only shows a message to an agent as the agent's own turn when the role is the agent's name, so set it to the agent that answers users. If omitted, MAWS uses the bot's `component_name` when it is an agent, or the only agent in the system; otherwise it falls back to `assistant` and logs a warning.

The endpoint is separate from the provider webhook and always requires `MAWS_HISTORY_WRITE_SECRET` (it answers `503` without it). It verifies `X-MAWS-Timestamp` and `X-MAWS-Signature`; the signature is `sha256=` plus the HMAC-SHA256 of `<timestamp>.<raw-body>`. Requests more than five minutes from the current time are rejected.

POST a JSON body containing the recipient's MAWS `chat_id`, a stable `event_id`, and the message text. MAWS queues a worker job that appends the message under the normal per-user lock and S3 history import/export lifecycle. It does not call `process_webhook_update`. The event ID is stored in block metadata so retries do not append the same message twice.

```json
{
  "chat_id": "5491100000000",
  "event_id": "wamid.outbound-1",
  "content": "Hola desde el envio masivo"
}
```

Sign the exact serialized request body and send both headers:

```js
const timestamp = String(Math.floor(Date.now() / 1000));
const signature = createHmac("sha256", historyWriteSecret)
  .update(`${timestamp}.${rawBody}`)
  .digest("hex");

headers: {
  "X-MAWS-Timestamp": timestamp,
  "X-MAWS-Signature": `sha256=${signature}`,
}
```

Store the same random secret in MAWS `.env.prod` (so it reaches Lambda through SSM) and in the sending service. A `202` response means the history write was accepted for processing, not that the WhatsApp message was delivered.

Notes for the sending service:

* `chat_id` must be exactly the id MAWS uses for that user (for WhatsApp, the `from` / `wa_id` of incoming webhooks). MAWS does not check that the user exists; a different id creates a separate history. With the WhatsApp Cloud API, prefer `contacts[0].wa_id` from the send response as `chat_id` and `messages[0].id` as `event_id`.
* Send the text the user actually received. A placeholder such as a template name gives the agent no useful context.
* Call the endpoint right after the provider accepts the message. The message is stored when the worker runs, so a late call can land after the user's reply.
* Appends are never dropped because a user is busy: the worker waits up to `MAWS_HISTORY_APPEND_LOCK_WAIT_SECONDS` for the user lock and otherwise fails so Lambda async retries (or SQS, in FIFO mode) try again. With `history_mode="shared"` every append waits on one global lock; use `busy_policy="fifo"` for bulk sends.

---

## Loading environment from SSM

If `ENV_PARAMETER_NAME` is set, maws reads that **SecureString** parameter (in `.env` format) and exports **all** keys to `os.environ` **without** overriding already-present variables.

Accepted format:

```
# comment
KEY=VALUE
ANOTHER=VAL
```

`maws update` uploads `.env.prod` before `sam deploy` and passes the new parameter version to the stack, which sets `MAWS_ENV_PARAMETER_VERSION`; the function reads exactly that version (`<name>:<version>`). If the deploy fails, CloudFormation rolls back and the running function keeps reading the version it was deployed with, so it never runs with secrets meant for another deploy. Uploading after the deploy instead would let the new function cold-start with the old secrets. Every deploy therefore creates a new parameter version (Parameter Store keeps up to 100) and, because the environment changes, replaces the function's warm containers so new secrets take effect at once. Functions deployed by older maws versions read the latest version until their first deploy with this version. If the pinned version no longer exists, the runtime logs a warning and reads the latest one.

`maws update` itself never runs `.env.prod` as shell code: it reads the few values it needs (bot token, webhook secret) like the runtime does, and exports only deploy-tool settings found there (`AWS_*`, `SAM_CLI_*`, `PIP_*`, proxy variables) to `aws`/`sam`/`pip`.

---

## User histories and files

* User histories are stored as SQLite files in S3 under `history/<chat_id>.sqlite`.
* maws automatically imports/exports around processing.
* With `history_mode="shared"`, MAWS syncs MAS shared-history SQLite files under `history/shared/` and serializes workers globally so time-period/message-count rotation does not race.
* `maws pull-history` syncs all history files by default. Use `--user-id <id>` repeatedly or `--user-ids <id,id>` to download only selected user histories.
* By default, user media and generated files remain in Lambda `/tmp/files` only. Set `persist_files_s3=true` to sync per-user files under `files/<chat_id>/` in the history bucket. Each job starts from the S3 copy and, at the end, uploads the user's files and deletes from S3 the ones that were deleted during the job (for example by `/clear`), so cleared files do not come back. That needs `s3:DeleteObject`, which the template grants only on `<files_s3_prefix>/*` and only with `persist_files_s3=true`. If the job could not download the files first, nothing is deleted. With `infra.s3_versioning`, deleted files stay as noncurrent versions until `infra.s3_noncurrent_version_days`.
* For **tokens** defined in `SPECIAL_TOKEN_FILES_JSON`, it syncs S3 to `/tmp` (or package to `/tmp` if missing in S3) and can optionally expose paths via `TOKEN_ENV_MAP_JSON`.
* `infra.s3_lifecycle_days` expires objects under `history/`, `<files_s3_prefix>/` and `failed-events/` that were not written for that many days (an inactive user's history is deleted). Token files under `tokens_s3_prefix` (`secrets/`) never expire.
* With `infra.s3_versioning: true`, every history upload keeps the previous copy as a noncurrent version; they expire after `infra.s3_noncurrent_version_days` (default 30; `0` keeps them forever), and expired delete markers are removed.
* New projects created with `maws start` / `maws setup` get unguessable bucket names in `params.json` (`<project>-history-<random>`, `<project>-deploy-<random>`). Projects without names in `params.json` keep the old defaults (`<project>-history-bucket`, `<project>-deployment-bucket`): never change the history bucket of a deployed project, or the bot starts with empty histories. `maws update` and `maws pull-history` refuse a bucket owned by another AWS account (`--expected-bucket-owner`).

---

## Logging

* `VERBOSE=true` enables detailed traces in bots/MAS.
* The function logs (JSON format) go to the stack's log group `/aws/lambda/<stack>-maws` with a retention of `infra.log_retention_days` days, **30 by default**: long enough to investigate a problem reported a few weeks later, short enough that chat ids and (with `VERBOSE`) message contents are not kept forever. `0` keeps logs forever; other values must be a CloudWatch retention (1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, ...).
* The log group has its own name (set through Lambda's `LoggingConfig`) because Lambda already created `/aws/lambda/<function>` for every deployed function, and CloudFormation cannot create a log group that exists; that is why setting `log_retention_days` on an existing stack used to fail. Changing the retention later is a normal `maws update`. After upgrading, new logs go to the new group; the old `/aws/lambda/<function>` group keeps the older logs without retention (delete it or set a retention in CloudWatch). A stack that already manages `/aws/lambda/<function>` (deployed with `log_retention_days` on a fresh stack) keeps using it.
* Drops, failures and skipped repeated deliveries are also metrics: `MAWS/DroppedMessages`, `MAWS/FailedMessages` and `MAWS/DuplicateMessages`.

---

## Requirements

* Python 3.9+ for the library. Deploys default to the Lambda `python3.12` runtime (`infra.runtime` in `params.json`); `sam build` needs that Python version installed locally, so set `infra.runtime` to match your local `python3` if needed.
* The deploy tries to upgrade pip/setuptools/wheel and only warns if it cannot (for example an externally managed system Python, PEP 668).
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
mas[maws,telegram,whatsapp,env] @ git+https://github.com/mneuronico/multi-agent-system-library.git@<commit>
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
