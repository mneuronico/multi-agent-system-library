# Changelog

## [0.4.0] - 2026-11-02

Changes since 0.3.5 (`00ea15b`) from a production-readiness review. Several defaults and behaviours changed on purpose because the old ones were unsafe or surprising, so read **Upgrading** before you deploy.

### Upgrading

#### maws deployments

Do these before the first `maws update` with this version:

- **Add the webhook secret to `.env.prod`; the deploy stops without it.** Telegram: `TELEGRAM_WEBHOOK_SECRET_TOKEN` (1-256 characters from `A-Z a-z 0-9 _ -`, for example `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`). WhatsApp: `WHATSAPP_APP_SECRET` (Meta App Dashboard > App settings > Basic). The only opt-out is `"webhook_security": {"allow_unauthenticated": true}` in `params.json`, for experiments.
- **Keep `.env.prod` as plain `KEY=value` lines**: no quotes, no `export`, no comments on the same line. It is no longer `source`d as shell code.
- **Install Python 3.12 locally, or set `infra.runtime`.** The default Lambda runtime went from `python3.9` to `python3.12`, and `sam build` needs that version on your machine. To keep another version, set `"infra": {"runtime": "python3.X"}`.
- **Replace `requirements_source: "pypi"` with `"git"`.** The `mas` package on PyPI is a different project, so the deploy now stops on `pypi`, and on any value other than `git` or `local`. Pin `requirements_ref` to a commit SHA or an existing tag; without it every deploy ships whatever is on the default branch.
- **Set `history_mode` in `params.json`, not only in `config.json`.** The value in `params.json` (default `per_user`) now also wins inside the manager. Before, the manager could follow `config.json` while locks and S3 followed `params.json`.
- **Custom `template.yaml` or `samconfig.toml` files in the project folder are no longer used.** maws generates the template in `.maws_build/`; customise the stack through `params.json`. Your own `lambda_function.py` and `requirements.txt` are still deployed unchanged.

The first deploy also changes the stack:

- Logs go to a new log group, `/aws/lambda/<stack>-maws`, kept 30 days (`infra.log_retention_days`, `0` keeps them forever). The old `/aws/lambda/<function>` group keeps the older logs with no retention; delete it or set a retention in CloudWatch.
- `drop` mode gets a `<stack>-async-failures` SQS queue, and `fifo` mode gets a worker dead-letter queue (`<stack>-worker-dlq.fifo`, 3 receives). Both are on by default; see `failure_handling` in `maws/README.md`.
- The self-invoke IAM permission is scoped to the function itself. S3 lifecycle rules only expire `history/`, the files prefix and `failed-events/`.
- With `persist_files_s3`, the function gets `s3:DeleteObject` on the files prefix: files deleted during a job (for example by `/clear`) are deleted from S3 too, instead of being downloaded again by the next job.

#### WhatsApp bots run with `start_whatsapp_bot`

- **The Flask server refuses to start without the app secret.** Pass `app_secret=` or store `WHATSAPP_APP_SECRET` / `whatsapp_app_secret`. Unsigned POSTs get `403`. `verify_signature=False` is for local testing only.
- **If you run your own server**, check each POST with `bot.verify_webhook_signature(raw_body, request.headers.get("X-Hub-Signature-256"))` before parsing the JSON.
- **The Graph API version moved from `v18.0` to `v23.0`.** Change it with `api_version=` or `WHATSAPP_API_VERSION`.

#### Code that uses the library

- **`user_id=` arguments apply to that call only.** This covers `run`, `get_messages`, `add_message`, `add_blocks`, `set_variable`, `stt` and others. Code that relied on `run(user_id=X)` making `X` the thread's user for later calls must pass `user_id` every time, or call `set_current_user(X)`.
- **Constructor arguments now win over `general_parameters`, which win over library defaults.** Before, values in the JSON, including `base_directory`, replaced arguments you passed explicitly. `imports` from both places are combined.
- **`base_directory` defaults to the current directory when the manager is created**, no longer when `mas` was imported.
- **`.env` values no longer replace environment variables the process already has.** `manager.get_key()` still prefers the file, and keys belong to the manager that loaded them.
- **A `config` string that looks like a path but does not exist raises `FileNotFoundError`.** Before, a typo such as `"config"` was treated as a description: an LLM generated a new system and `history/` and `files/` were wiped. Bootstrapping from a description raises `FileExistsError` when `config.json` or `fns.py` already exist.
- **Agents may only emit variable blocks for variables declared in `general_parameters.variables`.** Component overrides (`component:parameter`) and other block shapes in model output are stored as plain text. Set overrides from Python, tools or processes.
- **A `while` step with only `condition` now runs while the condition is true**, checked before every iteration. Before, `condition` was used as both `start_condition` (run the first iteration only if it is true) and `end_condition` (stop after any iteration in which it is true). If you meant "repeat until X", write `end_condition: X`.
- **`while` stops after `max_iterations` (default `100`) and raises.** Set `"max_iterations": null` to remove the limit.
- **An automation step naming a component that does not exist is an error.** Systems loaded from JSON fail when they load; automations created in Python fail when the step runs. Before, the step was skipped.
- **A response cut off by the token limit counts as a failed model attempt**, and MAS tries the next model. Before, the incomplete JSON was repaired silently. Raise `max_tokens` if the logs say "Response truncated". It does not count against the model's health, so the model is not put in cooldown for other requests.
- **OpenAI receives `max_tokens` as `max_completion_tokens`.** Other `model_params` keys are not sent to any provider, and a warning names them once.
- **Tool parameters annotated `str` receive the text verbatim**, and empty strings stay empty strings. `"true"`, `"false"`, `"null"` and `"none"` are still converted for other parameters.
- **`file:` references are only read from inside `files_folder`.** A file copied in from a `.pkl` path is stored as `.bin` and loads as bytes; only objects pickled by MAS itself are unpickled.
- **Clearing history deletes files.** `clear_message_history(user)` also deletes that user's files, including TTS audio, now saved under `files/<user>/tts/`. `clear_global_history()` deletes the folder of every user whose history it clears, and leaves anything else in `files_folder`. `reset_system()` no longer deletes a history, files or logs folder that is the project directory, one of its parents or a filesystem root.
- **`get_cost_report()` raises** when the manager was created without `usage_logging=True`.
- **`add_message()` returns the new message number.**
- **`run()` without `component_name` prefers your own system's latest automation.** Imported automations are only used when you have none, and the default linear automation leaves out imported components.

#### Bots

Behaviour changes for Telegram and WhatsApp:

- **Local files can only be sent from `files_folder` or from folders listed in `media_dirs`.** A path elsewhere raises `PermissionError` and nothing is uploaded. Add the folder to `media_dirs`, or copy the file in with `manager.save_file()`.
- **Without `on_complete`, every block of the latest message is sent**, each as its own message: text, images, audio, documents. Before, only one text was sent.
- **A user's messages that arrive while their previous one is running are queued**, in order, instead of ignored.
- **Telegram processes up to 256 updates at once** (`concurrent_updates=True`). Each user's messages still run one at a time. Pass `concurrent_updates=False` for the old one-by-one behaviour.
- **On Telegram, `audio` blocks are sent as audio files** unless marked `"is_voice_note": true`. Audio the user sent as a voice note keeps the flag.
- **The raw platform payload is no longer stored in message metadata**, because it holds names, usernames and phone numbers. Pass `store_original_payload=True` to keep storing it. Callbacks still receive it in `params["original_payload"]`.
- **When processing fails, the user receives `error_msg`.** Pass `None` to send nothing.
- **Incoming files over 20 MB are not downloaded**, and the user receives `media_too_large_msg`. Change the limit with `max_media_bytes`.
- **Admin commands are checked against the sender, also in Telegram groups.**
- **Invalid custom command names raise `ValueError`.** A valid name is `/` followed by 1-32 letters, digits or underscores.
- **Direct send methods raise when the platform rejects the request**: `TelegramError` or `mas.bots.WhatsappAPIError`.
- **`python-telegram-bot` must be 22.x** (`>=22,<23`), the series the bot was tested with. Before, any version was accepted, so a new major release could break the bot on the next deploy.

#### Standard library, dashboard and CLI

- **`markdown_to_pdf` returns two blocks**: a text block with `pdf_file_path`, and a `document` block. It saves to `files/<user>/<title>.pdf`.
- **Standard-library functions raise when a service fails**, so the tool logs the error and returns its `default_output` (`Error: ...`). Before, some put the error text inside their normal result, or under an `error` key they did not declare.
- **`weather_search` and `google_places_search` return the outputs they declare.**
- **`calendar_search` includes `end_date`**, and `current_date` uses the manager's timezone.
- **`mas dashboard` with a non-loopback `--host` requires a token** (`--token` or `MAS_DASHBOARD_TOKEN`; one is generated if missing). The dashboard no longer imports `fns.py`.
- **`maws start` no longer installs packages with `sudo` or runs `aws configure` by default.** Pass `--install-deps` / `--run-config`. It writes `.gitignore` instead of `.samignore`, random S3 bucket names, and a Telegram webhook secret.
- **`maws update --set` keeps values as text** unless they are exact JSON literals (`--set requirements_ref=1.10` stays `"1.10"`).
- **An invalid `params.json` stops the command.** Booleans must be `true`/`false` (or `1`/`0`, `yes`/`no`, `on`/`off`).

### Security

- WhatsApp webhooks verify `X-Hub-Signature-256`, and maws makes webhook authentication mandatory for both platforms.
- The Lambda package is staged in `.maws_package/` with only project code: `.env*`, `api_keys.json`, histories and deploy files are never packaged. Secrets reach Lambda only through an SSM parameter version pinned at deploy time.
- Model output and user text cannot read arbitrary files or trigger `pickle.load` through `file:` references.
- Bots cannot be made to upload files outside `files_folder` / `media_dirs`, for example your `.env`.
- The dashboard checks the `Host` header (DNS rebinding) and requires a token off loopback.
- `markdown_to_pdf` shows raw HTML as text and renders without JavaScript or local file access.
- API keys are scrubbed from standard-library errors. Webhook secrets and tokens are redacted from maws failed-event captures. On Lambda, the Telegram bot token is redacted from httpx's request logs, which used to write it to CloudWatch on every reply.
- `maws remove` checks that the stack exists in the chosen region before emptying buckets. It waits for the stack to be deleted before deleting the SSM parameter and the deployment artifacts (`--no-wait` keeps them and prints the commands), and deletes the deployment bucket only when nothing else is in it.
- Deploy and history buckets owned by another AWS account are refused.

### Added

- `max_context_messages`, for the manager and per agent: the most history messages an agent reads. The default `None` keeps the whole history, so cost grows with every message unless you set it. The trimmed conversation always starts with a user turn.
- `while` conditions can call functions: `"fn:name"`, or `"fn:name:<input spec>"`, from JSON as documented.
- Bots: `allowed_user_ids`, `media_dirs`, `max_media_bytes`, `dedup_ttl`, `send_queue_timeout`, `maintenance_timeout` and `error_msg`. `sender_id` and `message_id` are added to the metadata. Telegram groups accept `/command@YourBot`.
- Bots: `/clear_all_users` and `/reset_system` wait for messages in progress (`maintenance_timeout`) and run one at a time. `WhatsappBot.create_flask_app()` builds the webhook app for your own WSGI server.
- Bots split texts over 4096 characters and captions over 1024. Duplicate deliveries are ignored for an hour, and each user's replies are sent in order.
- maws:
  - Drop mode waits up to `runtime.busy_wait_seconds` (default 30 s) for a busy user before dropping the update; `0` restores the immediate drop.
  - Each message is processed once. Workers record processed messages in the locks table for `runtime.dedup_retention_days` days (default 8; `0` turns it off) and skip repeated deliveries. A message is identified by Telegram's `update_id`, or by WhatsApp's message id and timestamp; one without an id is always processed. No new AWS resources or permissions; see "Repeated deliveries" in `maws/README.md`.
  - CloudWatch metrics `MAWS/DroppedMessages`, `MAWS/FailedMessages` and `MAWS/DuplicateMessages`.
  - `infra.log_retention_days` and `infra.s3_noncurrent_version_days`.
- `add_message(..., user_id=...)`.

### Fixed

- Concurrent requests or parallel automation steps for the same user could read rows from another thread's query, returning truncated or mixed history with no error. Each pooled SQLite connection is now used under a lock.
- Concurrent users could see each other's agent metadata (the answering model, usage, provider response).
- Usage logs record tokens and costs without `return_token_count`; `summary.log` refreshes every 100 entries instead of re-reading the log on every run.
- Function conditions (`fn:...`) never worked from JSON.
- `on_update` for a single component now sees that component's output.
- Threads reconnect after `import_history`, `reset_system` and `clear_global_history`; imports discard a stale write-ahead log.
- Imports no longer create folders in the current directory or leave temp files. Imported agents use the importing system's `default_models`, `timeout`, `include_timestamp` and `general_system_description`.
- Failures of a model, tool or process are always logged, not only with `verbose`.
- The in-memory file cache is bounded (256 files / 64 MB).
- maws:
  - FIFO retries a busy job instead of deleting it.
  - Locks belong to their worker and are sized from the Lambda timeout.
  - A failed hand-off answers `5xx`, so the platform retries. The chats of that webhook that had already been handed off are no longer processed, and answered, a second time.
  - A history deleted in S3 is no longer brought back by a warm container.
  - Token files are written back only when changed, with a conditional write.
  - A rejected Telegram `setWebhook` makes `maws update` fail and prints Telegram's error.
  - WhatsApp's verification GET is answered from the verify token in the environment without building the manager. On a cold start, building it could take longer than the ~6 s Meta waits, so setting or changing the callback URL failed.
  - `requirements_source: "local"` with a path to a wheel, sdist or folder installs the Lambda extras, so Telegram bots no longer fail with `ImportError: python-telegram-bot is required`.
  - A `requirements.txt` that maws 0.3.x wrote in the project folder is replaced in the package even after `requirements_ref` or `extra_requirements` change. It used to count as hand-written then, so upgrading a project kept deploying the old mas version.
- `mas start` never overwrites files; `mas` and `maws` commands return proper exit codes; API keys are typed without echo.
- `youtube_search` works with `youtube-transcript-api` 1.x.

### Removed

- maws `requirements_source: "pypi"`: the PyPI `mas` package is not this library.
