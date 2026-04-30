# Code Review Improvement Plan

Review date: 2026-04-30

Scope: MAS core package, MAWS AWS helper, packaging metadata, examples, tests, bundled shell resources, and documentation alignment. This plan intentionally focuses on fixes, hardening, maintainability, and best practices for existing behavior, not new product features.

## Executive Summary

The repository has a useful core concept, but the implementation is currently high-risk in three areas:

1. Packaging and supported-runtime metadata are inconsistent with the source tree, so installed CLI entry points and Python 3.8 support can break.
2. Runtime behavior has several correctness and reliability bugs around indexing, provider fallback, callbacks, AWS webhook acknowledgment, and standard-library tool definitions.
3. The codebase is dominated by a 6k-line `mas/mas.py` module and a 1.2k-line `maws/maws.py` module, making regression testing, error handling, and API boundaries harder than they need to be.

## Highest Priority Fixes

### 0. Fix programmatic `AgentSystemManager` initialization without `config`

File:

- `mas/mas.py`

Problem:

The new test suite surfaced a runtime failure in documented programmatic usage. `AgentSystemManager(base_directory=...)` with no JSON config raises:

```text
AttributeError: 'AgentSystemManager' object has no attribute 'general_system_description'
```

Relevant area:

- `AgentSystemManager.__init__()` checks `self.general_system_description` before the attribute is initialized when no config file calls `build_from_json()`.

Impact:

- Programmatic construction, direct component creation, persistence tests, and several README examples are broken unless a config file is provided.
- This blocks lower-level tests for tools, processes, and history from exercising their intended behavior.

Recommended fix:

- Initialize `self.general_system_description`, `self.on_update`, and `self.on_complete` from constructor defaults before optional `build_from_json()`.
- Add tests for config-less manager construction, explicit constructor defaults, and config-file overrides.

### 1. Fix package inclusion for `mas.cli` and `mas.lib`

Files:

- `pyproject.toml`
- `mas/cli/cli.py`
- `mas/lib/std.py`
- `mas/lib/std.json`

Problem:

`pyproject.toml` declares console scripts:

- `mas = "mas.cli.cli:main"`
- `maws = "maws.cli:main"`

But `[tool.setuptools].packages` only includes `["mas", "maws", "maws.resources"]`. It omits `mas.cli` and `mas.lib`. `setuptools.find_packages()` would discover `['mas', 'maws', 'mas.cli', 'mas.lib', 'maws.resources']`, but the hard-coded package list prevents those two packages from being installed.

Impact:

- The installed `mas` console script can fail with `ModuleNotFoundError: No module named 'mas.cli'`.
- The documented standard library import path can fail because `mas.lib` is not installed as an importable package.
- `MANIFEST.in` includes `mas/lib`, but package inclusion for wheels still needs the package listed or discovered.

Recommended fix:

- Replace the hard-coded package list with setuptools package discovery, or add `mas.cli` and `mas.lib` explicitly.
- Add a packaging smoke test that builds a wheel, installs it into a temporary venv, and checks:
  - `python -c "import mas, maws, mas.cli.cli, mas.lib.std"`
  - `mas --help`
  - `maws --help`

### 2. Resolve the Python version contract

Files:

- `pyproject.toml`
- `mas/mas.py`
- `mas/cli/cli.py`
- `maws/maws.py`

Problem:

`pyproject.toml` says `requires-python = ">=3.8"`, but the code uses PEP 585 built-in generic annotations such as:

- `list[str]` in `mas/mas.py`
- `tuple[str, int]` in `mas/mas.py`
- `list[str]` in `mas/cli/cli.py`
- `list[str]` in `maws/maws.py`

Without `from __future__ import annotations`, these annotations are not compatible with Python 3.8.

Impact:

Users on Python 3.8 can hit import-time/type-annotation failures despite the package claiming support.

Recommended fix:

- Either bump `requires-python` to `>=3.9`, or add `from __future__ import annotations` and audit all 3.9+ syntax.
- Add CI for each claimed Python version.

### 3. Correct index bounds handling

Files:

- `mas/mas.py`

Problem:

Several `_handle_index` implementations check `len(subset) >= abs(index)` before doing `subset[index]`. Positive indices equal to `len(subset)` pass the check but are out of range.

Examples:

- `Agent._handle_index` around lines 517-523
- `Process._handle_index` around lines 2001-2015
- Tool custom input handling around lines 1830-1837 has the same pattern

Impact:

Valid-looking target syntax can raise uncaught `IndexError`, especially when users request the Nth item in a collection with N elements using 0-based indexing incorrectly. The library should return no match or a controlled error consistently.

Recommended fix:

- Centralize index handling in one helper.
- Use `-len(subset) <= index < len(subset)` for integer indexes.
- Add tests for `-1`, `0`, `len - 1`, `len`, `-(len)`, and `-(len + 1)`.

### 4. Fix `maws` webhook acknowledgment path

File:

- `maws/maws.py`

Problem:

`MawsRuntime.handle_apigw_event()` calls `self.initialize_system()` before it checks whether the incoming event is an initial API Gateway POST or a background worker event. The intended self-invocation pattern is to acknowledge webhook POSTs quickly, but the current path can initialize MAS, load config, load bot clients, and contact services before returning the initial 200.

Relevant area:

- `handle_apigw_event()` around lines 268-304

Impact:

- API Gateway/Meta/Telegram webhook acknowledgments can be delayed by cold-start initialization and dependency loading.
- Bad bot credentials or config can break the initial webhook acknowledgment instead of only failing the background worker.

Recommended fix:

- Parse `http_method` first.
- For initial API Gateway POST, self-invoke and return 200 before `initialize_system()`.
- Initialize MAS only for GET verification and background worker events.
- Add Lambda handler unit tests for GET, initial POST, and worker event paths with mocked AWS clients.

### 5. Fix standard library tool mismatches and secret leakage

Files:

- `mas/lib/std.py`
- `mas/lib/std.json`

Problems:

- `read_google_doc()` prints `GOOGLE_DRIVE_API` and `GOOGLE_DOC_ID` to stdout, leaking secrets or identifiers.
- `create_payment_url()` takes `qty`, but `std.json` defines the input as `quantity`; the tool will report missing `qty`.
- `create_payment_url()` appears to swap MercadoPago `quantity` and `unit_price`: it sets `"quantity": price` and `"unit_price": qty`.
- `weather_search` documents `standard` as an accepted unit, while `weather_query()` only accepts `metric` and `imperial`.
- `youtube_search` defines `n_results` in `std.json`, while the function parameter is `max_results`.

Impact:

The bundled standard library can fail at runtime even when the agent produces the documented fields. One function can leak sensitive configuration in logs.

Recommended fix:

- Remove secret/debug prints.
- Align JSON component input names with function signatures.
- Add unit tests for each std tool/process with mocked HTTP/API clients.
- Add optional dependency extras for standard-library integrations.

## Core Runtime Reliability

### Provider calls should fail over consistently

File:

- `mas/mas.py`

Problem:

`Agent.run()` only catches `requests.exceptions.RequestException`, `json.JSONDecodeError`, and `ValueError` around provider calls. Response-shape errors such as `KeyError`, `IndexError`, or `TypeError` can escape and prevent fallback to later configured models/providers.

Relevant area:

- `Agent.run()` around lines 209-333
- Provider response indexing around lines 987-989, 1047-1048, 1287-1288, 1346-1348, 1412-1413, 1484-1485

Impact:

A malformed provider response can halt the whole agent even when fallback providers are configured.

Recommended fix:

- Normalize provider responses through a shared parser.
- Catch response-shape exceptions as provider failures, with clear logging.
- Add tests for malformed responses and fallback behavior.

### Add timeouts to all network calls

Files:

- `mas/mas.py`
- `mas/lib/std.py`

Problem:

Most provider calls have timeouts, but some network calls do not:

- Image URL fetching in provider formatting uses `requests.get(src["url"]).content` without a timeout.
- ElevenLabs voice-list request uses `requests.get(voices_url, headers=headers)` without a timeout.
- Several standard-library HTTP calls in `mas/lib/std.py` have no timeout.

Impact:

The process can hang indefinitely on slow third-party endpoints.

Recommended fix:

- Use a shared timeout default.
- Apply `timeout=...` to every `requests.get/post`.
- Add tests with mocked timeout exceptions.

### Sanitize user IDs before using them as paths or S3 keys

Files:

- `mas/mas.py`
- `maws/maws.py`

Problem:

User IDs are interpolated directly into local filesystem paths and S3 keys:

- `AgentSystemManager._get_db_path_for_user()` creates `history/<user_id>.sqlite`.
- `_store_file()` creates `files/<user_id>/...`.
- `MawsRuntime.s3_sqlite_key()` creates `history/<chat_id>.sqlite`.

Impact:

If an externally supplied chat/user ID contains path separators or traversal sequences, local writes can escape the intended folder, and S3 keys can become ambiguous.

Recommended fix:

- Introduce a single `safe_user_id()` helper.
- Allow only a narrow character set or encode IDs with URL-safe base64/hex.
- Store original IDs separately if needed for display.
- Add tests for IDs containing `/`, `\`, `..`, `:`, and Unicode.

### Add SQLite schema migration and connection safety

File:

- `mas/mas.py`

Problems:

- `_create_table()` creates the current schema, but existing SQLite files are not migrated if columns are missing.
- `_ensure_user_db()` uses `check_same_thread=False`, but there is no per-connection lock around writes.
- `_save_message()` starts `BEGIN IMMEDIATE` without rollback handling on insert/serialization failure.

Impact:

- Older histories can fail when newer code expects columns like `model`.
- Concurrent runs can conflict on a shared connection.
- Failed writes can leave transactions in a bad state.

Recommended fix:

- Add `PRAGMA user_version` migrations.
- Use per-user write locks or one connection per thread.
- Wrap writes in `try/except` with rollback.
- Add tests for importing old schemas and concurrent writes.

### Do not classify every byte string as an image

File:

- `mas/mas.py`

Problem:

`_is_image_bytes()` returns `True` for any `bytes` or `bytearray`, so `_to_blocks()` labels arbitrary bytes as an image block.

Impact:

Audio, PDFs, binary blobs, and pickled objects can be mislabeled as images. Downstream provider formatting then tries to send them as image inputs.

Recommended fix:

- Use magic-byte detection and MIME inference before assigning block type.
- Add an explicit generic `file` block type or keep non-image bytes as persisted file references in text blocks.
- Add tests for PNG bytes, MP3/Ogg bytes, PDF bytes, and arbitrary bytes.

## API and Behavior Consistency

### Make tool return validation explicit

File:

- `mas/mas.py`

Problem:

`Tool.run()` expects the function result to behave like a dictionary but checks `len(result)` before validating its type, then calls `result.setdefault(...)`.

Relevant area:

- `Tool.run()` around lines 1537-1632

Impact:

Non-dict returns fail indirectly and are swallowed as `default_output`, making the root cause hard to diagnose.

Recommended fix:

- Require tool functions to return a dict.
- Validate required output keys by name instead of comparing only dictionary length.
- Preserve extra metadata only if documented.
- Emit clear errors in non-verbose mode for developer mistakes.

### Avoid silent empty-system builds

File:

- `mas/mas.py`

Problem:

`build_from_json()` logs JSON/open errors and then continues with `system_definition = {}`.

Relevant area:

- `build_from_json()` around lines 4176-4185

Impact:

Invalid configs can produce an empty manager rather than failing fast. Later errors become harder to connect to the configuration problem.

Recommended fix:

- Raise on missing or invalid config by default.
- If lenient behavior is needed, make it an explicit option.
- Add tests for missing, malformed, and empty config files.

### Normalize callback signatures

Files:

- `mas/mas.py`
- `README.md`

Problem:

Callback invocation is inconsistent across manager and automation paths. Some lambdas pass `(messages, manager)` while others pass `(messages, manager, params)`. The README says callbacks must include `messages` and `manager`, with params optional, but code has several wrapper variants.

Impact:

User callbacks can work in one path and fail in another, especially inside automations and bot wrappers.

Recommended fix:

- Define one callback protocol.
- Add an adapter that inspects the callable signature once.
- Test callbacks with 2 args, 3 args, and no params across direct runs, automations, and bots.

## MAWS AWS Helper

### Fix return values for background worker paths

File:

- `maws/maws.py`

Problem:

Background processing paths return `None` in several cases, including successful worker completion.

Impact:

Direct Lambda invocation receives `null`, which is usable but makes tests, observability, and caller expectations less clear.

Recommended fix:

- Return explicit dictionaries such as `{"statusCode": 200, "body": "Processed"}` for all paths.
- Add tests for unrecognized, no-chat-id, locked, failed, and successful worker events.

### Validate required environment early and clearly

File:

- `maws/maws.py`

Problem:

`BUCKET_NAME` defaults to `"MISSING_BUCKET"`, and AWS clients are created eagerly in `MawsRuntime.__init__()`.

Impact:

Misconfiguration produces AWS errors later instead of a clear startup diagnostic.

Recommended fix:

- Validate required env vars for the path being executed.
- Consider lazy AWS client creation or injectable clients for tests.
- Replace sentinel defaults with explicit errors.

### Fix documentation/install mismatch

Files:

- `maws/README.md`
- `pyproject.toml`

Problem:

`maws/README.md` documents installing `maws` from `#subdirectory=maws`, but there is no standalone package metadata under `maws/`. The root package includes `maws`.

Impact:

Users following the MAWS README can get install failures or a different install shape than intended.

Recommended fix:

- Update MAWS docs to install from the repository root, or add standalone packaging metadata under `maws/`.
- Add a CI install smoke test for every documented install command.

### Repair mojibake in MAWS output/docs

Files:

- `maws/maws.py`
- `maws/resources/bootstrap.sh`
- `maws/resources/pull_history.sh`
- `maws/README.md`

Problem:

Several strings contain mojibake such as `âœ…`, `âš ï¸`, and `ðŸ”§`.

Impact:

CLI output and docs look corrupted on normal UTF-8 terminals.

Recommended fix:

- Replace mojibake with valid UTF-8 or plain ASCII.
- Prefer ASCII in shell output for portability.
- Add a check that flags replacement-character/mojibake patterns.

## Packaging and Dependencies

### Split core and integration dependencies

File:

- `pyproject.toml`

Problem:

The core `mas` import pulls bot/web dependencies at module import time:

- `telegram`
- `Flask`
- `requests`
- `dotenv`

The dependency list also includes `boto3` even for users who only want local MAS core.

Impact:

Core users install and import more than they need. Source checkouts without all deps cannot import `mas` at all.

Recommended fix:

- Split bot integrations into submodules imported lazily.
- Move optional integrations to extras, for example:
  - `mas[telegram]`
  - `mas[whatsapp]`
  - `mas[aws]`
  - `mas[std]`
  - `mas[dev]`
- Keep `requests` in core only if all core providers require it.

### Add dev/test extras and CI

Files:

- `pyproject.toml`
- `tests/`

Problem:

Tests use `pytest`, but there is no dev extra or CI configuration in the repo. The only current test file covers Wavespeed.

Recommended fix:

- Add `[project.optional-dependencies].dev` with `pytest`, coverage tooling, and lint/type tools.
- Add CI steps:
  - install package with `.[dev]`
  - run unit tests
  - run packaging smoke tests
  - run syntax/import checks on all supported Python versions

### Remove unused runtime dependencies

File:

- `pyproject.toml`

Problem:

`pandas` is listed as a runtime dependency but is not imported anywhere in the repository.

Recommended fix:

- Remove it unless it is a documented extension dependency.
- If needed for optional user functions, move it to an extra.

## Code Organization

### Break up `mas/mas.py`

File:

- `mas/mas.py`

Problem:

The file contains core components, provider clients, persistence, parser logic, bot integrations, STT/TTS, and utility functions in one large module.

Impact:

Small fixes have a high regression risk, tests need broad imports, and optional dependencies become mandatory.

Recommended split:

- `mas/core/components.py`: `Component`, `Agent`, `Tool`, `Process`, `Automation`
- `mas/core/manager.py`: `AgentSystemManager`
- `mas/core/parser.py`: input syntax parser
- `mas/core/blocks.py`: block conversion and file persistence helpers
- `mas/providers/*.py`: OpenAI/OpenRouter/Google/etc. clients
- `mas/bots/telegram.py`
- `mas/bots/whatsapp.py`
- `mas/audio/stt.py`
- `mas/audio/tts.py`

Do this incrementally with re-export compatibility from `mas/__init__.py`.

### Break up `maws/maws.py`

File:

- `maws/maws.py`

Recommended split:

- `maws/runtime.py`: Lambda runtime and handler
- `maws/aws_clients.py`: S3, SSM, Lambda, DynamoDB wrappers
- `maws/cli_actions.py`: start/update/setup/describe/list/remove
- `maws/config.py`: params/env parsing and validation

## Test Plan

Focused tests were added before refactors:

1. Packaging:
   - Build/install wheel.
   - Import `mas.cli.cli`, `mas.lib.std`, and `maws.cli`.
   - Execute `mas --help` and `maws --help`.

2. Python version:
   - CI matrix for every supported version.
   - If keeping 3.8, import all modules under 3.8.

3. Indexing:
   - Agent/tool/process target indexes around all boundaries.

4. Providers:
   - Mock successful responses.
   - Mock malformed responses.
   - Mock HTTP failures and fallback.

5. Persistence:
   - New DB creation.
   - Old DB migration.
   - Import/export round trip.
   - User ID sanitization.

6. Standard library:
   - Mock Google Drive, WeatherAPI, YouTube, Places, Calendar, MercadoPago.
   - Verify function signatures match `std.json`.

7. MAWS:
   - Initial POST returns fast without manager initialization.
   - Worker initializes and persists history.
   - Lock-acquired and lock-denied paths.
   - Missing env vars produce clear errors.

Added test files:

- `tests/conftest.py`
- `tests/test_agent_fallback.py`
- `tests/test_indexing_and_blocks.py`
- `tests/test_manager_persistence.py`
- `tests/test_maws_runtime.py`
- `tests/test_packaging_contracts.py`
- `tests/test_parser_and_components.py`
- `tests/test_standard_library_contracts.py`
- Existing `tests/test_wavespeed_provider.py`

Current test status before fixes:

```text
20 failed, 8 passed, 1 skipped
```

Most failures correspond to findings already listed in this plan: package inclusion, Python version metadata, provider fallback, index bounds, byte classification, MAWS POST acknowledgment, S3/local path sanitization, stdlib secret leakage, stdlib signature mismatches, MercadoPago value mapping, and weather docs mismatch. The new finding from the suite is config-less `AgentSystemManager` initialization failing before defaults are set.

## Verification Performed During Review

- Read repository structure, package metadata, core modules, MAWS modules, shell resources, tests, examples, and documentation headings.
- Ran `py_compile` successfully on:
  - `mas/mas.py`
  - `mas/cli/cli.py`
  - `mas/lib/std.py`
  - `maws/maws.py`
  - `maws/cli.py`
- Installed `pytest` into the bundled local runtime after sandboxed network access was blocked and escalation was approved.
- Added offline import stubs for optional integrations in `tests/conftest.py`, so tests do not need real Telegram, Flask, requests, boto3, botocore, or network access for the covered paths.
- Ran `python -m pytest -q --tb=short`; current result is `20 failed, 8 passed, 1 skipped`.
