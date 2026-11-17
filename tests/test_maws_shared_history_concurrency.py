"""Shared history with workers for different users running at the same time.

Each worker has its own /tmp (its own MAS manager and history folder) and they share one S3 bucket.
Uploads are conditional on the ETag each worker synced; a worker that loses the race merges its
own writes into the newer files instead of overwriting them.
"""
from __future__ import annotations

import hashlib
import os
import types

import pytest

import maws.runtime as runtime_module
from maws.maws import MawsRuntime

BUCKET = "history-bucket"
PREFIX = "history/shared"


class _ConditionalS3:
    """In-memory S3 with ETags and conditional PUTs (If-Match / If-None-Match)."""

    def __init__(self):
        self.objects = {}
        self.puts = []
        self.before_put = None  # hook(key) run before each PUT is evaluated

    @staticmethod
    def _etag(data):
        return '"%s"' % hashlib.md5(data).hexdigest()

    def list_objects_v2(self, Bucket, Prefix):
        return {"Contents": [
            {"Key": key, "ETag": self._etag(data)}
            for (bucket, key), data in sorted(self.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]}

    def download_file(self, bucket, key, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as fp:
            fp.write(self.objects[(bucket, key)])

    def put_object(self, Bucket, Key, Body, IfMatch=None, IfNoneMatch=None):
        if self.before_put is not None:
            self.before_put(Key)
        current = self.objects.get((Bucket, Key))
        if IfNoneMatch == "*" and current is not None:
            self._fail()
        if IfMatch is not None and (current is None or self._etag(current) != IfMatch):
            self._fail()
        self.objects[(Bucket, Key)] = Body
        self.puts.append(Key.rsplit("/", 1)[-1])
        return {"ETag": self._etag(Body)}

    @staticmethod
    def _fail():
        raise runtime_module.botocore.exceptions.ClientError(
            {"Error": {"Code": "PreconditionFailed"}}, "PutObject"
        )


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    monkeypatch.setattr(runtime_module, "SHARED_HISTORY_RETRY_BASE_SECONDS", 0)


def _worker(tmp_path, s3, name, **manager_kwargs):
    from mas.manager import AgentSystemManager

    base = tmp_path / name
    manager = AgentSystemManager(base_directory=str(base), history_mode="shared", **manager_kwargs)
    runtime = object.__new__(MawsRuntime)
    runtime.bucket_name = BUCKET
    runtime.TMP_DIR = str(base)
    runtime.shared_history_prefix = PREFIX
    runtime.history_mode = "shared"
    runtime.s3 = s3
    runtime.manager = manager
    assert os.path.abspath(manager.history_folder) == os.path.abspath(base / "history")
    return runtime, manager


def _start_job(runtime, user_id):
    MawsRuntime._import_history_for(runtime, user_id, "", "")


def _write(manager, user_id, *texts):
    for text in texts:
        manager.add_blocks({"response": text}, user_id=user_id)


def _texts(manager, user_id):
    return [m["message"][0]["content"]["response"] for m in manager.get_messages(user_id)]


_readers = iter(range(1, 1000))


def _fresh_view(tmp_path, s3, **manager_kwargs):
    """A new worker (cold container) that has just synced S3."""
    runtime, manager = _worker(tmp_path, s3, f"reader{next(_readers)}", **manager_kwargs)
    MawsRuntime._sync_shared_history_from_s3(runtime)
    return manager


def test_lock_and_fifo_group_are_per_user_in_shared_mode():
    runtime = object.__new__(MawsRuntime)
    runtime.history_mode = "shared"
    runtime.manager = types.SimpleNamespace(_history_mode_is_shared=lambda: True)

    assert MawsRuntime._lock_id_for(runtime, "alice") == "alice"
    assert MawsRuntime._lock_id_for(runtime, "bob") == "bob"
    assert MawsRuntime._queue_group_id_for(runtime, {"chat_id": "alice"}) != (
        MawsRuntime._queue_group_id_for(runtime, {"chat_id": "bob"})
    )
    groups = MawsRuntime._group_jobs_by_lock(runtime, [
        {"chat_id": "alice", "n": 1}, {"chat_id": "bob", "n": 2}, {"chat_id": "alice", "n": 3},
    ])
    assert [[job["n"] for job in group] for group in groups] == [[1, 3], [2]]


def test_concurrent_workers_merge_instead_of_overwriting(workspace_tmp_path):
    s3 = _ConditionalS3()
    a, manager_a = _worker(workspace_tmp_path, s3, "a")
    b, manager_b = _worker(workspace_tmp_path, s3, "b")

    # Both start from the same (empty) S3, process different users and upload.
    _start_job(a, "alice")
    _start_job(b, "bob")
    _write(manager_a, "alice", "a1", "a2")
    _write(manager_b, "bob", "b1")
    MawsRuntime._upload_shared_history_to_s3(a)
    MawsRuntime._upload_shared_history_to_s3(b)  # loses the race on shared_history_000001

    reader = _fresh_view(workspace_tmp_path, s3)
    assert _texts(reader, "alice") == ["a1", "a2"]
    assert _texts(reader, "bob") == ["b1"]

    # The next job in worker A downloads what B merged and keeps going.
    _start_job(a, "alice")
    _write(manager_a, "alice", "a3")
    MawsRuntime._upload_shared_history_to_s3(a)
    reader = _fresh_view(workspace_tmp_path, s3)
    assert _texts(reader, "alice") == ["a1", "a2", "a3"]
    assert _texts(reader, "bob") == ["b1"]


def test_merge_follows_a_rotation_made_by_the_other_worker(workspace_tmp_path):
    s3 = _ConditionalS3()
    a, manager_a = _worker(workspace_tmp_path, s3, "a", history_max_messages=3)
    b, manager_b = _worker(workspace_tmp_path, s3, "b", history_max_messages=3)

    _start_job(a, "alice")
    _write(manager_a, "alice", "a1", "a2")
    MawsRuntime._upload_shared_history_to_s3(a)

    _start_job(a, "alice")
    _start_job(b, "bob")
    _write(manager_a, "alice", "a3", "a4")  # fills file 1 and rotates to file 2
    _write(manager_b, "bob", "b1")  # written to file 1, which A closes first
    MawsRuntime._upload_shared_history_to_s3(a)
    MawsRuntime._upload_shared_history_to_s3(b)

    reader = _fresh_view(workspace_tmp_path, s3, history_max_messages=3)
    assert _texts(reader, "alice") == ["a1", "a2", "a3", "a4"]
    assert _texts(reader, "bob") == ["b1"]
    # B's row went to the open file instead of reopening the closed one.
    assert sorted(name for _, name in [k for k in s3.objects]) == [
        f"{PREFIX}/shared_history_000001.sqlite",
        f"{PREFIX}/shared_history_000002.sqlite",
    ]
    file_one = reader._shared_history_db_path(1)
    conn, close = reader._open_history_db_for_read(file_one)
    try:
        users = [row[0] for row in conn.execute("SELECT user_id FROM message_history ORDER BY id")]
    finally:
        if close:
            conn.close()
    assert users == ["alice", "alice", "alice"]


def test_a_merge_does_not_duplicate_rows_already_uploaded(workspace_tmp_path):
    s3 = _ConditionalS3()
    b, manager_b = _worker(workspace_tmp_path, s3, "b", history_max_messages=2)
    _start_job(b, "bob")
    _write(manager_b, "bob", "b1", "b2", "b3")  # file 1 (b1, b2) and file 2 (b3)

    # File 1 uploads; just before file 2 does, another worker creates it in S3.
    other, manager_other = _worker(workspace_tmp_path, s3, "other", history_max_messages=2)
    _start_job(other, "carol")
    _write(manager_other, "carol", "c1", "c2", "c3")
    for conn in manager_other._db_pool.values():
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    other_files = {
        name: open(manager_other._shared_history_db_path(n), "rb").read()
        for n, name in ((2, "shared_history_000002.sqlite"),)
    }

    def create_file_two_first(key):
        if key.endswith("shared_history_000002.sqlite") and s3.before_put is not None:
            s3.before_put = None
            s3.objects[(BUCKET, key)] = other_files["shared_history_000002.sqlite"]

    s3.before_put = create_file_two_first
    MawsRuntime._upload_shared_history_to_s3(b)

    reader = _fresh_view(workspace_tmp_path, s3, history_max_messages=2)
    assert _texts(reader, "bob") == ["b1", "b2", "b3"]
    assert _texts(reader, "carol") == ["c3"]


def test_replaying_a_journal_twice_changes_nothing_the_second_time(workspace_tmp_path):
    s3 = _ConditionalS3()
    _, manager = _worker(workspace_tmp_path, s3, "a")
    manager.start_shared_history_journal()
    _write(manager, "alice", "a1")
    manager.clear_message_history("alice")
    _write(manager, "alice", "a2", "a3")
    journal = manager.take_shared_history_journal()
    assert [op for op, _ in journal] == ["insert", "clear", "insert", "insert"]
    assert manager.take_shared_history_journal() is None

    # Replaying onto files that already hold the result leaves them as they are...
    manager.replay_shared_history_journal(journal)
    assert _texts(manager, "alice") == ["a2", "a3"]
    # ...and onto files that never saw the job rebuilds it.
    for path in manager._list_shared_history_db_paths():
        manager._db_pool.pop(manager._shared_history_db_key(path)).close()
        os.remove(path)
    manager.replay_shared_history_journal(journal)
    assert _texts(manager, "alice") == ["a2", "a3"]
    # Replaying records nothing.
    assert manager.take_shared_history_journal() is None


def test_a_cleared_history_stays_cleared_after_a_merge(workspace_tmp_path):
    s3 = _ConditionalS3()
    a, manager_a = _worker(workspace_tmp_path, s3, "a")
    _start_job(a, "alice")
    _write(manager_a, "alice", "old")
    MawsRuntime._upload_shared_history_to_s3(a)

    b, manager_b = _worker(workspace_tmp_path, s3, "b")
    _start_job(a, "alice")
    _start_job(b, "bob")
    manager_a.clear_message_history("alice")
    _write(manager_a, "alice", "new")
    _write(manager_b, "bob", "b1")
    MawsRuntime._upload_shared_history_to_s3(b)
    MawsRuntime._upload_shared_history_to_s3(a)  # conflicts with B and merges

    reader = _fresh_view(workspace_tmp_path, s3)
    assert _texts(reader, "alice") == ["new"]
    assert _texts(reader, "bob") == ["b1"]


def test_gives_up_without_overwriting_when_the_file_keeps_changing(workspace_tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "SHARED_HISTORY_UPLOAD_ATTEMPTS", 3)
    s3 = _ConditionalS3()
    a, manager_a = _worker(workspace_tmp_path, s3, "a")
    other, manager_other = _worker(workspace_tmp_path, s3, "other")
    _start_job(other, "carol")
    _write(manager_other, "carol", "c1")
    MawsRuntime._upload_shared_history_to_s3(other)

    _start_job(a, "alice")
    _write(manager_a, "alice", "a1")
    counter = {"n": 0}

    def another_worker_always_wins(key):
        counter["n"] += 1
        _start_job(other, "carol")
        _write(manager_other, "carol", f"c{counter['n'] + 1}")
        s3.before_put = None
        MawsRuntime._upload_shared_history_to_s3(other)
        s3.before_put = another_worker_always_wins

    s3.before_put = another_worker_always_wins
    with pytest.raises(runtime_module.SharedHistoryUploadError):
        MawsRuntime._upload_shared_history_to_s3(a)
    s3.before_put = None

    reader = _fresh_view(workspace_tmp_path, s3)
    assert _texts(reader, "carol") == ["c1", "c2", "c3", "c4"]
    assert _texts(reader, "alice") == []


def test_a_conflict_without_a_journal_is_not_overwritten(workspace_tmp_path):
    s3 = _ConditionalS3()
    a, manager_a = _worker(workspace_tmp_path, s3, "a")
    MawsRuntime._sync_shared_history_from_s3(a)  # no journal started
    _write(manager_a, "alice", "a1")
    s3.objects[(BUCKET, f"{PREFIX}/shared_history_000001.sqlite")] = b"another worker's file"

    with pytest.raises(runtime_module.SharedHistoryUploadError):
        MawsRuntime._upload_shared_history_to_s3(a)
    assert s3.objects[(BUCKET, f"{PREFIX}/shared_history_000001.sqlite")] == b"another worker's file"


def test_old_botocore_without_conditional_writes_uploads_unconditionally(workspace_tmp_path):
    class ParamValidationError(Exception):
        pass

    class _OldS3(_ConditionalS3):
        def put_object(self, Bucket, Key, Body, **conditions):
            if conditions:
                raise ParamValidationError("Unknown parameter IfNoneMatch")
            return super().put_object(Bucket, Key, Body)

    s3 = _OldS3()
    a, manager_a = _worker(workspace_tmp_path, s3, "a")
    _start_job(a, "alice")
    _write(manager_a, "alice", "a1")
    MawsRuntime._upload_shared_history_to_s3(a)
    assert s3.puts == ["shared_history_000001.sqlite"]
