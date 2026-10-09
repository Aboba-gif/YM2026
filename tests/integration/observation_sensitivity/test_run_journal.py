"""Проверки блокировок журнала, прерываний записи и атомарной замены файлов."""
from copy import deepcopy
import errno
import os
from pathlib import Path
import subprocess
import sys
from queue import Queue, Empty
from threading import Thread
from uuid import uuid4

import pytest

from adrkit.config.validation import canonical_bytes, digest, strict_json
from experiments.observation_sensitivity.admission import Admission
from experiments.observation_sensitivity.direct import DirectContractError
from experiments.observation_sensitivity import journal as jn
from experiments.observation_sensitivity import lifecycle as life
from experiments.file_locks import FileLock, FileLockBusy


def test_read_only_does_not_create_directory_or_lock(tmp_path):
    output = tmp_path / "e06"
    admitted = admission(output)
    with pytest.raises(jn.JournalError, match="directory does not exist"):
        jn.Freeze(output, admitted, read_only=True)
    assert not output.exists()
    output.mkdir()
    with pytest.raises(jn.JournalError, match="lock file does not exist"):
        jn.Freeze(output, admitted, read_only=True)
    assert not list(output.iterdir())


@pytest.mark.parametrize("value", [None, 0, 1, "true", []])
def test_read_only_is_explicit_boolean_before_any_write(tmp_path, value):
    output = tmp_path / "e06"
    with pytest.raises(jn.JournalError, match="read_only must be boolean"):
        jn.Freeze(output, admission(output), read_only=value)
    assert not output.exists()


def test_read_only_contexts_reject_every_write_and_preserve_bytes(setup):
    output, admitted = setup
    with jn.DirectJournal(output, admitted) as direct:
        direct.start()
        direct.finish({"test_only": "no numerical calculation"})
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
    with jn.Freeze(output, admitted, read_only=True) as frozen:
        assert frozen.require_frozen()["admission_sha256"] == admitted.sha256
        calls = [frozen.freeze, frozen.commit_pending,
                 lambda: frozen._write("blocked.json", {}),
                 lambda: frozen._commit("blocked.json", {})]
        for call in calls:
            with pytest.raises(jn.JournalError, match="Read-only"):
                call()
    with jn.DirectJournal(output, admitted, read_only=True) as direct:
        assert direct.record["status"] == "completed"
        for call in (direct.start, lambda: direct.finish({}), direct.commit_pending):
            with pytest.raises(jn.JournalError, match="Read-only"):
                call()
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}


def test_read_only_keeps_pending_and_identity_gates(setup):
    output, admitted = setup
    pending = output / "freeze.json.pending"
    pending.write_bytes((output / "freeze.json").read_bytes())
    with jn.Freeze(output, admitted, read_only=True) as frozen:
        assert frozen.pending_exists
        with pytest.raises(jn.PendingRecoveryRequired):
            frozen.require_frozen()
    assert pending.exists()
    pending.unlink()
    with pytest.raises(jn.JournalError, match="admission|identity"):
        jn.Freeze(output, admission(output, fingerprint="different"), read_only=True)


@pytest.mark.parametrize("shared, requested_shared", [(False, False), (False, True), (True, False)])
def test_os_lock_incompatible_process_cannot_change_bytes(tmp_path, shared, requested_shared):
    lock = tmp_path / "lock"
    lock.write_bytes(b"unchanged")
    code = """import sys
from experiments.file_locks import FileLock, FileLockBusy
try:
 with FileLock(sys.argv[1], shared=sys.argv[2]=='True'):
  print('unexpected', flush=True)
except FileLockBusy:
 print('blocked', flush=True)
"""
    with FileLock(lock, shared=shared) as held:
        assert not os.get_inheritable(held.fd)
        process = subprocess.run([sys.executable, "-B", "-c", code, str(lock), str(requested_shared)],
            capture_output=True, text=True, timeout=20)
        assert process.returncode == 0 and process.stdout.strip() == "blocked", process.stderr
        if shared:
            assert lock.read_bytes() == b"unchanged"
    assert lock.read_bytes() == b"unchanged"
    with FileLock(lock):
        pass


@pytest.mark.parametrize("cls", ["Freeze", "DirectJournal"])
def test_two_os_reader_processes_overlap_and_exclude_writer(setup, tmp_path, cls):
    """Оба процесса сообщают о входе, удерживая настоящую совместную блокировку."""
    output, admitted = setup
    if cls == "DirectJournal":
        with jn.DirectJournal(output, admitted) as direct:
            direct.start()
            direct.finish({"test_only": "no numerical calculation"})
    config = tmp_path / "admission.json"
    config.write_bytes(admitted._content)
    code = """import sys
from pathlib import Path
from experiments.observation_sensitivity.admission import Admission
from experiments.observation_sensitivity import journal
admitted=Admission(Path(sys.argv[2]).read_bytes(), ())
with getattr(journal, sys.argv[3])(sys.argv[1], admitted, read_only=True):
 print('READING', flush=True)
 sys.stdin.readline()
"""
    processes = []
    events = Queue()
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
    def collect(process, index):
        for line in process.stdout:
            events.put((index, line.strip()))
    try:
        for index in range(2):
            process = subprocess.Popen([sys.executable, "-B", "-c", code, str(output), str(config), cls],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            processes.append(process)
            Thread(target=collect, args=(process, index), daemon=True).start()
        observed = {events.get(timeout=25), events.get(timeout=25)}
        assert observed == {(0, "READING"), (1, "READING")}, observed
        for journal_cls in (jn.Freeze, jn.DirectJournal):
            with pytest.raises(life.LeaseBusyError):
                journal_cls(output, admitted)
        assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
    finally:
        for process in processes:
            if process.poll() is None:
                process.stdin.write("\n")
                process.stdin.flush()
            process.wait(timeout=25)
            assert process.returncode == 0
    with jn.Freeze(output, admitted) as frozen:
        frozen.require_frozen()


def test_exclusive_writer_blocks_existing_reader_without_initializing_empty_lock(tmp_path):
    lock = tmp_path / "empty.lock"
    lock.touch()
    with FileLock(lock, create=False):
        with pytest.raises(FileLockBusy):
            FileLock(lock, shared=True, create=False)
        with pytest.raises(FileLockBusy):
            FileLock(lock)
    assert lock.read_bytes() == b""


def admission(output, *, fingerprint="original"):
    record = current_admission(output).to_dict()
    files = {"adrkit/baseline.py": digest(fingerprint)}
    record["baseline"]["bindings"].update(code_files=files, code_sha256=digest(files))
    return Admission(canonical_bytes(record), ())


@pytest.fixture
def setup(tmp_path):
    output = tmp_path/"e06"
    admitted = admission(output)
    with jn.Freeze(output, admitted) as frozen:
        frozen.freeze()
    return output, admitted


def read(path):
    return strict_json(path.read_bytes())


def write_record(path, value):
    value = deepcopy(value)
    value["content_sha256"] = digest({key: item for key, item in value.items()
                                       if key != "content_sha256"})
    path.write_bytes(canonical_bytes(value)+b"\n")


def current_admission(output):
    from tests.experiments.observation_sensitivity.fixtures.paired_records import make_admission
    from experiments.observation_sensitivity.design import build_design
    sources = {name: dict(record={"name": name}, sha256=digest({"name": name}))
               for name in ("PG10", "SB150", "EC04", "EC06", "NEW-J2", "NEW-S2")}
    return Admission(canonical_bytes(make_admission({}, sources, build_design(), output=output.resolve())), ())


def test_current_freeze_and_direct_journal_use_explicit_version_without_runtime_calls(tmp_path):
    output = tmp_path / "project/runs/current"
    admitted = current_admission(output)
    with jn.Freeze(output, admitted) as frozen:
        assert frozen.freeze()["version"] == 3
    with jn.DirectJournal(output, admitted) as journal:
        assert journal.start()["version"] == 3
        assert journal.finish({"synthetic": "no calculation"})["version"] == 3
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    historical = admitted.to_dict()
    historical["version"] = 1
    with pytest.raises(jn.JournalError, match="version-3"):
        jn.Freeze(output, Admission(canonical_bytes(historical), ()))
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_current_admission_cannot_resume_a_historical_freeze(tmp_path):
    output = tmp_path / "project/runs/current"
    # Старый формат создаём напрямую, чтобы проверить отказ текущего журнала.
    
    output.mkdir(parents=True)
    old = dict(schema="ym2026.observation_sensitivity.admission", version=1,
        status="prepared_not_frozen", output=str(output.resolve()))
    write_record(output / "freeze.json", dict(schema="ym2026.observation_sensitivity.freeze",
        version=1, admission=old, admission_sha256=digest(old), frozen_at="2026-09-27T00:00:00+00:00"))
    before = (output / "freeze.json").read_bytes()
    with pytest.raises(jn.JournalError, match="current admission"):
        jn.Freeze(output, current_admission(output))
    assert (output / "freeze.json").read_bytes() == before


@pytest.mark.parametrize("cls", [jn.Freeze, jn.DirectJournal])
def test_historical_admission_has_no_write_path(tmp_path, cls):
    output = tmp_path / "project/runs/current"
    old = current_admission(output).to_dict()
    old["version"] = 1
    with pytest.raises(jn.JournalError, match="canonical version-3"):
        cls(output, Admission(canonical_bytes(old), ()))
    assert not output.exists()


@pytest.mark.parametrize("alter", ["missing", "code_digest", "file_hash", "version", "science_hash", "group_hash", "group_missing", "path_order", "paths", "extra", "output_overlap"])
def test_journal_refuses_invalid_input_binding_before_writes(tmp_path, alter):
    output = tmp_path / "project/runs/current"
    record = current_admission(output).to_dict()
    baseline = record["baseline"]
    if alter == "missing": record.pop("sources")
    if alter == "code_digest": baseline["bindings"]["code_sha256"] = "b" * 64
    if alter == "file_hash": baseline["bindings"]["code_files"]["adrkit/baseline.py"] = "invalid"
    if alter == "version": record["version"] = 3.0
    if alter == "science_hash": record["science_spec_sha256"] = "b" * 64
    if alter == "group_hash": baseline["group_files"][next(iter(baseline["group_files"]))] = "invalid"
    if alter == "group_missing": baseline["group_files"].pop(next(iter(baseline["group_files"])))
    if alter == "path_order": record["expected_paths"]["PG10/r1"].reverse()
    if alter == "paths": record["config_path"] = "relative.json"
    if alter == "extra": record["current_execution"] = {}
    if alter == "output_overlap": record["output"] = str(Path(baseline["run_manifest_path"]).parent)
    with pytest.raises(jn.JournalError, match="Invalid|version-3"):
        jn.Freeze(output, Admission(canonical_bytes(record), ()))
    assert not output.exists()


def refuse_replace(*_):
    raise OSError(errno.EIO, "injected replacement failure")


def test_freeze_is_explicit_and_identical_freeze_keeps_bytes_and_timestamp(tmp_path, monkeypatch):
    output, admitted = tmp_path/"e06", admission(tmp_path/"e06")
    clock = iter(["2026-09-27T10:00:00.000000+00:00"])
    monkeypatch.setattr(jn, "_now", lambda: next(clock))
    with jn.Freeze(output, admitted) as frozen:
        assert frozen.record is None and frozen.pending_exists is False
        assert not (output/"freeze.json").exists()
        with pytest.raises(jn.JournalError):
            frozen.require_frozen()
        original = frozen.freeze()
        raw = (output/"freeze.json").read_bytes()
        original["admission"]["baseline"]["bindings"]["code_sha256"] = "caller mutation"
        assert frozen.freeze() == frozen.require_frozen()
        assert frozen.record["admission_sha256"] == admitted.sha256
    with jn.Freeze(output, admitted) as frozen:
        assert frozen.freeze()["frozen_at"] == "2026-09-27T10:00:00.000000+00:00"
        assert (output/"freeze.json").read_bytes() == raw
    assert {p.name for p in output.iterdir()} == {".journal.lock", "freeze.json"}


@pytest.mark.parametrize("cls", [jn.Freeze, jn.DirectJournal])
def test_changed_admission_forbids_resume_without_replacing_evidence(setup, cls):
    output, _ = setup
    before = (output/"freeze.json").read_bytes()
    with pytest.raises(jn.JournalError, match="current admission"):
        cls(output, admission(output, fingerprint="changed code or parameters"))
    assert (output/"freeze.json").read_bytes() == before
    assert not (output/"direct.json").exists()


def test_output_must_match_admission_and_direct_requires_explicit_freeze(tmp_path):
    expected, other = tmp_path/"e06", tmp_path/"other"
    admitted = admission(expected)
    with pytest.raises(jn.JournalError, match="admitted E06 output"):
        jn.Freeze(other, admitted)
    assert not other.exists() and not expected.exists()
    with pytest.raises(jn.JournalError, match="current admission"):
        jn.DirectJournal(expected, admitted)
    assert not (expected/"freeze.json").exists()
    
    with jn.Freeze(expected, admitted) as frozen:
        frozen.freeze()


def test_shared_os_lease_excludes_both_context_types_and_other_process(setup):
    output, admitted = setup
    with jn.Freeze(output, admitted):
        with pytest.raises(life.LeaseBusyError):
            jn.DirectJournal(output, admitted)
        with pytest.raises(life.LeaseBusyError):
            jn.Freeze(output, admitted)
        program = """
import sys
from pathlib import Path
from experiments.observation_sensitivity.lifecycle import _Lease, LeaseBusyError
try:
    lease = _Lease(Path(sys.argv[1]))
except LeaseBusyError:
    raise SystemExit(0)
else:
    lease.close()
    raise SystemExit(91)
"""
        child = subprocess.run([sys.executable, "-c", program, str(output/".journal.lock")],
                               capture_output=True, text=True, timeout=20)
        assert child.returncode == 0, child.stderr
    with jn.DirectJournal(output, admitted) as direct:
        assert direct.record is None
    assert (output/".journal.lock").exists()


def test_start_is_committed_before_caller_collects_and_completed_result_is_owned(setup):
    output, admitted = setup
    calls = []

    def collect():
        calls.append(read(output/"direct.json")["status"])
        return {"status": "incomplete", "rows": [{"value": 4.5}]}

    with jn.DirectJournal(output, admitted) as direct:
        direct.start()
        result = collect()
        complete = direct.finish(result)
        assert calls == ["started"]
        assert complete["status"] == "completed"
        # Статус результата хранится отдельно от статуса вызова.
        assert complete["result"]["status"] == "incomplete"
        result["rows"][0]["value"] = -1
        complete["result"]["rows"][0]["value"] = -2
        assert direct.record["result"]["rows"][0]["value"] == 4.5
        with pytest.raises(AttributeError):
            direct.record = {}
        with pytest.raises(AttributeError):
            direct.pending_exists = True
    with jn.DirectJournal(output, admitted) as direct:
        assert direct.require_terminal()["result"]["rows"][0]["value"] == 4.5
        with pytest.raises(jn.JournalError, match="no rerun"):
            direct.start()
        with pytest.raises(jn.UnresolvedStart):
            direct.finish({"status": "replacement"})
    assert len(calls) == 1


def test_cli_can_store_caught_contract_error_as_owned_partial_without_solver_coupling(setup):
    output, admitted = setup
    partial = {"status": "contract_failure", "fields": [{"status": "failed", "attempts": 1}]}
    with jn.DirectJournal(output, admitted) as direct:
        direct.start()
        try:
            raise DirectContractError("counter contract", partial)
        except DirectContractError as error:
            saved = direct.finish(error.partial_record, partial=True)
            error.partial_record["fields"].clear()
        partial["fields"].clear()
        assert saved["status"] == "partial"
        assert len(direct.require_terminal()["result"]["fields"]) == 1
    with jn.DirectJournal(output, admitted) as direct:
        assert direct.require_terminal()["status"] == "partial"
        with pytest.raises(jn.JournalError):
            direct.start()


def test_abandoned_committed_start_cannot_start_or_finish_after_reopen(setup):
    output, admitted = setup
    with pytest.raises(KeyboardInterrupt):
        with jn.DirectJournal(output, admitted) as direct:
            direct.start()
            raise KeyboardInterrupt("caller interrupted during work")
    before = (output/"direct.json").read_bytes()
    with jn.DirectJournal(output, admitted) as direct:
        assert direct.record["status"] == "started" and not direct.pending_exists
        with pytest.raises(jn.UnresolvedStart):
            direct.start()
        with pytest.raises(jn.UnresolvedStart):
            direct.finish({"status": "guessed output"})
        with pytest.raises(jn.UnresolvedStart):
            direct.require_terminal()
    assert (output/"direct.json").read_bytes() == before


def test_fsync_precedes_atomic_start_publication(setup, monkeypatch):
    output, admitted = setup
    events = []
    real_sync, real_replace = os.fsync, os.replace

    def sync(fd):
        events.append("fsync")
        return real_sync(fd)

    def replace(source, target):
        assert events and events[-1] == "fsync"
        assert read(Path(source))["status"] == "started"
        assert not Path(target).exists()
        events.append("replace")
        return real_replace(source, target)

    with jn.DirectJournal(output, admitted) as direct:
        with monkeypatch.context() as local:
            local.setattr(os, "fsync", sync)
            local.setattr(os, "replace", replace)
            direct.start()
        assert events[:2] == ["fsync", "replace"]
        assert direct.pending_exists is False


def test_prepared_freeze_requires_explicit_commit_and_preserves_timestamp(tmp_path, monkeypatch):
    output, admitted = tmp_path/"e06", admission(tmp_path/"e06")
    with jn.Freeze(output, admitted) as frozen:
        with monkeypatch.context() as local:
            local.setattr(jn, "_replace_pending", refuse_replace)
            with pytest.raises(life.PersistenceError):
                frozen.freeze()
        assert frozen.record is None
        with pytest.raises(life.PersistenceError, match="had a failed write"):
            frozen.freeze()
    prepared = read(output/"freeze.json.pending")
    with jn.Freeze(output, admitted) as frozen:
        assert frozen.pending_exists and frozen.record is None
        with pytest.raises(life.PendingRecoveryRequired):
            frozen.freeze()
        with pytest.raises(life.PendingRecoveryRequired):
            frozen.require_frozen()
        assert frozen.commit_pending() == prepared
        assert not frozen.pending_exists
        assert frozen.require_frozen() == prepared
    assert not (output/"freeze.json.pending").exists()


def test_recovered_prepared_start_never_grants_a_new_computation(setup, monkeypatch):
    output, admitted = setup
    with jn.DirectJournal(output, admitted) as direct:
        with monkeypatch.context() as local:
            local.setattr(jn, "_replace_pending", refuse_replace)
            with pytest.raises(life.PersistenceError):
                direct.start()
        assert direct.record is None
    with jn.DirectJournal(output, admitted) as direct:
        assert direct.record is None and direct.pending_exists
        with pytest.raises(life.PendingRecoveryRequired):
            direct.start()
        recovered = direct.commit_pending()
        assert recovered["status"] == "started"
        with pytest.raises(jn.UnresolvedStart):
            direct.start()
        with pytest.raises(jn.UnresolvedStart):
            direct.finish({"status": "not computed"})


@pytest.mark.parametrize("partial", [False, True])
def test_prepared_result_recovers_exact_payload_without_second_collection(setup, monkeypatch, partial):
    output, admitted = setup
    calls = 0
    with jn.DirectJournal(output, admitted) as direct:
        direct.start()
        calls += 1  
        result = {"status": "contract_failure" if partial else "complete", "rows": [1, 2, 3]}
        with monkeypatch.context() as local:
            local.setattr(jn, "_replace_pending", refuse_replace)
            with pytest.raises(life.PersistenceError):
                direct.finish(result, partial=partial)
        assert direct.record["status"] == "started"
        with pytest.raises(life.PersistenceError, match="had a failed write"):
            direct.finish(result, partial=partial)
    prepared = read(output/"direct.json.pending")
    with jn.DirectJournal(output, admitted) as direct:
        assert direct.record["status"] == "started" and direct.pending_exists
        for operation in (direct.start, direct.require_terminal):
            with pytest.raises(life.PendingRecoveryRequired):
                operation()
        assert direct.commit_pending() == prepared
        assert direct.require_terminal()["result"] == result
        assert direct.record["status"] == ("partial" if partial else "completed")
    assert calls == 1


@pytest.mark.parametrize("kind", ["sharing", "io"])
def test_replace_retries_are_bounded_and_never_retry_numerical_work(setup, monkeypatch, kind):
    output, admitted = setup
    attempts, waits = [], []

    def failed(source, target):
        attempts.append((source, target))
        if kind == "sharing":
            raise PermissionError(errno.EACCES, "injected sharing violation")
        raise OSError(errno.EIO, "injected non-retryable IO error")

    with jn.DirectJournal(output, admitted, replace_attempts=3, retry_delay=.01) as direct:
        with monkeypatch.context() as local:
            local.setattr(os, "replace", failed)
            local.setattr(life.time, "sleep", waits.append)
            with pytest.raises(life.PersistenceError):
                direct.start()
        assert len(attempts) == (3 if kind == "sharing" else 1)
        assert waits == ([.01, .02] if kind == "sharing" else [])
    assert (output/"direct.json.pending").exists()
    assert not (output/"direct.json").exists()


def test_transient_sharing_retry_publishes_same_pending_bytes(setup, monkeypatch):
    output, admitted = setup
    seen = []
    real_replace = os.replace

    def intermittent(source, target):
        seen.append(Path(source).read_bytes())
        if len(seen) == 1:
            raise PermissionError(errno.EACCES, "injected sharing violation")
        return real_replace(source, target)

    with jn.DirectJournal(output, admitted, retry_delay=0) as direct:
        with monkeypatch.context() as local:
            local.setattr(os, "replace", intermittent)
            committed = direct.start()
        assert len(seen) == 2 and seen[0] == seen[1]
        assert read(output/"direct.json") == committed


def test_interruption_after_preparation_is_recovered_without_new_timestamp(tmp_path, monkeypatch):
    output, admitted = tmp_path/"e06", admission(tmp_path/"e06")

    def interrupt(*_):
        raise KeyboardInterrupt("interrupted between fsync and replace")

    with jn.Freeze(output, admitted) as frozen:
        with monkeypatch.context() as local:
            local.setattr(jn, "_replace_pending", interrupt)
            with pytest.raises(KeyboardInterrupt):
                frozen.freeze()
        assert frozen.record is None and frozen.pending_exists
        with pytest.raises(life.PendingRecoveryRequired):
            frozen.freeze()
    prepared = read(output/"freeze.json.pending")
    with jn.Freeze(output, admitted) as frozen:
        assert frozen.commit_pending() == prepared


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(admission_sha256="a"*64),
    lambda row: row.update(freeze_sha256="a"*64),
    lambda row: row.update(version=True),
    lambda row: row.update(run_id=str(uuid4())),
    lambda row: row.update(started_pid=0),
    lambda row: row.update(started_at="2026-09-27T10:00:00"),
    lambda row: row.update(result={}),
    lambda row: row.update(unplanned_field=1),
])
def test_invalid_prepared_result_preserves_committed_start_and_pending_evidence(setup, monkeypatch, mutation):
    output, admitted = setup
    with jn.DirectJournal(output, admitted) as direct:
        direct.start()
        with monkeypatch.context() as local:
            local.setattr(jn, "_replace_pending", refuse_replace)
            with pytest.raises(life.PersistenceError):
                direct.finish({"status": "complete"})
    target, pending = output/"direct.json", output/"direct.json.pending"
    original = target.read_bytes()
    changed = read(pending)
    mutation(changed)
    write_record(pending, changed)
    evidence = pending.read_bytes()
    with jn.DirectJournal(output, admitted) as direct:
        with pytest.raises(jn.JournalError):
            direct.commit_pending()
    assert target.read_bytes() == original and pending.read_bytes() == evidence


def test_corrupt_prepared_json_is_not_discarded_or_recomputed(setup):
    output, admitted = setup
    pending = output/"direct.json.pending"
    pending.write_bytes(b'{"incomplete":')
    with jn.DirectJournal(output, admitted) as direct:
        assert direct.pending_exists
        with pytest.raises(life.PendingRecoveryRequired):
            direct.start()
        with pytest.raises(jn.JournalError):
            direct.commit_pending()
    assert pending.read_bytes() == b'{"incomplete":'
    assert not (output/"direct.json").exists()


def test_prepared_result_without_committed_start_is_rejected(setup):
    output, admitted = setup
    with jn.DirectJournal(output, admitted) as direct:
        start = direct.start()
    # Имитируем утрату записи о начале расчёта.
    target = output/"direct.json"
    start.update(status="completed", finished_at=start["started_at"], result={"status": "complete"})
    write_record(output/"direct.json.pending", start)
    target.unlink()
    with jn.DirectJournal(output, admitted) as direct:
        with pytest.raises(jn.JournalError, match="without its start marker"):
            direct.commit_pending()
    assert not target.exists()


def test_prepared_different_terminal_cannot_replace_committed_terminal(setup):
    output, admitted = setup
    with jn.DirectJournal(output, admitted) as direct:
        direct.start()
        complete = direct.finish({"status": "complete", "value": 1})
    before = (output/"direct.json").read_bytes()
    complete["result"]["value"] = 2
    write_record(output/"direct.json.pending", complete)
    with jn.DirectJournal(output, admitted) as direct:
        with pytest.raises(jn.JournalError, match="does not extend"):
            direct.commit_pending()
    assert (output/"direct.json").read_bytes() == before


def test_freeze_appearing_during_empty_context_is_not_overwritten(tmp_path):
    output, admitted = tmp_path/"e06", admission(tmp_path/"e06")
    with jn.Freeze(output, admitted) as frozen:
        record = dict(schema="ym2026.observation_sensitivity.freeze", version=3,
                      admission=admitted.to_dict(), admission_sha256=admitted.sha256,
                      frozen_at="2026-09-27T00:00:00+00:00")
        write_record(output/"freeze.json", record)
        before = (output/"freeze.json").read_bytes()
        with pytest.raises(jn.JournalError, match="appeared"):
            frozen.freeze()
        assert (output/"freeze.json").read_bytes() == before


@pytest.mark.parametrize("operation", ["start", "finish", "commit_pending"])
def test_changed_freeze_during_direct_lease_blocks_every_publication(setup, operation):
    output, admitted = setup
    with jn.DirectJournal(output, admitted) as direct:
        if operation != "start":
            direct.start()
        if operation == "commit_pending":
            record = direct.record
            record.update(status="completed", finished_at=record["started_at"], result={"status": "complete"})
            write_record(output/"direct.json.pending", record)
        changed = read(output/"freeze.json")
        changed["frozen_at"] = "2026-09-28T00:00:00+00:00"
        write_record(output/"freeze.json", changed)
        with pytest.raises(jn.JournalError, match="Freeze changed"):
            if operation == "finish":
                direct.finish({"status": "complete"})
            else:
                getattr(direct, operation)()


def test_prepared_freeze_blocks_direct_actions_even_after_context_opened(setup):
    output, admitted = setup
    with jn.DirectJournal(output, admitted) as direct:
        (output/"freeze.json.pending").write_bytes((output/"freeze.json").read_bytes())
        with pytest.raises(life.PendingRecoveryRequired):
            direct.start()


def test_missing_pending_does_not_create_files(setup):
    output, admitted = setup
    for cls in (jn.Freeze, jn.DirectJournal):
        with cls(output, admitted) as journal:
            with pytest.raises(jn.JournalError):
                journal.commit_pending()
    assert {p.name for p in output.iterdir()} == {".journal.lock", "freeze.json"}


def test_closed_context_cannot_write_but_owned_snapshot_remains_readable(setup):
    output, admitted = setup
    with jn.DirectJournal(output, admitted) as direct:
        direct.start()
    assert direct.record["status"] == "started"
    with pytest.raises(life.PersistenceError, match="closed"):
        direct.finish({"status": "complete"})


@pytest.mark.parametrize("options", [
    {"replace_attempts": True}, {"replace_attempts": 0},
    {"replace_attempts": -1}, {"replace_attempts": 1.5},
    {"retry_delay": True}, {"retry_delay": -1},
    {"retry_delay": float("nan")}, {"retry_delay": float("inf")},
])
@pytest.mark.parametrize("journal_type", [jn.Freeze, jn.DirectJournal])
def test_invalid_retry_policy_rejected_before_any_directory_write(tmp_path, options, journal_type):
    output, admitted = tmp_path/"e06", admission(tmp_path/"e06")
    with pytest.raises(jn.JournalError, match="retry count|delay"):
        journal_type(output, admitted, **options)
    assert not output.exists()


def test_explicit_larger_retry_settings_publish_same_journal_bytes(setup, monkeypatch):
    output, admitted = setup
    payloads, waits = [], []
    real_replace = os.replace

    def intermittent(source, target):
        payloads.append(Path(source).read_bytes())
        if len(payloads) < 9:
            raise PermissionError(errno.EACCES, "sharing violation")
        return real_replace(source, target)

    with jn.DirectJournal(output, admitted, replace_attempts=9, retry_delay=1.1) as direct:
        with monkeypatch.context() as local:
            local.setattr(os, "replace", intermittent)
            local.setattr(life.time, "sleep", waits.append)
            committed = direct.start()
        assert len(payloads) == 9 and len(set(payloads)) == 1
        assert waits == [1.1 * 2**index for index in range(8)]
        assert read(output/"direct.json") == committed
        assert not (output/"direct.json.pending").exists()


def test_journal_child_alias_cannot_write_outside_admitted_directory(tmp_path):
    output, admitted = tmp_path/"e06", admission(tmp_path/"e06")
    output.mkdir()
    elsewhere = tmp_path/"outside.json"
    elsewhere.write_bytes(b"external content")
    try:
        (output/"freeze.json.pending").symlink_to(elsewhere)
    except (OSError, NotImplementedError):
        pytest.skip("This OS/user cannot create a file symlink")
    with jn.Freeze(output, admitted) as frozen:
        with pytest.raises(jn.JournalError, match="aliases"):
            frozen.freeze()
    assert elsewhere.read_bytes() == b"external content"


def test_blocking_lock_waits_for_native_owner_without_changing_existing_bytes(tmp_path):
    lock = tmp_path / "shared.lock"
    lock.write_bytes(b"existing lock metadata")
    events = Queue()
    code = """import os,sys
from experiments.file_locks import FileLock
print('REQUEST',flush=True)
with FileLock(sys.argv[1],blocking=True) as held:
 assert not os.get_inheritable(held.fd)
 print('ACQUIRED',flush=True)
 sys.stdin.readline()
"""
    def collect(process):
        for line in process.stdout:
            events.put(line.strip())
        events.put("<process-ended>")
    process = collector = None
    try:
        with FileLock(lock):
            process = subprocess.Popen([sys.executable, "-B", "-c", code, str(lock)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            collector = Thread(target=collect, args=(process,), daemon=True)
            collector.start()
            assert events.get(timeout=20) == "REQUEST"
            with pytest.raises(Empty):
                events.get(timeout=.2)
            assert process.poll() is None
        assert events.get(timeout=20) == "ACQUIRED"
        process.stdin.write('finish\n')
        process.stdin.flush()
        assert process.wait(timeout=20) == 0, process.stderr.read()
        assert lock.read_bytes() == b"existing lock metadata"
        with FileLock(lock):
            pass
    finally:
        if process is not None:
            try:
                process.stdin.write('stop\n')
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            process.wait(timeout=20)
            if collector is not None and collector.ident is not None:
                collector.join(timeout=20)
                assert not collector.is_alive(), 'Native waiter did not close stdout'


@pytest.mark.parametrize("blocking", [None, 0, 1, "true"])
def test_blocking_option_is_boolean_and_invalid_value_never_creates_lock(tmp_path, blocking):
    lock = tmp_path / "new.lock"
    with pytest.raises(ValueError, match="boolean"):
        FileLock(lock, blocking=blocking)
    assert not lock.exists()
