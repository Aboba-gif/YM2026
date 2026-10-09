"""Проверки состояния расчёта, блокировок и восстановления незавершённых записей."""
import hashlib
from copy import deepcopy
import errno
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from adrkit.config.validation import canonical_bytes, digest
from experiments.observation_sensitivity import lifecycle as life


BINDINGS = {"source": "PG10", "replicate": 1, "config_sha256": "a"*64,
            "versions": {"python": "test"}}
IDS = ["matched/PG10/r1/L2", "matched/PG10/r1/H1"]


@pytest.mark.parametrize("options", [
    {"replace_attempts": True}, {"replace_attempts": 0},
    {"replace_attempts": -1}, {"replace_attempts": 1.5},
    {"retry_delay": True}, {"retry_delay": -1},
    {"retry_delay": float("nan")}, {"retry_delay": float("inf")},
])
def test_invalid_retry_settings_do_not_create_checkpoint_files(tmp_path, options):
    target = tmp_path / "new_group/replicate_1.json"
    with pytest.raises(life.CheckpointError, match="retry count|delay"):
        life.Checkpoint(target, BINDINGS, IDS, **options)
    assert not target.parent.exists()


def test_explicit_larger_retry_settings_publish_same_checkpoint_bytes(tmp_path, monkeypatch):
    target = tmp_path / "replicate_1.json"
    payloads, waits = [], []
    real_replace = os.replace

    def intermittent(source, destination):
        payloads.append(Path(source).read_bytes())
        if len(payloads) < 9:
            raise PermissionError(errno.EACCES, "sharing violation")
        return real_replace(source, destination)

    with monkeypatch.context() as local:
        local.setattr(os, "replace", intermittent)
        local.setattr(life.time, "sleep", waits.append)
        with life.Checkpoint(target, BINDINGS, IDS, replace_attempts=9, retry_delay=1.1) as checkpoint:
            assert len(payloads) == 9 and len(set(payloads)) == 1
            assert waits == [1.1 * 2**index for index in range(8)]
            assert json.loads(target.read_bytes()) == checkpoint.record
            assert not target.with_suffix(".json.pending").exists()


def candidate(exponent, *, accepted=True):
    return dict(exponent=exponent, alpha=10.*10.**exponent, accepted=accepted,
                coefficients=[1., 2.], selection_mse=3.)


def path_record(count=0, *, finalized=False):
    return dict(candidates={str(e): candidate(e) for e in life.ALPHA_EXPONENTS[:count]},
                alpha_reference=10., provenance={"input_hash": "b"*64}, finalized=finalized)


def complete(cp):
    cp.record["paths"] = {pid: path_record(25, finalized=True) for pid in IDS}
    cp.save()


def raw_write(path, record, *, rehash=True):
    record = deepcopy(record)
    if rehash:
        record["content_sha256"] = digest({k: v for k, v in record.items() if k != "content_sha256"})
    path.write_bytes(canonical_bytes(record))


def test_checkpoint_preserves_origin_and_refuses_its_replacement(tmp_path):
    target = tmp_path / "checkpoint.json"
    with life.Checkpoint(target, BINDINGS, IDS) as checkpoint:
        record = deepcopy(checkpoint.record)
    origin = dict(artifact_sha256="a" * 64, content_sha256="b" * 64, selection_seal="c" * 64)
    record["origin"] = origin
    raw_write(target, record)
    with life.Checkpoint(target, BINDINGS, IDS) as checkpoint:
        assert checkpoint.record["origin"] == origin
        checkpoint.record["paths"][IDS[0]] = path_record(1)
        checkpoint.save()
    saved = target.read_bytes()
    with life.Checkpoint(target, BINDINGS, IDS) as checkpoint:
        checkpoint.record["origin"]["artifact_sha256"] = "d" * 64
        with pytest.raises(life.CheckpointError, match="origin cannot change"):
            checkpoint.save()
    assert target.read_bytes() == saved


@pytest.mark.parametrize("origin", [None, {}, {"artifact_sha256": "a" * 64},
    dict(artifact_sha256=True, content_sha256="b" * 64, selection_seal="c" * 64)])
def test_malformed_checkpoint_origin_is_rejected(tmp_path, origin):
    target = tmp_path / "checkpoint.json"
    with life.Checkpoint(target, BINDINGS, IDS) as checkpoint:
        record = deepcopy(checkpoint.record)
    record["origin"] = origin
    raw_write(target, record)
    saved = target.read_bytes()
    with pytest.raises(life.CheckpointError, match="origin"):
        life.Checkpoint(target, BINDINGS, IDS)
    assert target.read_bytes() == saved


def test_interrupted_resume_retains_rejections_and_scores_all_paths(tmp_path):
    target = tmp_path/"checkpoint.json"
    binding = deepcopy(BINDINGS)
    with life.Checkpoint(target, binding, IDS) as cp:
        binding["versions"]["python"] = "caller mutation"
        row = path_record(2)
        row["candidates"]["-7.5"]["accepted"] = False
        cp.record["paths"][IDS[0]] = row
        cp.save()
    with life.Checkpoint(target, BINDINGS, IDS) as cp:
        assert len(cp.record["paths"][IDS[0]]["candidates"]) == 2
        assert cp.record["paths"][IDS[0]]["candidates"]["-7.5"]["accepted"] is False
        for exponent in life.ALPHA_EXPONENTS[2:]:
            cp.record["paths"][IDS[0]]["candidates"][str(exponent)] = candidate(exponent)
        cp.record["paths"][IDS[0]]["finalized"] = True
        cp.record["paths"][IDS[1]] = dict(candidates={}, finalized=True,
            calibration_failure="covariance could not be estimated", procedure_accepted=False)
        cp.save()
        seal = cp.seal()
        assert cp.require_sealed() == seal
        cp.record["scores"] = {IDS[0]: {"status": "scored", "E_q": 1.2}}
        cp.save()
    with life.Checkpoint(target, BINDINGS, IDS) as cp:
        cp.require_sealed()
        cp.record["scores"][IDS[1]] = {"status": "unavailable", "reason": "calibration_failure"}
        cp.record["stage"] = "scored"
        cp.save()
    with life.Checkpoint(target, BINDINGS, IDS) as cp:
        assert cp.record["stage"] == "scored"
        assert cp.seal() == seal
        assert len(cp.record["scores"]) == 2
    assert target.with_name(target.name+".lock").exists()


@pytest.mark.parametrize("edit", [
    lambda cp: cp.record.update(stage="scored", scores={}),
    lambda cp: cp.record.update(stage="sealed", selection_seal="x"),
    lambda cp: cp.record.update(scores={}),
    lambda cp: cp.record["exponents"].append(4.5),
    lambda cp: cp.record["bindings"].update(replicate=2),
    lambda cp: cp.record["expected_paths"].reverse(),
    lambda cp: cp.record.update(version=True),
    lambda cp: cp.record.update(revision=50),
])
def test_invalid_transition_does_not_replace_committed_file(tmp_path, edit):
    target = tmp_path/"cp.json"
    with life.Checkpoint(target, BINDINGS, IDS) as cp:
        before = target.read_bytes()
        edit(cp)
        with pytest.raises(ValueError):
            cp.save()
        assert target.read_bytes() == before


@pytest.mark.parametrize("edit", [
    lambda row: row["candidates"]["-8.0"].update(accepted=True),
    lambda row: row["candidates"].pop("-8.0"),
    lambda row: row["provenance"].update(input_hash="changed"),
])
def test_committed_failed_candidate_and_provenance_are_append_only(tmp_path, edit):
    with life.Checkpoint(tmp_path/"cp.json", BINDINGS, IDS) as cp:
        row = path_record(1)
        row["candidates"]["-8.0"]["accepted"] = False
        cp.record["paths"][IDS[0]] = row
        cp.save()
        edit(row)
        with pytest.raises(ValueError):
            cp.save()


@pytest.mark.parametrize("row", [
    path_record(24, finalized=True),
    {"candidates": {}, "finalized": 1},
    {"candidates": {}, "finalized": True, "calibration_failure": "bad", "procedure_accepted": True},
    {"candidates": {}, "finalized": False, "calibration_failure": "bad", "procedure_accepted": False},
    {"candidates": {"4.5": candidate(4.5)}, "finalized": False},
    {"candidates": {"-8.0": candidate(-7.5)}, "finalized": False},
    {"candidates": {"-8.0": dict(candidate(-8.), alpha=float("nan"))}, "finalized": False},
    {"candidates": {"-8.0": dict(candidate(-8.), accepted=1)}, "finalized": False},
])
def test_malformed_path_or_grid_rejected(tmp_path, row):
    with life.Checkpoint(tmp_path/"cp.json", BINDINGS, IDS) as cp:
        cp.record["paths"][IDS[0]] = row
        with pytest.raises(ValueError):
            cp.save()


def test_finalized_path_is_immutable_even_before_group_seal(tmp_path):
    with life.Checkpoint(tmp_path/"cp.json", BINDINGS, IDS) as cp:
        cp.record["paths"][IDS[0]] = path_record(25, finalized=True)
        cp.save()
        cp.record["paths"][IDS[0]]["new_score"] = 1.
        with pytest.raises(ValueError, match="finalized paths"):
            cp.save()


def test_seal_requires_every_final_record_and_detects_unsaved_tamper(tmp_path):
    with life.Checkpoint(tmp_path/"cp.json", BINDINGS, IDS) as cp:
        with pytest.raises(ValueError):
            cp.require_sealed()
        cp.record["paths"][IDS[0]] = path_record(25, finalized=True)
        with pytest.raises(ValueError):
            cp.seal()
        cp.record["paths"][IDS[1]] = path_record(24)
        with pytest.raises(ValueError):
            cp.seal()
        cp.record["paths"][IDS[1]] = path_record(25, finalized=True)
        cp.seal()
        cp.record["paths"][IDS[0]]["candidates"]["-8.0"]["coefficients"][0] = 99.
        with pytest.raises(ValueError):
            cp.require_sealed()
        with pytest.raises(ValueError):
            cp.save()


def test_score_resume_append_only_and_complete_coverage(tmp_path):
    with life.Checkpoint(tmp_path/"cp.json", BINDINGS, IDS) as cp:
        complete(cp)
        cp.seal()
        cp.record["scores"] = {IDS[0]: {"status": "unavailable"}}
        cp.record["stage"] = "scored"
        with pytest.raises(ValueError, match="every planned path"):
            cp.save()
        cp.record["stage"] = "sealed"
        cp.save()
        cp.record["scores"][IDS[0]]["status"] = "scored"
        with pytest.raises(ValueError, match="scores cannot change"):
            cp.save()


@pytest.mark.parametrize("change", ["digest", "seal", "version", "paths", "grid", "extra"])
def test_resume_rejects_tamper_even_after_envelope_rehash(tmp_path, change):
    target = tmp_path/"cp.json"
    with life.Checkpoint(target, BINDINGS, IDS) as cp:
        complete(cp)
        cp.seal()
        record = deepcopy(cp.record)
    if change == "digest":
        record["stage"] = "scored"
        record["scores"] = {pid: {"status": "unavailable"} for pid in IDS}
    elif change == "seal":
        record["paths"][IDS[0]]["candidates"]["-8.0"]["coefficients"][0] = 99
    elif change == "version":
        record["version"] = True
    elif change == "paths":
        record["expected_paths"].reverse()
    elif change == "grid":
        record["exponents"][0] = -9.
    else:
        record["mystery"] = 1
    raw_write(target, record, rehash=change != "digest")
    with pytest.raises(ValueError):
        life.Checkpoint(target, BINDINGS, IDS)


@pytest.mark.parametrize("suffix", [b',"stage":"estimating"}', b',"bogus":NaN}', b',"bogus":1e999}'])
def test_resume_rejects_duplicate_keys_and_nonfinite_json(tmp_path, suffix):
    target = tmp_path/"cp.json"
    with life.Checkpoint(target, BINDINGS, IDS):
        pass
    target.write_bytes(target.read_bytes().rstrip()[:-1]+suffix)
    with pytest.raises(ValueError):
        life.Checkpoint(target, BINDINGS, IDS)


def test_same_process_lease_and_closed_instance(tmp_path):
    target = tmp_path/"cp.json"
    with life.Checkpoint(target, BINDINGS, IDS) as cp:
        with pytest.raises(life.LeaseBusyError):
            life.Checkpoint(target, BINDINGS, IDS)
    with pytest.raises(life.PersistenceError, match="closed"):
        cp.save()
    with life.Checkpoint(target, BINDINGS, IDS):
        pass


def run_child(target, body):
    code = ("from experiments.observation_sensitivity.lifecycle import Checkpoint, LeaseBusyError\n"
            "import json, os, sys\n"
            "path, bindings, ids = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])\n" + body)
    return subprocess.run([sys.executable, "-B", "-c", code, str(target), json.dumps(BINDINGS), json.dumps(IDS)],
                          capture_output=True, text=True, timeout=15)


def test_os_lease_blocks_other_process_then_process_death_releases_it(tmp_path):
    target = tmp_path/"cp.json"
    with life.Checkpoint(target, BINDINGS, IDS):
        result = run_child(target,
            "try:\n cp=Checkpoint(path,bindings,ids)\nexcept LeaseBusyError:\n sys.exit(7)\nsys.exit(9)\n")
        assert result.returncode == 7, result.stderr
    # os._exit завершает процесс без выхода из контекстного менеджера.
    
    result = run_child(target, "cp=Checkpoint(path,bindings,ids)\nos._exit(0)\n")
    assert result.returncode == 0, result.stderr
    with life.Checkpoint(target, BINDINGS, IDS):
        pass


def test_replace_retry_only_repeats_same_prepared_payload(tmp_path, monkeypatch):
    target = tmp_path/"cp.json"
    with life.Checkpoint(target, BINDINGS, IDS, retry_delay=0) as cp:
        original = life.os.replace
        seen = []
        def replace(source, destination):
            seen.append(Path(source).read_bytes())
            if len(seen) < 3:
                raise PermissionError(errno.EACCES, "temporary sharing conflict")
            return original(source, destination)
        monkeypatch.setattr(life.os, "replace", replace)
        cp.record["paths"][IDS[0]] = path_record(1)
        cp.save()
        assert len(seen) == 3 and len(set(seen)) == 1
        assert not cp.pending_path.exists()


@pytest.mark.parametrize("error, expected_calls", [(PermissionError(errno.EACCES, "sharing"), 3),
                                                   (OSError(errno.ENOSPC, "disk full"), 1)])
def test_failed_commit_poison_pending_explicit_recovery_no_reroll(tmp_path, monkeypatch, error, expected_calls):
    target = tmp_path/"cp.json"
    with life.Checkpoint(target, BINDINGS, IDS, replace_attempts=3, retry_delay=0) as cp:
        committed = target.read_bytes()
        original = life.os.replace
        calls = []
        def fail(*args):
            calls.append(args)
            raise error
        monkeypatch.setattr(life.os, "replace", fail)
        cp.record["paths"][IDS[0]] = path_record(1)
        with pytest.raises(life.PersistenceError):
            cp.save()
        assert len(calls) == expected_calls
        assert target.read_bytes() == committed
        pending = cp.pending_path.read_bytes()
        with pytest.raises(life.PersistenceError, match="do not reroll"):
            cp.save()
        monkeypatch.setattr(life.os, "replace", original)
    with pytest.raises(life.PendingRecoveryRequired):
        life.Checkpoint(target, BINDINGS, IDS)
    with life.Checkpoint(target, BINDINGS, IDS, recover_pending=True) as cp:
        assert target.read_bytes() == pending
        assert len(cp.record["paths"][IDS[0]]["candidates"]) == 1


def test_truncated_pending_cannot_be_discarded_or_recovered(tmp_path):
    target = tmp_path/"cp.json"
    with life.Checkpoint(target, BINDINGS, IDS) as cp:
        pending = cp.pending_path
    pending.write_bytes(b'{"schema":')
    with pytest.raises(life.PendingRecoveryRequired):
        life.Checkpoint(target, BINDINGS, IDS)
    with pytest.raises(ValueError):
        life.Checkpoint(target, BINDINGS, IDS, recover_pending=True)
    assert pending.read_bytes() == b'{"schema":'


def test_pending_cannot_modify_a_committed_candidate_even_with_valid_hash(tmp_path):
    target = tmp_path/"cp.json"
    with life.Checkpoint(target, BINDINGS, IDS) as cp:
        cp.record["paths"][IDS[0]] = path_record(1)
        cp.save()
        pending = deepcopy(cp.record)
        pending["revision"] += 1
        pending["paths"][IDS[0]]["candidates"]["-8.0"]["coefficients"][0] = 4.
        pending_path = cp.pending_path
    raw_write(pending_path, pending)
    with pytest.raises(ValueError, match="persisted candidate"):
        life.Checkpoint(target, BINDINGS, IDS, recover_pending=True)


def test_fsync_happens_before_replace(tmp_path, monkeypatch):
    target = tmp_path/"cp.json"
    events = []
    original_sync, original_replace = life.os.fsync, life.os.replace
    def sync(fd):
        events.append("fsync")
        return original_sync(fd)
    def replace(*args):
        assert events[-1] == "fsync"
        events.append("replace")
        return original_replace(*args)
    monkeypatch.setattr(life.os, "fsync", sync)
    monkeypatch.setattr(life.os, "replace", replace)
    with life.Checkpoint(target, BINDINGS, IDS):
        pass
    assert events[:2] == ["fsync", "replace"]


@pytest.mark.skipif(os.name != "nt", reason="real Windows sharing behaviour")
def test_actual_windows_reader_conflict_retries_commit_after_release(tmp_path, monkeypatch):
    target = tmp_path/"cp.json"
    with life.Checkpoint(target, BINDINGS, IDS, retry_delay=0) as cp:
        reader = target.open("rb")
        original = life.os.replace
        errors = []
        def replace(*args):
            try:
                return original(*args)
            except PermissionError as error:
                errors.append(error)
                reader.close()
                raise
        monkeypatch.setattr(life.os, "replace", replace)
        try:
            cp.record["paths"][IDS[0]] = path_record(1)
            cp.save()
        finally:
            reader.close()
        assert len(errors) == 1
        assert len(cp.record["paths"][IDS[0]]["candidates"]) == 1


def v2_record():
    paths = {"main/L2": path_record(25, finalized=True),
             "temporal_average/H1": dict(candidates={}, finalized=True,
                calibration_failure="terminal covariance failure", procedure_accepted=False),
             "single/fixed_alpha": dict(finalized=True, candidates={}, candidate=candidate(-3.))}
    return dict(bindings=deepcopy(BINDINGS), expected_paths=list(paths), paths=paths,
                stage="scored", exponents=list(life.ALPHA_EXPONENTS), selection_seal=digest(paths), scores={})


def test_terminal_v2_checks_raw_file_pin_with_one_read(tmp_path, monkeypatch):
    target = tmp_path / "v2.json"
    raw = canonical_bytes(v2_record())
    target.write_bytes(raw)
    pin = hashlib.sha256(raw).hexdigest()
    events, original = [], Path.read_bytes
    def read(path):
        events.append(path)
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    result, actual_pin = life.read_terminal_v2(target, expected_file_sha256=pin, include_file_hash=True)
    assert result["stage"] == "scored" and actual_pin == pin
    assert events == [target]
    events.clear()
    with pytest.raises(life.CheckpointError, match="SHA|sha|identity|hash|digest|changed"):
        life.read_terminal_v2(target, expected_file_sha256="0" * 64)
    assert events == [target]


@pytest.mark.parametrize("kind", ["seal", "stage", "missing", "grid", "candidate", "single", "file_changed"])
def test_terminal_v2_rejects_incomplete_or_changed_records(tmp_path, kind):
    target = tmp_path/"v2.json"
    record = v2_record()
    if kind == "seal":
        record["paths"]["main/L2"]["candidates"]["-8.0"]["alpha"] *= 2
    elif kind == "stage":
        record["stage"] = "sealed"
    elif kind == "missing":
        record["paths"].pop("main/L2")
    elif kind == "grid":
        record["exponents"].pop()
    elif kind == "candidate":
        record["paths"]["main/L2"]["candidates"].pop("-8.0")
        record["selection_seal"] = digest(record["paths"])
    elif kind == "single":
        record["paths"]["single/fixed_alpha"]["finalized"] = False
        record["selection_seal"] = digest(record["paths"])
    target.write_bytes(canonical_bytes(record))
    pin = hashlib.sha256(target.read_bytes()).hexdigest()
    if kind == "file_changed": target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError):
        life.read_terminal_v2(target, expected_file_sha256=pin)


@pytest.mark.parametrize("ids", [IDS[::-1]+[IDS[0]], ["matched/EC04/r1/L2"],
                                   ["matched/PG10/r2/L2"], ["../PG10/r1/L2"], set(IDS), []])
def test_bad_path_manifest_is_rejected_before_checkpoint_creation(tmp_path, ids):
    target = tmp_path/"cp.json"
    with pytest.raises(ValueError):
        life.Checkpoint(target, BINDINGS, ids)
    assert not target.exists()


@pytest.mark.parametrize("field,value", [("alpha", 2.), ("alpha_reference", 0.),
                                        ("alpha_reference", True)])
def test_fixed_physical_alpha_matches_scale(tmp_path, field, value):
    with life.Checkpoint(tmp_path/"cp.json", BINDINGS, IDS) as cp:
        row = path_record(1)
        if field == "alpha":
            row["candidates"]["-8.0"][field] = value
        else:
            row[field] = value
        cp.record["paths"][IDS[0]] = row
        with pytest.raises(ValueError):
            cp.save()


def test_initial_pending_save_is_explicitly_recoverable(tmp_path, monkeypatch):
    target = tmp_path/"cp.json"
    original = life.os.replace
    def fail(*args):
        raise PermissionError("sharing")
    monkeypatch.setattr(life.os, "replace", fail)
    with pytest.raises(life.PersistenceError):
        life.Checkpoint(target, BINDINGS, IDS, replace_attempts=1)
    monkeypatch.setattr(life.os, "replace", original)
    assert not target.exists()
    with pytest.raises(life.PendingRecoveryRequired):
        life.Checkpoint(target, BINDINGS, IDS)
    with life.Checkpoint(target, BINDINGS, IDS, recover_pending=True) as cp:
        assert cp.record["revision"] == 0 and cp.record["paths"] == {}


def test_seal_write_failure_cannot_expose_truth_until_pending_commit(tmp_path, monkeypatch):
    target = tmp_path/"cp.json"
    with life.Checkpoint(target, BINDINGS, IDS, replace_attempts=1) as cp:
        complete(cp)
        original = life.os.replace
        def fail(*args):
            raise PermissionError("sharing")
        monkeypatch.setattr(life.os, "replace", fail)
        with pytest.raises(life.PersistenceError):
            cp.seal()
        with pytest.raises(life.PersistenceError):
            cp.require_sealed()
        assert json.loads(target.read_bytes())["stage"] == "estimating"
        monkeypatch.setattr(life.os, "replace", original)
    with life.Checkpoint(target, BINDINGS, IDS, recover_pending=True) as cp:
        cp.require_sealed()
        assert cp.record["stage"] == "sealed"


def test_file_fsync_failure_never_calls_replace(tmp_path, monkeypatch):
    target = tmp_path/"cp.json"
    with life.Checkpoint(target, BINDINGS, IDS) as cp:
        before = target.read_bytes()
        def fail(fd):
            raise OSError(errno.EIO, "fsync failed")
        def forbidden(*args):
            pytest.fail("replace must not run after failed file fsync")
        monkeypatch.setattr(life.os, "fsync", fail)
        monkeypatch.setattr(life.os, "replace", forbidden)
        cp.record["paths"][IDS[0]] = path_record(1)
        with pytest.raises(life.PersistenceError):
            cp.save()
        assert target.read_bytes() == before
        assert cp.pending_path.exists()


def test_lease_conflict_does_not_read_checkpoint(tmp_path, monkeypatch):
    target = tmp_path/"cp.json"
    with life.Checkpoint(target, BINDINGS, IDS):
        def forbidden(*args):
            pytest.fail("a second writer must not read the active checkpoint")
        monkeypatch.setattr(Path, "read_bytes", forbidden)
        with pytest.raises(life.LeaseBusyError):
            life.Checkpoint(target, BINDINGS, IDS)


def test_loaded_bindings_must_match_exact_json_types(tmp_path):
    target = tmp_path/"cp.json"
    with life.Checkpoint(target, {**BINDINGS, "some_parameter": 1}, IDS):
        pass
    with pytest.raises(ValueError, match="bindings"):
        life.Checkpoint(target, {**BINDINGS, "some_parameter": True}, IDS)


@pytest.mark.parametrize("component", ["target", "pending"])
def test_checkpoint_rejects_hard_link_before_read_or_replace(tmp_path, monkeypatch, component):
    target = tmp_path / "checkpoint.json"
    with life.Checkpoint(target, BINDINGS, IDS) as checkpoint:
        pending_record = deepcopy(checkpoint.record)
        pending_record["revision"] += 1
    committed = target.read_bytes()
    outside = tmp_path / "outside.json"
    raw_write(outside, pending_record)
    outside_bytes = outside.read_bytes()
    link = target if component == "target" else target.with_name(target.name + ".pending")
    if component == "target":
        target.unlink()
    os.link(outside, link)
    def forbidden(*args):
        pytest.fail("aliased checkpoint files must not be read or replaced")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(life.os, "replace", forbidden)
    with pytest.raises(life.CheckpointError, match="hard links"):
        life.Checkpoint(target, BINDINGS, IDS, recover_pending=True)
    # open не подменён: проверка не затронула внешний файл или прежнюю цель.
    with outside.open("rb") as stream:
        assert stream.read() == outside_bytes
    if component == "pending":
        with target.open("rb") as stream:
            assert stream.read() == committed


@pytest.mark.parametrize("broken", [False, True])
def test_pending_symbolic_link_is_rejected_including_broken_link(tmp_path, broken):
    target = tmp_path / "checkpoint.json"
    with life.Checkpoint(target, BINDINGS, IDS) as checkpoint:
        pending_record = deepcopy(checkpoint.record)
        pending_record["revision"] += 1
    before = target.read_bytes()
    outside = tmp_path / "outside.json"
    if not broken:
        raw_write(outside, pending_record)
    pending = target.with_name(target.name + ".pending")
    try:
        pending.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"OS does not permit symbolic links in this test directory: {error}")
    with pytest.raises(life.CheckpointError, match="aliases"):
        life.Checkpoint(target, BINDINGS, IDS, recover_pending=True)
    assert target.read_bytes() == before and pending.is_symlink()


def test_pending_directory_is_rejected_before_recovery(tmp_path):
    target = tmp_path / "checkpoint.json"
    with life.Checkpoint(target, BINDINGS, IDS):
        pass
    before = target.read_bytes()
    target.with_name(target.name + ".pending").mkdir()
    with pytest.raises(life.CheckpointError, match="ordinary file"):
        life.Checkpoint(target, BINDINGS, IDS, recover_pending=True)
    assert target.read_bytes() == before


def test_checkpoint_rechecks_pending_after_lease_acquisition(tmp_path, monkeypatch):
    target = tmp_path / "checkpoint.json"
    with life.Checkpoint(target, BINDINGS, IDS) as checkpoint:
        pending_record = deepcopy(checkpoint.record)
        pending_record["revision"] += 1
    before = target.read_bytes()
    outside = tmp_path / "outside.json"
    raw_write(outside, pending_record)
    pending = target.with_name(target.name + ".pending")
    lease = life._Lease
    def acquire(*args, **kwargs):
        owner = lease(*args, **kwargs)
        os.link(outside, pending)
        return owner
    monkeypatch.setattr(life, "_Lease", acquire)
    with pytest.raises(life.CheckpointError, match="hard links"):
        life.Checkpoint(target, BINDINGS, IDS, recover_pending=True)
    assert target.read_bytes() == before
