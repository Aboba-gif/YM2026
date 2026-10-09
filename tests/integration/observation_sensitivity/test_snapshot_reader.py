"""Проверки чтения синтетических журналов и исключения писателя блокировками ОС."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from adrkit.config.validation import canonical_bytes, digest
from experiments.observation_sensitivity.analysis import read_results as mod
from experiments.observation_sensitivity.design import build_design


def full_groups():
    return tuple(dict.fromkeys((p.source, p.replicate) for p in build_design().paths))


def save(path, value):
    raw = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def campaign(tmp_path):
    project = tmp_path / "project"
    run = project / "runs/e06"
    files = [run / "freeze.json", run / "direct.json"]
    from tests.integration.observation_sensitivity.test_run_journal import current_admission
    pin = save(files[0], {"admission": current_admission(run).to_dict()})
    save(files[1], {"status": "completed", "result": {"status": "incomplete"}})
    locks = [run / ".journal.lock"]
    for source, rep in full_groups():
        path = run / source / f"replicate_{rep}.json"
        save(path, {"stage": "scored", "marker": f"{source}/{rep}"})
        files.append(path)
        locks.append(path.with_suffix(".json.lock"))
    for path in locks:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0preserved metadata")
    def load(**kwargs):
        return mod.load_records(run, expected_freeze_sha256=kwargs.pop("pin", pin), **kwargs)
    return dict(project=project, run=run, files=files, locks=locks, pin=pin, load=load)



def test_terminal_read_is_owned_and_does_not_write(campaign):
    c = campaign
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in c["files"] + c["locks"]}
    result = c["load"]()
    assert len(result["groups"]) == 8 and len(result["raw_file_sha256"]) == 10
    assert result["raw_file_sha256"]["freeze"] == c["pin"]
    assert result["direct"]["result"]["status"] == "incomplete"
    result["groups"]["PG10/r1"]["marker"] = "caller mutation"
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


@pytest.mark.parametrize("index", [0, 1, 5])
def test_missing_or_busy_lock_prevents_all_scientific_reads(campaign, monkeypatch, index):
    c = campaign
    read = mod._read
    def frozen_only(path):
        assert path == c["files"][0], "group/direct bytes read before all declared group locks held"
        return read(path)
    monkeypatch.setattr(mod, "_read", frozen_only)
    c["locks"][index].unlink()
    with pytest.raises(mod.SnapshotError, match="lease unavailable"):
        c["load"]()
    assert not c["locks"][index].exists()


@pytest.mark.parametrize("index", [0, 1, 9])
def test_pending_blocks_without_recovery(campaign, monkeypatch, index):
    c = campaign
    pending = c["files"][index].with_suffix(".json.pending")
    pending.write_bytes(b"uncommitted")
    read = mod._read
    def frozen_only(path):
        assert index == 9 and path == c["files"][0], "scientific bytes read despite pending"
        return read(path)
    monkeypatch.setattr(mod, "_read", frozen_only)
    with pytest.raises(mod.SnapshotError, match="pending"):
        c["load"]()
    assert pending.read_bytes() == b"uncommitted"








@pytest.mark.parametrize("change", ["pin", "bytes", "duplicate_key", "nonfinite", "stage", "direct", "missing"])
def test_bad_or_nonterminal_files_fail(campaign, change):
    c = campaign
    if change == "pin":
        with pytest.raises(mod.SnapshotError, match="pin"):
            c["load"](pin="0"*64)
        return
    target = c["files"][-1]
    if change == "bytes":
        target.write_bytes(b'{ "stage": "scored" }\n')
    elif change == "duplicate_key":
        target.write_bytes(b'{"stage":"estimating","stage":"scored"}\n')
    elif change == "nonfinite":
        target.write_bytes(b'{"stage":"scored","value":NaN}\n')
    elif change == "stage":
        save(target, {"stage": "sealed"})
    elif change == "direct":
        save(c["files"][1], {"status": "partial"})
    else:
        target.unlink()
    with pytest.raises(mod.SnapshotError):
        c["load"]()


def test_mismatched_output_denied_even_with_matching_pin(campaign):
    c = campaign
    pin = save(c["files"][0], {"admission": {"output": str(c["run"].parent / "elsewhere")}})
    with pytest.raises(mod.SnapshotError, match="different result directory"):
        c["load"](pin=pin)


def test_empty_lock_is_valid_but_hardlink_is_denied(tmp_path):
    lock = tmp_path / "lease"
    lock.touch()
    before = lock.stat().st_mtime_ns
    with mod._ExistingLease(lock) as lease:
        assert not os.get_inheritable(lease.fd)
    assert lock.read_bytes() == b"" and lock.stat().st_mtime_ns == before
    alias = tmp_path / "alias"
    os.link(lock, alias)
    with pytest.raises(mod.SnapshotError, match="unaliased"):
        mod._ExistingLease(lock)


def test_hardlinked_scientific_input_is_rejected_before_read(tmp_path, monkeypatch):
    path = tmp_path / "scientific.json"
    save(path, {"stage": "scored"})
    os.link(path, tmp_path / "elsewhere.json")
    original = Path.open
    class Spy:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def fileno(self):
            return self.stream.fileno()
        def read(self):
            pytest.fail("aliased scientific bytes must not be read")
    monkeypatch.setattr(Path, "open", lambda p, *a, **kw: Spy(original(p, *a, **kw)))
    with pytest.raises(mod.SnapshotError, match="unaliased"):
        mod._read(path)


@pytest.mark.parametrize("shared", [False, True])
def test_actual_e06_holder_blocks_existing_only_reader(tmp_path, shared):
    lock = tmp_path / "lease"
    code = """import sys
from experiments.file_locks import FileLock
with FileLock(sys.argv[1],shared=sys.argv[2]=='True'):
 print('locked',flush=True);sys.stdin.readline()
"""
    process = subprocess.Popen([sys.executable, "-B", "-c", code, str(lock), str(shared)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(mod.SnapshotError, match="lease unavailable"):
            mod._ExistingLease(lock)
    finally:
        out, err = process.communicate("\n", timeout=15)
        assert process.returncode == 0, (out, err)
    before = lock.read_bytes()
    with mod._ExistingLease(lock):
        pass
    assert lock.read_bytes() == before


@pytest.mark.skipif(os.name != "nt", reason="Native Windows writer uses msvcrt")
def test_existing_only_reader_blocks_checkpoint_writer(tmp_path):
    lock = tmp_path / "lease"
    lock.write_bytes(b"\0")
    code = """import sys,msvcrt
with open(sys.argv[1],'r+b') as f:
 try: msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
 except OSError: print('blocked')
 else: raise SystemExit('writer wrongly admitted')
"""
    with mod._ExistingLease(lock):
        process = subprocess.run([sys.executable, "-B", "-c", code, str(lock)],
            capture_output=True, text=True, timeout=15)
        assert process.returncode == 0 and process.stdout.strip() == "blocked", process.stderr




def test_cli_never_overwrites_existing_or_project_file(campaign, tmp_path, monkeypatch):
    c = campaign
    freeze = json.loads(c["files"][0].read_text(encoding="utf-8"))
    monkeypatch.setattr(mod, "load_records", lambda *a, **kw: dict(freeze=freeze))
    for target in (c["run"] / "new-summary.json", tmp_path / "existing.json"):
        if target.name == "existing.json":
            target.write_text("retain", encoding="utf-8")
        with pytest.raises(mod.SnapshotError, match="new output"):
            mod.main(["--run", str(c["run"]),
                      "--freeze-sha256", c["pin"], "--output", str(target)])
    assert (tmp_path / "existing.json").read_text() == "retain"


def test_cli_combines_safe_snapshot_and_full_pure_aggregation(campaign, tmp_path, monkeypatch):
    from tests.experiments.observation_sensitivity.fixtures.paired_records import make_records, seal, reseal
    c = campaign
    freeze, direct, groups = make_records()
    freeze["admission"].update(output=str(c["run"]))
    config = freeze["admission"]["configuration"]
    config["output"] = str(c["run"])
    freeze["admission"]["config_sha256"] = digest(config)
    freeze["admission"]["config_canonical_sha256"] = digest(config)
    freeze["admission_sha256"] = digest(freeze["admission"])
    seal(freeze)
    direct.update(admission_sha256=freeze["admission_sha256"], freeze_sha256=freeze["content_sha256"])
    seal(direct)
    pin = save(c["files"][0], freeze)
    save(c["files"][1], direct)
    for source, r in full_groups():
        group = groups[f"{source}/r{r}"]
        group["bindings"].update(admission_sha256=freeze["admission_sha256"], direct_record_sha256=digest(direct))
        reseal(group)
        save(c["run"] / source / f"replicate_{r}.json", group)
    target = tmp_path / "synthetic-test-summary.json"
    mod.main(["--run", str(c["run"]),
              "--freeze-sha256", pin, "--output", str(target)])
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["status"] == "complete_records"
    assert len(result["raw_outcomes"]) == 176 and len(result["paired_outcomes"]) == 224
    assert len(result["contrast_summaries"]) == 60
    assert result["inputs"]["freeze_file_sha256"] == result["raw_input_file_sha256"]["freeze"] == pin
    assert len(result["analyzer_sha256"]) == len(result["snapshot_reader_sha256"]) == 64



def test_selected_reader_locks_and_reads_only_the_declared_groups(campaign):
    from tests.experiments.observation_sensitivity.fixtures.paired_records import make_records, select_records
    c = campaign
    ids = ["spatial_G1_matched/PG10/r1/L2", "spatial_G1_matched/PG10/r1/H1",
           "spatial_G2_matched/PG10/r1/L2", "spatial_G2_matched/PG10/r1/H1"]
    records = make_records()
    admitted = records[0]["admission"]
    admitted["output"] = admitted["configuration"]["output"] = str(c["run"])
    freeze, direct, groups = select_records(records, ids, figures=[])
    expected = save(c["files"][0], freeze)
    save(c["files"][1], direct)
    target = c["run"] / "PG10/replicate_1.json"
    save(target, groups["PG10/r1"])
    for path in c["files"][3:] + c["locks"][2:]: path.unlink()
    before = {p: p.read_bytes() for p in c["run"].rglob("*") if p.is_file()}
    snapshot = c["load"](pin=expected)
    assert set(snapshot["groups"]) == {"PG10/r1"}
    assert set(snapshot["raw_file_sha256"]) == {"freeze", "direct", "PG10/r1"}
    assert before == {p: p.read_bytes() for p in before}
    target.with_suffix(".json.lock").unlink()
    with pytest.raises(mod.SnapshotError, match="lease unavailable"): c["load"](pin=expected)
