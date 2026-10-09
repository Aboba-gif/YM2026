"""Проверки допуска метаданных и записей E05 для плана E06."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from adrkit.config.validation import canonical_bytes, digest, strict_json
from experiments.source_recovery.sources import ScaledSource, TwoPulse
from experiments.observation_sensitivity import admission as mod
from experiments.observation_sensitivity.design import build_design
from experiments.observation_sensitivity.lifecycle import ALPHA_EXPONENTS


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))
    return sha(path.read_bytes())


def rejected_path():
    return dict(finalized=True, alpha_reference=1., procedure_accepted=False,
        candidates={str(float(x)): dict(exponent=x, alpha=10.**x, accepted=False)
                    for x in ALPHA_EXPONENTS})


@pytest.fixture
def tree(tmp_path, monkeypatch):
    project = tmp_path / "project"
    base = tmp_path / "config/source_recovery.json"
    registered = tmp_path / "accepted/experiment.json"
    run = tmp_path / "campaign/source_recovery/run.json"
    config = tmp_path / "config/observation_sensitivity.json"
    code, new_code = project / "src/adrkit", project / "experiments"
    code.mkdir(parents=True)
    new_code.mkdir(parents=True)
    for index in range(3):
        (code / f"module{index:02}.py").write_text(f"INDEX = {index}\n", encoding="utf-8")
    for name in ("source_recovery", "source_comparison"):
        folder = new_code / name
        folder.mkdir()
        (folder / "run.py").write_text("RUN = True\n", encoding="utf-8")
    (new_code / "worker.py").write_text("WORKER = True\n", encoding="utf-8")
    protocol = tmp_path / "inputs/protocol.json"
    protocol_sha = put(protocol, {"fixture_protocol": True})
    input_paths = [tmp_path / "inputs" / f"static{i}.json" for i in range(2)]
    for index, target in enumerate(input_paths):
        put(target, {"static": index})
    spec = strict_json(mod._registered_config_path().read_bytes())
    spec["output"] = str(run.parent)
    spec["protected_roots"] = [str(tmp_path / "inputs")]
    spec["source_protocol"] = dict(path=str(protocol), sha256=protocol_sha)
    spec["input_files"] = [dict(path=str(p), sha256=sha(p.read_bytes())) for p in input_paths]
    put(registered, spec)
    monkeypatch.setattr(mod, "_registered_config_path", lambda: registered)
    runtime = dict(versions=dict(python="fixture-python", numpy="fixture-numpy", scipy="fixture-scipy"),
                   threads={key: "1" for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})
    origins = {"adrkit": str(code), "experiments": str(new_code)}
    source_factory = lambda: {name: ScaledSource(TwoPulse("jump" if i % 2 == 0 else "cosine"),
        spec["source_mass"] / 100) for i, name in enumerate(mod.SOURCE_IDS)}
    bindings = dict(config_sha256=put(base, spec),
        input_files={str(p): sha(p.read_bytes()) for p in [protocol, *input_paths]},
        code_files={"adrkit/module00.py": "a" * 64}, **deepcopy(runtime))
    bindings["code_sha256"] = digest(bindings["code_files"])
    run_record = dict(schema=spec["schema"], configuration=deepcopy(spec),
                      bindings=deepcopy(bindings), initial_state="estimating")
    put(run, run_record)
    groups = {}
    for source in ("PG10", "EC04"):
        for replicate in range(1, 5):
            expected = mod._baseline_paths(spec, source, replicate)
            paths = {pid: dict(finalized=True) if pid.startswith("single/") else rejected_path()
                     for pid in expected}
            group = dict(stage="scored", exponents=list(ALPHA_EXPONENTS), paths=paths,
                expected_paths=expected, selection_seal=digest(paths),
                bindings=dict(bindings, source=source, replicate=replicate,
                    source_record_sha256=mod.source_record_hash(source_factory()[source])))
            target = run.parent / source / f"replicate_{replicate}.json"
            put(target, group)
            groups[target] = group
    declaration = dict(schema=mod.CONFIG_SCHEMA, version=3, study=mod.STUDY,
        output=str(tmp_path / "campaign/observation_sensitivity"),
        design_sha256=digest(mod.design_record()),
        baseline=dict(kind="completed_run", config_path=str(base), run_manifest_path=str(run)))
    put(config, declaration)
    events = []
    def protocol_loader(path, *, expected_sha256):
        events.append("protocol")
        assert path == protocol and sha(path.read_bytes()) == expected_sha256
        return strict_json(path.read_bytes())
    def sources(protocol, *, mass):
        events.append("sources")
        assert protocol == {"fixture_protocol": True} and mass == spec["source_mass"]
        return source_factory()
    monkeypatch.setattr(mod, "load_protocol", protocol_loader)
    monkeypatch.setattr(mod, "make_sources", sources)
    return SimpleNamespace(root=tmp_path, project=project, base=base, run=run, config=config,
        registered=registered, protocol=protocol, input_paths=input_paths, code=code, new_code=new_code,
        groups=groups, spec=spec, declaration=declaration, run_record=run_record, bindings=bindings,
        runtime=runtime, events=events, build=lambda: mod.build_admission(config))


def refresh_run(tree):
    tree.bindings["config_sha256"] = put(tree.base, tree.spec)
    tree.run_record["configuration"] = deepcopy(tree.spec)
    tree.run_record["bindings"] = deepcopy(tree.bindings)
    put(tree.run, tree.run_record)
    for path, group in tree.groups.items():
        group["bindings"].update(deepcopy(tree.bindings))
        put(path, group)


def test_complete_owned_manifest_and_no_writes(tree):
    before = {str(p): p.read_bytes() for p in tree.root.rglob("*") if p.is_file()}
    admitted = tree.build()
    assert before == {str(p): p.read_bytes() for p in tree.root.rglob("*") if p.is_file()}
    data = admitted.to_dict()
    assert data["status"] == "prepared_not_frozen" and "frozen" not in data["configuration"]
    assert data["baseline"]["bindings"] == tree.bindings
    assert data["baseline"]["kind"] == "completed_run"
    assert not {"reference_root", "locations", "logical_project_root"} & set(data)
    assert len(data["baseline"]["group_files"]) == 8
    assert all(sha(Path(p).read_bytes()) == value for p, value in data["baseline"]["group_files"].items())
    assert data["counts"] == dict(paths=176, contrasts=224, new_paths=152, reused_paths=24,
        new_fit_attempts=3800, reused_fit_attempts=600, inverse_groups=8, source_profiles=6)
    assert not {"resource_policy", "coordination_root", "current_execution", "implementation", "runtime", "native_thread_pools"} & set(data)
    assert data["science_spec_sha256"] == digest(mod.scientific_configuration(tree.spec))
    assert tuple(data["sources"]) == tuple(sorted(mod.SOURCE_IDS))
    assert all(digest(v["record"]) == v["sha256"] for v in data["sources"].values())
    assert admitted.sha256 == digest(data) == tree.build().sha256
    data["sources"].clear()
    admitted.spec["source_mass"] = -1
    assert len(admitted.to_dict()["sources"]) == 6 and admitted.spec["source_mass"] > 0
    with pytest.raises(FrozenInstanceError):
        admitted._content = b"{}"


def test_fresh_admission_and_freeze_bind_the_actual_completed_group_bytes(tree):
    from experiments.observation_sensitivity.journal import Freeze, JournalError
    admitted = tree.build()
    with Freeze(tree.declaration["output"], admitted) as journal:
        frozen = journal.freeze()
    with Freeze(tree.declaration["output"], tree.build()) as journal:
        assert journal.require_frozen() == frozen
    group = next(iter(tree.groups))
    group.write_bytes(group.read_bytes() + b"\n")
    changed = tree.build()
    assert changed.sha256 != admitted.sha256
    with pytest.raises(JournalError, match="admission|identity"):
        with Freeze(tree.declaration["output"], changed):
            pass


def test_fresh_admission_reads_only_explicit_current_inputs(tree, monkeypatch):
    reads, original = [], mod._read
    def read(path):
        assert Path(path).is_relative_to(tree.root)
        reads.append(Path(path))
        return original(path)
    monkeypatch.setattr(mod, "_read", read)
    tree.build()
    assert tree.base in reads and tree.run in reads
    assert set(tree.groups) <= set(reads)
    assert not any("locations" in path.name for path in reads)


def test_real_current_settings_sources_and_source_inventory_form_a_complete_admission(tmp_path):
    """Настоящие входы и генераторы; кандидаты искусственно отклонены без решателя."""
    registered = mod._registered_config_path()
    spec = strict_json(registered.read_bytes())
    spec["output"] = str(tmp_path / "source_recovery")
    spec["protected_roots"] = []
    for item in [spec["source_protocol"], *spec["input_files"]]:
        item["path"] = str((registered.parent / item["path"]).resolve())
    baseline = tmp_path / "source_recovery.json"
    config_sha = put(baseline, spec)
    code = {"adrkit/recorded_baseline.py": "a" * 64}
    bindings = dict(config_sha256=config_sha, code_files=code, code_sha256=digest(code),
        input_files={item["path"]: item["sha256"] for item in [spec["source_protocol"], *spec["input_files"]]},
        versions={"python": "recorded-python", "numpy": "recorded-numpy", "scipy": "recorded-scipy"},
        threads={"OPENBLAS_NUM_THREADS": "8", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": None})
    run = Path(spec["output"]) / "run.json"
    put(run, dict(schema=spec["schema"], configuration=spec, bindings=bindings, initial_state="estimating"))
    protocol = mod.load_protocol(Path(spec["source_protocol"]["path"]),
                                 expected_sha256=spec["source_protocol"]["sha256"])
    sources = mod.make_sources(protocol, mass=spec["source_mass"])
    for source in ("PG10", "EC04"):
        for replicate in range(1, 5):
            expected = mod._baseline_paths(spec, source, replicate)
            paths = {pid: dict(finalized=True) if pid.startswith("single/") else rejected_path()
                     for pid in expected}
            put(run.parent / source / f"replicate_{replicate}.json",
                dict(stage="scored", exponents=list(ALPHA_EXPONENTS), expected_paths=expected,
                    paths=paths, selection_seal=digest(paths), bindings=dict(bindings, source=source,
                        replicate=replicate, source_record_sha256=mod.source_record_hash(sources[source]))))
    config = tmp_path / "observation_sensitivity.json"
    put(config, dict(schema=mod.CONFIG_SCHEMA, version=3, study=mod.STUDY,
        output=str(tmp_path / "observation_sensitivity"),
        design_sha256=digest(mod.design_record()), baseline=dict(kind="completed_run",
            config_path=str(baseline), run_manifest_path=str(run))))
    before = {str(p): sha(p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file()}
    admitted = mod.build_admission(config)
    assert len(admitted.to_dict()["baseline"]["group_files"]) == 8
    assert admitted.to_dict()["baseline"]["bindings"] == bindings
    assert before == {str(p): sha(p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file()}


def test_exact_group_paths_bindings_and_ownership(tree):
    admitted = tree.build()
    for source in ("PG10", "EC04"):
        for r in range(1, 5):
            assert admitted.paths_for(source, r) == tuple(p for p in build_design().paths
                if p.source == source and p.replicate == r)
            assert len(admitted.paths_for(source, r)) == (24 if r <= 2 else 20)
            assert admitted.expected_baseline_bindings(source, r) == dict(tree.bindings,
                source=source, replicate=r, source_record_sha256=admitted.to_dict()["sources"][source]["sha256"])
            assert admitted.expected_baseline_paths(source, r) == tuple(mod._baseline_paths(tree.spec, source, r))
            admitted.expected_baseline_bindings(source, r)["code_files"].clear()
            assert admitted.expected_baseline_bindings(source, r)["code_files"]


@pytest.mark.parametrize("source,r", [("SB150", 1), ("PG10", True), ("EC04", 1.0),
    ("PG10", 0), ("EC04", 5), (None, 1), ("pg10", 1)])
def test_only_registered_inverse_groups(tree, source, r):
    admitted = tree.build()
    for method in (admitted.paths_for, admitted.expected_baseline_bindings, admitted.expected_baseline_paths):
        with pytest.raises(mod.AdmissionError):
            method(source, r)


@pytest.mark.parametrize("change", [lambda c: c.update(version=True), lambda c: c.update(version=1.0),
    lambda c: c.update(version=1), lambda c: c.update(frozen=True), lambda c: c.update(overrides={}),
    lambda c: c.update(study="other"), lambda c: c.update(schema="other"),
    lambda c: c.pop("output"), lambda c: c.update(design_sha256="0"*64),
    lambda c: c["baseline"].update(extra="unregistered"),
    lambda c: c["baseline"].update(kind="archive"),
    lambda c: c["baseline"].update(config_sha256="a"*64),
    lambda c: c.update(reference_root="archive"), lambda c: c.update(locations={})])
def test_closed_config_requires_completed_run(tree, change):
    change(tree.declaration)
    put(tree.config, tree.declaration)
    with pytest.raises(mod.AdmissionError):
        tree.build()
    assert "sources" not in tree.events


@pytest.mark.parametrize("target", ["config", "base", "run", "group"])
@pytest.mark.parametrize("payload", [b'{"schema":1,"schema":2}', b'{"x":NaN}', b'{"x":1e999}', b'\xff'])
def test_strict_json_before_acceptance(tree, target, payload):
    path = next(iter(tree.groups)) if target == "group" else getattr(tree, target)
    path.write_bytes(payload)
    with pytest.raises(mod.AdmissionError):
        tree.build()


@pytest.mark.parametrize("change", ["mass", "alpha", "stream", "grid", "noise", "condition",
    "condition_order", "source_order", "protocol_sha", "input_sha", "replicates", "solver"])
def test_scientific_changes_rejected_even_with_consistent_fresh_bindings(tree, change):
    spec = tree.spec
    if change == "mass": spec["source_mass"] += 1
    if change == "alpha": spec["alpha"]["tie_atol"] *= 2
    if change == "stream": spec["stream"]["seed"] += 1
    if change == "grid": spec["grids"]["G0"]["steps"] *= 2
    if change == "noise": spec["noise"]["corr"]["station_sd"][0] += 1
    if change == "condition": spec["conditions"][0]["tau_hours"] += .1
    if change == "condition_order": spec["conditions"].reverse()
    if change == "source_order": spec["sources"].reverse()
    if change == "protocol_sha": spec["source_protocol"]["sha256"] = "a"*64
    if change == "input_sha": spec["input_files"][0]["sha256"] = "b"*64
    if change == "replicates": spec["replicates"].reverse()
    if change == "solver": spec["solver"]["max_iterations"] += 1
    refresh_run(tree)
    with pytest.raises(mod.AdmissionError, match="scientific configuration"):
        tree.build()
    assert not tree.events


def test_deployment_paths_and_resource_settings_are_not_scientific_inputs(tree):
    value = deepcopy(tree.spec)
    value["output"] = "elsewhere"
    value["resources"] = {"workers": 9, "max_trajectory_bytes": 1}
    value["protected_roots"] = ["another input directory"]
    value["source_protocol"]["path"] = "another protocol path"
    for item in value["input_files"]:
        item["path"] = "another file"
    assert mod.scientific_configuration(value) == mod.scientific_configuration(tree.spec)
    value["additional_science"] = True
    assert mod.scientific_configuration(value) != mod.scientific_configuration(tree.spec)










@pytest.mark.parametrize("output", ["baseline_parent", "baseline_run", "baseline_child",
    "empty", "config"])
def test_output_isolated_before_baseline_open(tree, monkeypatch, output):
    tree.declaration["output"] = {"baseline_parent": str(tree.run.parent.parent),
        "baseline_run": str(tree.run.parent), "baseline_child": str(tree.run.parent / "child"),
        "empty": "", "config": str(tree.base)}[output]
    put(tree.config, tree.declaration)
    reads, original = [], mod._read
    monkeypatch.setattr(mod, "_read", lambda p: (reads.append(p), original(p))[1])
    with pytest.raises(mod.AdmissionError):
        tree.build()
    assert reads == [tree.config.resolve()]


def test_output_cannot_contain_static_inputs(tree):
    tree.declaration["output"] = str(tree.protocol.parent)
    put(tree.config, tree.declaration)
    with pytest.raises(mod.AdmissionError, match="static input|protected input"):
        tree.build()
    assert not tree.events


@pytest.mark.parametrize("change", ["missing", "stage", "path_missing", "single_missing", "order",
    "binding", "source_hash", "config_hash", "seal", "incomplete_candidates", "alpha", "grid"])
def test_complete_terminal_groups_required(tree, change):
    path, group = next(iter(tree.groups.items()))
    if change == "missing": path.unlink()
    if change == "stage": group["stage"] = "sealed"
    if change == "path_missing": group["paths"].pop(next(iter(group["paths"])))
    if change == "single_missing":
        single = next(p for p in group["paths"] if p.startswith("single/"))
        group["paths"].pop(single)
        group["expected_paths"].remove(single)
        group["selection_seal"] = digest(group["paths"])
    if change == "order": group["expected_paths"].reverse()
    if change == "binding": group["bindings"]["replicate"] = 3
    if change == "source_hash": group["bindings"]["source_record_sha256"] = "a"*64
    if change == "config_hash": group["bindings"]["config_sha256"] = "a"*64
    if change == "seal": group["selection_seal"] = "a"*64
    if change == "incomplete_candidates":
        next(row for pid, row in group["paths"].items() if not pid.startswith("single/"))["candidates"].pop("-8.0")
        group["selection_seal"] = digest(group["paths"])
    if change == "alpha":
        next(row for pid, row in group["paths"].items() if not pid.startswith("single/"))["candidates"]["-8.0"]["alpha"] = 1
        group["selection_seal"] = digest(group["paths"])
    if change == "grid": group["exponents"].pop()
    if change != "missing": put(path, group)
    with pytest.raises(mod.AdmissionError):
        tree.build()


@pytest.mark.parametrize("target", ["input", "config", "base", "run", "registered", "group"])
def test_inputs_changed_during_admission_fail(tree, monkeypatch, target):
    original = mod._read
    final_group = list(tree.groups)[-1]
    mutated = False
    def read(path):
        nonlocal mutated
        raw = original(path)
        if path == final_group and not mutated:
            mutated = True
            changed = {"input": tree.input_paths[0], "config": tree.config, "base": tree.base,
                "run": tree.run, "registered": tree.registered, "group": next(iter(tree.groups))}[target]
            changed.write_bytes(changed.read_bytes() + b" ")
        return raw
    monkeypatch.setattr(mod, "_read", read)
    with pytest.raises(mod.AdmissionError, match="changed while"):
        tree.build()


@pytest.mark.parametrize("target", ["protocol", "input0", "input1"])
def test_any_static_byte_change_fails_before_sources(tree, target):
    path = tree.input_paths[int(target[-1])] if target.startswith("input") else tree.protocol
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(mod.AdmissionError, match="Static input bytes changed"):
        tree.build()
    assert not tree.events


@pytest.mark.parametrize("target", ["duplicate", "checkpoint"])
def test_static_input_cannot_alias_another_input_or_a_group(tree, monkeypatch, target):
    tree.spec["input_files"][0]["path"] = str(tree.protocol if target == "duplicate" else next(iter(tree.groups)))
    refresh_run(tree)
    reads, original = [], mod._read
    monkeypatch.setattr(mod, "_read", lambda p: (reads.append(p), original(p))[1])
    with pytest.raises(mod.AdmissionError, match="Duplicate input|checkpoint"):
        tree.build()
    assert not any(p.name == "replicate_1.json" for p in reads)
    assert not tree.events


@pytest.mark.parametrize("change", ["configuration", "config_hash", "input_hash", "code_digest"])
def test_run_bindings_cannot_relabel_different_configuration_or_inputs(tree, change):
    if change == "configuration": tree.run_record["configuration"]["source_mass"] += 1
    if change == "config_hash": tree.run_record["bindings"]["config_sha256"] = "a"*64
    if change == "input_hash": tree.run_record["bindings"]["input_files"][str(tree.protocol)] = "a"*64
    if change == "code_digest": tree.run_record["bindings"]["code_sha256"] = "a"*64
    put(tree.run, tree.run_record)
    with pytest.raises(mod.AdmissionError):
        tree.build()
    if change != "code_digest":
        assert not tree.events


def test_configuration_output_must_identify_the_provided_completed_run(tree):
    tree.spec["output"] = str(tree.root / "other_run")
    refresh_run(tree)
    with pytest.raises(mod.AdmissionError, match="output differs"):
        tree.build()


def test_missing_source_factory_and_source_hash_mismatch_fail(tree, monkeypatch):
    original = mod.make_sources
    monkeypatch.setattr(mod, "make_sources", lambda *a, **k: dict(list(original(*a, **k).items())[:-1]))
    with pytest.raises(mod.AdmissionError, match="six"):
        tree.build()
    monkeypatch.setattr(mod, "make_sources", original)
    monkeypatch.setattr(mod, "source_record_hash", lambda _: "0"*64)
    with pytest.raises(mod.AdmissionError, match="serialization"):
        tree.build()


def test_all_six_sources_serialized_even_for_noninverse_profile(tree, monkeypatch):
    original, count = mod.source_record, []
    def fail_last(source):
        count.append(source)
        if len(count) == 6: raise ValueError("malformed noninverse NEW-S2")
        return original(source)
    monkeypatch.setattr(mod, "source_record", fail_last)
    with pytest.raises(mod.AdmissionError, match="NEW-S2"):
        tree.build()
    assert len(count) == 6


def test_config_byte_identity_includes_whitespace(tree):
    before = tree.build()
    tree.config.write_bytes(tree.config.read_bytes() + b"\n")
    after = tree.build()
    assert before.sha256 != after.sha256
    assert before.to_dict()["config_canonical_sha256"] == after.to_dict()["config_canonical_sha256"]


def test_changed_declarative_path_changes_whole_design_hash(tree, monkeypatch):
    old = build_design()
    changed = deepcopy(old)
    altered = replace(old.paths[0], inverse_h=replace(old.paths[0].inverse_h, width_km=2.))
    object.__setattr__(changed, "paths", (altered, *old.paths[1:]))
    monkeypatch.setattr(mod, "build_design", lambda: changed)
    with pytest.raises(mod.AdmissionError, match="design SHA"):
        tree.build()












def test_baseline_symlink_cannot_redirect_static_manifest_to_checkpoint(tree):
    target = tree.run.parent / "fake-checkpoint.json"
    target.write_bytes(tree.run.read_bytes())
    tree.run.unlink()
    try:
        tree.run.symlink_to(target)
    except OSError:
        pytest.skip("OS does not permit creating this test symlink")
    with pytest.raises(mod.AdmissionError, match="aliases"):
        tree.build()









def test_recorded_baseline_survives_current_code_and_environment_changes(tree, monkeypatch):
    before = tree.build()
    (tree.code / "module00.py").write_text("CURRENT_CODE = 'changed'", encoding="utf-8")
    for name, value in (("OMP_NUM_THREADS", "8"), ("OPENBLAS_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "2")):
        monkeypatch.setenv(name, value)
    after = tree.build()
    assert before.sha256 == after.sha256
    assert after.to_dict()["baseline"]["bindings"] == tree.bindings
    assert after.spec == mod.scientific_configuration(tree.spec)
    assert "resources" not in after.spec


def test_recorded_resource_configuration_does_not_change_scientific_baseline(tree):
    tree.spec["resources"] = {"workers": 4, "max_trajectory_bytes": 1}
    tree.bindings["threads"].update(OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS=None)
    refresh_run(tree)
    admitted = tree.build()
    assert admitted.spec == mod.scientific_configuration(tree.spec)
    assert admitted.to_dict()["baseline"]["bindings"] == tree.bindings



def selected_tree(tree):
    from tests.experiments.observation_sensitivity.test_design import selected_configuration
    spec = tree.spec
    spec["sources"], spec["replicates"] = ["PG10"], [1]
    spec["grids"]["G0"].update(spacing_km=1., steps=168)
    spec["conditions"] = [c for c in spec["conditions"] if c["id"] in ("main", "weight_W01")]
    for condition in spec["conditions"]:
        condition.update(sources=["PG10"], replicates=[1])
    spec["single_fits"] = []
    spec["analysis_contrasts"] = [dict(id="weight_W01", baseline="main", variant="weight_W01", factor="weight_W01")]
    spec["planned_counts"] = dict(paths=4, single_fits=0, nominal_candidates=100, maximum_candidates=100)
    refresh_run(tree)
    target = tree.run.parent / "PG10/replicate_1.json"
    group = tree.groups[target]
    group["expected_paths"] = mod._baseline_paths(spec, "PG10", 1)
    group["paths"] = {p: rejected_path() for p in group["expected_paths"]}
    group["selection_seal"] = digest(group["paths"])
    put(target, group)
    tree.declaration.update(selected_configuration())
    put(tree.config, tree.declaration)
    return target


def test_selected_admission_pins_entire_four_path_e05_group_and_no_other_groups(tree):
    target = selected_tree(tree)
    for path in tree.groups:
        if path != target: path.unlink()
    before = {p: p.read_bytes() for p in tree.root.rglob("*") if p.is_file()}
    admitted = tree.build()
    assert before == {p: p.read_bytes() for p in before}
    record = admitted.to_dict()
    assert record["version"] == 4 and record["counts"]["new_fit_attempts"] == record["counts"]["reused_fit_attempts"] == 50
    assert record["baseline"]["group_files"] == {str(target): sha(target.read_bytes())}
    assert admitted.expected_baseline_paths("PG10", 1) == ("main/L2", "main/H1", "weight_W01/L2", "weight_W01/H1")
    assert all(p.reuse is None or p.reuse.condition == "main" for p in admitted.paths_for("PG10", 1))
    with pytest.raises(mod.AdmissionError, match="absent"): admitted.paths_for("EC04", 1)


def test_unused_condition_in_reused_group_remains_subject_to_seal_and_raw_identity(tree):
    target = selected_tree(tree)
    admitted = tree.build()
    group = strict_json(target.read_bytes())
    group["paths"]["weight_W01/L2"]["candidates"]["-8.0"]["alpha"] = 7.
    put(target, group)
    with pytest.raises(mod.AdmissionError): tree.build()
    assert admitted.to_dict()["baseline"]["group_files"][str(target)] != sha(target.read_bytes())


def test_wrong_main_weight_is_rejected_before_source_generation(tree):
    selected_tree(tree)
    tree.spec["conditions"][0]["weight"] = "W02"
    refresh_run(tree)
    tree.events.clear()
    with pytest.raises(mod.AdmissionError, match="reuse path"): tree.build()
    assert tree.events == []


@pytest.mark.parametrize("field", ["direct", "figures", "expected_paths", "counts"])
def test_selected_admission_cannot_reseal_changed_plan_fields(tree, field):
    from experiments.observation_sensitivity.input_binding import validate_input_binding
    selected_tree(tree)
    record = tree.build().to_dict()
    if field == "direct": record[field]["grids"]["G0"]["steps"] *= 2
    elif field == "figures": record[field][0]["contrast_keys"] = []
    elif field == "expected_paths": record[field]["PG10/r1"].reverse()
    else: record[field]["paths"] += 1
    with pytest.raises(ValueError): validate_input_binding(record)


@pytest.mark.parametrize("value", [None, [], 1, "invalid"])
def test_input_binding_rejects_non_object_admission(value):
    from experiments.observation_sensitivity.input_binding import validate_input_binding
    with pytest.raises(ValueError, match="Admission object"):
        validate_input_binding(value)


@pytest.mark.parametrize("consumer", ["pairs", "direct", "figures"])
def test_analysis_cli_protects_all_admitted_scientific_inputs(tree, monkeypatch, consumer):
    from experiments.observation_sensitivity.input_binding import scientific_input_roots
    from experiments.observation_sensitivity.analysis import read_results, prediction_error, plots
    selected_tree(tree)
    extra = tree.root / "protected-observations"
    extra.mkdir()
    tree.spec["protected_roots"] = [str(extra)]
    refresh_run(tree)
    admitted = tree.build().to_dict()
    roots = scientific_input_roots(admitted)
    assert tree.run.parent in roots and tree.config in roots
    assert set(map(Path, admitted["static_input_paths"].values())) <= set(roots)
    assert extra in roots
    monkeypatch.setattr(read_results, "load_records", lambda *a, **kw: dict(freeze=dict(admission=admitted)))
    module = dict(pairs=read_results, direct=prediction_error, figures=plots)[consumer]
    output_flag = "--output-dir" if consumer == "figures" else "--output"
    before = {p: p.read_bytes() for p in tree.root.rglob("*") if p.is_file()}
    for root in roots:
        target = root / "forbidden-export" if root.is_dir() else root
        existed = target.exists()
        original = target.read_bytes() if existed else None
        with pytest.raises(read_results.SnapshotError, match="new output"):
            module.main(["--run", admitted["output"], "--freeze-sha256", "a"*64,
                output_flag, str(target)])
        assert target.exists() == existed
        if existed:
            assert target.read_bytes() == original
    assert before == {p: p.read_bytes() for p in before}
