"""Проверки повторного использования оценок на искусственных группах E05."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import pytest

from adrkit.config.validation import canonical_bytes, digest
from experiments.observation_sensitivity.design import ALPHA_EXPONENTS, ObservationSpec, StreamSpec, build_design
from experiments.observation_sensitivity.lifecycle import Checkpoint, CheckpointError, read_terminal_v2
from experiments.observation_sensitivity.reuse import VerifiedBaseline


BINDINGS = dict(source="PG10", replicate=1, config_sha256="a"*64,
                code_sha256="b"*64, source_record_sha256="c"*64,
                versions={"python": "test"}, input_files={"protocol.json": "d"*64})
# Ожидаемый порядок задан в тесте отдельно от читаемой группы.
EXPECTED_PATHS = ("main/L2", "main/H1", "temporal_average/L2", "single/audit")
PROVENANCE = dict(fit_y_sha256="1"*64, selection_y_sha256="2"*64,
                 selected_covariance_sha256="3"*64, jacobian_sha256="4"*64,
                 offset_sha256="5"*64, panel_records={"fit": {"hash": "f"*64},
                 "selection": {"hash": "e"*64}}, calibration={"status": "accepted", "panels": 32},
                 design={"rows": [0, 1, 2], "mask": "none"}, truth_forward_residual=1e-10)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec(condition="spatial_G1_matched", penalty="L2", source="PG10", replicate=1):
    return next(p for p in build_design().paths if (p.condition,p.penalty,p.source,p.replicate) ==
                (condition,penalty,source,replicate))


def original_path(condition="main", penalty="L2", *, failure=False, rejected=False):
    row = dict(condition=condition, penalty=penalty, candidates={}, finalized=True,
               selected_exponent=None, procedure_accepted=False)
    if failure:
        row["calibration_failure"] = "independently reproducible covariance rejection"
        return row
    row.update(alpha_reference=10., provenance=deepcopy(PROVENANCE),
        candidates={str(e): dict(exponent=e, alpha=10.*10.**e, accepted=not rejected,
                    coefficients=[1., 2.], selection_mse=1., status="rejected" if rejected else "accepted")
                    for e in ALPHA_EXPONENTS},
        complete_path=not rejected, candidate_count=25, accepted_count=0 if rejected else 25)
    if not rejected:
        row.update(selected_exponent=-3., procedure_accepted=True,
                   final_certificate={"accepted": True, "residual": 1e-10})
    return row


def write_v2(target, *, failure=False, rejected=False, transform=None):
    paths = {"main/L2": original_path(failure=failure,rejected=rejected),
             "main/H1": original_path(penalty="H1"),
             "temporal_average/L2": original_path("temporal_average"),
             "single/audit": {"finalized": True, "candidates": {},
                 "status": "baseline_selection_unavailable", "procedure_accepted": False}}
    record = dict(bindings=deepcopy(BINDINGS), paths=paths, expected_paths=list(paths),
        exponents=list(ALPHA_EXPONENTS), stage="scored",
        scores={"main/L2": {"E_q": 999., "test": "HIDDEN_TEST_PAYLOAD"}},
        source_scores={"sentinel": "HIDDEN_SOURCE_PAYLOAD"})
    if transform:
        transform(record)
    record["selection_seal"] = digest(record["paths"])
    # Байты файла намеренно записаны не в канонической форме: указанная контрольная сумма должна
    # охватывать именно их, а не последующую сериализацию JSON.
    raw = json.dumps(record, ensure_ascii=False, indent=3).encode("utf-8")+b"\n"
    target.write_bytes(raw)
    return record, raw


def test_exact_estimate_copy_raw_hash_and_no_scores_retained(tmp_path):
    target = tmp_path/"v2.json"
    original, raw = write_v2(target)
    baseline = VerifiedBaseline.from_terminal(target, expected_file_sha256=file_sha(target), expected_bindings=BINDINGS, expected_paths=EXPECTED_PATHS)
    result = baseline.estimate(spec(), provenance=deepcopy(PROVENANCE), alpha_reference=10.)
    reuse = result.pop("reuse")
    assert result == original["paths"]["main/L2"]
    assert reuse["original_path_sha256"] == digest(result)
    assert reuse["source_file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert reuse["source_file_sha256"] != digest(original)
    assert reuse["selection_seal"] == original["selection_seal"]
    assert reuse["bindings"] == BINDINGS
    assert reuse["new_path_id"] == spec().id
    assert b"HIDDEN_TEST_PAYLOAD" not in baseline._payload
    assert b"HIDDEN_SOURCE_PAYLOAD" not in baseline._payload
    assert b"single/audit" not in baseline._payload
    result["candidates"]["-8.0"]["coefficients"][0] = 99.
    assert baseline.estimate(spec(), provenance=PROVENANCE, alpha_reference=10.)["candidates"]["-8.0"]["coefficients"][0] == 1.
    with pytest.raises(FrozenInstanceError):
        baseline._payload = b"{}"
    with pytest.raises(TypeError):
        VerifiedBaseline()


def test_pinned_baseline_reads_once_and_reuse_never_reopens_file(tmp_path, monkeypatch):
    target = tmp_path / "v2.json"
    _, raw = write_v2(target)
    pin = hashlib.sha256(raw).hexdigest()
    events, original = [], Path.read_bytes
    def read(path):
        events.append(path)
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    baseline = VerifiedBaseline.from_terminal(target, expected_bindings=BINDINGS,
        expected_paths=EXPECTED_PATHS, expected_file_sha256=pin)
    assert events == [target]
    monkeypatch.setattr(Path, "read_bytes", lambda *args: pytest.fail("Reuse must not reopen a pinned baseline"))
    baseline.estimate(spec(), provenance=PROVENANCE, alpha_reference=10.)


def test_expected_paths_keyword_is_mandatory_before_any_io(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "read_bytes", lambda *args: pytest.fail("Missing expectation must fail before IO"))
    with pytest.raises(TypeError, match="expected_paths"):
        VerifiedBaseline.from_terminal(tmp_path / "missing.json", expected_bindings=BINDINGS,
                                       expected_file_sha256="a" * 64)


@pytest.mark.parametrize("expected", [
    None, True, False, "main/L2", b"main/L2", [], (), {},
    {"main/L2": True}, {"main/L2"}, frozenset({"main/L2"}),
    [True], [1], [""], [None], [["main/L2"]], ["main/L2", "main/L2"],
])
def test_invalid_expected_paths_fail_before_read(tmp_path, monkeypatch, expected):
    def forbidden(*args):
        pytest.fail("malformed admission expectation must fail before a file read")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(CheckpointError, match="expected_paths"):
        VerifiedBaseline.from_terminal(tmp_path/"missing.json", expected_bindings=BINDINGS, expected_file_sha256="a"*64,
                                       expected_paths=expected)


@pytest.mark.parametrize("expected", [EXPECTED_PATHS[:-1], EXPECTED_PATHS[::-1]])
def test_valid_but_different_admitted_group_is_rejected(tmp_path, expected):
    target = tmp_path/"v2.json"
    write_v2(target)
    with pytest.raises(CheckpointError, match="complete admitted ordered list"):
        VerifiedBaseline.from_terminal(target, expected_file_sha256=file_sha(target), expected_bindings=BINDINGS,
                                       expected_paths=expected)


@pytest.mark.parametrize("removed", ["main/H1", "single/audit"])
def test_internally_resealed_truncation_cannot_replace_admitted_full_group(tmp_path, removed):
    target = tmp_path/"v2.json"
    def truncate(record):
        del record["paths"][removed]
        record["expected_paths"].remove(removed)
    record, _ = write_v2(target, transform=truncate)
    # Условия внутренней согласованности записи v2 допускают эту завершённую запись.
    assert read_terminal_v2(target, expected_file_sha256=file_sha(target)) == record
    with pytest.raises(CheckpointError, match="complete admitted ordered list"):
        VerifiedBaseline.from_terminal(target, expected_file_sha256=file_sha(target), expected_bindings=BINDINGS,
                                       expected_paths=EXPECTED_PATHS)


def test_checkpoint_expected_order_cannot_override_admitted_order(tmp_path):
    target = tmp_path/"v2.json"
    record, _ = write_v2(target, transform=lambda r: r["expected_paths"].reverse())
    assert read_terminal_v2(target, expected_file_sha256=file_sha(target)) == record
    with pytest.raises(CheckpointError, match="complete admitted ordered list"):
        VerifiedBaseline.from_terminal(target, expected_file_sha256=file_sha(target), expected_bindings=BINDINGS,
                                       expected_paths=EXPECTED_PATHS)


@pytest.mark.parametrize("corrupt", [
    lambda r: r["expected_paths"].append("main/L2"),
    lambda r: r.update(expected_paths=True),
    lambda r: r.update(expected_paths="main/L2"),
    lambda r: r["expected_paths"].__setitem__(0, True),
])
def test_corrupt_checkpoint_expected_paths_fail_closed(tmp_path, corrupt):
    target = tmp_path/"v2.json"
    write_v2(target, transform=corrupt)
    with pytest.raises(CheckpointError):
        VerifiedBaseline.from_terminal(target, expected_file_sha256=file_sha(target), expected_bindings=BINDINGS,
                                       expected_paths=EXPECTED_PATHS)


def test_list_expectation_is_owned_before_single_read(tmp_path, monkeypatch):
    target = tmp_path / "v2.json"
    _, raw = write_v2(target)
    expected = list(EXPECTED_PATHS)
    original_read, events = Path.read_bytes, []
    def read(path):
        events.append(path)
        expected.clear()
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    baseline = VerifiedBaseline.from_terminal(target, expected_bindings=BINDINGS,
        expected_paths=expected, expected_file_sha256=hashlib.sha256(raw).hexdigest())
    assert events == [target]
    assert baseline.estimate(spec(), provenance=PROVENANCE, alpha_reference=10.)["penalty"] == "L2"


@pytest.mark.parametrize("field", ["config_sha256", "code_sha256", "source_record_sha256",
                                   "versions", "input_files", "source", "replicate"])
def test_every_full_binding_is_exact(tmp_path, field):
    target = tmp_path/"v2.json"
    write_v2(target)
    expected = deepcopy(BINDINGS)
    expected[field] = 2 if field == "replicate" else "changed"
    with pytest.raises(ValueError, match="bindings"):
        VerifiedBaseline.from_terminal(target, expected_file_sha256=file_sha(target), expected_bindings=expected, expected_paths=EXPECTED_PATHS)


@pytest.mark.parametrize("field", ["fit_y_sha256", "selection_y_sha256", "selected_covariance_sha256",
    "jacobian_sha256", "offset_sha256", "panel_records", "calibration", "design"])
def test_permitted_provenance_mismatches_never_fallback(tmp_path, field):
    target = tmp_path/"v2.json"
    write_v2(target)
    baseline = VerifiedBaseline.from_terminal(target, expected_file_sha256=file_sha(target), expected_bindings=BINDINGS, expected_paths=EXPECTED_PATHS)
    changed = deepcopy(PROVENANCE)
    if field.endswith("sha256"):
        changed[field] = "9"*64
    elif field == "panel_records":
        changed[field]["fit"]["hash"] = "9"*64
    else:
        changed[field]["changed"] = True
    with pytest.raises(ValueError, match=field):
        baseline.estimate(spec(), provenance=changed, alpha_reference=10.)


@pytest.mark.parametrize("alpha", [None, True, 0., float("inf"), 10.000000000000002, 10])
def test_alpha_reference_is_exact_not_close(tmp_path, alpha):
    target = tmp_path/"v2.json"
    write_v2(target)
    baseline = VerifiedBaseline.from_terminal(target, expected_file_sha256=file_sha(target), expected_bindings=BINDINGS, expected_paths=EXPECTED_PATHS)
    with pytest.raises(ValueError, match="alpha_reference"):
        baseline.estimate(spec(), provenance=PROVENANCE, alpha_reference=alpha)


@pytest.mark.parametrize("change", ["missing_hash", "non_hash", "test_panel", "score", "missing_design"])
def test_incomplete_or_forbidden_permitted_inputs_fail_closed(tmp_path, change):
    target = tmp_path/"v2.json"
    write_v2(target)
    baseline = VerifiedBaseline.from_terminal(target, expected_file_sha256=file_sha(target), expected_bindings=BINDINGS, expected_paths=EXPECTED_PATHS)
    provided = deepcopy(PROVENANCE)
    if change == "missing_hash":
        provided.pop("jacobian_sha256")
    elif change == "non_hash":
        provided["fit_y_sha256"] = "not a hash"
    elif change == "test_panel":
        provided["panel_records"]["test"] = {"hash": "c"*64}
    elif change == "score":
        provided["E_q"] = .1
    else:
        provided.pop("design")
    with pytest.raises(ValueError):
        baseline.estimate(spec(), provenance=provided, alpha_reference=10.)


def test_rejected_paths_remain_rejected_and_can_enter_e06_checkpoint(tmp_path):
    target = tmp_path/"v2.json"
    original, _ = write_v2(target,rejected=True)
    baseline = VerifiedBaseline.from_terminal(target, expected_file_sha256=file_sha(target), expected_bindings=BINDINGS, expected_paths=EXPECTED_PATHS)
    result = baseline.estimate(spec(), provenance=PROVENANCE, alpha_reference=10.)
    assert result["procedure_accepted"] is False
    assert result["selected_exponent"] is None
    assert result["candidates"] == original["paths"]["main/L2"]["candidates"]
    with Checkpoint(tmp_path/"e06.json", BINDINGS, [spec().id]) as cp:
        cp.record["paths"][spec().id] = result
        cp.seal()
        cp.require_sealed()


def test_same_terminal_calibration_failure_preserved_with_explicit_limit(tmp_path):
    target = tmp_path/"v2.json"
    original, _ = write_v2(target,failure=True)
    row = original["paths"]["main/L2"]
    baseline = VerifiedBaseline.from_terminal(target, expected_file_sha256=file_sha(target), expected_bindings=BINDINGS, expected_paths=EXPECTED_PATHS)
    result = baseline.estimate(spec(), provenance={"calibration_failure": row["calibration_failure"]}, alpha_reference=None)
    assert result["candidates"] == {} and result["procedure_accepted"] is False
    assert "not byte-compared" in result["reuse"]["comparison"]
    assert "fit_y_sha256" not in result["reuse"]["matched_fields"]
    with Checkpoint(tmp_path/"e06.json", BINDINGS, [spec().id]) as cp:
        cp.record["paths"][spec().id] = result
        cp.seal()


@pytest.mark.parametrize("kind", ["successful", "different_failure", "invented_alpha", "extra_inputs"])
def test_terminal_failure_must_independently_recur_exactly(tmp_path, kind):
    target = tmp_path/"v2.json"
    original, _ = write_v2(target,failure=True)
    baseline = VerifiedBaseline.from_terminal(target, expected_file_sha256=file_sha(target), expected_bindings=BINDINGS, expected_paths=EXPECTED_PATHS)
    provenance = {"calibration_failure": original["paths"]["main/L2"]["calibration_failure"]}
    alpha = None
    if kind == "successful":
        provenance, alpha = deepcopy(PROVENANCE), 10.
    elif kind == "different_failure":
        provenance["calibration_failure"] = "other failure"
    elif kind == "invented_alpha":
        alpha = 10.
    else:
        provenance["fit_y_sha256"] = "a"*64
    with pytest.raises(ValueError, match="independently reproduced"):
        baseline.estimate(spec(), provenance=provenance, alpha_reference=alpha)


@pytest.mark.parametrize("new_spec", [spec(source="EC04"), spec(replicate=2),
                                    spec(condition="spatial_G2_matched")])
def test_reference_requires_registered_exact_source_replicate_and_reuse(new_spec, tmp_path):
    target = tmp_path/"v2.json"
    write_v2(target)
    baseline = VerifiedBaseline.from_terminal(target, expected_file_sha256=file_sha(target), expected_bindings=BINDINGS, expected_paths=EXPECTED_PATHS)
    with pytest.raises(ValueError):
        baseline.estimate(new_spec, provenance=PROVENANCE, alpha_reference=10.)


def test_exact_condition_and_penalty_lookup(tmp_path):
    target = tmp_path/"v2.json"
    write_v2(target)
    baseline = VerifiedBaseline.from_terminal(target, expected_file_sha256=file_sha(target), expected_bindings=BINDINGS, expected_paths=EXPECTED_PATHS)
    result = baseline.estimate(spec(penalty="H1"), provenance=PROVENANCE, alpha_reference=10.)
    assert result["penalty"] == "H1" and result["reuse"]["original_path_id"] == "main/H1"
    result = baseline.estimate(spec("temporal_average_matched"), provenance=PROVENANCE, alpha_reference=10.)
    assert result["condition"] == "temporal_average"
    with pytest.raises(ValueError, match="absent"):
        baseline.estimate(spec("temporal_average_matched", penalty="H1"), provenance=PROVENANCE, alpha_reference=10.)


@pytest.mark.parametrize("field,value", [("condition", "wrong"), ("penalty", "H1"), ("E_q", .01),
                                        ("scores", {"test": "hidden"})])
def test_body_locator_or_injected_score_mismatch_is_rejected(tmp_path, field, value):
    target = tmp_path/"v2.json"
    write_v2(target, transform=lambda r: r["paths"]["main/L2"].update({field: value}))
    baseline = VerifiedBaseline.from_terminal(target, expected_file_sha256=file_sha(target), expected_bindings=BINDINGS, expected_paths=EXPECTED_PATHS)
    with pytest.raises(ValueError):
        baseline.estimate(spec(), provenance=PROVENANCE, alpha_reference=10.)


@pytest.mark.parametrize("altered", [
    replace(spec(), inverse_h=ObservationSpec("gaussian", 2.)),
    replace(spec(), true_h=ObservationSpec("gaussian", .5)),
    replace(spec(), stream=StreamSpec(20260926, 3)),
    replace(spec(), noise=replace(spec().noise, station_sd=(2., 1.5, 2., 1.))),
    replace(spec(), weight="W_exp_population"),
    replace(spec(), condition="unregistered_reuse"),
    replace(spec(), reuse=replace(spec().reuse, condition="temporal_average")),
])
def test_matching_hashes_cannot_authorize_changed_declarative_path(tmp_path, altered):
    target = tmp_path/"v2.json"
    write_v2(target)
    baseline = VerifiedBaseline.from_terminal(target, expected_file_sha256=file_sha(target), expected_bindings=BINDINGS, expected_paths=EXPECTED_PATHS)
    with pytest.raises(ValueError, match="registered finite reuse design"):
        baseline.estimate(altered, provenance=PROVENANCE, alpha_reference=10.)


@pytest.mark.parametrize("measurements", [False, True])
def test_reuse_preserves_old_candidates_with_or_without_optional_measurements(tmp_path, measurements):
    target = tmp_path/"v2-optional-measurements.json"
    def add_measurements(record):
        if measurements:
            for candidate in record["paths"]["main/L2"]["candidates"].values():
                candidate.update(forward_calls=7, adjoint_calls=3, seconds=.125)
    original, raw = write_v2(target, transform=add_measurements)
    expected_sha = hashlib.sha256(raw).hexdigest()
    baseline = VerifiedBaseline.from_terminal(target, expected_file_sha256=expected_sha,
        expected_bindings=BINDINGS, expected_paths=EXPECTED_PATHS)
    result = baseline.estimate(spec(), provenance=deepcopy(PROVENANCE), alpha_reference=10.)
    assert result["candidates"] == original["paths"]["main/L2"]["candidates"]
    assert result["reuse"]["original_path_sha256"] == digest(original["paths"]["main/L2"])
    assert target.read_bytes() == raw
    with Checkpoint(tmp_path/"e06.json", BINDINGS, [spec().id]) as cp:
        cp.record["paths"][spec().id] = result
        cp.seal()
        cp.require_sealed()
        assert cp.record["paths"][spec().id]["candidates"] == original["paths"]["main/L2"]["candidates"]
    assert target.read_bytes() == raw
