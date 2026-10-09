"""Проверки диагностики ковариации на синтетических записях E06.

Матрицы и их хеши вычислены; хеши выборок невязок фиктивны.
"""

from tests.experiments.observation_sensitivity.fixtures.covariance_records import ROWS
from tests.experiments.observation_sensitivity.fixtures.covariance_records import pin, make_records
from copy import deepcopy
import json

import numpy as np
import pytest

from adrkit.config.validation import canonical_bytes, digest
from experiments.source_comparison.calibration import array_hash
from experiments.observation_sensitivity.analysis.covariance import CovarianceAnalysisError, summarize_covariances


@pytest.fixture(scope="module")
def original():
    return make_records()


@pytest.fixture
def records(original):
    # Эталонные численные матрицы неизменны; JSON копируется.
    freeze, summary, matrices = original
    return deepcopy(freeze), deepcopy(summary), matrices


def run(records):
    freeze, summary, _ = records
    return summarize_covariances(freeze, summary, expected_freeze_sha256=pin(freeze))


def test_covariance_analysis_rejects_resealed_scientific_input_change(records):
    original = run(records)
    assert original["counts"]["available_sets"] == 16
    freeze, summary, _ = records
    freeze["admission"]["science_spec_sha256"] = "b" * 64
    freeze["admission_sha256"] = digest(freeze["admission"])
    freeze["content_sha256"] = digest({k: v for k, v in freeze.items() if k != "content_sha256"})
    summary["inputs"].update(freeze_file_sha256=pin(freeze), freeze_content_sha256=freeze["content_sha256"],
                             admission_sha256=freeze["admission_sha256"])
    with pytest.raises(CovarianceAnalysisError, match="specification digest"):
        run(records)


def outcome(records, weight="W03", replicate=1, *, new=True):
    return next(r for r in records[1]["raw_outcomes"] if r["replicate"] == replicate
                and r["estimate"]["provenance"]["calibration"]["family"] == weight
                and (not new or r["origin"] == "new"))


def test_deduplicated_sets_and_calibration_counts(original):
    report = run(original)
    assert report["counts"] == dict(expected_sets=16, available_sets=16, unavailable_sets=0, represented_paths=176)
    assert [r["member_count"] for r in report["shared_sets"] if r["weight"] == "W03"] == [36, 36, 28, 28]
    assert all(r["member_count"] == 4 for r in report["shared_sets"] if r["weight"] != "W03")
    assert sum(r["origin_counts"]["reused"] for r in report["shared_sets"]) == 24
    assert report["layout"]["primary_rows"] == ROWS.tolist()
    json.dumps(report, allow_nan=False)


def test_independent_kl_orientation_and_whitening_reference(original):
    report = run(original)
    for row in report["shared_sets"]:
        dense, assumed = original[2][row["member_path_ids"][0]]
        truth = dense[np.ix_(ROWS, ROWS)]
        chol = np.linalg.cholesky(assumed)
        left = np.linalg.solve(chol, truth)
        whitened = np.linalg.solve(chol, left.T).T
        kl = .5 * (np.trace(np.linalg.solve(assumed, truth)) - 36
                   + np.linalg.slogdet(assumed)[1] - np.linalg.slogdet(truth)[1])
        actual = row["diagnostics"]
        assert actual["kl_true_to_assumed"] == pytest.approx(kl, abs=3e-13, rel=2e-12)
        assert actual["whitening_spectral"] == pytest.approx(np.linalg.norm(whitened - np.eye(36), 2), abs=5e-14)
        assert actual["whitening_frobenius"] == pytest.approx(np.linalg.norm(whitened - np.eye(36)), abs=5e-14)
        if row["weight"] == "W_mix_oracle":
            assert actual["kl_true_to_assumed"] < 1e-24
            assert actual["whitening_frobenius"] < 1e-13
        else:
            assert actual["kl_true_to_assumed"] > .01
    row = next(r for r in report["shared_sets"] if r["weight"] == "W_exp_estimated")
    dense, assumed = original[2][row["member_path_ids"][0]]
    truth = dense[np.ix_(ROWS, ROWS)]
    reverse = .5 * (np.trace(np.linalg.solve(truth, assumed)) - 36
                   + np.linalg.slogdet(truth)[1] - np.linalg.slogdet(assumed)[1])
    assert abs(reverse - row["diagnostics"]["kl_true_to_assumed"]) > .01


def test_marginalize_before_precision_and_report_parameters(original):
    report = run(original)
    row = next(r for r in report["shared_sets"] if r["weight"] == "W03")
    dense, working = original[2][row["member_path_ids"][0]]
    assert not np.allclose(np.linalg.inv(dense)[np.ix_(ROWS, ROWS)], np.linalg.inv(dense[np.ix_(ROWS, ROWS)]))
    params, flags = row["parameters"], row["fitted_parameter_flags"]
    assert params["station_variances"] == [1.17, 2.33, 4.12, .93]
    assert flags["length_grid_indices"] == [16, 18, 20, 22]
    assert flags["length_grid_endpoint"] == [False] * 4
    assert row["working_primary_covariance_sha256"] == array_hash(working)


@pytest.mark.parametrize("location", ["selected", "dense", "parent", "generation", "parent_time", "parent_settings"])
def test_hash_mismatch_rejected(records, location):
    # Все копии меняем одинаково: ошибка должна обнаружиться по хешу матрицы.
    
    for row in records[1]["raw_outcomes"]:
        if row["replicate"] != 1:
            continue
        for pkey in ("provenance", "e06_provenance"):
            prov = row["estimate"].get(pkey)
            if not prov or prov["calibration"]["family"] != "W03":
                continue
            parent = prov["calibration"]["parent_coarse_calibration"]
            target, key = dict(
                selected=(prov, "selected_covariance_sha256"), dense=(prov["calibration"], "dense_covariance_sha256"),
                parent=(parent, "covariance_sha256"), generation=(prov["panel_records"]["fit"], "covariance_sha256"),
                parent_time=(parent, "times_sha256"), parent_settings=(parent, "settings_sha256"))[location]
            target[key] = "f" * 64
    with pytest.raises(CovarianceAnalysisError, match="hash mismatch|metadata mismatch|generating covariance"):
        run(records)


@pytest.mark.parametrize("location", ["estimator", "test"])
def test_primary_layout_cannot_be_changed(records, location):
    row = outcome(records)
    prov = row["estimate"]["provenance"] if location == "estimator" else row["score"]["test_provenance"]
    prov["design"]["rows"][0] += 1
    with pytest.raises(CovarianceAnalysisError, match="rows|restriction"):
        run(records)


@pytest.mark.parametrize("value", [True, 0., -1., float("nan"), "1.17"])
def test_bad_recorded_variance_is_never_repaired(records, value):
    # Изменены все записи, чтобы пройти предшествующую проверку их взаимной согласованности.
    for row in records[1]["raw_outcomes"]:
        for key in ("provenance", "e06_provenance"):
            p = row["estimate"].get(key)
            if p and p["calibration"]["family"] == "W03" and row["replicate"] == 1:
                p["calibration"]["parent_coarse_calibration"]["station_variances"][0] = value
    with pytest.raises(CovarianceAnalysisError):
        run(records)


@pytest.mark.parametrize("length", [.2, 1 / 60, 1.5])
def test_off_grid_and_endpoint_fitted_length_fail(records, length):
    for row in records[1]["raw_outcomes"]:
        for key in ("provenance", "e06_provenance"):
            p = row["estimate"].get(key)
            if p and p["calibration"]["family"] == "W03" and row["replicate"] == 1:
                p["calibration"]["parent_coarse_calibration"]["ell_hours"][0] = length
    with pytest.raises(CovarianceAnalysisError, match="off-grid|endpoint"):
        run(records)


def test_one_shared_calibration_may_not_disagree(records):
    p = outcome(records)["estimate"]["provenance"]["calibration"]
    p["parent_coarse_calibration"]["station_variances"][0] += .1
    with pytest.raises(CovarianceAnalysisError, match="Shared calibration"):
        run(records)


def set_failure(row, reason="recorded terminal rejection"):
    estimate = row["estimate"]
    estimate.update(calibration_failure=reason, procedure_accepted=False, candidates={})
    estimate.pop("provenance", None)
    if row["origin"] == "reused":
        estimate["e06_provenance"] = dict(calibration_failure=reason)
    row["score"] = dict(status="unavailable", reason="calibration_failure")


def test_shared_failure_is_retained_without_matrices(records):
    ids = []
    for row in records[1]["raw_outcomes"]:
        if row["replicate"] == 1 and row["estimate"]["provenance"]["calibration"]["family"] == "W03":
            ids.append(row["path_id"])
            set_failure(row)
    report = run(records)
    assert report["counts"]["available_sets"] == 15
    failed, = [r for r in report["shared_sets"] if r["status"] == "unavailable"]
    assert failed["member_path_ids"] == ids
    assert failed["member_count"] == 36 and failed["diagnostics"] is None
    assert "working_primary_covariance_sha256" not in failed
    assert failed["parameters"] is None


def test_success_failure_disagreement_is_not_averaged(records):
    set_failure(outcome(records))
    with pytest.raises(CovarianceAnalysisError, match="both success and failure"):
        run(records)


def test_failure_reason_cannot_differ_inside_shared_set(records):
    targets = [r for r in records[1]["raw_outcomes"] if r["replicate"] == 1
               and r["estimate"]["provenance"]["calibration"]["family"] == "W03"]
    for row in targets:
        set_failure(row)
    set_failure(targets[-1], "a different failure")
    with pytest.raises(CovarianceAnalysisError, match="inconsistent failure reasons"):
        run(records)


def test_variance_bits_survive_without_sqrt_resquare(original):
    variance = 1.17
    assert np.sqrt(variance) ** 2 != variance
    report = run(original)
    row = next(r for r in report["shared_sets"] if r["weight"] == "W03")
    matrix = original[2][row["member_path_ids"][0]][1]
    assert matrix[0, 0] == variance
    assert row["working_primary_covariance_sha256"] == array_hash(matrix)


def test_fit_rejection_and_unavailable_scores_do_not_drop_calibration(records):
    baseline = run(records)
    for row in records[1]["raw_outcomes"]:
        row["estimate"]["procedure_accepted"] = False
        row["score"] = dict(status="unavailable", reason="no_accepted_candidate")
    actual = run(records)
    assert actual["shared_sets"] == baseline["shared_sets"]
    assert actual["counts"]["available_sets"] == 16


@pytest.mark.parametrize("purpose", ["fit", "selection", "test", "calibration"])
def test_wrong_stream_or_panel_index_rejected(records, purpose):
    row = outcome(records)
    p = row["estimate"]["provenance"]
    target = (row["score"]["test_provenance"]["panel_record"] if purpose == "test"
              else p["calibration"]["panel_records"][0] if purpose == "calibration" else p["panel_records"][purpose])
    target["seed"][1] = 3
    with pytest.raises(CovarianceAnalysisError):
        run(records)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "noise", "binding", "reuse", "pin"])
def test_structural_and_freeze_corruption(records, corruption):
    summary = records[1]
    row = summary["raw_outcomes"][0]
    if corruption == "missing":
        summary["raw_outcomes"].pop()
    elif corruption == "duplicate":
        summary["raw_outcomes"][-1] = deepcopy(row)
    elif corruption == "noise":
        row["noise"]["station_sd"][0] = 2.
    elif corruption == "binding":
        row["estimate"]["driver_binding"]["spec_sha256"] = "f" * 64
    elif corruption == "reuse":
        row["origin"] = "new" if row["origin"] == "reused" else "reused"
    else:
        with pytest.raises(CovarianceAnalysisError, match="external pin"):
            summarize_covariances(records[0], summary, expected_freeze_sha256="f" * 64)
        return
    with pytest.raises(CovarianceAnalysisError):
        run(records)


def test_population_parameters_are_a_known_reference_not_an_estimate(records):
    for row in records[1]["raw_outcomes"]:
        p = row["estimate"]["provenance"]
        if p["calibration"]["family"] == "W_exp_population":
            p["calibration"]["ell_hours"] = [.25] * 4
    with pytest.raises(CovarianceAnalysisError, match="population reference|Population reference"):
        run(records)


def test_pure_no_file_rng_or_calibration_and_owned_output(records, monkeypatch):
    import builtins
    import experiments.source_comparison.calibration as calibration_module
    def forbidden(*args, **kwargs):
        pytest.fail("Diagnostic analysis must not read files, draw RNG or fit covariance")
    # До вызова завершены все импорты и подготовка тестовых данных.
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(np.random, "Generator", forbidden)
        patch.setattr(calibration_module, "fit_covariance", forbidden)
        first = run(records)
    saved = canonical_bytes(list(records[:2]))
    expected = deepcopy(first)
    first["layout"]["primary_rows"].clear()
    first["shared_sets"][0]["parameters"]["station_variances"].clear()
    first["counts"]["available_sets"] = 0
    assert run(records) == expected
    assert canonical_bytes(list(records[:2])) == saved



def test_selected_covariance_scope_has_one_shared_set_and_four_members(records):
    from tests.experiments.observation_sensitivity.fixtures.paired_records import select_admission, seal
    from adrkit.config.validation import digest
    freeze, summary, _ = records
    ids = ["spatial_G1_matched/PG10/r1/L2", "spatial_G1_matched/PG10/r1/H1",
           "spatial_G2_matched/PG10/r1/L2", "spatial_G2_matched/PG10/r1/H1"]
    admitted = select_admission(freeze["admission"], ids, figures=[])
    freeze.update(version=4, admission=admitted, admission_sha256=digest(admitted))
    seal(freeze)
    summary["inputs"].update(freeze_file_sha256=pin(freeze), freeze_content_sha256=freeze["content_sha256"], admission_sha256=digest(admitted))
    summary["raw_outcomes"] = [r for r in summary["raw_outcomes"] if r["path_id"] in ids]
    report = summarize_covariances(freeze, summary, expected_freeze_sha256=pin(freeze))
    assert report["counts"] == dict(expected_sets=1, available_sets=1, unavailable_sets=0, represented_paths=4)
    assert report["shared_sets"][0]["member_count"] == 4 and report["shared_sets"][0]["origin_counts"] == dict(new=2, reused=2)
    summary["raw_outcomes"].pop()
    with pytest.raises(CovarianceAnalysisError): summarize_covariances(freeze, summary, expected_freeze_sha256=pin(freeze))
