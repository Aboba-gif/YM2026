"""Проверки подгонки, фиксации выбора и продолжения группы E06.

Используются искусственные наблюдения и линейные модели; PDE не решается.
"""
import experiments.source_recovery.config as recovery_config
import experiments.source_recovery.driver as recovery_driver
import experiments.source_recovery.panels as recovery_panels
import hashlib
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from adrkit.config.validation import JSONRecord, canonical_bytes, digest
from experiments.source_comparison.calibration import CalibrationFailure, array_hash
from experiments.source_comparison.inverse import penalty_grams
from adrkit.inverse.misfit import CovarianceMetric
from adrkit.sources import P1Basis
from adrkit.spaces import ArraySpace

from adrkit.inverse.projected import fit as frozen_fit
from experiments.observation_sensitivity import driver
from experiments.observation_sensitivity.backend import PRIMARY_ROWS
from experiments.observation_sensitivity.design import ALPHA_EXPONENTS, build_design
from experiments.observation_sensitivity.lifecycle import Checkpoint, CheckpointError, PersistenceError
from experiments.observation_sensitivity.reuse import VerifiedBaseline


def path_for(condition="spatial_G05_matched", penalty="L2"):
    return next(p for p in build_design().paths if (p.condition, p.penalty, p.source, p.replicate)
                == (condition, penalty, "PG10", 1))


def bindings():
    return dict(source="PG10", replicate=1, fixture="tiny synthetic orchestration")


def layout():
    return ArraySpace((36,), axes=("observation",),
        coordinates=(tuple(f"dense_row_{r}" for r in PRIMARY_ROWS),), units="ug/m^3")


class Prediction:
    def __init__(self):
        basis = P1Basis(np.linspace(0, 3, 73), time_unit="h")
        self.domain = ArraySpace((basis.size,), axes=("coefficient",),
                                 coordinates=(basis.knots,), units="dimensionless")
        self.codomain = layout()
        self.invalidations = 0
        self.trajectory = SimpleNamespace(max_scaled_residual=0.)

    def invalidate(self):
        self.invalidations += 1


class Backend:
    def __init__(self):
        self.basis = P1Basis(np.linspace(0, 3, 73), time_unit="h")
        self.jacobian = np.eye(36, 73)
        self.offset = np.full(36, .3)
        self.prepared, self.scoring = [], []
        self.signal_shift = 0.
        self.truth_residual, self.score_residual = 0., 0.
        self.source_failure = False
        self.checkpoint = None

    def truth(self, path):
        return np.full(288, .5+self.signal_shift), self.truth_residual

    def prepare(self, path):
        prediction = Prediction()
        self.prepared.append(prediction)
        return SimpleNamespace(restricted_prediction=prediction, basis=self.basis,
            jacobian=self.jacobian.copy(), offset=self.offset.copy())

    def score_source(self, path, point):
        self.checkpoint.require_sealed()
        self.scoring.append((path.id, "source"))
        if self.source_failure:
            raise RuntimeError("synthetic source score failure")
        return dict(E_q=.4, relative_L2=.7)

    def score_prediction(self, path, point):
        self.checkpoint.require_sealed()
        self.scoring.append((path.id, "prediction"))
        # Различные проекции позволяют проверить, что RMSE с истинным и предполагаемым H не
        # смешиваются.
        return np.full(288, 1.5), np.full(288, .75), self.score_residual


class Panels:
    def __init__(self):
        self.estimation_calls, self.test_calls = [], []
        self.checkpoint = None
        self.failure = None
        self.fit_shift, self.selection_shift, self.covariance_scale = 0., 0., 1.

    def estimation(self, path, signal):
        self.estimation_calls.append(path.id)
        if self.failure:
            raise CalibrationFailure(self.failure)
        fit = np.asarray(signal)[list(PRIMARY_ROWS)]+1.+self.fit_shift
        selection = np.asarray(signal)[list(PRIMARY_ROWS)]+2.+self.selection_shift
        covariance = self.covariance_scale*np.eye(36)
        for value in (fit, selection):
            value.flags.writeable = False
        provenance = dict(fit_y_sha256=array_hash(fit), selection_y_sha256=array_hash(selection),
            selected_covariance_sha256=array_hash(covariance),
            panel_records={"fit": {"panel_code": 211}, "selection": {"panel_code": 307}},
            calibration={"status": "synthetic fixture", "panels": 32},
            design=dict(rows=list(PRIMARY_ROWS), retained_count=36, mask="none", removed_primary_tick_indices=[]),
            selected_covariance_guards={"checked": True})
        return SimpleNamespace(fit=fit, selection=selection,
            metric=CovarianceMetric(covariance, layout=layout()), selected_covariance=covariance.copy(),
            provenance=JSONRecord(provenance))

    def test(self, path, signal):
        self.checkpoint.require_sealed()
        self.test_calls.append(path.id)
        values = np.asarray(signal)[list(PRIMARY_ROWS)]+.5
        return SimpleNamespace(values=values,
            provenance=JSONRecord({"test_y_sha256": array_hash(values), "panel_record": {"panel_code": 401}}))


class Fitter:
    def __init__(self, *, target=-2., failures=(), tied=(), all_fail=False):
        self.target, self.failures, self.tied = target, set(failures), set(tied)
        self.all_fail, self.calls, self.inputs = all_fail, [], []
        self.raise_at = None
        self.bad_value = None
        self.forward_residual = 0.

    def __call__(self, prediction, observations, metric, gram, alpha, jacobian, offset, **options):
        assert options == dict(tolerance=1e-6, max_iterations=80)
        assert not hasattr(prediction, "truth") and not hasattr(prediction, "source")
        assert prediction.invalidations > 0
        assert not observations.flags.writeable
        white = metric.whiten_matrix(jacobian)
        reference = float(np.trace(np.linalg.solve(gram, white.T@white))/len(gram))
        exponent = round(float(np.log10(alpha/reference)), 8)
        self.calls.append(exponent)
        self.inputs.append((prediction, observations.copy(), metric, gram.copy(), jacobian.copy(), offset.copy()))
        if self.raise_at is not None and len(self.calls) == self.raise_at:
            raise KeyboardInterrupt("synthetic process interruption before a result")
        if self.all_fail or exponent in self.failures:
            raise RuntimeError("synthetic numerical failure")
        mismatch = 0. if exponent in self.tied else abs(exponent-self.target)
        prediction.trajectory = SimpleNamespace(max_scaled_residual=self.forward_residual)
        result = dict(alpha=alpha, accepted=True, status="accepted", initialization="affine_nnls",
            coefficients=np.full(73, .1), prediction=np.full(36, 2.5+mismatch),
            residual_norm=1.+alpha, penalty_norm=1.)
        if self.bad_value is not None:
            self.bad_value(result)
        return result


class Certificate:
    def __init__(self, *, accepted=True, raises=False):
        self.accepted, self.raises, self.calls = accepted, raises, []

    def __call__(self, prediction, observations, metric, gram, candidate, tolerance):
        assert tolerance == 1e-11
        self.calls.append(candidate["exponent"])
        prediction.invalidate()
        if self.raises:
            raise RuntimeError("synthetic fresh certificate failure")
        return dict(accepted=self.accepted, fresh=True)


def run(cp, *, paths=None, spec=None, backend=None, panels=None, fitter=None, certificate=None, baseline=None):
    paths = (path_for(),) if paths is None else paths
    spec = recovery_config.template_config(100.) if spec is None else spec
    backend = Backend() if backend is None else backend
    panels = Panels() if panels is None else panels
    backend.checkpoint = panels.checkpoint = cp
    fitter = Fitter() if fitter is None else fitter
    certificate = Certificate() if certificate is None else certificate
    result = driver.run_group(spec, paths, cp, backend, panels, baseline,
        fitter=fitter, certificate=certificate)
    return result, backend, panels, fitter, certificate


def checkpoint(tmp_path, paths=None):
    paths = (path_for(),) if paths is None else paths
    return Checkpoint(tmp_path/"group.json", bindings(), [p.id for p in paths])


def test_complete_group_fixed_cold_grid_seal_and_distinct_scores(tmp_path):
    paths = (path_for(), path_for(penalty="H1"))
    with checkpoint(tmp_path, paths) as cp:
        result, backend, panels, fitter, cert = run(cp, paths=paths)
        assert result["stage"] == "scored"
        assert fitter.calls == list(ALPHA_EXPONENTS)*2
        assert cert.calls == [-2., -2.]
        assert set(result["scores"]) == set(result["expected_paths"])
        assert panels.test_calls == [p.id for p in paths]
        for path in paths:
            record, score = result["paths"][path.id], result["scores"][path.id]
            assert record["complete_path"] and record["procedure_accepted"]
            assert record["selected_exponent"] == -2.
            assert score["status"] == "available" and not score["diagnostic_only"]
            assert score["source"] == dict(E_q=.4, relative_L2=.7)
            assert score["assumed_H_test_rmse"] == .5
            assert score["prediction_under_true_H_test_rmse"] == .25
            assert score["assumed_H_noiseless_rmse"] == 1.
            assert score["prediction_under_true_H_noiseless_rmse"] == .25
            assert score["observation_model_discrepancy_rmse"] == .75
        assert all(np.array_equal(row[1], np.full(36, 1.5)) for row in fitter.inputs)
        assert all(pred.invalidations >= 27 for pred in backend.prepared)
        cp.require_sealed()


def test_injected_fitter_does_not_change_declared_defaults():
    assert driver.run_group.__kwdefaults__["fitter"] is frozen_fit
    assert driver.run_group.__kwdefaults__["certificate"] is recovery_driver.final_certificate


def test_interruption_after_committed_candidate_resumes_without_repeating(tmp_path):
    fitter = Fitter()
    with checkpoint(tmp_path) as cp:
        save = cp.save
        def interrupt_after_save():
            save()
            if len(cp.record["paths"][path_for().id]["candidates"]) == 7:
                raise KeyboardInterrupt("after durable candidate seven")
        cp.save = interrupt_after_save
        with pytest.raises(KeyboardInterrupt):
            run(cp, fitter=fitter)
        assert fitter.calls == list(ALPHA_EXPONENTS[:7])
    second = Fitter()
    with checkpoint(tmp_path) as cp:
        result, *_ = run(cp, fitter=second)
        assert second.calls == list(ALPHA_EXPONENTS[7:])
        assert result["paths"][path_for().id]["procedure_accepted"]


@pytest.mark.parametrize("all_fail", [False, True])
def test_candidate_failure_terminal_no_retry_all_scores_present(tmp_path, all_fail):
    fitter = Fitter(failures={-3.}, all_fail=all_fail)
    with checkpoint(tmp_path) as cp:
        result, _, panels, _, cert = run(cp, fitter=fitter)
        record, score = result["paths"][path_for().id], result["scores"][path_for().id]
        assert len(fitter.calls) == 25 and record["candidate_count"] == 25
        assert not record["complete_path"] and not record["procedure_accepted"]
        assert record["candidates"]["-3.0"]["status"] == "numerical_failure"
        assert score["diagnostic_only"]
        if all_fail:
            assert score["status"] == "unavailable" and not panels.test_calls and not cert.calls
        else:
            assert score["status"] == "available"
    with checkpoint(tmp_path) as cp:
        _, _, panels, fitter, _ = run(cp)
        assert not fitter.calls and not panels.test_calls


@pytest.mark.parametrize("target", [-8., 4.])
def test_boundary_selection_is_unresolved_even_with_fresh_certificate(tmp_path, target):
    with checkpoint(tmp_path) as cp:
        result, *_ = run(cp, fitter=Fitter(target=target))
        record = result["paths"][path_for().id]
        assert record["complete_path"] and record["final_certificate"]["accepted"]
        assert record["tuning_unresolved"] and not record["procedure_accepted"]


def test_tie_uses_frozen_largest_physical_alpha_rule(tmp_path):
    with checkpoint(tmp_path) as cp:
        result, *_ = run(cp, fitter=Fitter(tied={-2., -.5}))
        assert result["paths"][path_for().id]["selected_exponent"] == -.5


@pytest.mark.parametrize("raises", [False, True])
def test_fresh_certificate_failure_remains_failure(tmp_path, raises):
    cert = Certificate(accepted=False, raises=raises)
    with checkpoint(tmp_path) as cp:
        result, *_ = run(cp, certificate=cert)
        record = result["paths"][path_for().id]
        assert cert.calls == [-2.] and not record["procedure_accepted"]
        assert not record["final_certificate"]["accepted"]
        assert result["scores"][path_for().id]["diagnostic_only"]


def test_calibration_failure_has_no_fit_or_test_and_is_rechecked_on_resume(tmp_path):
    panels = Panels()
    panels.failure = "fixed panel covariance guard failure"
    with checkpoint(tmp_path) as cp:
        result, backend, _, fitter, cert = run(cp, panels=panels)
        row = result["paths"][path_for().id]
        assert row["calibration_failure"] == panels.failure
        assert row["candidates"] == {} and row["finalized"]
        assert not backend.prepared and not fitter.calls and not cert.calls and not panels.test_calls
        assert result["scores"][path_for().id]["reason"] == "calibration_failure"
    with checkpoint(tmp_path) as cp:
        run(cp, panels=panels)
        panels.failure = "different failure"
        with pytest.raises(CheckpointError, match="calibration failure differs"):
            run(cp, panels=panels)


def test_seal_canary_blocks_source_test_scoring_even_after_all_paths_finalize(tmp_path):
    panels, backend = Panels(), Backend()
    with checkpoint(tmp_path) as cp:
        cp.seal = lambda: "fake seal must grant no access"
        with pytest.raises(CheckpointError, match="before selection seal"):
            run(cp, panels=panels, backend=backend)
        assert not panels.test_calls and not backend.scoring
        assert all(row["finalized"] for row in cp.record["paths"].values())


def test_interruption_after_seal_resumes_scoring_without_fits(tmp_path):
    with checkpoint(tmp_path) as cp:
        seal = cp.seal
        def interrupt():
            seal()
            raise KeyboardInterrupt("after committed group seal")
        cp.seal = interrupt
        with pytest.raises(KeyboardInterrupt):
            run(cp)
    with checkpoint(tmp_path) as cp:
        result, _, panels, fitter, cert = run(cp)
        assert not fitter.calls and not cert.calls and panels.test_calls == [path_for().id]
        assert result["stage"] == "scored"


def test_scoring_rows_commit_individually_and_resume_skips_committed_scores(tmp_path):
    paths = (path_for(), path_for(penalty="H1"))
    with checkpoint(tmp_path, paths) as cp:
        save = cp.save
        def interrupt():
            save()
            if len(cp.record.get("scores", {})) == 1:
                raise KeyboardInterrupt("after first score commit")
        cp.save = interrupt
        with pytest.raises(KeyboardInterrupt):
            run(cp, paths=paths)
    with checkpoint(tmp_path, paths) as cp:
        result, _, panels, fitter, _ = run(cp, paths=paths)
        assert panels.test_calls == [paths[1].id] and not fitter.calls
        assert len(result["scores"]) == 2


@pytest.mark.parametrize("where", ["candidate", "seal", "score"])
def test_persistence_failure_propagates_without_numerical_reroll(tmp_path, where):
    fitter, panels = Fitter(), Panels()
    with checkpoint(tmp_path) as cp:
        save, seal = cp.save, cp.seal
        def failing_save():
            if where == "candidate" or (where == "score" and cp.record.get("scores")):
                raise PersistenceError("synthetic disk failure")
            save()
        def failing_seal():
            if where == "seal":
                raise PersistenceError("synthetic seal write failure")
            return seal()
        cp.save, cp.seal = failing_save, failing_seal
        with pytest.raises(PersistenceError, match="synthetic"):
            run(cp, fitter=fitter, panels=panels)
        assert len(fitter.calls) == (1 if where == "candidate" else 25)
        assert len(panels.test_calls) == (1 if where == "score" else 0)


@pytest.mark.parametrize("field", ["fit", "selection", "covariance", "jacobian", "offset", "truth"])
def test_scored_resume_rejects_changed_estimator_inputs_before_scores(tmp_path, field):
    with checkpoint(tmp_path) as cp:
        run(cp)
    backend, panels, fitter = Backend(), Panels(), Fitter()
    if field == "fit":
        panels.fit_shift = .1
    elif field == "selection":
        panels.selection_shift = .1
    elif field == "covariance":
        panels.covariance_scale = 2.
    elif field == "jacobian":
        backend.jacobian[0, 0] += .1
    elif field == "offset":
        backend.offset[0] += .1
    else:
        backend.signal_shift = .1
    with checkpoint(tmp_path) as cp, pytest.raises(CheckpointError, match="inputs differ"):
        run(cp, backend=backend, panels=panels, fitter=fitter)
    assert not fitter.calls and not panels.test_calls and not backend.scoring


def test_resume_binds_full_spec_and_path_even_if_id_unchanged(tmp_path):
    with checkpoint(tmp_path) as cp:
        run(cp)
    spec = recovery_config.template_config(100.)
    spec["model"]["diffusion"] *= 2
    with checkpoint(tmp_path) as cp, pytest.raises(CheckpointError, match="binding changed"):
        run(cp, spec=spec)
    changed = replace(path_for(), inverse_h=replace(path_for().inverse_h, width_km=.6))
    with checkpoint(tmp_path) as cp, pytest.raises(CheckpointError, match="binding changed"):
        run(cp, paths=(changed,))


@pytest.mark.parametrize("change", ["order", "omit", "replicate", "solver", "alpha"])
def test_invalid_group_or_method_rejected_before_any_numerics(tmp_path, change):
    paths = (path_for(), path_for(penalty="H1"))
    supplied, spec = paths, recovery_config.template_config(100.)
    if change == "order":
        supplied = paths[::-1]
    elif change == "omit":
        supplied = paths[:1]
    elif change == "replicate":
        supplied = (replace(paths[0], replicate=2), paths[1])
    elif change == "solver":
        spec["solver"]["max_iterations"] = 81
    else:
        spec["alpha"]["tie_rtol"] = 0.
    backend, panels = Backend(), Panels()
    with checkpoint(tmp_path, paths) as cp, pytest.raises(ValueError):
        run(cp, paths=supplied, spec=spec, backend=backend, panels=panels)
    assert not backend.prepared and not panels.estimation_calls


@pytest.mark.parametrize("bad", ["nan", "negative", "wrong_alpha", "bad_residual"])
def test_invalid_candidate_cannot_enter_an_accepted_procedure(tmp_path, bad):
    fitter = Fitter()
    def mutate(result):
        if bad == "nan":
            result["prediction"][0] = float("nan")
        elif bad == "negative":
            result["coefficients"][0] = -1.
        elif bad == "wrong_alpha":
            result["alpha"] *= 2
    fitter.bad_value = mutate
    if bad == "bad_residual":
        fitter.forward_residual = 1e-4
    with checkpoint(tmp_path) as cp:
        result, *_ = run(cp, fitter=fitter)
        record = result["paths"][path_for().id]
        assert not record["procedure_accepted"] and record["accepted_count"] == 0
        assert result["scores"][path_for().id]["status"] == "unavailable"


@pytest.mark.parametrize("failure", ["source", "residual"])
def test_scoring_failure_is_a_terminal_explicit_row(tmp_path, failure):
    backend = Backend()
    if failure == "source":
        backend.source_failure = True
    else:
        backend.score_residual = 1e-4
    with checkpoint(tmp_path) as cp:
        result, _, panels, *_ = run(cp, backend=backend)
        assert result["scores"][path_for().id]["reason"] == "scoring_failure"
        assert result["stage"] == "scored" and not panels.test_calls


def baseline_fixture(tmp_path, path, *, rejected=False, calibration_failure=None, backend=None):
    """Создать искусственную группу E05 с происхождением входов тестового оценщика."""
    spec, panels = recovery_config.template_config(100.), Panels()
    backend = Backend() if backend is None else backend
    if calibration_failure:
        row = dict(condition=path.reuse.condition, penalty=path.penalty, candidates={},
            finalized=True, procedure_accepted=False, calibration_failure=calibration_failure)
    else:
        signal, _ = backend.truth(path)
        data = panels.estimation(path, signal)
        grams = penalty_grams(backend.basis, spec["Qref"], tau_hours=path.tau_hours)
        white = data.metric.whiten_matrix(backend.jacobian)
        reference = float(np.trace(np.linalg.solve(grams[path.penalty], white.T@white))/73)
        provenance = data.provenance.to_dict()
        provenance.pop("selected_covariance_guards")
        provenance.update(jacobian_sha256=array_hash(backend.jacobian), offset_sha256=array_hash(backend.offset),
            observation_diagnostic=recovery_panels.observation_diagnostics(white, grams["L2"]), truth_forward_residual=0.)
        row = dict(condition=path.reuse.condition, penalty=path.penalty, alpha_reference=reference,
            provenance=provenance, candidates={}, finalized=True, complete_path=not rejected,
            candidate_count=25, accepted_count=0 if rejected else 25,
            selected_exponent=None if rejected else -2., lcurve_diagnostic=None,
            tuning_unresolved=False, procedure_accepted=not rejected)
        for exponent in ALPHA_EXPONENTS:
            row["candidates"][str(exponent)] = dict(exponent=exponent, alpha=reference*10.**exponent,
                accepted=not rejected, coefficients=[.1]*73, selection_mse=(exponent+2.)**2,
                status="numerical_failure" if rejected else "accepted", forward_residual=0.)
        if not rejected:
            row["final_certificate"] = {"accepted": True, "frozen_v2": True}
    original_id = f"{path.reuse.condition}/{path.penalty}"
    record = dict(bindings=bindings(), paths={original_id: row}, expected_paths=[original_id],
        exponents=list(ALPHA_EXPONENTS), stage="scored", scores={original_id: {"HIDDEN_TEST": 123}})
    record["selection_seal"] = digest(record["paths"])
    target = tmp_path/"synthetic-terminal-v2.json"
    target.write_bytes(canonical_bytes(record))
    return VerifiedBaseline.from_terminal(target, expected_bindings=bindings(),
        expected_file_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        expected_paths=(original_id,)), row


@pytest.mark.parametrize("rejected", [False, True])
def test_exact_v2_reuse_never_refits_even_a_rejected_baseline(tmp_path, rejected):
    path = path_for("spatial_G1_matched")
    baseline, original = baseline_fixture(tmp_path, path, rejected=rejected)
    with checkpoint(tmp_path, (path,)) as cp:
        result, _, panels, fitter, cert = run(cp, paths=(path,), baseline=baseline)
        row = result["paths"][path.id]
        assert not fitter.calls and not cert.calls
        assert row["candidates"] == original["candidates"]
        assert row["procedure_accepted"] is not rejected
        assert "HIDDEN_TEST" not in str(row)
        assert row["reuse"]["original_path_id"] == "main/L2"
        assert result["scores"][path.id]["status"] == ("unavailable" if rejected else "available")
        assert bool(panels.test_calls) is not rejected
    with checkpoint(tmp_path, (path,)) as cp:
        _, _, panels, fitter, cert = run(cp, paths=(path,), baseline=baseline)
        assert not fitter.calls and not cert.calls and not panels.test_calls


def test_reused_calibration_failure_is_reproduced_without_fitting(tmp_path):
    path = path_for("spatial_G1_matched")
    baseline, _ = baseline_fixture(tmp_path, path, calibration_failure="same guard failure")
    panels = Panels()
    panels.failure = "same guard failure"
    with checkpoint(tmp_path, (path,)) as cp:
        result, _, _, fitter, cert = run(cp, paths=(path,), baseline=baseline, panels=panels)
        assert not fitter.calls and not cert.calls and not panels.test_calls
        assert result["scores"][path.id]["reason"] == "calibration_failure"


def test_missing_or_mismatched_baseline_fails_closed_without_fit(tmp_path):
    path = path_for("spatial_G1_matched")
    baseline, _ = baseline_fixture(tmp_path, path)
    panels, fitter = Panels(), Fitter()
    with checkpoint(tmp_path, (path,)) as cp:
        with pytest.raises(CheckpointError, match="VerifiedBaseline"):
            run(cp, paths=(path,), fitter=fitter)
        panels.fit_shift = .1
        with pytest.raises(CheckpointError, match="input mismatch"):
            run(cp, paths=(path,), baseline=baseline, panels=panels, fitter=fitter)
        assert not fitter.calls and not cp.record["paths"] and not panels.test_calls


def test_checkpoint_error_from_seal_check_is_not_numerical_score_failure(tmp_path):
    panels = Panels()
    with checkpoint(tmp_path) as cp:
        def rejected_test(path, signal):
            raise CheckpointError("synthetic seal-integrity error")
        panels.test = rejected_test
        with pytest.raises(CheckpointError, match="seal-integrity"):
            run(cp, panels=panels)
        assert cp.record["stage"] == "sealed" and not cp.record.get("scores")


@pytest.mark.parametrize("panel", ["fit", "selection", "test"])
def test_reported_panel_hash_must_match_actual_values(tmp_path, panel):
    panels, fitter = Panels(), Fitter()
    method = panels.test if panel == "test" else panels.estimation
    def corrupt(path, signal):
        result = method(path, signal)
        field = "values" if panel == "test" else panel
        setattr(result, field, getattr(result, field)+.01)
        return result
    if panel == "test":
        panels.test = corrupt
    else:
        panels.estimation = corrupt
    with checkpoint(tmp_path) as cp, pytest.raises(CheckpointError, match="bytes disagree"):
        run(cp, panels=panels, fitter=fitter)
    assert len(fitter.calls) == (25 if panel == "test" else 0)


def test_actual_frozen_optimizer_certificate_and_panel_factory_linear_integration(tmp_path):
    """Проверить общий оптимизатор, сертификат и генератор данных на линейной модели."""
    from experiments.observation_sensitivity.data import PanelFactory

    class LinearPrediction(Prediction):
        def __init__(self, jacobian, offset):
            super().__init__()
            self.matrix, self.offset = jacobian.copy(), offset.copy()
            self.trajectory = SimpleNamespace(max_scaled_residual=0.)

        def predict(self, point):
            return self.matrix@point+self.offset

        def vjp(self, point, cotangent):
            return self.matrix.T@cotangent

    class LinearBackend(Backend):
        def prepare(self, path):
            prepared = super().prepare(path)
            prepared.restricted_prediction = LinearPrediction(prepared.jacobian, prepared.offset)
            return prepared

    path = path_for("covariance_mix_oracle")
    spec, backend = recovery_config.template_config(100.), LinearBackend()
    panels = PanelFactory(spec)
    with checkpoint(tmp_path, (path,)) as cp:
        backend.checkpoint = cp
        result = driver.run_group(spec, (path,), cp, backend, panels, None)
        row = result["paths"][path.id]
        assert row["candidate_count"] == row["accepted_count"] == 25
        assert all(candidate["initialization"] == "affine_nnls" for candidate in row["candidates"].values())
        assert row["final_certificate"]["accepted"]
        assert set(row["final_certificate"]["norms"]) == {
            "primal", "dual", "stationarity", "complementarity", "free_coordinate"}
        assert result["stage"] == "scored" and result["scores"][path.id]["status"] == "available"


@pytest.mark.parametrize("defect,match", [
    ("metric_only", "actual metric factor"),
    ("raw_only_stale_hash", "raw working covariance bytes"),
    ("raw_and_metric_stale_hash", "raw working covariance bytes"),
    ("raw_metric_and_new_hash", "input mismatch: selected_covariance_sha256"),
])
def test_exact_reuse_checks_covariance_and_actual_factor_even_when_alpha_scale_unchanged(tmp_path, defect, match):
    # J ранга один не чувствителен к изменению дисперсии координаты 1. Совпадение J, сдвига и
    # alpha_reference не заменяет проверку фактической W.
    backend = Backend()
    backend.jacobian[:] = 0.
    backend.jacobian[0, 0] = 1.
    path = path_for("spatial_G1_matched")
    baseline, _ = baseline_fixture(tmp_path, path, backend=backend)
    panels = Panels()
    original = panels.estimation
    def replace_metric_and_or_raw(path, signal):
        result = original(path, signal)
        changed = np.eye(36)
        changed[1, 1] = 2.
        if defect != "raw_only_stale_hash":
            result.metric = CovarianceMetric(changed, layout=layout())
        if defect != "metric_only":
            result.selected_covariance = changed.copy()
        if defect == "raw_metric_and_new_hash":
            provenance = result.provenance.to_dict()
            provenance["selected_covariance_sha256"] = array_hash(changed)
            result.provenance = JSONRecord(provenance)
        return result
    panels.estimation = replace_metric_and_or_raw
    fitter = Fitter()
    with checkpoint(tmp_path, (path,)) as cp, pytest.raises(CheckpointError, match=match):
        run(cp, paths=(path,), backend=backend, panels=panels, fitter=fitter, baseline=baseline)
    assert not fitter.calls and not panels.test_calls
    assert len(backend.prepared) == (1 if defect == "raw_metric_and_new_hash" else 0)


def test_actual_W03_raw_byte_hash_and_factor_pass_without_covariance_round_trip():
    from experiments.observation_sensitivity.data import PanelFactory

    spec, backend, path = recovery_config.template_config(100.), Backend(), path_for()
    args = driver._inputs(spec, path, backend, PanelFactory(spec))
    data, provenance = args[1], args[-1]
    covariance288, _ = recovery_panels.calibrate_dense(spec, 1, "corr", "W03")
    expected = covariance288[np.ix_(recovery_config.PRIMARY_ROWS, recovery_config.PRIMARY_ROWS)]
    assert np.array_equal(data.selected_covariance, expected)
    assert provenance["selected_covariance_sha256"] == array_hash(expected)
    assert data.metric.covariance_factor.tobytes(order="C") == (
        CovarianceMetric(expected, layout=data.metric.layout).covariance_factor.tobytes(order="C"))
    # Коррелированная ковариация не обязана сохраняться побитно после L@L.T. Проверка выше принимает
    # исходные байты, не требуя такого равенства.
    assert not np.array_equal(data.metric.covariance, expected)


@pytest.mark.parametrize("where", ["candidate", "certificate", "score"])
def test_memory_error_propagates_and_resume_uses_last_committed_checkpoint(tmp_path, where):
    error = MemoryError("synthetic allocation failure")
    fitter, certificate, panels = Fitter(), Certificate(), Panels()
    if where == "candidate":
        original = fitter
        def fail_candidate(*args, **kwargs):
            if len(original.calls) == 3:
                raise error
            return original(*args, **kwargs)
        fitter = fail_candidate
    elif where == "certificate":
        def fail_certificate(*args, **kwargs):
            raise error
        certificate = fail_certificate
    else:
        def fail_score(*args, **kwargs):
            raise error
        panels.test = fail_score
    with checkpoint(tmp_path) as cp:
        with pytest.raises(MemoryError) as caught:
            run(cp, fitter=fitter, certificate=certificate, panels=panels)
        assert caught.value is error
    fresh_fitter, fresh_certificate = Fitter(), Certificate()
    with checkpoint(tmp_path) as cp:
        row = cp.record["paths"][path_for().id]
        assert len(row["candidates"]) == (3 if where == "candidate" else 25)
        assert all(candidate["accepted"] for candidate in row["candidates"].values())
        result, *_ = run(cp, fitter=fresh_fitter, certificate=fresh_certificate)
        assert result["stage"] == "scored"
    expected_remaining = list(ALPHA_EXPONENTS[3:]) if where == "candidate" else []
    assert fresh_fitter.calls == expected_remaining
    assert fresh_certificate.calls == ([] if where == "score" else [-2.])


@pytest.mark.parametrize("residual,local_accepted,status,accepted", [
    (0., True, "accepted", True), (1e-11, True, "accepted", True),
    (1.1e-11, True, "accepted", False), (-1e-12, True, "accepted", False),
    (np.nan, True, "accepted", False), (np.inf, True, "accepted", False),
    (0., False, "accepted", False), (0., True, "iteration_limit", False),
])
def test_attempt_reads_final_state_and_keeps_acceptance_conjunction(
        residual, local_accepted, status, accepted):
    spec = recovery_config.template_config(100.)
    prediction = Prediction()
    data = SimpleNamespace(fit=np.zeros(36), selection=np.ones(36), metric=object())
    calls = []
    def estimate(model, *args, **options):
        assert model is prediction
        assert options == dict(tolerance=1e-6, max_iterations=80)
        calls.append("fit")
        model.trajectory = SimpleNamespace(max_scaled_residual=residual)
        return dict(alpha=1., status=status, accepted=local_accepted,
                    coefficients=np.full(73, .1), prediction=np.ones(36))
    result = driver._attempt(spec,
        (prediction, data, np.eye(73), np.eye(36, 73), np.zeros(36), None, None),
        1., estimate)
    assert calls == ["fit"] and prediction.invalidations == 1
    assert result["accepted"] is accepted
    if np.isfinite(residual):
        assert result["forward_residual"] == residual
        assert result["coefficients"] == [.1]*73 and result["prediction"] == [1.]*36
        assert result["selection_mse"] == 0.
        canonical_bytes(result)
    else:
        assert result["status"] == "numerical_failure"
    assert not {"seconds", "forward_calls", "adjoint_calls"} & result.keys()
