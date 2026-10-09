"""Восстановление E06 с фиксацией выбора группы перед независимой оценкой ошибок."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json

import numpy as np

from adrkit.config.validation import canonical_bytes, digest
from experiments.source_comparison.calibration import CalibrationFailure, array_hash
from experiments.source_comparison.inverse import penalty_grams
from adrkit.inverse.misfit import CovarianceMetric
from experiments.source_recovery.driver import final_certificate, select_candidate
from experiments.source_recovery.panels import observation_diagnostics
from adrkit.inverse.projected import fit, lcurve_corner

from .backend import PRIMARY_ROWS
from .design import ALPHA_EXPONENTS, PathSpec
from .lifecycle import CheckpointError, PersistenceError
from .reuse import VerifiedBaseline


def _equal(left, right):
    return canonical_bytes(left) == canonical_bytes(right)


def _vector(value, size, name):
    raw = np.asarray(value)
    if raw.shape != (size,) or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be a finite real vector of length {size}")
    return np.asarray(raw, dtype=np.float64)


def _residual(value, tolerance):
    if not np.isfinite(value) or not 0 <= value <= tolerance:
        raise ValueError("forward residual certificate exceeds the fixed tolerance")
    return float(value)


def _error(error):
    return dict(error_type=type(error).__name__, error=str(error))


def _binding(spec, path):
    # Кортежи dataclass преобразуются в списки JSON без изменения чисел.
    declared = json.loads(json.dumps(asdict(path), allow_nan=False))
    return dict(spec_sha256=digest(spec), path_sha256=digest(declared))


def _validate(spec, paths, checkpoint, baselines):
    canonical_bytes(spec)
    if isinstance(paths, (str, bytes, set, frozenset, dict)):
        raise ValueError("paths must be an explicitly ordered sequence")
    paths = tuple(paths)
    if not paths or any(not isinstance(path, PathSpec) for path in paths):
        raise ValueError("a nonempty sequence of PathSpec is required")
    cp = checkpoint.record
    if ([path.id for path in paths] != cp["expected_paths"]
            or tuple(cp["exponents"]) != ALPHA_EXPONENTS
            or any((path.source, path.replicate) !=
                   (cp["bindings"]["source"], cp["bindings"]["replicate"]) for path in paths)):
        raise CheckpointError("ordered paths must exactly match the source/replicate checkpoint")
    if len({path.id for path in paths}) != len(paths):
        raise ValueError("duplicate path identity")
    expected_solver = dict(kkt_tolerance=1e-6, max_iterations=80, forward_tolerance=1e-11)
    if not _equal(spec["solver"], expected_solver):
        raise ValueError("E06 uses the unchanged v2 solver and its fixed tolerances")
    alpha = spec["alpha"]
    if (alpha["policy"] != "fixed_common_grid" or tuple(alpha["exponents"]) != ALPHA_EXPONENTS
            or alpha["tie_atol"] != 1e-12 or alpha["tie_rtol"] != 1e-10):
        raise ValueError("E06 uses the fixed 25-alpha grid and v2 tie tolerances")
    if any(path.reuse is not None for path in paths) and not isinstance(baselines, VerifiedBaseline):
        raise CheckpointError("reused paths require an admitted VerifiedBaseline; no fit fallback")
    return paths


def _inputs(spec, path, backend, panel_factory):
    signal, residual = backend.truth(path)
    signal = _vector(signal, 288, "noiseless observations")
    residual = _residual(residual, spec["solver"]["forward_tolerance"])
    data = panel_factory.estimation(path, signal)
    provenance = data.provenance.to_dict()
    for name in ("fit", "selection"):
        _vector(getattr(data, name), 36, f"{name} observations")
        if provenance.get(f"{name}_y_sha256") != array_hash(getattr(data, name)):
            raise CheckpointError(f"{name} panel bytes disagree with their declared provenance")
    raw_covariance = np.asarray(data.selected_covariance)
    if (raw_covariance.shape != (36, 36) or raw_covariance.dtype != np.dtype(np.float64)
            or not np.isfinite(raw_covariance).all()):
        raise CheckpointError("working covariance must have 36 by 36 finite float64 entries")
    if provenance.get("selected_covariance_sha256") != array_hash(raw_covariance):
        raise CheckpointError("raw working covariance bytes disagree with their declared provenance")
    # Разложение строится по точным исходным байтам, не по восстановленной L@L.T, которая может
    # отличаться из-за округления.
    checked_metric = CovarianceMetric(raw_covariance, layout=data.metric.layout)
    expected_factor_hash = sha256(b"covariance-factor:"+checked_metric.covariance_factor.tobytes(order="C")).hexdigest()
    actual_factor_hash = sha256(b"covariance-factor:"+data.metric.covariance_factor.tobytes(order="C")).hexdigest()
    if expected_factor_hash != actual_factor_hash:
        raise CheckpointError("actual metric factor differs from the declared raw working covariance")
    prepared = backend.prepare(path)
    prediction = prepared.restricted_prediction
    data.metric.layout.require_compatible(prediction.codomain)
    jacobian, offset = prepared.jacobian, prepared.offset
    if (np.shape(jacobian) != (36, 73) or np.iscomplexobj(jacobian)
            or not np.isfinite(jacobian).all()):
        raise ValueError("initializer Jacobian must have 36 by 73 real finite entries")
    _vector(offset, 36, "initializer offset")
    grams = penalty_grams(prepared.basis, spec["Qref"], tau_hours=path.tau_hours)
    white = data.metric.whiten_matrix(jacobian)
    information = white.T @ white
    gram = grams[path.penalty]
    reference = float(np.trace(np.linalg.solve(gram, information))/len(gram))
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("nonpositive alpha information scale")
    provenance.update(jacobian_sha256=array_hash(jacobian), offset_sha256=array_hash(offset),
        observation_diagnostic=observation_diagnostics(white, grams["L2"]),
        truth_forward_residual=residual)
    canonical_bytes(provenance)
    return prediction, data, gram, jacobian, offset, reference, provenance


def _attempt(spec, args, alpha, fitter):
    prediction, data, gram, jacobian, offset, _, _ = args
    try:
        prediction.invalidate()
        result = deepcopy(fitter(prediction, data.fit, data.metric, gram, alpha, jacobian, offset,
            tolerance=spec["solver"]["kkt_tolerance"], max_iterations=80))
        if result["alpha"] != alpha:
            raise ValueError("fitter changed the requested physical alpha")
        fitted = _vector(result["prediction"], 36, "fitted observations")
        point = _vector(result["coefficients"], 73, "fitted coefficients")
        if np.any(point < 0):
            raise ValueError("fitted coefficients violate nonnegativity")
        result["forward_residual"] = float(prediction.trajectory.max_scaled_residual)
        result["coefficients"] = point.tolist()
        result["prediction"] = fitted.tolist()
        result["accepted"] = bool(result["accepted"] and result["status"] == "accepted"
            and np.isfinite(result["forward_residual"])
            and 0 <= result["forward_residual"] <= spec["solver"]["forward_tolerance"])
        result["selection_mse"] = float(np.mean((fitted-data.selection)**2))
        canonical_bytes(result)
        return result
    except (CheckpointError, PersistenceError):
        raise
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        return dict(alpha=alpha, accepted=False, status="numerical_failure", **_error(error))


def _finalize(spec, record, args, certificate):
    candidates = sorted(record["candidates"].values(), key=lambda row: row["exponent"])
    selected = select_candidate(candidates, atol=spec["alpha"]["tie_atol"], rtol=spec["alpha"]["tie_rtol"])
    complete = len(candidates) == 25 and all(row["accepted"] for row in candidates)
    boundary = selected is not None and selected["exponent"] in (ALPHA_EXPONENTS[0], ALPHA_EXPONENTS[-1])
    update = dict(finalized=True, candidate_count=len(candidates),
        accepted_count=sum(row["accepted"] for row in candidates), complete_path=complete,
        selected_exponent=None if selected is None else selected["exponent"],
        lcurve_diagnostic=lcurve_corner(candidates), tuning_unresolved=boundary,
        procedure_accepted=False)
    if selected is not None:
        prediction, data, gram, *_ = args
        try:
            # Проверка стационарности использует отдельную траекторию прогноза.
            cert = deepcopy(certificate(prediction, data.fit, data.metric, gram, selected,
                spec["solver"]["forward_tolerance"]))
            if type(cert.get("accepted")) is not bool:
                raise ValueError("final certificate must return boolean accepted")
            canonical_bytes(cert)
        except (CheckpointError, PersistenceError):
            raise
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            cert = dict(accepted=False, **_error(error))
        update.update(final_certificate=cert,
            procedure_accepted=bool(complete and cert["accepted"] and not boundary))
    record.update(update)


def _reuse(baselines, path, provenance, reference, binding):
    permitted = {key: value for key, value in provenance.items()
                 if key != "selected_covariance_guards"}
    record = baselines.estimate(path, provenance=permitted, alpha_reference=reference)
    record["driver_binding"] = binding
    record["e06_provenance"] = provenance
    return record


def _score(spec, path, record, checkpoint, backend, panel_factory):
    # Эта проверка остаётся вне обработчика численных отказов.
    checkpoint.require_sealed()
    common = dict(procedure_accepted=record["procedure_accepted"],
        diagnostic_only=not record["procedure_accepted"], score_row_count=36)
    exponent = record.get("selected_exponent")
    if exponent is None:
        return dict(common, status="unavailable", reason=("calibration_failure"
            if "calibration_failure" in record else "no_accepted_candidate"))
    candidate = record["candidates"][str(exponent)]
    if not candidate["accepted"]:
        raise CheckpointError("selected candidate was not accepted")
    try:
        point = _vector(candidate["coefficients"], 73, "selected coefficients")
        source = backend.score_source(path, point)
        assumed, true_h, residual = backend.score_prediction(path, point)
        assumed = _vector(assumed, 288, "prediction under assumed H")[list(PRIMARY_ROWS)]
        true_h = _vector(true_h, 288, "prediction under true H")[list(PRIMARY_ROWS)]
        residual = _residual(residual, spec["solver"]["forward_tolerance"])
        signal, truth_residual = backend.truth(path)
        signal = _vector(signal, 288, "noiseless observations")
        _residual(truth_residual, spec["solver"]["forward_tolerance"])
        # Повторно проверить сохранённый выбор перед запросом проверочной выборки.
        checkpoint.require_sealed()
        test = panel_factory.test(path, signal)
        values = _vector(test.values, 36, "test observations")
        test_provenance = test.provenance.to_dict()
        if test_provenance.get("test_y_sha256") != array_hash(test.values):
            raise CheckpointError("test panel bytes disagree with their declared provenance")
        truth = signal[list(PRIMARY_ROWS)]
        rms = lambda difference: float(np.sqrt(np.mean(difference**2)))
        result = dict(common, status="available", source=source,
            assumed_H_test_rmse=rms(assumed-values),
            prediction_under_true_H_test_rmse=rms(true_h-values),
            assumed_H_noiseless_rmse=rms(assumed-truth),
            prediction_under_true_H_noiseless_rmse=rms(true_h-truth),
            observation_model_discrepancy_rmse=rms(assumed-true_h),
            forward_residual=residual, test_provenance=test_provenance,
            score_design=dict(primary_rows=list(PRIMARY_ROWS),
                target="noiseless observations generated under this path's true H",
                test_target="the same true-H mean plus its independent test residual",
                prediction_under_true_H="reproject the fitted state through the generating H",
                assumed_H="reproject the same fitted state through the fitted H",
                source_error="analytic temporal source error, distinct from observation RMSE"))
        canonical_bytes(result)
        return result
    except (CheckpointError, PersistenceError):
        raise
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        return dict(common, status="unavailable", reason="scoring_failure", **_error(error))


def run_group(spec, paths, checkpoint, backend, panel_factory, baselines, *,
              fitter=fit, certificate=final_certificate):
    """Выполнить или продолжить группу оценок E06.

    Сохранённые кандидаты и ошибки повторно не вычисляются. Перед запросом
    проверочных данных закрепляется выбор группы. Численные отказы
    сохраняются как результаты попыток; ошибки привязок и сохранения
    распространяются вызывающей программе.

    Parameters
    ----------
    spec : dict
        Научная конфигурация с фиксированными настройками решателя и сеткой
        регуляризации.
    paths : sequence of PathSpec
        Непустая упорядоченная группа, точно совпадающая с checkpoint.
    checkpoint : Checkpoint
        Открытая контрольная запись группы.
    backend : object
        Модель с методами truth, prepare, score_source и score_prediction;
        prepare возвращает прогноз, базис и линеаризацию.
    panel_factory : object
        Фабрика с методами estimation и test для данных подгонки, выбора и
        проверки.
    baselines : VerifiedBaseline or None
        Проверенный снимок E05, обязательный для последовательностей с reuse.
    fitter : callable, optional
        Подгонка с интерфейсом adrkit.inverse.projected.fit; возвращает
        запись кандидата.
    certificate : callable, optional
        Проверка выбранной оценки с интерфейсом final_certificate; возвращает
        запись с логическим accepted.

    Returns
    -------
    dict
        Запись checkpoint.record после сохранения оценок и проверочных
        ошибок; возвращается тот же изменяемый объект.
    """
    paths = _validate(spec, paths, checkpoint, baselines)
    cp = checkpoint.record
    if cp["stage"] != "estimating":
        checkpoint.require_sealed()
    for path in paths:
        existing = cp["paths"].get(path.id)
        binding = _binding(spec, path)
        if existing is not None and not _equal(existing.get("driver_binding"), binding):
            raise CheckpointError("driver spec/path binding changed on resume")
        try:
            args = _inputs(spec, path, backend, panel_factory)
        except CalibrationFailure as error:
            provenance = dict(calibration_failure=str(error))
            if path.reuse is not None:
                terminal = _reuse(baselines, path, provenance, None, binding)
            else:
                terminal = dict(condition=path.condition, penalty=path.penalty,
                    candidates={}, finalized=True, procedure_accepted=False,
                    calibration_failure=str(error), driver_binding=binding)
            if existing is not None:
                if not _equal(existing, terminal):
                    raise CheckpointError("regenerated calibration failure differs from checkpoint")
            else:
                cp["paths"][path.id] = terminal
                checkpoint.save()
            continue
        prediction, _, _, _, _, reference, provenance = args
        if path.reuse is not None:
            terminal = _reuse(baselines, path, provenance, reference, binding)
            if existing is not None:
                if not _equal(existing, terminal):
                    raise CheckpointError("regenerated reused estimate differs from checkpoint")
            else:
                cp["paths"][path.id] = terminal
                checkpoint.save()
        else:
            if existing is None:
                existing = dict(condition=path.condition, penalty=path.penalty,
                    alpha_reference=reference, provenance=provenance, candidates={},
                    finalized=False, driver_binding=binding)
                cp["paths"][path.id] = existing
            elif (not _equal(existing.get("alpha_reference"), reference)
                    or not _equal(existing.get("provenance"), provenance)):
                raise CheckpointError("regenerated estimator inputs differ from checkpoint")
            if not existing["finalized"]:
                for exponent in ALPHA_EXPONENTS:
                    if str(exponent) in existing["candidates"]:
                        continue
                    candidate = _attempt(spec, args, float(reference*10.**exponent), fitter)
                    candidate["exponent"] = exponent
                    existing["candidates"][str(exponent)] = candidate
                    checkpoint.save()
                _finalize(spec, existing, args, certificate)
                checkpoint.save()
        prediction.invalidate()
        del args, prediction
    checkpoint.seal()
    checkpoint.require_sealed()
    if cp["stage"] == "scored":
        return cp
    for path in paths:
        if path.id in cp.get("scores", {}):
            continue
        score = _score(spec, path, cp["paths"][path.id], checkpoint, backend, panel_factory)
        cp.setdefault("scores", {})[path.id] = score
        checkpoint.save()
    cp["stage"] = "scored"
    checkpoint.save()
    return cp
