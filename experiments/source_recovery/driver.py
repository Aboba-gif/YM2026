"""Подбор регуляризации и оценка источника в закреплённых условиях опыта."""
from __future__ import annotations

import numpy as np
from adrkit.config.validation import JSONRecord
from adrkit.inverse.projected import fit, lcurve_corner
from adrkit.predictions import SelectedPrediction
from experiments.source_comparison.calibration import CalibrationFailure, array_hash, noise_covariance
from .checkpoints import canonical_hash
from .config import DENSE_TIMES, PRIMARY_ROWS, BASE_EXPONENTS
from .panels import draw_residual, selected_rows, restricted_metric, observation_diagnostics, calibrate_dense


def select_candidate(path, *, atol=1e-12, rtol=1e-10):
    """Выбрать принятую оценку с минимальной ошибкой выбора.

    Parameters
    ----------
    path : iterable of dict
        Кандидаты с accepted, selection_mse и alpha; MSE задана в квадрате
        единицы концентрации.
    atol : float, optional
        Абсолютный допуск равенства MSE в тех же единицах.
    rtol : float, optional
        Безразмерный относительный допуск равенства MSE.

    Returns
    -------
    candidate : dict or None
        Исходная запись с минимумом MSE; среди близких минимумов берётся
        наибольшая alpha. None при отсутствии подходящих оценок.
    """
    candidates = [r for r in path if r.get("accepted") and np.isfinite(r.get("selection_mse", np.nan))]
    if not candidates:
        return None
    best = min(r["selection_mse"] for r in candidates)
    tied = [r for r in candidates if r["selection_mse"]-best <= atol+rtol*abs(best)]
    return max(tied, key=lambda row: row["alpha"])


def final_certificate(prediction, fit_y, metric, gram, candidate, tolerance):
    """Проверить невязки условий ККТ выбранной оценки и прямой задачи.

    Parameters
    ----------
    prediction : GridPrediction or SelectedPrediction
        Прогноз с VJP и диагностикой состояния.
    fit_y : array_like, shape (n_observations,)
        Концентрации подгонки в мкг/м³.
    metric : CovarianceMetric
        Метрика ошибок в пространстве выбранных наблюдений.
    gram : ndarray, shape (n_coefficients, n_coefficients)
        Матрица квадратичного штрафа.
    candidate : dict
        Выбранные безразмерные коэффициенты, alpha и статус оптимизатора.
    tolerance : float
        Допуск масштабированной прямой невязки.

    Returns
    -------
    certificate : dict
        Показатели ККТ и реальная невязка повторного прямого решения.

    See Also
    --------
    experiments.source_comparison.inverse.kkt_certificate
    """
    from experiments.source_comparison.inverse import kkt_certificate
    point = np.asarray(candidate["coefficients"])
    prediction.invalidate()
    residual = prediction.predict(point)-fit_y
    precision = metric.apply_precision(residual)
    gd = prediction.vjp(point,precision)
    gp = candidate["alpha"]*(gram@point)
    return kkt_certificate(point,gd,gp,data_term=.5*float(residual@precision),
        penalty_term=.5*candidate["alpha"]*float(point@gram@point),
        optimizer_success=candidate["status"] == "accepted",
        forward_residual=prediction.trajectory.max_scaled_residual,
        forward_tolerance=tolerance)


def _path_list(record):
    return sorted(record.get("candidates",{}).values(),key=lambda r:r["exponent"])


def _conditions(spec, source_id, replicate):
    return [c for c in spec["conditions"] if source_id in c["sources"] and replicate in c["replicates"]]


def run_group(spec, source_id, replicate, checkpoint, backend, *, fitter=fit):
    """Восстановить источник во всех условиях одной реализации шума.

    Parameters
    ----------
    spec : dict
        Конфигурация условий и сетки регуляризации.
    source_id : str
        Идентификатор моделируемого источника.
    replicate : int
        Номер реализации случайных ошибок.
    checkpoint : Checkpoint
        Состояние расчёта для сохранения и продолжения оценок.
    backend : ProductionBackend or compatible object
        Объект с методами ``truth``, ``prediction`` и ``score_source``.
    fitter : callable, optional
        Процедура получения одной оценки при фиксированной регуляризации.

    Returns
    -------
    record : dict
        Обновлённая запись `checkpoint` с оценками и проверочными ошибками.
    """
    from experiments.source_comparison.inverse import penalty_grams
    conditions = _conditions(spec,source_id,replicate)
    by_id = {c["id"]:c for c in spec["conditions"]}
    singles = [s for s in spec.get("single_fits",[]) if source_id in s["sources"] and replicate in s["replicates"]]
    cp = checkpoint.record
    if cp["stage"] == "scored":
        checkpoint.require_sealed()
        return cp
    covariance_cache = {}

    def inputs(condition):
        key = (condition["noise"],condition["weight"])
        if key not in covariance_cache:
            try:
                covariance_cache[key] = calibrate_dense(spec,replicate,*key)
            except CalibrationFailure as error:
                covariance_cache[key] = error
        calibrated = covariance_cache[key]
        if isinstance(calibrated,Exception):
            raise calibrated
        covariance, calibration = calibrated
        rows, row_record = selected_rows(condition,spec["stream"],replicate)
        metric = restricted_metric(covariance,rows)
        truth, truth_residual = backend.truth(condition)
        if truth_residual > spec["solver"]["forward_tolerance"]:
            raise RuntimeError("Truth forward residual failed; no replacement inputs")
        generating = noise_covariance(DENSE_TIMES,spec["noise"][condition["noise"]])
        observations, panel_records = {}, {}
        for name in ("fit","selection"):
            residual, provenance = draw_residual(spec["stream"],replicate,name,generating)
            observations[name] = (truth+residual)[rows]
            panel_records[name] = provenance
        prediction = backend.prediction(condition)
        restricted = SelectedPrediction(prediction,rows)
        zero = np.zeros(prediction.basis.size)
        offset = restricted.predict(zero).copy()
        jacobian = prediction.jacobian(zero)[rows]
        white = metric.whiten_matrix(jacobian)
        information = white.T@white
        grams = penalty_grams(prediction.basis,spec["Qref"],tau_hours=condition["tau_hours"])
        provenance = dict(calibration=calibration,design=row_record,panel_records=panel_records,
            fit_y_sha256=array_hash(observations["fit"]),selection_y_sha256=array_hash(observations["selection"]),
            selected_covariance_sha256=array_hash(covariance[np.ix_(rows,rows)]),
            jacobian_sha256=array_hash(jacobian),offset_sha256=array_hash(offset),
            observation_diagnostic=observation_diagnostics(white,grams["L2"]),
            truth_forward_residual=truth_residual)
        return restricted,observations,metric,grams,jacobian,offset,information,provenance

    def attempt(args, gram, alpha, initial=None):
        prediction,observations,metric,_,jacobian,offset,_,_ = args
        try:
            options = dict(tolerance=spec["solver"]["kkt_tolerance"],
                           max_iterations=spec["solver"]["max_iterations"])
            if initial is not None:
                options["initial_point"] = initial
            result = fitter(prediction,observations["fit"],metric,gram,alpha,jacobian,offset,**options)
            result["forward_residual"] = float(prediction.trajectory.max_scaled_residual)
            result["coefficients"] = result["coefficients"].tolist()
            result["prediction"] = result["prediction"].tolist()
            result["accepted"] = bool(result["accepted"] and
                0 <= result["forward_residual"] <= spec["solver"]["forward_tolerance"])
            result["selection_mse"] = float(np.mean((np.asarray(result["prediction"])-observations["selection"])**2))
            return result
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            return dict(alpha=alpha,accepted=False,status="numerical_failure",error_type=type(error).__name__,error=str(error))

    if cp["stage"] == "estimating":
        if tuple(cp["exponents"]) != BASE_EXPONENTS:
            raise ValueError("Checkpoint does not contain the fixed common alpha grid")
        for condition in conditions:
            path_ids = [condition["id"]+"/"+arm for arm in condition["penalties"]]
            unfinished = any(any(str(e) not in cp["paths"].get(pid,{}).get("candidates",{})
                                 for e in cp["exponents"]) for pid in path_ids)
            if not unfinished:
                continue
            try:
                args = inputs(condition)
            except CalibrationFailure as error:
                for pid,arm in zip(path_ids,condition["penalties"]):
                    cp["paths"][pid] = dict(condition=condition["id"],penalty=arm,candidates={},
                        calibration_failure=str(error),finalized=False)
                checkpoint.save()
                continue
            for pid,arm in zip(path_ids,condition["penalties"]):
                gram = args[3][arm]
                scale = float(np.trace(np.linalg.solve(gram,args[6]))/len(gram))
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError("Nonpositive alpha information scale")
                record = cp["paths"].setdefault(pid,dict(condition=condition["id"],penalty=arm,
                    alpha_reference=scale,provenance=args[7],candidates={},finalized=False))
                if record.get("alpha_reference") != scale or record.get("provenance") != args[7]:
                    raise ValueError("Regenerated estimator inputs differ from checkpoint")
                for exponent in cp["exponents"]:
                    if str(exponent) in record["candidates"]:
                        continue
                    candidate = attempt(args,gram,float(scale*10.**exponent))
                    candidate["exponent"] = exponent
                    record["candidates"][str(exponent)] = candidate
                    checkpoint.save()
            del args  # Снять ссылку на входы этого условия перед переходом к следующему.
        # Итоговую диагностику выполнить после расчёта всех кандидатов.
        # Сетка регуляризации общая для всех условий.
        for condition in conditions:
            records = [cp["paths"][condition["id"]+"/"+arm] for arm in condition["penalties"]]
            if all(r.get("finalized") for r in records):
                continue
            args = None if all(r.get("calibration_failure") for r in records) else inputs(condition)
            for record in records:
                if record.get("finalized"):
                    continue
                path = _path_list(record)
                selected = select_candidate(path,atol=spec["alpha"]["tie_atol"],rtol=spec["alpha"]["tie_rtol"])
                record.update(finalized=True,accepted_count=sum(bool(r["accepted"]) for r in path),
                    candidate_count=len(path),complete_path=len(path)==len(cp["exponents"]) and all(r["accepted"] for r in path),
                    selected_exponent=None if selected is None else selected["exponent"],
                    lcurve_diagnostic=lcurve_corner(path),tuning_unresolved=selected is not None and
                        selected["exponent"] in (min(cp["exponents"]),max(cp["exponents"])))
                if selected is not None:
                    try:
                        certificate = final_certificate(args[0],args[1]["fit"],args[2],args[3][record["penalty"]],selected,
                                                        spec["solver"]["forward_tolerance"])
                    except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
                        certificate = dict(accepted=False,error_type=type(error).__name__,error=str(error))
                    record["final_certificate"] = certificate
                    record["procedure_accepted"] = bool(record["complete_path"] and certificate["accepted"] and not record["tuning_unresolved"])
                else:
                    record["procedure_accepted"] = False
                checkpoint.save()
            del args
        # Для отдельных проверок взять коэффициент регуляризации из выбранной
        # базовой оценки, без повторного выбора.
        for single in singles:
            pid = "single/"+single["id"]
            if cp["paths"].get(pid,{}).get("finalized"):
                continue
            base = cp["paths"][single["baseline_condition"]+"/"+single["penalty"]]
            record = dict(condition=single["condition"],penalty=single["penalty"],kind=single["kind"],
                          baseline=single["baseline_condition"],baseline_procedure_accepted=base["procedure_accepted"],
                          finalized=True,candidates={})
            if base.get("selected_exponent") is None:
                record.update(status="baseline_selection_unavailable",procedure_accepted=False)
            else:
                condition = by_id[single["condition"]]
                try:
                    args = inputs(condition)
                    alpha = base["candidates"][str(base["selected_exponent"])]["alpha"]
                    initial = np.zeros(len(args[3][single["penalty"]])) if single["kind"] == "zero_start" else None
                    candidate = attempt(args,args[3][single["penalty"]],alpha,initial)
                    record.update(candidate=candidate,provenance=args[7],alpha=alpha)
                    if candidate["accepted"]:
                        try:
                            certificate = final_certificate(args[0],args[1]["fit"],args[2],args[3][single["penalty"]],candidate,
                                                            spec["solver"]["forward_tolerance"])
                        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
                            certificate = dict(accepted=False,error_type=type(error).__name__,error=str(error))
                        record.update(final_certificate=certificate,procedure_accepted=bool(certificate["accepted"] and base["procedure_accepted"]))
                    else:
                        record["procedure_accepted"] = False
                    del args
                except CalibrationFailure as error:
                    record.update(calibration_failure=str(error),procedure_accepted=False)
            cp["paths"][pid] = record
            checkpoint.save()
        checkpoint.seal()
    checkpoint.require_sealed()
    scores = {}
    for pid,record in cp["paths"].items():
        candidate = record.get("candidate")
        if candidate is None and record.get("selected_exponent") is not None:
            candidate = record["candidates"][str(record["selected_exponent"])]
        if candidate is None or not candidate.get("accepted"):
            continue
        condition = by_id[record["condition"]]
        truth,_ = backend.truth(condition)
        covariance = noise_covariance(DENSE_TIMES,spec["noise"][condition["noise"]])
        test, provenance = draw_residual(spec["stream"],replicate,"test",covariance)
        prediction = backend.prediction(condition)
        predicted = prediction.predict(np.asarray(candidate["coefficients"]))
        rows,_ = selected_rows(condition,spec["stream"],replicate)
        scores[pid] = dict(backend.score_source(condition,candidate["coefficients"]),
            test_provenance=provenance,full_primary_test_rmse=float(np.sqrt(np.mean((predicted-truth-test)[PRIMARY_ROWS]**2))),
            available_test_rmse=float(np.sqrt(np.mean((predicted-truth-test)[rows]**2))),
            full_primary_noiseless_rmse=float(np.sqrt(np.mean((predicted-truth)[PRIMARY_ROWS]**2))),
            available_noiseless_rmse=float(np.sqrt(np.mean((predicted-truth)[rows]**2))),
            score_row_count=36,diagnostic_only=not record["procedure_accepted"],
            score_design=dict(primary_rows=PRIMARY_ROWS.tolist(),temporal_H=condition["temporal_H"],
                relocation_km=condition["relocation_km"],truth_gamma=condition["truth_gamma"],
                truth_grid=condition.get("truth_grid",spec["truth_grid"]),
                estimand="all36 rows of this condition's observation operator; not a common baseline target if H/geometry/truth model changes"))
        del prediction
    cp.update(scores=scores,stage="scored")
    checkpoint.save()
    return cp
