"""Диагностика ковариаций E06 по сохранённым параметрам.

Ковариации сравниваются на 36 основных координатах наблюдений.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import re

import numpy as np
from scipy.linalg import block_diag

from adrkit.config.validation import canonical_bytes, digest, strict_json
from experiments.source_comparison.calibration import array_hash, noise_covariance
from experiments.observation_sensitivity.covariance import (
    CovarianceGuards, ExponentialMixture, covariance_diagnostics, mixture_covariance,
)
from experiments.observation_sensitivity.design import (
    FULL_TIMES_HOURS, POPULATION_LENGTH_HOURS, design_from_record,
)
from experiments.observation_sensitivity.admission import scientific_configuration
from experiments.observation_sensitivity.input_binding import validate_input_binding


SCHEMA = "ym2026.observation_sensitivity.covariance_diagnostics"
_TIMES = np.asarray(FULL_TIMES_HOURS, dtype=np.float64)
_TICKS = np.arange(7, 72, 8)
_ROWS = np.concatenate([72 * station + _TICKS for station in range(4)])
_PRIMARY_TIMES = _TIMES[_TICKS]
_PRIMARY_DESIGN = dict(removed_primary_tick_indices=[], rows=_ROWS.tolist(), retained_count=36, mask="none")
for _array in (_TIMES, _TICKS, _ROWS, _PRIMARY_TIMES):
    _array.flags.writeable = False


class CovarianceAnalysisError(ValueError):
    """Неполные или несогласованные записи для диагностики ковариации."""


def _require(condition, message):
    if not condition:
        raise CovarianceAnalysisError(message)


def _same(left, right):
    return canonical_bytes(left) == canonical_bytes(right)


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _is_sha256(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _vector(value, name):
    _require(type(value) is list and len(value) == 4
             and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in value),
             f"{name} requires four positive finite real values")
    return np.asarray(value, dtype=np.float64)


def _hash_matches(matrix, expected, label):
    actual = array_hash(matrix)
    _require(_is_sha256(expected) and actual == expected, f"{label} byte hash mismatch")
    return actual


def _parameters_matrix(times, variances, lengths):
    # Дисперсии берём из записи: пересчёт через SD может изменить байты матрицы.
    
    lag = np.abs(times[:, None] - times[None, :])
    return block_diag(*[variance * np.exp(-lag / length)
                        for variance, length in zip(variances, lengths)])


def _settings(spec):
    settings = spec["calibration"]
    _require(settings["panels"] == {"calib_noise": 32}
             and _same(settings["sd_bounds"], [.001, 100.])
             and _same(settings["ell_grid_hours"], np.geomspace(1 / 60, 1.5, 33).tolist())
             and _same(settings["guards"], asdict(CovarianceGuards())),
             "Changed frozen calibration settings")
    return settings, CovarianceGuards(**settings["guards"])


def _frozen_inputs(freeze, summary, expected):
    _require(_is_sha256(expected), "Independently recorded raw freeze SHA256 required")
    _require(hashlib.sha256(canonical_bytes(freeze) + b"\n").hexdigest() == expected,
             "Raw freeze differs from the external pin")
    _require(freeze["schema"] == "ym2026.observation_sensitivity.freeze"
             and type(freeze["version"]) is int and freeze["version"] in (3, 4),
             "Unsupported freeze schema")
    _require(freeze["content_sha256"] == digest({k: v for k, v in freeze.items() if k != "content_sha256"}),
             "Freeze content digest mismatch")
    admission = freeze["admission"]
    _require(freeze["admission_sha256"] == digest(admission), "Admission digest mismatch")
    _require(type(admission.get("version")) is int and admission["version"] == freeze["version"],
             "Admission/freeze execution version mismatch")
    validate_input_binding(admission)
    design = design_from_record(admission["design"])
    _require(summary["schema"] == "ym2026.observation_sensitivity.summary"
             and type(summary["version"]) is int and summary["version"] == 1
             and summary["status"] == "complete_records", "Complete post-seal summary required")
    inputs = summary["inputs"]
    _require(inputs["freeze_file_sha256"] == expected
             and inputs["freeze_content_sha256"] == freeze["content_sha256"]
             and inputs["admission_sha256"] == freeze["admission_sha256"],
             "Summary/freeze binding mismatch")
    rows = summary["raw_outcomes"]
    _require(type(rows) is list and len(rows) == len(design["paths"]), "All admitted raw outcomes required")
    indexed = {row["path_id"]: row for row in rows}
    _require(len(indexed) == len(rows) and set(indexed) == {path.id for path in design["paths"]},
             "Missing, duplicate or unknown path identity")
    return admission, design, indexed


def _provenance(row, path, spec):
    _require(all(_same(row[name], expected) for name, expected in (
        ("source", path.source), ("replicate", path.replicate), ("condition", path.condition),
        ("penalty", path.penalty), ("noise", _json(asdict(path.noise))),
        ("stream", asdict(path.stream)), ("true_h", asdict(path.true_h)),
        ("inverse_h", asdict(path.inverse_h)))), "Outcome differs from canonical PathSpec")
    _require(row["origin"] == ("new" if path.reuse is None else "reused"), "New/reuse origin changed")
    estimate = row["estimate"]
    _require(estimate["finalized"] is True and estimate["penalty"] == path.penalty,
             "Estimator has not been finalized")
    _require(_same(estimate["driver_binding"], dict(spec_sha256=digest(spec),
             path_sha256=digest(_json(asdict(path))))), "Estimator scientific binding mismatch")
    provenance = estimate.get("e06_provenance") if path.reuse else estimate.get("provenance")
    if "calibration_failure" in estimate:
        reason = estimate["calibration_failure"]
        _require(type(reason) is str and bool(reason) and estimate["procedure_accepted"] is False
                 and estimate["candidates"] == {}, "Invalid terminal calibration failure")
        if path.reuse:
            _require(_same(provenance, {"calibration_failure": reason}), "Reused failure changed")
        return dict(status="unavailable", reason=reason)
    _require(type(provenance) is dict and _same(provenance["design"], _PRIMARY_DESIGN),
             "Covariance must use exactly the 36 primary rows")
    if path.reuse:
        for key in ("calibration", "selected_covariance_sha256", "design", "panel_records"):
            _require(_same(provenance[key], estimate["provenance"][key]), "Reused covariance provenance changed")
    _require(type(provenance["calibration"]) is dict and _is_sha256(provenance["selected_covariance_sha256"]),
             "Missing recorded covariance parameters/hash")
    return dict(status="available", provenance=provenance)


def _true_matrix(path, guards):
    noise = path.noise
    if noise.family == "station_exponential":
        matrix = noise_covariance(_TIMES, dict(type="station_exponential", station_sd=list(noise.station_sd),
                                              ell_hours=list(noise.lengths_hours)))
    else:
        mixture = ExponentialMixture(noise.fast_hours, noise.slow_hours, noise.fast_weight)
        matrix = mixture_covariance(_TIMES, list(noise.station_sd), mixture, guards=guards)
    return matrix, matrix[np.ix_(_ROWS, _ROWS)]


def _panel(panel, path, purpose, full_hash, index=0):
    codes = {"calibration": 101, "fit": 211, "selection": 307, "test": 401}
    _require(type(panel) is dict and panel.get("panel") == purpose and panel.get("generator") == "PCG64"
             and _same(panel.get("seed"), [path.stream.seed, path.stream.version, path.replicate, codes[purpose], index])
             and panel.get("covariance_sha256") == full_hash,
             "Panel does not bind the declared 288-row generating covariance/stream")
    _require(all(_is_sha256(panel.get(k)) for k in ("standard_normal_sha256", "residual_sha256")),
             "Missing panel byte identities")


def _parent_parameters(parent, path, settings, calibration, full_hash):
    _require(parent["family"] == "W03" and parent["status"] == "accepted"
             and parent["units"] == "(ug/m^3)^2"
             and parent["times_sha256"] == array_hash(_PRIMARY_TIMES)
             and parent["settings_sha256"] == digest(settings)
             and _is_sha256(parent["residuals_sha256"]), "Parent calibration metadata mismatch")
    _require(_same(calibration["calibration_shape"], [32, 4, 9]), "Calibration is not 32x4x9")
    panels = calibration["panel_records"]
    _require(type(panels) is list and len(panels) == 32, "All 32 recorded calibration panels required")
    for index, panel in enumerate(panels):
        _panel(panel, path, "calibration", full_hash, index)
    variances = _vector(parent["station_variances"], "Recorded station variances")
    lengths = _vector(parent["ell_hours"], "Recorded correlation lengths")
    bounds, grid = settings["sd_bounds"], settings["ell_grid_hours"]
    sd = np.sqrt(variances)
    _require(np.all((sd >= bounds[0]) & (sd <= bounds[1])), "Recorded SD violates calibration bounds")
    _require(all(float(length) in grid[1:-1] for length in lengths),
             "Accepted fitted correlation length is off-grid or at an endpoint")
    matrix = _parameters_matrix(_PRIMARY_TIMES, variances, lengths)
    _hash_matches(matrix, parent["covariance_sha256"], "Parent primary covariance")
    return matrix, variances, lengths, dict(
        sd_bounds=list(bounds), length_grid_bounds=[grid[0], grid[-1]],
        length_grid_indices=[grid.index(float(length)) for length in lengths],
        length_grid_endpoint=[False] * 4,
        sd_at_lower_bound=(sd == bounds[0]).tolist(), sd_at_upper_bound=(sd == bounds[1]).tolist(),
    )


def _working(path, provenance, settings, true_full, true_primary):
    record = provenance["calibration"]
    full_hash = array_hash(true_full)
    _require(record.get("family") == path.weight, "Recorded covariance family does not match the path")
    flags = None
    if path.weight == "W03":
        _require(record["status"] == "accepted", "W03 calibration is not accepted")
        parent, variances, lengths, flags = _parent_parameters(
            record["parent_coarse_calibration"], path, settings, record, full_hash)
        dense = _parameters_matrix(_TIMES, variances, lengths)
        _hash_matches(dense, record["dense_covariance_sha256"], "Dense W03 parametric lift")
        matrix = dense[np.ix_(_ROWS, _ROWS)]
        _require(np.allclose(matrix, parent, rtol=2e-12, atol=2e-14), "Dense lift differs from primary calibration")
        parameters = dict(station_variances=variances.tolist(), ell_hours=lengths.tolist(),
                          role="recorded finite-calibration parameters")
    elif path.weight == "W_exp_estimated":
        _require(record["status"] == "accepted" and record["calibration_inputs_used"] is True,
                 "Estimated covariance lacks accepted calibration provenance")
        matrix, variances, lengths, flags = _parent_parameters(
            record["parent_coarse_calibration"], path, settings, record, full_hash)
        _hash_matches(matrix, record["covariance_sha256"], "Estimated primary covariance")
        parameters = dict(station_variances=variances.tolist(), ell_hours=lengths.tolist(),
                          role="recorded finite-calibration parameters in the misspecified family")
    elif path.weight == "W_exp_population":
        variances = [float(sd * sd) for sd in path.noise.station_sd]
        lengths = [POPULATION_LENGTH_HOURS] * 4
        _require(record["status"] == "population" and record["calibration_inputs_used"] is False
                 and _same(record["station_variances"], variances) and _same(record["ell_hours"], lengths),
                 "Population reference differs from the declared known-variance optimum")
        dense = noise_covariance(_TIMES, dict(type="station_exponential", station_sd=list(path.noise.station_sd),
                                              ell_hours=lengths))
        matrix = dense[np.ix_(_ROWS, _ROWS)]
        _hash_matches(matrix, record["covariance_sha256"], "Population primary covariance")
        parameters = dict(station_variances=variances, ell_hours=lengths,
                          role="known-variance population control; no parameter estimation")
    else:
        _require(path.weight == "W_mix_oracle" and record["status"] == "oracle"
                 and record["calibration_inputs_used"] is False, "Invalid mixture oracle")
        matrix = true_primary
        _hash_matches(matrix, record["covariance_sha256"], "Oracle primary covariance")
        parameters = dict(station_variances=[float(sd * sd) for sd in path.noise.station_sd],
            fast_hours=path.noise.fast_hours, slow_hours=path.noise.slow_hours,
            fast_weight=path.noise.fast_weight, role="known synthetic generating covariance")
    _hash_matches(matrix, provenance["selected_covariance_sha256"], "Estimator raw primary covariance")
    return matrix, parameters, flags


def summarize_covariances(freeze, summary, *, expected_freeze_sha256):
    """Восстановить и сравнить общие наборы ковариаций выбранных условий.

    Parameters
    ----------
    freeze, summary : dict
        Закреплённый план E06 и полная сводка ``summarize_records``.
    expected_freeze_sha256 : str
        Независимо записанный SHA-256 канонического ``freeze.json`` с конечным LF.

    Returns
    -------
    dict
        Параметры, матричные диагностики и контрольные суммы каждого набора.
        Ковариации заданы в (мкг/м³)², SD — в мкг/м³, длины корреляции — в часах.
        Отказ калибровки сохраняется как недоступный набор. Ковариации доступны
        и при отклонённой оценке источника, если калибровка сохранена.
    """
    try:
        freeze, summary = strict_json(canonical_bytes([freeze, summary]))
        return _summarize(freeze, summary, expected_freeze_sha256)
    except CovarianceAnalysisError:
        raise
    except (KeyError, TypeError, ValueError, ArithmeticError) as error:
        raise CovarianceAnalysisError(f"Unsupported or inconsistent covariance records: {error}") from error


def _summarize(freeze, summary, expected):
    admission, design, indexed = _frozen_inputs(freeze, summary, expected)
    spec = scientific_configuration(admission["baseline"]["configuration"])
    settings, guards = _settings(spec)
    buckets = defaultdict(list)
    for path in design["paths"]:
        row = indexed[path.id]
        state = _provenance(row, path, spec)
        buckets[(path.noise, path.stream, path.replicate, path.weight)].append((path, row, state))
    cache, rows = {}, []
    for (_, _, _, weight), members in buckets.items():
        path, _, state = members[0]
        scientific_key = dict(noise=_json(asdict(path.noise)), stream=asdict(path.stream),
                              replicate=path.replicate, weight=weight)
        result = dict(id=digest(scientific_key), **scientific_key,
            member_path_ids=[p.id for p, _, _ in members], member_count=len(members),
            origin_counts={name: sum(r["origin"] == name for _, r, _ in members) for name in ("new", "reused")},
            status=state["status"], source_and_penalty_are_not_independent_calibrations=True)
        _require(all(s["status"] == state["status"] for _, _, s in members),
                 "One shared calibration has both success and failure outcomes")
        if state["status"] == "unavailable":
            _require(all(s["reason"] == state["reason"] for _, _, s in members),
                     "One shared calibration has inconsistent failure reasons")
            result.update(reason=state["reason"], diagnostics=None, parameters=None, fitted_parameter_flags=None,
                          covariance_hash_verification="unavailable: failed calibration has no recorded matrices")
            rows.append(result)
            continue
        provenance = state["provenance"]
        for _, _, other in members:
            _require(_same(other["provenance"]["calibration"], provenance["calibration"])
                     and other["provenance"]["selected_covariance_sha256"] == provenance["selected_covariance_sha256"],
                     "Shared calibration parameters/matrix hashes disagree")
        if path.noise not in cache:
            cache[path.noise] = _true_matrix(path, guards)
        true_full, true_primary = cache[path.noise]
        full_hash = array_hash(true_full)
        for member_path, row, member_state in members:
            p = member_state["provenance"]
            _require(set(p["panel_records"]) == {"fit", "selection"}, "Missing permitted residual panel provenance")
            for purpose in ("fit", "selection"):
                _panel(p["panel_records"][purpose], member_path, purpose, full_hash)
            if row["score"]["status"] == "available":
                test = row["score"]["test_provenance"]
                _require(_same(test["design"], _PRIMARY_DESIGN), "Test covariance uses another row restriction")
                _panel(test["panel_record"], member_path, "test", full_hash)
        working, parameters, flags = _working(path, provenance, settings, true_full, true_primary)
        diagnostics = covariance_diagnostics(true_primary, working, guards=guards)
        result.update(parameters=parameters, fitted_parameter_flags=flags, diagnostics=diagnostics,
            true_full_covariance_sha256=full_hash, true_primary_covariance_sha256=array_hash(true_primary),
            working_primary_covariance_sha256=array_hash(working), covariance_shape=[36, 36],
            calibration_record_sha256=digest(provenance["calibration"]),
            covariance_hash_verification="exact recorded generating/working/parent/lift hashes as applicable",
            residual_values_recomputed=False, calibration_repeated=False)
        rows.append(result)
    return dict(schema=SCHEMA, version=1, status="complete_records",
        inputs=dict(freeze_file_sha256=expected, admission_sha256=freeze["admission_sha256"],
                    accepted_summary_sha256=digest(summary)),
        units=dict(covariance="(ug/m^3)^2", lengths="hour", standard_deviation="ug/m^3", kl="dimensionless"),
        layout=dict(station_order=spec["observations"]["station_names"], dense_rows=288,
                    primary_rows=_ROWS.tolist(), primary_times_hours=_PRIMARY_TIMES.tolist(), dtype="float64"),
        definitions=dict(kl="KL(N(0,Sigma_true)||N(0,Sigma_assumed)) = .5*(tr(B)-logdet(B)-36)",
            B="L_assumed^-1 Sigma_true L_assumed^-T, with lower Cholesky L_assumed",
            whitening_spectral="||B-I||_2", whitening_frobenius="||B-I||_F"),
        counts=dict(expected_sets=len(buckets), available_sets=sum(r["status"] == "available" for r in rows),
                    unavailable_sets=sum(r["status"] == "unavailable" for r in rows), represented_paths=len(design["paths"])),
        shared_sets=rows)
