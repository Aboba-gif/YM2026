"""Синтетические записи ковариации с детерминированными параметрами.

FAKE_BYTES_HASH обозначает искусственный идентификатор: соответствующие
выборки невязок не генерируются. Матричные величины вычисляются
по заданным параметрам для проверки анализатора.
"""
from copy import deepcopy
from tests.experiments.observation_sensitivity.fixtures.paired_records import make_admission
from experiments.observation_sensitivity.input_binding import scientific_configuration

from dataclasses import asdict

import hashlib

import json

import numpy as np

import pytest

from scipy.linalg import block_diag

from adrkit.config.validation import canonical_bytes, digest

from experiments.source_comparison.calibration import array_hash, noise_covariance

from experiments.observation_sensitivity.analysis.covariance import CovarianceAnalysisError, summarize_covariances

from experiments.observation_sensitivity.covariance import CovarianceGuards, ExponentialMixture, mixture_covariance

from experiments.observation_sensitivity.design import FULL_TIMES_HOURS, POPULATION_LENGTH_HOURS, build_design

TIMES = np.array(FULL_TIMES_HOURS)

TICKS = np.arange(7, 72, 8)

ROWS = np.concatenate([72 * j + TICKS for j in range(4)])

LAYOUT = dict(removed_primary_tick_indices=[], rows=ROWS.tolist(), retained_count=36, mask="none")

FAKE_BYTES_HASH = hashlib.sha256(b"artificial absent residual bytes").hexdigest()

def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))

def pin(freeze):
    return hashlib.sha256(canonical_bytes(freeze) + b"\n").hexdigest()

def parameters_matrix(times, variances, lengths):
    lag = np.abs(times[:, None] - times[None, :])
    return block_diag(*[v * np.exp(-lag / ell) for v, ell in zip(variances, lengths)])

def true_matrix(path):
    n = path.noise
    if n.family == "station_exponential":
        return noise_covariance(TIMES, dict(type=n.family, station_sd=n.station_sd, ell_hours=n.lengths_hours))
    return mixture_covariance(TIMES, n.station_sd, ExponentialMixture(n.fast_hours, n.slow_hours, n.fast_weight))

def panel(path, name, full_hash, index=0):
    return dict(panel=name, generator="PCG64", seed=[path.stream.seed, path.stream.version,
        path.replicate, dict(calibration=101, fit=211, selection=307, test=401)[name], index],
        covariance_sha256=full_hash, standard_normal_sha256=FAKE_BYTES_HASH, residual_sha256=FAKE_BYTES_HASH)

def calibration(path, settings, true_full):
    primary = true_full[np.ix_(ROWS, ROWS)]
    if path.weight == "W_mix_oracle":
        return dict(family=path.weight, status="oracle", calibration_inputs_used=False,
                    covariance_sha256=array_hash(primary)), primary
    if path.weight == "W_exp_population":
        lengths = [POPULATION_LENGTH_HOURS] * 4
        full = noise_covariance(TIMES, dict(type="station_exponential", station_sd=path.noise.station_sd,
                                           ell_hours=lengths))
        matrix = full[np.ix_(ROWS, ROWS)]
        return dict(family=path.weight, status="population", calibration_inputs_used=False,
            station_variances=[s * s for s in path.noise.station_sd], ell_hours=lengths,
            covariance_sha256=array_hash(matrix)), matrix
    # Параметры задаются детерминированно, без калибровочной выборки.
    
    variances = [1.17, 2.33, 4.12, .93]
    lengths = [settings["ell_grid_hours"][i + path.replicate] for i in (15, 17, 19, 21)]
    matrix = parameters_matrix(TIMES[TICKS], variances, lengths)
    parent = dict(family="W03", status="accepted", units="(ug/m^3)^2",
        residuals_sha256=FAKE_BYTES_HASH, times_sha256=array_hash(TIMES[TICKS]),
        settings_sha256=digest(settings), covariance_sha256=array_hash(matrix),
        station_variances=variances, ell_hours=lengths,
        correlation_scores=[[(float(i) - settings["ell_grid_hours"].index(ell)) ** 2
                              for i in range(33)] for ell in lengths])
    result = dict(family=path.weight, status="accepted", parent_coarse_calibration=parent,
        calibration_shape=[32, 4, 9], panel_records=[panel(path, "calibration", array_hash(true_full), i)
                                                    for i in range(32)])
    if path.weight == "W03":
        dense = parameters_matrix(TIMES, variances, lengths)
        result["dense_covariance_sha256"] = array_hash(dense)
        matrix = dense[np.ix_(ROWS, ROWS)]
    else:
        result.update(calibration_inputs_used=True, covariance_sha256=array_hash(matrix))
    return result, matrix

def make_records():
    design = build_design()
    settings = dict(panels={"calib_noise": 32}, sd_bounds=[.001, 100.],
                    ell_grid_hours=np.geomspace(1 / 60, 1.5, 33).tolist(), guards=asdict(CovarianceGuards()))
    spec = dict(calibration=settings, observations=dict(station_names=["Severny", "Peschanka", "Soloncy", "KrAZ"]))
    declared = normalized(asdict(design))
    sources = {name: dict(record={"name": name}, sha256=digest({"name": name}))
               for name in ("PG10", "SB150", "EC04", "EC06", "NEW-J2", "NEW-S2")}
    admission = make_admission(spec, sources, design)
    spec = scientific_configuration(admission["baseline"]["configuration"])
    freeze = dict(schema="ym2026.observation_sensitivity.freeze", version=3,
                  admission=admission, admission_sha256=digest(admission))
    freeze["content_sha256"] = digest(freeze)
    summary = dict(schema="ym2026.observation_sensitivity.summary", version=1, status="complete_records",
        inputs=dict(freeze_file_sha256=pin(freeze), freeze_content_sha256=freeze["content_sha256"],
                    admission_sha256=freeze["admission_sha256"]), raw_outcomes=[])
    true_cache, fitted_cache, matrices = {}, {}, {}
    for path in design.paths:
        if path.noise not in true_cache:
            true_cache[path.noise] = true_matrix(path)
        full = true_cache[path.noise]
        key = (path.noise, path.replicate, path.weight)
        if key not in fitted_cache:
            fitted_cache[key] = calibration(path, settings, full)
        record, working = fitted_cache[key]
        p = dict(calibration=deepcopy(record), design=deepcopy(LAYOUT),
            panel_records={n: panel(path, n, array_hash(full)) for n in ("fit", "selection")},
            selected_covariance_sha256=array_hash(working))
        estimate = dict(finalized=True, penalty=path.penalty, procedure_accepted=True,
            driver_binding=dict(spec_sha256=digest(spec), path_sha256=digest(normalized(asdict(path)))),
            provenance=p, candidates={})
        if path.reuse:
            estimate["e06_provenance"] = deepcopy(p)
        score = dict(status="available", test_provenance=dict(design=deepcopy(LAYOUT),
                    panel_record=panel(path, "test", array_hash(full))))
        summary["raw_outcomes"].append(dict(path_id=path.id, source=path.source, replicate=path.replicate,
            condition=path.condition, penalty=path.penalty, origin="new" if path.reuse is None else "reused",
            noise=normalized(asdict(path.noise)), stream=asdict(path.stream), true_h=asdict(path.true_h),
            inverse_h=asdict(path.inverse_h), estimate=estimate, score=score))
        matrices[path.id] = (full, working)
    return freeze, summary, matrices
