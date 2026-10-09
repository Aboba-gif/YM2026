"""Проверки генерации шума, рабочих ковариаций и кэшей выборок E06."""
import experiments.source_recovery.config as recovery_config
import experiments.source_recovery.panels as recovery_panels
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import block_diag

from adrkit.config.validation import JSONRecord
from experiments.source_comparison.calibration import CalibrationFailure, array_hash, fit_covariance, noise_covariance

from experiments.observation_sensitivity import data
from experiments.observation_sensitivity.design import NoiseSpec, POPULATION_LENGTH_HOURS, StreamSpec, build_design


def _path(condition="spatial_G1_matched", replicate=1, source="PG10", penalty="L2"):
    return next(p for p in build_design().paths
                if (p.condition, p.replicate, p.source, p.penalty)
                == (condition, replicate, source, penalty))


def _signal():
    return .2*np.arange(288)+np.sin(np.arange(288)/7)


def _true_mixture(path):
    noise = path.noise
    lags = np.abs(recovery_config.DENSE_TIMES[:, None]-recovery_config.DENSE_TIMES[None, :])
    corr = (noise.fast_weight*np.exp(-lags/noise.fast_hours)
            + (1-noise.fast_weight)*np.exp(-lags/noise.slow_hours))
    return block_diag(*[sd*sd*corr for sd in noise.station_sd])


@pytest.mark.parametrize("replicate", [1, 2, 3, 4])
def test_W03_inputs_metric_and_required_provenance_are_bitwise_v2(replicate):
    spec = recovery_config.template_config(100.)
    path = _path(replicate=replicate)
    signal = _signal()
    result = data.PanelFactory(spec).estimation(path, signal)
    dense, calibration = recovery_panels.calibrate_dense(spec, replicate, "corr", "W03")
    generating = noise_covariance(recovery_config.DENSE_TIMES, spec["noise"]["corr"])
    records = {}
    for name in ("fit", "selection"):
        residual, records[name] = recovery_panels.draw_residual(spec["stream"], replicate, name, generating)
        expected = (signal+residual)[recovery_config.PRIMARY_ROWS]
        np.testing.assert_array_equal(getattr(result, name), expected)
    metric = recovery_panels.restricted_metric(dense, recovery_config.PRIMARY_ROWS)
    np.testing.assert_array_equal(result.selected_covariance, dense[np.ix_(recovery_config.PRIMARY_ROWS, recovery_config.PRIMARY_ROWS)])
    assert not result.selected_covariance.flags.writeable
    assert result.selected_covariance.flags.owndata
    assert result.metric.covariance_factor.tobytes(order="C") == metric.covariance_factor.tobytes(order="C")
    np.testing.assert_array_equal(result.metric.whiten(result.fit), metric.whiten(result.fit))
    provenance = result.provenance.to_dict()
    assert provenance["calibration"] == calibration
    assert provenance["panel_records"] == records
    assert provenance["fit_y_sha256"] == array_hash(result.fit)
    assert provenance["selection_y_sha256"] == array_hash(result.selection)
    assert provenance["selected_covariance_sha256"] == array_hash(
        dense[np.ix_(recovery_config.PRIMARY_ROWS, recovery_config.PRIMARY_ROWS)])


def test_generation_is_dense_then_primary_not_cholesky_of_36_rows():
    spec = recovery_config.template_config(100.)
    path = _path()
    result = data.PanelFactory(spec).estimation(path, np.zeros(288))
    generating = noise_covariance(recovery_config.DENSE_TIMES, spec["noise"]["corr"])
    z, _ = recovery_panels.standard_panel(spec["stream"], 1, "fit")
    expected = (np.linalg.cholesky(generating) @ z.ravel())[recovery_config.PRIMARY_ROWS]
    wrong = (np.linalg.cholesky(generating[np.ix_(recovery_config.PRIMARY_ROWS, recovery_config.PRIMARY_ROWS)])
             @ z.ravel()[recovery_config.PRIMARY_ROWS])
    np.testing.assert_array_equal(result.fit, expected)
    assert not np.allclose(result.fit, wrong)


def test_panel_codes_are_separate_and_test_is_generated_only_on_explicit_call(monkeypatch):
    calls = []
    original = recovery_panels.standard_panel

    def spy(stream, replicate, panel, index=0):
        result = original(stream, replicate, panel, index)
        calls.append((panel, index, result[1]["standard_normal_sha256"], result[1]["seed"]))
        return result

    monkeypatch.setattr(recovery_panels, "standard_panel", spy)
    factory = data.PanelFactory(recovery_config.template_config(100.))
    result = factory.estimation(_path(), _signal())
    assert len(calls) == 34
    assert {(name, index) for name, index, _, _ in calls} == (
        {("calibration", i) for i in range(32)} | {("fit", 0), ("selection", 0)})
    assert len({r[2] for r in calls}) == 34
    assert {r[3][3] for r in calls} == {101, 211, 307}
    assert not hasattr(result, "test")
    assert {f.name for f in fields(result)} == {"fit", "selection", "metric", "selected_covariance", "provenance"}
    test = factory.test(_path(), _signal())
    assert calls[-1][0] == "test" and calls[-1][3][3] == 401
    assert len({r[2] for r in calls}) == 35
    assert test.provenance.to_dict()["test_y_sha256"] == array_hash(test.values)
    assert not np.array_equal(test.values, result.fit)
    assert not np.array_equal(test.values, result.selection)


def test_cache_excludes_source_true_H_inverse_H_penalty_and_signal(monkeypatch):
    calls = []
    original = recovery_panels.calibrate_dense

    def spy(*args, **kwargs):
        calls.append(args[1:])
        return original(*args, **kwargs)

    monkeypatch.setattr(recovery_panels, "calibrate_dense", spy)
    factory = data.PanelFactory(recovery_config.template_config(100.))
    signal = _signal()
    paths = [_path(), _path("spatial_G2_matched", source="EC04", penalty="H1"),
             _path("spatial_C1_assumed_G1"), _path("temporal_average_matched"),
             _path("temporal_average_assumed_snapshot")]
    results = [factory.estimation(p, signal) for p in paths]
    assert len(calls) == 1
    for result in results[1:]:
        np.testing.assert_array_equal(result.fit, results[0].fit)
        np.testing.assert_array_equal(result.selection, results[0].selection)
        assert result.provenance.to_dict() == results[0].provenance.to_dict()
    changed = signal.copy()
    hidden = np.ones(288, dtype=bool)
    hidden[recovery_config.PRIMARY_ROWS] = False
    changed[hidden] = 1e12
    unchanged = factory.estimation(paths[-1], changed)
    np.testing.assert_array_equal(unchanged.fit, results[0].fit)
    assert unchanged.provenance.to_dict() == results[0].provenance.to_dict()
    shifted = factory.estimation(paths[0], signal+17.)
    np.testing.assert_allclose(shifted.fit-results[0].fit, 17., rtol=0, atol=2e-14)
    assert shifted.metric.covariance_factor.tobytes(order="C") == results[0].metric.covariance_factor.tobytes(order="C")
    assert len(calls) == 1
    factory.estimation(_path(replicate=2), signal)
    assert len(calls) == 2


def test_cache_key_includes_complete_noise_parameters():
    factory = data.PanelFactory(recovery_config.template_config(100.))
    original = _path()
    doubled = replace(original, noise=NoiseSpec(
        "station_exponential", tuple(2*s for s in original.noise.station_sd),
        original.noise.lengths_hours))
    first = factory.estimation(original, np.zeros(288))
    second = factory.estimation(doubled, np.zeros(288))
    np.testing.assert_array_equal(second.fit, 2*first.fit)
    np.testing.assert_array_equal(second.selection, 2*first.selection)
    np.testing.assert_allclose(second.metric.covariance, 4*first.metric.covariance,
                               rtol=2e-14, atol=2e-14)
    assert second.metric.covariance_factor.tobytes(order="C") != first.metric.covariance_factor.tobytes(order="C")


def test_factory_reads_calibration_settings_without_reading_source_parameters():
    settings = recovery_config.template_config(100.)["calibration"]

    class CalibrationOnly:
        def __getitem__(self, key):
            assert key == "calibration", "truth/source settings were consulted"
            return settings

    result = data.PanelFactory(CalibrationOnly()).estimation(_path(), _signal())
    assert result.fit.shape == result.selection.shape == (36,)


def test_mixture_three_weights_share_arrays_but_isolate_family_and_estimation(monkeypatch):
    spec = recovery_config.template_config(100.)
    factory = data.PanelFactory(spec)
    signal = _signal()
    oracle = _path("covariance_mix_oracle")
    population = _path("covariance_exp_population")
    estimated = _path("covariance_exp_estimated")
    seen = []
    original = data.fit_covariance

    def spy(residuals, times, *, family, specification):
        seen.append((residuals.copy(), times.copy(), family, specification))
        return original(residuals, times, family=family, specification=specification)

    monkeypatch.setattr(data, "fit_covariance", spy)
    results = [factory.estimation(path, signal) for path in (oracle, population)]
    assert not seen  # Эти два веса заданы популяционно.
                     
    results.append(factory.estimation(estimated, signal))
    assert len(seen) == 1
    e, times, family, settings = seen[0]
    assert e.shape == (32, 4, 9) and family == "W03"
    np.testing.assert_array_equal(times, recovery_config.DENSE_TIMES[recovery_config.PRIMARY_TICKS])
    assert settings == spec["calibration"]
    true_cov = _true_mixture(oracle)
    stream3 = dict(seed=20260926, version=3)
    expected_panels = np.stack([
        recovery_panels.draw_residual(stream3, 1, "calibration", true_cov, index)[0]
        .reshape(4, 72)[:, recovery_config.PRIMARY_TICKS] for index in range(32)])
    np.testing.assert_array_equal(e, expected_panels)
    ref_fit = fit_covariance(expected_panels, times, family="W03", specification=settings)
    population_cov = noise_covariance(recovery_config.DENSE_TIMES, dict(
        type="station_exponential", station_sd=list(population.noise.station_sd),
        ell_hours=[POPULATION_LENGTH_HOURS]*4))
    expected_covariances = [
        true_cov[np.ix_(recovery_config.PRIMARY_ROWS, recovery_config.PRIMARY_ROWS)],
        population_cov[np.ix_(recovery_config.PRIMARY_ROWS, recovery_config.PRIMARY_ROWS)],
        ref_fit.covariance,
    ]
    for result, cov in zip(results, expected_covariances):
        assert result.provenance.to_dict()["selected_covariance_sha256"] == array_hash(cov)
        np.testing.assert_allclose(result.metric.covariance, cov, rtol=2e-15, atol=2e-15)
        np.testing.assert_array_equal(result.fit, results[0].fit)
        np.testing.assert_array_equal(result.selection, results[0].selection)
        assert result.provenance.to_dict()["panel_records"] == results[0].provenance.to_dict()["panel_records"]
    assert len({r.metric.covariance_factor.tobytes(order="C") for r in results}) == 3
    cal = results[2].provenance.to_dict()["calibration"]
    assert cal["parent_coarse_calibration"] == ref_fit.provenance.to_dict()
    assert cal["calibration_shape"] == [32, 4, 9]
    assert len({r["standard_normal_sha256"] for r in cal["panel_records"]}) == 32
    z3 = results[0].provenance.to_dict()["panel_records"]["fit"]["standard_normal_sha256"]
    _, z2 = recovery_panels.standard_panel(dict(seed=20260926, version=2), 1, "fit")
    assert z3 != z2["standard_normal_sha256"]
    # Другой сигнал не должен поступать в подгонку калибровки или заменять её кеш.
    changed = factory.estimation(estimated, np.linspace(-1000., 2000., 288))
    assert len(seen) == 1
    assert changed.metric.covariance_factor.tobytes(order="C") == results[2].metric.covariance_factor.tobytes(order="C")


@pytest.mark.parametrize("condition", ["spatial_G1_matched", "covariance_exp_estimated"])
@pytest.mark.parametrize("kind, message", [("zero", "SD outside bounds"),
                                           ("constant_time", "grid endpoint")])
def test_actual_calibration_failure_is_terminal_without_resampling(monkeypatch, condition, kind, message):
    calls = []
    original = recovery_panels.draw_residual

    def zero_calibration(stream, replicate, panel, covariance, index=0):
        calls.append((panel, index))
        if panel == "calibration":
            values = (np.zeros(288) if kind == "zero" else
                      np.repeat([1., 1.5, 2., 1.], 72)*(-1.)**index)
            return values, dict(panel=panel, index=index, synthetic_test_kind=kind)
        return original(stream, replicate, panel, covariance, index)

    monkeypatch.setattr(recovery_panels, "draw_residual", zero_calibration)
    factory = data.PanelFactory(recovery_config.template_config(100.))
    path = _path(condition)
    for signal in (_signal(), np.zeros(288)):
        with pytest.raises(CalibrationFailure, match=message) as error:
            factory.estimation(path, signal)
        assert error.value.status == "unidentifiable_calibration"
    assert calls == [("calibration", index) for index in range(32)]


def test_selected_covariance_guard_rejects_without_repair_and_is_cached(monkeypatch):
    calls = []

    def rejected_fit(residuals, times, **kwargs):
        calls.append(residuals.shape)
        return SimpleNamespace(covariance=np.zeros((36, 36)),
                               provenance=JSONRecord(dict(synthetic_test_fit=True)))

    monkeypatch.setattr(data, "fit_covariance", rejected_fit)
    factory = data.PanelFactory(recovery_config.template_config(100.))
    for penalty in ("L2", "H1"):
        with pytest.raises(CalibrationFailure, match="zero covariance"):
            factory.estimation(_path("covariance_exp_estimated", penalty=penalty), _signal())
    assert calls == [(32, 4, 9)]


def test_owned_settings_outputs_and_provenance_do_not_mutate_caches():
    spec = recovery_config.template_config(100.)
    factory = data.PanelFactory(spec)
    spec["calibration"]["ell_grid_hours"][:] = [99.]
    spec["calibration"]["guards"]["cond2_max"] = 1.
    path, signal = _path(), _signal()
    result = factory.estimation(path, signal)
    expected = result.fit.copy()
    expected_covariance = result.selected_covariance.copy()
    signal[:] = -999.
    with pytest.raises(ValueError):
        result.fit[:] = 0.
    with pytest.raises(ValueError):
        result.selection[:] = 0.
    with pytest.raises(ValueError):
        result.selected_covariance[:] = 0.
    # Вызывающая сторона может сделать свои массивы доступными для записи; это не должно повреждать
    # кеш.
    result.fit.flags.writeable = True
    result.fit[:] = -999.
    result.selected_covariance.flags.writeable = True
    result.selected_covariance[:] = -999.
    record = result.provenance.to_dict()
    record["calibration"].clear()
    again = factory.estimation(path, _signal())
    np.testing.assert_array_equal(again.fit, expected)
    np.testing.assert_array_equal(again.selected_covariance, expected_covariance)
    assert not np.shares_memory(again.selected_covariance, result.selected_covariance)
    assert again.selected_covariance.shape == (36, 36)
    assert again.provenance.to_dict()["calibration"]["status"] == "accepted"
    test = factory.test(path, _signal())
    with pytest.raises(ValueError):
        test.values[:] = 0.


@pytest.mark.parametrize("bad", [
    np.zeros(36), np.zeros((4, 72)), np.full(288, np.nan),
    np.zeros(288, dtype=complex), np.zeros(288, dtype=bool), ["0"]*288,
])
def test_signal_validation_precedes_random_generation(monkeypatch, bad):
    monkeypatch.setattr(recovery_panels, "draw_residual", lambda *a, **k: pytest.fail("noise was generated"))
    factory = data.PanelFactory(recovery_config.template_config(100.))
    with pytest.raises(ValueError):
        factory.estimation(_path(), bad)
    with pytest.raises(ValueError):
        factory.test(_path(), bad)


@pytest.mark.parametrize("mutation", ["panels", "grid", "bounds", "guards"])
def test_protocol_calibration_settings_cannot_be_silently_changed(mutation):
    spec = recovery_config.template_config(100.)
    if mutation == "panels":
        spec["calibration"]["panels"]["calib_noise"] = 31
    elif mutation == "grid":
        spec["calibration"]["ell_grid_hours"][10] *= 1.001
    elif mutation == "bounds":
        spec["calibration"]["sd_bounds"][0] = .0001
    else:
        spec["calibration"]["guards"]["cond2_max"] = 1e12
    with pytest.raises(ValueError, match="unchanged v2"):
        data.PanelFactory(spec)


@pytest.mark.parametrize("path", [
    replace(_path(), stream=StreamSpec(20260926, 3)),
    replace(_path("covariance_mix_oracle"), stream=StreamSpec(20260926, 2)),
    replace(_path(), weight="W_exp_estimated"),
    replace(_path("covariance_mix_oracle"), weight="W03"),
])
def test_incompatible_noise_weight_or_stream_fails_before_generation(monkeypatch, path):
    monkeypatch.setattr(recovery_panels, "draw_residual", lambda *a, **k: pytest.fail("noise was generated"))
    factory = data.PanelFactory(recovery_config.template_config(100.))
    with pytest.raises(ValueError):
        factory.estimation(path, _signal())
