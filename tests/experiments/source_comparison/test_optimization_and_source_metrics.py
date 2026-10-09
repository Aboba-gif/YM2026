"""Проверки оптимизатора, метрик источника и диагностики L-кривой."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import cholesky
from scipy.optimize import lsq_linear

from adrkit.inverse.misfit import covariance_metric
from adrkit.spaces import ArraySpace
from adrkit.inverse.projected import fit, lcurve_corner
from experiments.source_comparison.run import source_metrics
from experiments.source_comparison.truth import FiniteRelease
from adrkit.sources import P1Basis


class AffinePrediction:
    """Аффинная модель для независимой проверки оптимизатора."""

    def __init__(self, matrix, offset):
        self.matrix, self.offset = matrix, offset

    def predict(self, point):
        return self.matrix @ point+self.offset

    def vjp(self, point, dual):
        return self.matrix.T @ dual


def test_optimizer_matches_independent_augmented_least_squares():
    rng = np.random.default_rng(771)
    matrix, data, offset = rng.normal(size=(15, 6)), rng.normal(size=15), rng.normal(size=15)
    noise = rng.normal(size=(15, 15))
    covariance = noise @ noise.T+np.eye(15)
    layout = ArraySpace((15,), axes=("observation",), coordinates=(np.arange(15),), units="C")
    metric = covariance_metric(covariance, layout=layout)
    gram = np.diag(np.arange(1, 7))
    for alpha in (.001, .3, 100.):
        model = AffinePrediction(matrix, offset)
        result = fit(model, data, metric, gram, alpha, matrix, offset)
        augmented = np.vstack((metric.whiten_matrix(matrix), np.sqrt(alpha)*cholesky(gram)))
        expected = lsq_linear(augmented, np.r_[metric.whiten(data-offset), np.zeros(6)],
                              bounds=(0, np.inf), tol=1e-13)
        assert result["accepted"]
        np.testing.assert_allclose(result["coefficients"], expected.x, rtol=1e-6, atol=1e-8)


def test_jump_metrics_integrate_at_event_boundaries():
    source = FiniteRelease(.41, .17, 600.)
    basis = P1Basis(np.linspace(0, 3, 73), time_unit="h")
    result = source_metrics(source, basis, np.zeros(73))
    assert result["relative_L2"] == 1
    assert result["relative_mass_error"] == 1
    assert result["true_mass"] == source.integral(0, 3)


def test_optimizer_explicit_start_matches_independent_solution_and_preserves_input():
    matrix = np.array([[1., .2], [.1, 2.], [1., -.5]])
    offset = np.array([.1, -.2, 0.])
    data = np.array([1., -2., .5])
    gram = np.diag([1., 3.])
    layout = ArraySpace((3,), axes=('observation',), coordinates=(np.arange(3),), units='C')
    metric = covariance_metric(np.eye(3), layout=layout)
    alpha = .3
    expected = lsq_linear(np.vstack([matrix, np.sqrt(alpha)*cholesky(gram)]),
                          np.r_[data-offset, np.zeros(2)], bounds=(0, np.inf), tol=1e-13).x
    for initial in (np.zeros(2), np.array([8., 5.])):
        before = initial.copy()
        result = fit(AffinePrediction(matrix, offset), data, metric, gram, alpha,
                     matrix, offset, initial_point=initial)
        assert result['accepted'] and result['initialization'] == 'explicit_point'
        np.testing.assert_allclose(result['coefficients'], expected, atol=1e-7, rtol=1e-6)
        np.testing.assert_array_equal(initial, before)


@pytest.mark.parametrize('initial', [[-1., 0.], [np.nan, 0.], [np.inf, 0.],
                                   [0.], [[0., 0.]], [1j, 0.], [True, False]])
def test_optimizer_rejects_invalid_explicit_start(initial):
    matrix = np.eye(2)
    layout = ArraySpace((2,), axes=('observation',), coordinates=(np.arange(2),), units='C')
    metric = covariance_metric(np.eye(2), layout=layout)
    with pytest.raises(ValueError, match='initial_point'):
        fit(AffinePrediction(matrix, np.zeros(2)), np.ones(2), metric, np.eye(2), 1.,
            matrix, np.zeros(2), initial_point=initial)


def test_failed_path_cannot_supply_lcurve_corner():
    assert lcurve_corner([{"accepted": False}]*7) is None


def test_scalar_ridge_has_no_positive_lcurve_corner():
    alphas = np.logspace(-4, 4, 81)
    path = [dict(alpha=a, residual_norm=a/(1+a), penalty_norm=1/(1+a), accepted=True)
            for a in alphas]
    # В скалярной гребневой задаче кривизна с этим знаком отрицательна.
    assert lcurve_corner(path) is None


def test_lcurve_corner_matches_independent_analytic_curvature():
    singular = np.geomspace(1., 1e-6, 80)
    data = singular**2 + 1e-3*np.where(np.arange(80) % 2 == 0, 1., -1.)
    alphas = np.geomspace(1e-16, 1e2, 1601)
    a, s2 = alphas[:, None], singular[None, :]**2
    fraction = a/(s2+a)
    solution = singular*data/(s2+a)
    residual = data*fraction

    def derivatives(z, first, second):
        norm2 = np.sum(z*z, axis=1)
        n1 = 2*np.sum(z*first, axis=1)
        n2 = 2*np.sum(first*first+z*second, axis=1)
        return np.sqrt(norm2), .5*n1/norm2, .5*(n2/norm2-(n1/norm2)**2)

    rn, dx, ddx = derivatives(residual, residual*(1-fraction),
                             residual*(1-fraction)*(1-2*fraction))
    pn, dy, ddy = derivatives(solution, -solution*fraction,
                             solution*fraction*(2*fraction-1))
    exact = (dx*ddy-dy*ddx)/(dx*dx+dy*dy)**1.5
    path = [dict(alpha=a, residual_norm=r, penalty_norm=p, accepted=True)
            for a, r, p in zip(alphas, rn, pn)]
    index = lcurve_corner(path)
    best = int(np.argmax(exact[2:-2]))+2
    assert index is not None and exact[index] > 0
    assert abs(index-best) <= 2


def test_lcurve_rejects_invalid_or_flat_paths():
    path = [dict(alpha=a, residual_norm=1., penalty_norm=1., accepted=True)
            for a in np.geomspace(.01, 100, 9)]
    assert lcurve_corner(path) is None


@pytest.mark.parametrize('size', [9, 17, 81])
def test_lcurve_rejects_nonunit_flat_paths_without_cancellation(size):
    path = [dict(alpha=a, residual_norm=2., penalty_norm=3., accepted=True)
            for a in np.geomspace(.001, 1000, size)]
    assert lcurve_corner(path) is None
    # Разность только из-за округления представления также не позволяет выбрать максимум.
    path[len(path)//2]['residual_norm'] = np.nextafter(2., np.inf)
    assert lcurve_corner(path) is None


@pytest.mark.parametrize('accepted', ['false', 'true', 1, None])
def test_lcurve_requires_explicit_boolean_acceptance(accepted):
    path = [dict(alpha=a, residual_norm=1+a, penalty_norm=1+1/a, accepted=accepted)
            for a in np.geomspace(.001, 1000, 17)]
    assert lcurve_corner(path) is None


@pytest.mark.parametrize('invalid', ['duplicate', 'zero', 'reversed', 'shuffled'])
def test_lcurve_rejects_invalid_alpha_with_valid_accepted_records(invalid):
    path = [dict(alpha=a, residual_norm=1+a, penalty_norm=1+1/a, accepted=True)
            for a in np.geomspace(.001, 1000, 17)]
    if invalid == 'duplicate':
        path[3]['alpha'] = path[2]['alpha']
    elif invalid == 'zero':
        path[3]['alpha'] = 0
    elif invalid == 'reversed':
        path.reverse()
    else:
        path[2], path[3] = path[3], path[2]
    assert lcurve_corner(path) is None


@pytest.mark.parametrize("residual,local_accepted,accepted", [
    (0., True, True), (1e-11, True, True), (1.1e-11, True, False),
    (np.nan, True, False), (0., False, False),
])
def test_source_comparison_records_actual_final_residual_and_json_arrays(
        tmp_path, monkeypatch, residual, local_accepted, accepted):
    import experiments.source_comparison.run as comparison

    source = FiniteRelease(.4, .2, 2.)
    model = SimpleNamespace(basis=P1Basis([0., 3.], time_unit="h"),
        observations=SimpleNamespace(space=object()),
        predict=lambda point: np.zeros(2), jacobian=lambda point: np.eye(2))
    settings = dict(sources=["source"], inverse_spacing_km=.5, inverse_steps=84,
                    replicates=[1], noise="noise", weights=["weight"],
                    alpha_exponents=[0.], kkt_tolerance=1e-6, max_iterations=80)
    document = dict(execution_config=dict(basis=dict(knots_hours=[0.,3.],Qref=1.,tau_hours=.25),
        model=dict(extended_start_hours=-.5), observations=dict(full_times_hours=[1., 2.]),
        calibration=dict(panels=dict(calib_noise=32))),
        selector=dict(tau_hours=.25), calibration_and_splits=dict(panels=dict(calib_noise=32)),
        solver=dict(forward=dict(scaled_residual_tolerance=1e-11)))
    protocol = SimpleNamespace(document=SimpleNamespace(to_dict=lambda: document))
    monkeypatch.setattr(comparison, "setup", lambda _: (settings, protocol, tmp_path))
    monkeypatch.setattr(comparison, "build_truth", lambda *args: SimpleNamespace(source=source))
    def prediction(binding, solver, basis, load):
        assert binding is protocol
        assert np.array_equal(solver.times, [0.,3.5])
        assert np.array_equal(basis.knots, [0.,3.]) and load.q_reference == 1.
        model.basis = basis
        return model
    monkeypatch.setattr(comparison, "Prediction", prediction)
    monkeypatch.setattr(comparison, "state_solver", lambda *args: SimpleNamespace(times=[0.,3.5]))
    monkeypatch.setattr(comparison, "check_model", lambda *args, **kwargs: (np.ones(2), {}))
    monkeypatch.setattr(comparison, "penalty_grams", lambda *args, **kwargs: {"L2": np.eye(2)})
    monkeypatch.setattr(comparison, "draw_panel",
                        lambda *args, **kwargs: SimpleNamespace(values=np.zeros(2)))
    monkeypatch.setattr(comparison, "fit_covariance", lambda *args, **kwargs:
        SimpleNamespace(covariance=np.eye(2), provenance=SimpleNamespace(to_dict=lambda: {})))
    monkeypatch.setattr(comparison, "covariance_metric", lambda *args, **kwargs:
        SimpleNamespace(whiten_matrix=lambda matrix: matrix))
    seen = []
    def estimate(prediction, *args, **options):
        assert prediction is model
        assert options == dict(tolerance=1e-6, max_iterations=80)
        seen.append("fit")
        model.trajectory = SimpleNamespace(max_scaled_residual=residual)
        return dict(alpha=1., status="accepted", accepted=local_accepted,
                    coefficients=np.array([.2, .3]), prediction=np.array([1., 2.]),
                    residual_norm=1., penalty_norm=1.)
    monkeypatch.setattr(comparison, "fit", estimate)
    saved = []
    monkeypatch.setattr(comparison, "save_json", lambda path, result: saved.append(result))
    assert comparison.run_source("unused", "source", {}) == ("source", "completed")
    assert seen == ["fit"]
    result = saved[0]
    candidate = result["cases"][0]["path"][0]
    assert candidate["accepted"] is accepted
    if np.isnan(residual):
        assert np.isnan(candidate["forward_residual"])
    else:
        assert candidate["forward_residual"] == residual
    assert candidate["coefficients"] == [.2, .3]
    assert candidate["prediction"] == [1., 2.]
    assert not {"forward_calls", "adjoint_calls"} & result.keys()


def test_source_comparison_owns_explicit_components_and_returns_only_certificate(project_root):
    from dataclasses import FrozenInstanceError
    import gc
    import json
    import weakref
    from adrkit.loads import IntervalAverageLoad
    from adrkit.observations.grid import apply_observation
    from adrkit.predictions import StateCertificate
    from experiments.source_comparison.prediction import Prediction, state_solver

    document = json.loads((project_root/
        "experiments/source_comparison/configs/protocol.json").read_text(encoding="utf-8"))
    protocol = SimpleNamespace(document=SimpleNamespace(to_dict=lambda: document))
    spec = document["execution_config"]
    settings = dict(bounds_km=[-1.,1.,-1.,1.])
    solver = state_solver(protocol, settings, .5, 84)
    basis = P1Basis(np.linspace(0.,3.,4), time_unit="h")
    source = FiniteRelease(.4,.2,2.)
    source_reference = weakref.ref(source)
    load = IntervalAverageLoad(basis, solver.times,
        origin=spec["model"]["extended_start_hours"], q_reference=spec["basis"]["Qref"],
        history_integral=source.integral, history_initial=source.value(-.5),
        source_unit="C*km^2/hour")
    prediction = Prediction(protocol, solver, basis, load)
    assert not any(hasattr(prediction,name) for name in ("source","solver","load"))
    assert prediction.trajectory is None
    del source
    gc.collect()
    assert source_reference() is None
    point, direction = np.array([.1,.2,.3,.2]), np.array([.2,-.3,.1,.4])
    independent = solver.solve_controls(load.predict(point))
    expected = apply_observation(independent.states,
        prediction.observations.space_weights, prediction.observations.time_weights)
    np.testing.assert_array_equal(prediction.predict(point), expected)
    derivative = prediction.jvp(point, direction)
    certificate = prediction.trajectory
    assert isinstance(certificate,StateCertificate) and not hasattr(certificate,"states")
    assert certificate.max_scaled_residual == independent.max_scaled_residual
    with pytest.raises(FrozenInstanceError):
        certificate.max_scaled_residual = 999.
    solver.load *= 4.
    solver.gamma *= 2.
    load.q_reference *= 3.
    prediction.invalidate()
    assert prediction.trajectory is None
    np.testing.assert_array_equal(prediction.predict(point), expected)
    np.testing.assert_array_equal(prediction.jvp(point,direction), derivative)
    np.testing.assert_array_equal(prediction.project(point,prediction.observation), expected)


@pytest.mark.parametrize("q_reference,tau_hours,panel_count",
                         [(1., .25, 32), (200., .5, 16), (10., .75, 64)])
def test_source_comparison_test_panel_changes_score_but_not_selection(
        tmp_path, monkeypatch, q_reference, tau_hours, panel_count):
    """Изменение проверочных ошибок меняет RMSE, сохраняя оценивание и выбор."""
    import experiments.source_comparison.run as comparison

    source = FiniteRelease(.4, .2, 2.)
    model = SimpleNamespace(basis=P1Basis([0., 3.], time_unit="h"),
        observations=SimpleNamespace(space=object()),
        predict=lambda point: np.zeros(2), jacobian=lambda point: np.eye(2))
    settings = dict(sources=["source"], inverse_spacing_km=.5, inverse_steps=84, replicates=[1],
        noise="noise", weights=["weight"], alpha_exponents=[-1., 0., 1.],
        kkt_tolerance=1e-6, max_iterations=80)
    document = dict(execution_config=dict(basis=dict(knots_hours=[0., 3.],
        Qref=q_reference, tau_hours=tau_hours),
        model=dict(extended_start_hours=-.5),
        observations=dict(full_times_hours=[1., 2.]),
        calibration=dict(panels=dict(calib_noise=panel_count))),
        selector=dict(tau_hours=tau_hours),
        calibration_and_splits=dict(panels=dict(calib_noise=panel_count)),
        solver=dict(forward=dict(scaled_residual_tolerance=1e-11)))
    protocol = SimpleNamespace(document=SimpleNamespace(to_dict=lambda: document))
    monkeypatch.setattr(comparison, "setup", lambda _: (settings, protocol, tmp_path))
    monkeypatch.setattr(comparison, "build_truth", lambda *args: SimpleNamespace(source=source))

    def prediction(binding, solver, basis, load):
        model.basis = basis
        return model

    monkeypatch.setattr(comparison, "Prediction", prediction)
    monkeypatch.setattr(comparison, "state_solver", lambda *args: SimpleNamespace(times=[0., 3.5]))
    monkeypatch.setattr(comparison, "check_model", lambda *args, **kwargs: (np.ones(2), {}))
    def grams(basis, scale, *, tau_hours):
        assert scale == q_reference
        assert tau_hours == document["selector"]["tau_hours"]
        return {"L2": np.eye(2)}
    monkeypatch.setattr(comparison, "penalty_grams", grams)
    test_values = np.zeros(2)

    def panel(*args, panel, **kwargs):
        values = {"calib_noise": np.zeros(2), "fit_obs": np.array([.4, -.2]),
                  "select_obs": np.array([.25, -.25]), "test_obs": test_values}[panel]
        return SimpleNamespace(values=values.copy())

    calibration_inputs, fit_inputs, saved = [], [], []

    def calibration(values, *args, **kwargs):
        calibration_inputs.append(values.copy())
        return SimpleNamespace(covariance=np.eye(2), provenance=SimpleNamespace(to_dict=lambda: {}))

    def estimate(prediction, observations, metric, gram, alpha, jacobian, offset, **options):
        assert options == dict(tolerance=1e-6, max_iterations=80)
        fit_inputs.append(dict(observations=observations.copy(), gram=gram.copy(),
            alpha=alpha, jacobian=jacobian.copy(), offset=offset.copy()))
        model.trajectory = SimpleNamespace(max_scaled_residual=0.)
        value = float(round(np.log10(alpha)) + 1)
        return dict(alpha=alpha, status="accepted", accepted=True,
            coefficients=np.full(2, value), prediction=np.full(2, value),
            residual_norm=1. + alpha, penalty_norm=1. / (1. + alpha))

    monkeypatch.setattr(comparison, "draw_panel", panel)
    monkeypatch.setattr(comparison, "fit_covariance", calibration)
    monkeypatch.setattr(comparison, "covariance_metric", lambda *args, **kwargs:
        SimpleNamespace(whiten_matrix=lambda matrix: matrix))
    monkeypatch.setattr(comparison, "fit", estimate)
    monkeypatch.setattr(comparison, "save_json", lambda path, result: saved.append(result))
    for test_values in (np.zeros(2), np.array([100., -50.])):
        assert comparison.run_source("unused", "source", {}) == ("source", "completed")

    assert len(fit_inputs) == 6 and len(calibration_inputs) == 2
    assert [row["alpha"] for row in fit_inputs[:3]] == [.1, 1., 10.]
    np.testing.assert_array_equal(calibration_inputs[0], np.zeros((panel_count, 2)))
    assert all(result["q_reference"] == q_reference for result in saved)
    np.testing.assert_array_equal(calibration_inputs[0], calibration_inputs[1])
    for first, second in zip(fit_inputs[:3], fit_inputs[3:]):
        np.testing.assert_array_equal(first["observations"], [1.4, .8])
        for key in ("observations", "gram", "jacobian", "offset"):
            np.testing.assert_array_equal(first[key], second[key])
        assert first["alpha"] == second["alpha"]
    first, second = [result["cases"][0] for result in saved]
    assert first["path"] == second["path"]
    assert first["selected"] == second["selected"] == 1
    np.testing.assert_array_equal([row["validation_mse"] for row in first["path"]],
                                  [1.0625, .0625, 1.0625])
    assert first["path"][1]["alpha"] == 1.
    assert first["path"][1]["prediction"] == [1., 1.]
    assert first["metrics"]["test_rmse"] == 0.
    assert second["metrics"]["test_rmse"] == pytest.approx(np.sqrt(6250.))
    assert first["metrics"]["noiseless_prediction_rmse"] == second["metrics"]["noiseless_prediction_rmse"] == 0.
    assert {k: v for k, v in first["metrics"].items() if k != "test_rmse"} == {
        k: v for k, v in second["metrics"].items() if k != "test_rmse"}


@pytest.mark.parametrize("q_reference", [100., 200.])
def test_source_metrics_use_the_explicit_physical_scale(q_reference):
    source = FiniteRelease(0., 3., q_reference)
    basis = P1Basis([0., 1., 3.], time_unit="h")
    result = source_metrics(source, basis, np.ones(3), q_reference=q_reference)
    assert result["relative_L2"] == pytest.approx(0., abs=1e-14)
    assert result["relative_mass_error"] == pytest.approx(0., abs=1e-14)
    assert result["estimated_mass"] == result["true_mass"] == 3. * q_reference


@pytest.mark.parametrize("q_reference", [True, 0., -1., np.nan, np.inf])
def test_source_metrics_reject_invalid_physical_scale(q_reference):
    with pytest.raises(ValueError, match="q_reference"):
        source_metrics(FiniteRelease(0., 3., 100.),
                       P1Basis([0., 3.], time_unit="h"), np.ones(2),
                       q_reference=q_reference)
