"""Проверки смены оператора наблюдений, производных и повторного использования состояния."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.source_recovery.config import DENSE_TIMES
from experiments.source_recovery.config import template_config
from experiments.source_recovery.backend import Prediction, make_observations, make_solver
from adrkit.backends.cached import CachedSolver
from adrkit.loads import IntervalAverageLoad
from adrkit.spaces import ArraySpace
from experiments.observation_sensitivity.operators import ProjectedPrediction, grid_observation, temporal_weights, solver_state_space


@pytest.fixture
def solver_calls(monkeypatch):
    calls = {}
    for name in ("solve_controls", "tangent_controls", "adjoint_controls"):
        original = getattr(CachedSolver, name)

        def record(solver, *args, _name=name, _method=original):
            counts = calls.setdefault(id(solver),
                dict(solve_controls=0, tangent_controls=0, adjoint_controls=0))
            counts[_name] += 1
            return _method(solver, *args)

        monkeypatch.setattr(CachedSolver, name, record)
    return calls


def small_prediction(kind="snapshot"):
    spec = template_config(100.)
    spec["grids"]["test"] = dict(bounds_km=[-2.,2.,-2.,2.], spacing_km=.5, steps=84)
    condition = dict(nodes=7, temporal_H=kind, relocation_km=0.)
    solver = make_solver(spec, "test", spec["model"]["reaction_gamma"])

    def history(a, b):
        assert b <= 1e-12, "Future source queried"
        return 1.3*(b-a)

    disclosed = SimpleNamespace(integral=history, initial=1.3)
    base = Prediction(spec, condition, solver, disclosed)
    load = IntervalAverageLoad(base.basis, solver.times,
        origin=spec["model"]["origin_hours"], q_reference=spec["Qref"],
        history_integral=disclosed.integral, history_initial=disclosed.initial,
        source_unit="C*km^2/hour")
    state_space = solver_state_space(solver)
    row_count = base.space_weights.shape[0]*base.time_weights.shape[0]
    result_space = ArraySpace((row_count,), axes=("observation",),
                             coordinates=(np.arange(row_count),),
                             units=state_space.units, dtype=state_space.dtype)
    obs = grid_observation(base.space_weights, base.time_weights,
                           state_space=state_space, codomain=result_space)
    return base, ProjectedPrediction(solver, load, obs), spec, condition, solver


@pytest.mark.parametrize("kind", ["snapshot", "average20"])
def test_temporal_matrices_exactly_reproduce_frozen_arithmetic(kind):
    _, _, spec, condition, solver = small_prediction(kind)
    _, previous = make_observations(spec, solver, condition)
    current = temporal_weights(solver.times, DENSE_TIMES, origin_hours=-.5,
        kind="snapshot" if kind == "snapshot" else "average",
        window_hours=None if kind == "snapshot" else 1/3)
    assert np.array_equal(current, previous)


def test_average_on_nonuniform_clock_exact_for_constants_and_affine_fields():
    state_times = np.array([0., .2, .55, 1., 1.4])
    times = np.array([.25, .8])
    weights = temporal_weights(state_times, times, origin_hours=-.4,
                               kind="average", window_hours=.5)
    assert np.all(weights >= 0)
    assert np.allclose(weights @ np.ones(5), 1., rtol=0, atol=2e-15)
    field = 3*(state_times-.4)+2
    assert np.allclose(weights @ field, 3*(times-.25)+2, rtol=0, atol=2e-15)


@pytest.mark.parametrize("kwargs", [
    dict(kind="snapshot", observation_times=[.3]),
    dict(kind="average", observation_times=[.2], window_hours=.3),
    dict(kind="average", observation_times=[1.1], window_hours=.2),
    dict(kind="average", observation_times=[.5], window_hours=0),
    dict(kind="average", observation_times=[.5], window_hours=np.inf),
    dict(kind="snapshot", observation_times=[.5], window_hours=.2),
    dict(kind="snapshot", observation_times=[.5, .5]),
    dict(kind="snapshot", observation_times=[.5+1j]),
])
def test_invalid_temporal_operators_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        temporal_weights([0., .5, 1.], origin_hours=0., **kwargs)


def test_grid_observation_layout_adjoint_and_owned_weights():
    s = np.array([[1., 2., 0.], [0., 1., 3.]])
    w = np.array([[1., 0.], [.3, .7]])
    state_space = ArraySpace((2, 3), axes=("time", "node"),
                             coordinates=([0., 1.], [(0,0), (0,1), (1,0)]), units="ug/m^3")
    result_space = ArraySpace((4,), axes=("observation",),
                             coordinates=(np.arange(4),), units="ug/m^3",
                             dtype=state_space.dtype)
    obs = grid_observation(s, w, state_space=state_space, codomain=result_space)
    assert (obs.codomain.shape, obs.codomain.axes, obs.codomain.units) == (
        (4,), ("observation",), "ug/m^3")
    assert obs.codomain.coordinates == ((0, 1, 2, 3),)
    assert (obs.codomain.dtype, obs.codomain.scale, obs.codomain.pairing,
            obs.codomain.binding) == (np.dtype("float64"), 1., "euclidean", None)
    states = np.arange(6.).reshape(2, 3)
    manual = np.array([sum(w[k,t]*s[i,x]*states[t,x] for t in range(2) for x in range(3))
                       for i in range(2) for k in range(2)])
    dual = np.array([.2, -.5, 1.3, 2.])
    assert np.allclose(obs.predict(states), manual, rtol=0, atol=2e-15)
    assert np.allclose(manual @ dual, np.vdot(states, obs.vjp(states, dual)))
    s[:] = 0; w[:] = 0
    assert np.allclose(obs.predict(states), manual)


@pytest.mark.parametrize("kind", ["snapshot", "average20"])
def test_nonlinear_predictions_reproduce_existing_adapter_bitwise(kind):
    old, new, _, _, _ = small_prediction(kind)
    assert old.codomain.units == "ug/m^3"
    assert new.codomain.units == "unknown"
    assert new.codomain.shape == old.codomain.shape == (288,)
    assert new.codomain.axes == old.codomain.axes == ("observation",)
    assert new.codomain.coordinates == old.codomain.coordinates == (tuple(range(288)),)
    assert new.codomain.dtype == old.codomain.dtype == np.dtype("float64")
    assert new.codomain.scale == old.codomain.scale == 1.
    assert new.codomain.pairing == old.codomain.pairing == "euclidean"
    assert new.codomain.binding is old.codomain.binding is None
    point = np.linspace(.2, 1.1, 7)
    cotangent = np.sin(np.arange(288))
    assert np.array_equal(old.predict(point), new.predict(point))
    assert np.array_equal(old.vjp(point, cotangent), new.vjp(point, cotangent))
    assert np.array_equal(old.jacobian(point), new.jacobian(point))


def test_nonlinear_derivative_duality_and_independent_finite_difference():
    _, prediction, _, _, _ = small_prediction()
    point = np.linspace(.2, 1.1, 7)
    direction = np.cos(np.arange(7))
    cotangent = np.sin(np.arange(288))
    tangent = prediction.jvp(point, direction)
    adjoint = prediction.vjp(point, cotangent)
    assert abs(tangent @ cotangent-direction @ adjoint)/max(1., np.linalg.norm(tangent)) < 1e-10
    errors = []
    for step in (1e-3, 5e-4, 2.5e-4):
        fd = (prediction.predict(point+step*direction)-prediction.predict(point-step*direction))/(2*step)
        errors.append(np.linalg.norm(fd-tangent)/max(1., np.linalg.norm(tangent)))
    assert max(errors) < 2e-6
    assert errors[-1] < errors[0]


def test_cache_ownership_and_projection_reuse_without_changing_fit_operator(solver_calls):
    old, prediction, _, _, solver = small_prediction()
    point = np.linspace(.2, 1.1, 7)
    expected = prediction.predict(point)
    calls = next(iter(solver_calls.values()))
    returned = prediction.predict(point); returned[:] = -999
    assert np.array_equal(prediction.predict(point), expected)
    assert calls["solve_controls"] == 1
    state_space = solver_state_space(solver)
    row_count = old.space_weights.shape[0]*old.time_weights.shape[0]
    result_space = ArraySpace((row_count,), axes=("observation",),
                             coordinates=(np.arange(row_count),),
                             units=state_space.units, dtype=state_space.dtype)
    second = grid_observation(2*old.space_weights, old.time_weights,
                              state_space=state_space, codomain=result_space)
    assert second.codomain == prediction.observation.codomain
    assert second.codomain.units == "unknown"
    assert np.array_equal(prediction.project(point, second), 2*expected)
    assert calls["solve_controls"] == 1
    assert np.array_equal(prediction.predict(point), expected)
    matrices = prediction.projected_jacobians(point, [prediction.observation, second])
    assert calls["tangent_controls"] == 7
    assert np.array_equal(matrices[1], 2*matrices[0])
    assert np.array_equal(matrices[0], old.jacobian(point))
    point[0] += .1
    prediction.predict(point)
    assert calls["solve_controls"] == 2
    prediction.invalidate(); prediction.predict(point)
    assert calls["solve_controls"] == 3
