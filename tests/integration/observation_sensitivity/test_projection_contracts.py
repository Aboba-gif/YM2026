"""Проверки композиции прогноза и контрактов пространств на малых задачах ADR."""
from copy import copy

import numpy as np
import pytest

from adrkit.backends.bounded import BoundedSolver
from adrkit.coefficients import ADRUnits
from adrkit.loads import IntervalAverageLoad
from adrkit.observations.grid import GridObservation
from adrkit.sources import P1Basis
from adrkit.spaces import ArraySpace
from experiments.observation_sensitivity.operators import ProjectedPrediction, temporal_weights


@pytest.fixture
def solver_calls(monkeypatch):
    calls = {}
    for name in ("solve_controls", "tangent_controls", "adjoint_controls"):
        original = getattr(BoundedSolver, name)

        def record(solver, *args, _name=name, _method=original):
            counts = calls.setdefault(id(solver),
                dict(solve_controls=0, tangent_controls=0, adjoint_controls=0))
            counts[_name] += 1
            return _method(solver, *args)

        monkeypatch.setattr(BoundedSolver, name, record)
    return calls


def _problem():
    units = ADRUnits("km", "h", "ug/m^3")
    solver = BoundedSolver(
        dict(diffusion=.3, velocity=[.2, -.1], linear_loss=.15,
             reaction_gamma=.4, reaction_c_star=.7, horizon=1.,
             source_position=[0., 0.], background=0., units=units),
        interior_points=3, time_steps=10, domain=[-1., 1., -1., 1.],
    )
    load = IntervalAverageLoad(
        P1Basis([0., .4, 1.], time_unit="h"), solver.times,
        origin=0., q_reference=1.7,
        history_integral=lambda a, b: 0., history_initial=0.,
        source_unit=units.source,
    )
    space = ArraySpace((11, 9), axes=("time", "node"),
                       coordinates=(solver.times, solver.xy),
                       units=units.concentration)
    # Для диагностических действий H_theta добавлен приёмник со знаком.
    spatial = np.array([[.1, .2, .3, .2, .5, .1, 0., .1, .3],
                        [-.1, .2, 0., -.2, .4, .1, .1, 0., -.3]])
    snapshots = temporal_weights(solver.times, [.3, .8, 1.],
                                 origin_hours=0., kind="snapshot")
    averages = temporal_weights(solver.times, [.3, .8, 1.],
                                origin_hours=0., kind="average", window_hours=.25)
    output = ArraySpace((6,), axes=("observation",),
                        coordinates=(tuple((i, t) for i in ("A", "B")
                                           for t in (.3, .8, 1.)),),
                        units=units.concentration)
    snapshot = GridObservation(spatial, snapshots, domain=space, codomain=output)
    average = GridObservation(1.2*spatial, averages, domain=space, codomain=output)
    return solver, load, snapshot, average


def _modified_space(space, **changes):
    kwargs = dict(axes=space.axes, coordinates=space.coordinates,
                  units=space.units, scale=space.scale, pairing=space.pairing,
                  dtype=space.dtype, binding=space.binding)
    kwargs.update(changes)
    return ArraySpace(space.shape, **kwargs)


@pytest.mark.parametrize("defect", ["load_clock", "load_units", "state_clock",
                                   "state_geometry", "state_units", "state_dtype"])
def test_constructor_rejects_equal_shapes_with_incompatible_meaning(defect):
    solver, load, observation, _ = _problem()
    if defect.startswith("load_"):
        load = copy(load)
        changes = ({"coordinates": (solver.times+.05,)}
                   if defect == "load_clock" else {"units": "kg/s"})
        load.codomain = _modified_space(load.codomain, **changes)
    else:
        changes = {
            "state_clock": {"coordinates": (solver.times+.05, solver.xy)},
            "state_geometry": {"coordinates": (solver.times, solver.xy[::-1])},
            "state_units": {"units": "kg/m^3"},
            "state_dtype": {"dtype": np.float32},
        }[defect]
        state = _modified_space(observation.domain, **changes)
        output = _modified_space(observation.codomain, dtype=state.dtype)
        observation = GridObservation(observation.space_weights,
                                      observation.time_weights,
                                      domain=state, codomain=output)
    with pytest.raises(ValueError):
        ProjectedPrediction(solver, load, observation)


def test_cache_owns_bound_solver_load_and_has_no_public_writable_states():
    solver, load, observation, _ = _problem()
    prediction = ProjectedPrediction(solver, load, observation)
    point = np.array([.2, .9, .5])
    direction = np.array([.4, -.1, .7])
    expected = prediction.predict(point)
    derivative = prediction.jvp(point, direction)
    # Проверяем привязку после изменения входных объектов и сброса кеша.
    
    solver.load *= 4.
    solver.gamma *= 2.
    load.q_reference *= 3.
    prediction.invalidate()
    np.testing.assert_array_equal(prediction.predict(point), expected)
    np.testing.assert_array_equal(prediction.jvp(point, direction), derivative)
    certificate = prediction.trajectory
    assert np.isfinite(certificate.max_scaled_residual)
    assert not hasattr(certificate, "states")
    with pytest.raises((AttributeError, TypeError)):
        certificate.max_scaled_residual = 999.
    np.testing.assert_array_equal(prediction.project(point, observation), expected)


def test_bound_fit_observation_cannot_be_replaced_behind_the_cache():
    solver, load, snapshot, average = _problem()
    prediction = ProjectedPrediction(solver, load, snapshot)
    point = np.array([.2, .9, .5])
    expected = prediction.predict(point)
    with pytest.raises(AttributeError):
        prediction.observation = average
    np.testing.assert_array_equal(prediction.predict(point), expected)
    np.testing.assert_array_equal(prediction.project(point, snapshot), expected)


def test_cross_projection_rejects_other_geometry_before_any_state_solve(solver_calls):
    solver, load, snapshot, _ = _problem()
    prediction = ProjectedPrediction(solver, load, snapshot)
    wrong = GridObservation(
        snapshot.space_weights, snapshot.time_weights,
        domain=_modified_space(snapshot.domain,
                               coordinates=(solver.times, solver.xy[::-1])),
        codomain=snapshot.codomain,
    )
    point = np.array([.2, .9, .5])
    with pytest.raises(ValueError):
        prediction.project(point, wrong)
    with pytest.raises(ValueError):
        prediction.projected_jacobians(point, [snapshot, wrong])
    assert not solver_calls


class _ExponentialLoad:
    """Тестовая композиция базовой нагрузки с поэлементной экспонентой.

    JVP и VJP учитывают производную экспоненты по правилу цепочки.
    """

    def __init__(self, base):
        self.base = base
        self.domain, self.codomain = base.domain, base.codomain

    def predict(self, point):
        point = self.domain.array(point)
        return self.base.predict(np.exp(point))

    def jvp(self, point, direction):
        point = self.domain.array(point)
        direction = self.domain.array(direction)
        return self.base.jvp(np.exp(point), np.exp(point)*direction)

    def vjp(self, point, cotangent):
        point = self.domain.array(point)
        return np.exp(point)*self.base.vjp(np.exp(point), cotangent)


@pytest.mark.parametrize("point", [
    np.array([-.7, -.2, -.4]),
    np.array([.3, -.4, .2]),
])
def test_true_H_and_shared_jacobians_match_independent_nonlinear_solves(point, solver_calls):
    solver, linear_load, snapshot, average = _problem()
    load = _ExponentialLoad(linear_load)
    prediction = ProjectedPrediction(solver, load, snapshot)
    
    prediction.predict(np.array([.15, .25, .05]))
    calls = next(iter(solver_calls.values()))
    jacobians = prediction.projected_jacobians(point, [snapshot, average])
    assert calls["solve_controls"] == 2
    assert calls["tangent_controls"] == 3
    assert not np.array_equal(jacobians[0], jacobians[1])
    reference = solver.solve_controls(load.predict(point))
    np.testing.assert_array_equal(prediction.project(point, average),
                                  average.predict(reference.states))
    np.testing.assert_array_equal(prediction.predict(point),
                                  snapshot.predict(reference.states))
    assert calls["solve_controls"] == 2
    step = 2e-5
    for j, direction in enumerate(np.eye(3)):
        plus = solver.solve_controls(load.predict(point+step*direction)).states
        minus = solver.solve_controls(load.predict(point-step*direction)).states
        for observation, jacobian in zip((snapshot, average), jacobians):
            finite_difference = (observation.predict(plus)-observation.predict(minus))/(2*step)
            np.testing.assert_allclose(jacobian[:, j], finite_difference,
                                       rtol=2e-7, atol=2e-9)
    tangent_direction = np.array([.3, -.4, .8])
    cotangent = np.array([.1, -.3, .2, .4, -.2, .5])
    pullback = prediction.vjp(point, cotangent)
    np.testing.assert_allclose(pullback, jacobians[0].T @ cotangent,
                               rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(prediction.jvp(point, tangent_direction),
                               jacobians[0] @ tangent_direction,
                               rtol=2e-11, atol=2e-12)


def test_partial_nonuniform_averages_match_piecewise_affine_midpoint_integrals():
    states = np.array([0., .11, .4, .46, .91, 1.3])
    physical = states-.2
    values = np.array([2., -.3, .8, 4., -.7, 1.])
    times, width = np.array([.19, .61, 1.02]), .37
    weights = temporal_weights(states, times, origin_hours=-.2,
                               kind="average", window_hours=width)
    expected = []
    for end in times:
        start = end-width
        splits = np.r_[start, physical[(physical > start) & (physical < end)], end]
        mids = (splits[:-1]+splits[1:])/2
        # Правило средней точки точно для аффинной функции на каждом участке.
        
        expected.append(np.sum(np.diff(splits)*np.interp(mids, physical, values))/width)
    np.testing.assert_allclose(weights @ values, expected, rtol=0, atol=4e-15)
    np.testing.assert_allclose(weights.sum(axis=1), 1., rtol=0, atol=2e-15)
