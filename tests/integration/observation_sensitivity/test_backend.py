"""Проверки проекций, кеша и ошибок источника на малой сетке ADR."""
from dataclasses import FrozenInstanceError
import gc
import weakref

import numpy as np
import pytest

from adrkit.backends.cached import CachedSolver
from adrkit.config.validation import JSONRecord
from experiments.source_comparison.truth import BiExponential, FiniteRelease, interval_controls
from adrkit.observations.grid import apply_observation
from adrkit.loads import IntervalAverageLoad
from experiments.source_recovery.config import template_config
from experiments.source_recovery.backend import Prediction, disclose_history, make_observations, make_solver
from experiments.observation_sensitivity.backend import DEFAULT_OBSERVATIONS, PRIMARY_ROWS, ProductionBackend
from experiments.observation_sensitivity.design import ObservationSpec, build_design


PATHS = build_design().paths


def path_for(condition="spatial_G1_matched", **kwargs):
    return next(p for p in PATHS if p.condition == condition and
                p.source == kwargs.get("source", "PG10") and
                p.penalty == kwargs.get("penalty", "L2") and
                p.replicate == kwargs.get("replicate", 1))


@pytest.fixture
def spec():
    result = template_config(100.)
    # Малая тестовая сетка проверяет интеграцию, а не точность рабочей сетки E06.
    
    result["grids"]["G0"] = dict(bounds_km=[-1., 1., -1., 1.], spacing_km=.5, steps=84)
    result["grids"]["expanded_test"] = dict(bounds_km=[-1.5, 1.5, -1.5, 1.5],
                                               spacing_km=.5, steps=84)
    result["observations"]["centers_km"] = [[-.6, -.2], [.4, -.1], [-.8, -.4], [0., 0.]]
    return result


@pytest.fixture
def source():
    # Ненулевая заданная предыстория проверяет аффинный сдвиг и будущий q.
    return BiExponential((1.2, .7), (.4, 1.2))


class CountingSolver(CachedSolver):
    """Считать вызовы прямого, касательного и сопряжённого решателей без замены алгоритмов."""
    calls = None
    state_refs = None

    def solve_controls(self, controls):
        type(self).calls["forward"] += 1
        trajectory = super().solve_controls(controls)
        type(self).state_refs.append(weakref.ref(trajectory.states))
        return trajectory

    def tangent_controls(self, trajectory, direction):
        type(self).calls["tangent"] += 1
        return super().tangent_controls(trajectory, direction)

    def adjoint_controls(self, trajectory, dual):
        type(self).calls["adjoint"] += 1
        return super().adjoint_controls(trajectory, dual)


@pytest.fixture
def factory():
    CountingSolver.calls = dict(forward=0, tangent=0, adjoint=0)
    CountingSolver.state_refs = []
    returned = []

    def create(settings, grid, gamma):
        mesh = settings["grids"][grid]
        bounds = np.asarray(mesh["bounds_km"])
        shape = np.rint(np.diff(bounds.reshape(2, 2), axis=1).ravel()/mesh["spacing_km"]).astype(int)-1
        model = {key: settings["model"][key] for key in ("diffusion", "velocity", "linear_loss",
            "reaction_c_star", "source_position", "background", "horizon")}
        model["reaction_gamma"] = gamma
        solver = create.solver_type(model, interior_points=shape, time_steps=mesh["steps"],
                                domain=bounds, residual_tolerance=settings["solver"]["forward_tolerance"])
        returned.append(solver)
        return solver

    create.returned = returned
    create.solver_type = CountingSolver
    return create


def test_constructor_is_lazy_and_spec_is_owned(spec, source, factory):
    backend = ProductionBackend(spec, source, solver_factory=factory, source_id="PG10")
    assert not factory.returned
    assert backend.cache_info["solver_builds"] == 0
    spec["Qref"] = -5.
    spec["observations"]["centers_km"][0][0] = np.nan
    weights, temporal = backend.observation_weights(DEFAULT_OBSERVATIONS[0])
    assert weights.shape == (4, 9) and temporal.shape == (72, 85)
    assert np.isfinite(weights).all()
    assert CountingSolver.calls == dict(forward=0, tangent=0, adjoint=0)


def test_factory_solver_alias_cannot_mutate_the_bound_numeric_snapshot(spec, source, factory):
    backend = ProductionBackend(spec, source, solver_factory=factory)
    backend.observation_weights(DEFAULT_OBSERVATIONS[0])  # Привязка снимка выполняется без решения PDE.
    factory.returned[0].xy[:] = 1e9
    factory.returned[0].times[:] = 0.
    # Ранее непривязанный H использует время и геометрию скопированного решателя.
    actual = backend.observation_weights(DEFAULT_OBSERVATIONS[1])
    reference = ProductionBackend(spec, source, solver_factory=make_solver).observation_weights(
        DEFAULT_OBSERVATIONS[1])
    assert all(np.array_equal(a, b) for a, b in zip(actual, reference))
    assert CountingSolver.calls["forward"] == 0


@pytest.mark.parametrize("condition,old_kind", [
    ("spatial_G1_matched", "snapshot"), ("temporal_average_matched", "average20"),
])
def test_g1_initializer_prediction_and_adjoint_match_recovery_prediction_bitwise(spec, source, condition, old_kind):
    path = path_for(condition)
    backend = ProductionBackend(spec, source, solver_factory=make_solver, source_id="PG10")
    prepared = backend.prepare(path)
    solver = make_solver(spec, "G0", spec["model"]["reaction_gamma"])
    old_condition = dict(nodes=73, temporal_H=old_kind, relocation_km=0.)
    old = Prediction(spec, old_condition, solver, disclose_history(source, solver.times, -.5))
    zero = np.zeros(73)
    rows = np.array(PRIMARY_ROWS)
    assert np.array_equal(prepared.jacobian, old.jacobian(zero)[rows])
    assert np.array_equal(prepared.offset, old.predict(zero)[rows])
    assert prepared.jacobian.shape == (36, 73) and prepared.offset.shape == (36,)
    assert prepared.dense_prediction.trajectory is None
    point = np.linspace(.01, .05, 73)
    dual = np.sin(np.arange(36))
    scatter = np.zeros(288); scatter[rows] = dual
    assert np.array_equal(prepared.dense_prediction.predict(point), old.predict(point))
    assert np.array_equal(prepared.restricted_prediction.predict(point), old.predict(point)[rows])
    assert np.array_equal(prepared.restricted_prediction.vjp(point, dual), old.vjp(point, scatter))
    assert prepared.restricted_prediction.trajectory.max_scaled_residual < spec["solver"]["forward_tolerance"]
    spatial, temporal = backend.observation_weights(path.true_h)
    old_spatial, old_temporal = make_observations(spec, solver, old_condition)
    assert np.array_equal(spatial, old_spatial)
    assert np.array_equal(temporal, old_temporal)


def test_all_initializers_share_one_forward_and_73_tangents_and_release_state(spec, source, factory):
    backend = ProductionBackend(spec, source, solver_factory=factory, source_id="PG10")
    for observation in DEFAULT_OBSERVATIONS:
        full_j, offset = backend.initial_linearization(observation)
        assert full_j.shape == (288, 73) and offset.shape == (288,)
        assert not full_j.flags.writeable and not offset.flags.writeable
    info = backend.cache_info
    assert info["initializer_matrices"] == 5
    assert info["initializer_shapes"] == [[288, 73]]*5
    assert JSONRecord(info).to_dict() == info
    assert info["initializer_cache_bytes"] == 5*8*288*74
    assert CountingSolver.calls == dict(forward=1, tangent=73, adjoint=0)
    gc.collect()
    assert all(ref() is None for ref in CountingSolver.state_refs)
    # Изменение штрафа, повтора и веса не меняет фиксированные предысторию и H.
    for path in (path_for(), path_for(penalty="H1", replicate=4), path_for("covariance_mix_oracle")):
        prepared = backend.prepare(path)
        assert prepared.dense_prediction.trajectory is None
        assert prepared.restricted_prediction.trajectory is None
    assert CountingSolver.calls == dict(forward=1, tangent=73, adjoint=0)


def test_initializers_and_weights_return_owned_copies(spec, source):
    backend = ProductionBackend(spec, source, solver_factory=make_solver)
    prepared = backend.prepare(path_for())
    original_j, original_b = prepared.jacobian.copy(), prepared.offset.copy()
    prepared.jacobian.setflags(write=True); prepared.jacobian[:] = -100
    prepared.offset.setflags(write=True); prepared.offset[:] = -100
    again = backend.prepare(path_for())
    assert np.array_equal(again.jacobian, original_j)
    assert np.array_equal(again.offset, original_b)
    full_j, full_b = backend.initial_linearization(path_for().inverse_h)
    full_j.setflags(write=True); full_j[:] = -1
    full_b.setflags(write=True); full_b[:] = -1
    assert np.array_equal(backend.prepare(path_for()).jacobian, original_j)
    weights, temporal = backend.observation_weights(path_for().inverse_h)
    weights[:] = 0; temporal[:] = 0
    assert np.any(backend.observation_weights(path_for().inverse_h)[0])
    with pytest.raises(FrozenInstanceError):
        prepared.offset = np.zeros(36)


def test_truth_primes_ten_projections_one_solve_and_releases_state(spec, source, factory):
    backend = ProductionBackend(spec, source, solver_factory=factory, source_id="PG10")
    first, residual = backend.truth(path_for())
    assert first.shape == (288,) and residual >= 0
    assert CountingSolver.calls == dict(forward=1, tangent=0, adjoint=0)
    assert backend.cache_info["truth_projections"] == 10
    assert backend.cache_info["projection_cache_bytes"] == 10*288*8
    for path in PATHS:
        if path.source == "PG10":
            assert backend.truth(path)[0].shape == (288,)
    for h in DEFAULT_OBSERVATIONS:
        assert backend.project_truth(h, log_width_derivative=True)[0].shape == (288,)
    assert CountingSolver.calls["forward"] == 1
    first.setflags(write=True); first[:] = -100.
    assert np.all(backend.truth(path_for())[0] >= 0)
    gc.collect()
    assert all(ref() is None for ref in CountingSolver.state_refs)
    assert backend.cache_info["retained_trajectory_count"] == 0


@pytest.mark.parametrize("condition,old_kind", [
    ("spatial_G1_matched", "snapshot"), ("temporal_average_matched", "average20"),
])
def test_true_g1_projection_matches_direct_observation_bitwise(spec, source, condition, old_kind):
    backend = ProductionBackend(spec, source, solver_factory=make_solver)
    signal, residual = backend.truth(path_for(condition))
    solver = make_solver(spec, "G0", spec["model"]["reaction_gamma"])
    state = solver.solve_controls(interval_controls(source, solver.times, origin=-.5))
    s, w = make_observations(spec, solver, dict(temporal_H=old_kind, relocation_km=0.))
    assert np.array_equal(signal, apply_observation(state.states, s, w))
    assert residual == state.max_scaled_residual


def test_custom_preload_and_expanded_grid_are_explicit_and_do_not_keep_fields(spec, source, factory):
    backend = ProductionBackend(spec, source, solver_factory=factory)
    h = ObservationSpec("gaussian", .8)
    backend.prime_truth(grid_name="expanded_test", observations=[h], include_log_width_derivatives=False)
    assert backend.cache_info["truth_projections"] == 1
    assert backend.project_truth(h, grid_name="expanded_test")[0].shape == (288,)
    assert CountingSolver.calls["forward"] == 1
    # После освобождения поля недостающая производная требует нового прямого решения.
    
    backend.project_truth(h, grid_name="expanded_test", log_width_derivative=True)
    assert CountingSolver.calls["forward"] == 2
    backend.truth(path_for())
    assert CountingSolver.calls["forward"] == 3
    assert backend.cache_info["solver_grids"] == ["expanded_test", "G0"]
    gc.collect()
    assert all(ref() is None for ref in CountingSolver.state_refs)


@pytest.mark.parametrize("kind", ["gaussian", "compact"])
def test_projected_log_width_derivative_matches_independent_width_difference(spec, source, factory, kind):
    backend = ProductionBackend(spec, source, solver_factory=factory)
    width, step = .8, 1e-4
    center = ObservationSpec(kind, width)
    plus = ObservationSpec(kind, width*np.exp(step))
    minus = ObservationSpec(kind, width*np.exp(-step))
    backend.prime_truth(observations=[center, plus, minus])
    analytic = backend.project_truth(center, log_width_derivative=True)[0]
    difference = (backend.project_truth(plus)[0]-backend.project_truth(minus)[0])/(2*step)
    assert np.allclose(difference, analytic, rtol=3e-7, atol=2e-8)
    assert CountingSolver.calls["forward"] == 1


class HistoryOnly:
    """Тестовый источник, допускающий запросы только к заданной предыстории."""

    def __init__(self):
        self.queries = []

    def value(self, time):
        assert time <= 0., "Future source queried"
        self.queries.append((time, time))
        return 1.3

    def integral(self, a, b):
        assert b <= 0., "Future source queried"
        self.queries.append((a, b))
        return 1.3*(b-a)


def test_prepare_and_prediction_scoring_never_see_future_and_estimator_drops_source(spec, factory):
    source = HistoryOnly()
    reference = weakref.ref(source)
    backend = ProductionBackend(spec, source, solver_factory=factory, source_id="PG10")
    prepared = backend.prepare(path_for("spatial_G2_assumed_G1"))
    assert source.queries and max(right for _, right in source.queries) <= 0
    query_count = len(source.queries)
    point = np.full(73, .03)
    prepared.restricted_prediction.predict(point)
    backend.score_prediction(path_for("spatial_G2_assumed_G1"), point)
    assert len(source.queries) == query_count
    del source, backend
    gc.collect()
    assert reference() is None, "Returned estimator retained the unknown generator"
    assert prepared.restricted_prediction.predict(point*2).shape == (36,)


def test_score_prediction_projects_assumed_and_true_H_once_in_declared_order(spec, source, factory):
    backend = ProductionBackend(spec, source, solver_factory=factory)
    path = path_for("spatial_G2_assumed_G1")
    point = np.linspace(.02, .04, 73)
    assumed, actual, residual = backend.score_prediction(path, point)
    assert assumed.shape == actual.shape == (288,) and residual >= 0
    assert not np.allclose(assumed, actual)
    assert CountingSolver.calls == dict(forward=1, tangent=0, adjoint=0)
    assert backend.cache_info["initializer_matrices"] == 0
    solver = make_solver(spec, "G0", spec["model"]["reaction_gamma"])
    history = disclose_history(source, solver.times, -.5)
    old = Prediction(spec, dict(nodes=73, temporal_H="snapshot", relocation_km=0.),
                     solver, history)
    assert np.array_equal(assumed, old.predict(point))
    s, w = backend.observation_weights(path.true_h)
    load = IntervalAverageLoad(old.basis, solver.times,
        origin=spec["model"]["origin_hours"], q_reference=spec["Qref"],
        history_integral=history.integral, history_initial=history.initial,
        source_unit="C*km^2/hour")
    independent = solver.solve_controls(load.predict(point))
    assert np.array_equal(actual, apply_observation(independent.states, s, w))
    gc.collect()
    assert all(ref() is None for ref in CountingSolver.state_refs)


def test_restricted_derivative_duality_and_difference(spec, source):
    prediction = ProductionBackend(spec, source, solver_factory=make_solver).prepare(path_for()).restricted_prediction
    point = np.linspace(.02, .04, 73)
    direction = .01*np.cos(np.arange(73))
    dual = np.sin(np.arange(36))
    tangent = prediction.jvp(point, direction)
    assert np.isclose(tangent@dual, direction@prediction.vjp(point, dual), rtol=1e-11, atol=1e-11)
    step = 1e-3
    fd = (prediction.predict(point+step*direction)-prediction.predict(point-step*direction))/(2*step)
    assert np.allclose(fd, tangent, rtol=2e-6, atol=2e-9)
    prediction.invalidate()
    assert prediction.trajectory is None


def test_estimator_observation_layout_matches_actual_panel_metric(spec, source):
    from experiments.observation_sensitivity.data import PanelFactory
    path = path_for("covariance_mix_oracle")
    backend = ProductionBackend(spec, source, solver_factory=make_solver)
    prepared = backend.prepare(path)
    data = PanelFactory(spec).estimation(path, np.zeros(288))
    prepared.restricted_prediction.codomain.require_compatible(data.metric.layout)
    assert prepared.dense_prediction.codomain.coordinates[0] == tuple(
        f"dense_row_{i}" for i in range(288))


def test_source_error_uses_analytic_source_not_inverse_sampling_and_no_solver(spec, factory):
    source = FiniteRelease(.713, .127, 7.)
    backend = ProductionBackend(spec, source, solver_factory=factory, source_id="PG10")
    level = .02
    scores = backend.score_source(path_for(), np.full(73, level))
    physical_level = spec["Qref"]*level
    true_mass = 7.*.127
    squared = 3*physical_level**2-2*physical_level*true_mass+7**2*.127
    assert np.isclose(scores["E_q"], np.sqrt(squared)/(100*np.sqrt(3)), rtol=2e-14)
    assert np.isclose(scores["true_mass"], true_mass, rtol=2e-14)
    assert scores["estimated_mass"] == pytest.approx(3*physical_level)
    assert not factory.returned


@pytest.mark.parametrize("point", [np.zeros(72), np.zeros((73, 1)), np.full(73, -1.),
    np.full(73, np.nan), np.full(73, np.inf), np.ones(73, dtype=complex),
    np.ones(73, dtype=bool), np.full(73, "1")])
def test_bad_scoring_parameters_fail_before_source_or_solver(spec, factory, point):
    class Forbidden:
        def value(self, t):
            raise AssertionError("Source queried")
        def integral(self, a, b):
            raise AssertionError("Source queried")
    backend = ProductionBackend(spec, Forbidden(), solver_factory=factory)
    for method in (backend.score_source, backend.score_prediction):
        with pytest.raises(ValueError, match="Coefficients"):
            method(path_for(), point)
    assert not factory.returned


def test_wrong_source_and_wrong_path_fail_before_factory(spec, source, factory):
    backend = ProductionBackend(spec, source, solver_factory=factory, source_id="PG10")
    for method in (backend.prepare, backend.truth):
        with pytest.raises(ValueError, match="source_id"):
            method(path_for(source="EC04"))
        with pytest.raises(TypeError, match="PathSpec"):
            method({})
    assert not factory.returned


@pytest.mark.parametrize("mutation", [
    lambda s: s["model"].update(origin_hours=0.),
    lambda s: s["model"].update(horizon=3.),
    lambda s: s.update(Qref=0.),
    lambda s: s.update(Qref=True),
    lambda s: s["observations"].update(centers_km=[[0., 0.]]),
])
def test_wrong_layout_spec_rejected_without_factory(spec, source, factory, mutation):
    mutation(spec)
    with pytest.raises(ValueError):
        ProductionBackend(spec, source, solver_factory=factory)
    assert not factory.returned


def test_bad_factory_and_projection_requests(spec, source, factory):
    with pytest.raises(TypeError, match="callable"):
        ProductionBackend(spec, source, solver_factory=None)
    backend = ProductionBackend(spec, source, solver_factory=lambda *args: object())
    with pytest.raises(TypeError, match="CachedSolver"):
        backend.observation_weights(DEFAULT_OBSERVATIONS[0])
    backend = ProductionBackend(spec, source, solver_factory=factory)
    with pytest.raises(ValueError, match="declared"):
        backend.prime_truth(grid_name="missing")
    with pytest.raises(ValueError, match="nonempty"):
        backend.prime_truth(observations=[])
    with pytest.raises(TypeError, match="bool"):
        backend.project_truth(DEFAULT_OBSERVATIONS[0], log_width_derivative=1)
    with pytest.raises(TypeError, match="ObservationSpec"):
        backend.project_truth({})
    assert CountingSolver.calls["forward"] == 0


def test_invalid_history_does_not_trigger_pde_or_poison_initializer(spec, factory):
    class BadHistory(HistoryOnly):
        def integral(self, a, b):
            return np.nan
    backend = ProductionBackend(spec, BadHistory(), solver_factory=factory)
    with pytest.raises(ValueError, match="history"):
        backend.prepare(path_for())
    assert CountingSolver.calls["forward"] == 0
    assert backend.cache_info["initializer_matrices"] == 0


def test_failed_forward_leaves_no_partial_truth_cache_or_retained_state(spec, source, factory):
    class FailingSolver(CountingSolver):
        fail = True

        def solve_controls(self, controls):
            trajectory = super().solve_controls(controls)
            if type(self).fail:
                raise RuntimeError("Injected failure after solve, before projection")
            return trajectory

    factory.solver_type = FailingSolver
    backend = ProductionBackend(spec, source, solver_factory=factory)
    with pytest.raises(RuntimeError, match="Injected"):
        backend.truth(path_for())
    assert backend.cache_info["truth_projections"] == 0
    assert backend.cache_info["truth_forward_attempts"] == 1
    assert backend.cache_info["truth_forward_calls"] == 0
    gc.collect()
    assert all(ref() is None for ref in CountingSolver.state_refs)
    FailingSolver.fail = False
    assert backend.truth(path_for())[0].shape == (288,)
    assert backend.cache_info["truth_forward_attempts"] == 2
    assert backend.cache_info["truth_forward_calls"] == 1
    assert backend.cache_info["truth_projections"] == 10
