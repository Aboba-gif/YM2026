"""Условия прямой диагностики проверяются на искусственном решателе без решений PDE."""
from copy import deepcopy
import math

import numpy as np
import pytest
from scipy.integrate import dblquad, quad

from adrkit.config.validation import JSONRecord
from experiments.source_comparison.truth import FiniteRelease
from experiments.source_recovery.config import template_config
from experiments.source_recovery.run import source_record_hash
from experiments.source_recovery.sources import ScaledSource
from experiments.observation_sensitivity import direct
from experiments.observation_sensitivity.backend import DEFAULT_OBSERVATIONS
from experiments.observation_sensitivity.design import FULL_TIMES_HOURS
from experiments.observation_sensitivity.kernels import GaussianKernel, CompactKernel, spatial_weights
from experiments.observation_sensitivity.operators import temporal_weights


def moment_vector(record):
    first = record["raw_first_about_station_km"]
    second = record["raw_second_about_station_km2"]
    return np.array([record["raw_mass"], *first, second[0][0], second[0][1], second[1][1]])


def test_interior_nodes_match_bounded_solver_formula_and_exclude_boundaries():
    nodes = direct.interior_nodes([-1., 1., -1., 1.], .5)
    expected = np.array([[x, y] for y in (-.5, 0., .5) for x in (-.5, 0., .5)])
    assert np.array_equal(nodes, expected)
    assert np.all(np.abs(nodes) < 1.)


@pytest.mark.parametrize("bounds,spacing", [([0, 1, 0, 1], .3), ([0, 1, 0, 1], 1.),
    ([1, 0, 0, 1], .25), ([0, np.inf, 0, 1], .25), ([0, 1, 0, 1], True),
    ([0, 1, 0, 1], -.1), ([0, 1, 0, 1], 1j)])
def test_bad_quadrature_geometry_rejected(bounds, spacing):
    with pytest.raises(ValueError):
        direct.interior_nodes(bounds, spacing)


def test_gaussian_reference_agrees_with_independent_cartesian_integration():
    center, width = np.array([.2, -.3]), .8
    bounds = [-2., .7, -1.1, 2.3]
    actual = moment_vector(direct.reference_moments(GaussianKernel(width), center, bounds))
    expected = []
    for px, py in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)):
        def integrand(y, x):
            dx, dy = x-center[0], y-center[1]
            return dx**px*dy**py*math.exp(-(dx*dx+dy*dy)/(2*width**2))/(2*math.pi*width**2)
        expected.append(dblquad(integrand, bounds[0], bounds[1], lambda _: bounds[2],
            lambda _: bounds[3], epsabs=1e-11, epsrel=1e-11)[0])
    assert np.allclose(actual, expected, rtol=2e-12, atol=2e-13)


def test_compact_full_and_half_plane_moments_from_independent_radial_integrals():
    kernel = CompactKernel(1.)
    radius = kernel.support_radius
    full = direct.reference_moments(kernel, [0., 0.], [-radius, radius, -radius, radius])
    assert np.allclose(moment_vector(full), [1., 0., 0., 1., 0., 1.], atol=8e-11, rtol=0)
    # Полудиск x>=0 имеет половину массы и вторых моментов. Его первый момент по x равен 2∫ r²ρ(r)dr
    # независимо от декартовой квадратуры.
    radial = quad(lambda r: r*r*float(kernel.density([[r, 0.]])[0]),
                  0., radius, epsabs=1e-12, epsrel=1e-12)[0]
    half = direct.reference_moments(kernel, [0., 0.], [0., radius, -radius, radius])
    assert np.allclose(moment_vector(half), [.5, 2*radial, 0., .5, 0., .5], rtol=1e-10, atol=1e-11)
    assert half["conditional"]["centroid_shift_km"][0] == pytest.approx(4*radial, rel=1e-10)
    assert half["reference"]["epsabs"] == 2e-11
    assert "not rigorous" in half["reference"]["error_interpretation"]


def test_compact_rectangle_outside_support_has_zero_mass_and_no_conditional_moments():
    result = direct.reference_moments(CompactKernel(.5), [0., 0.], [5., 6., 5., 6.])
    assert np.array_equal(moment_vector(result), np.zeros(6))
    assert result["conditional"] is None


@pytest.mark.parametrize("kernel", [GaussianKernel(.6), CompactKernel(.6)])
def test_raw_quadrature_refines_towards_finite_domain_not_renormalized_R2(kernel):
    center, bounds = [0., 0.], [-1., .5, -1., 1.]
    reference = direct.reference_moments(kernel, center, bounds)
    errors = []
    for spacing in (.25, .125, .0625):
        raw = direct.kernel_moments(kernel, center, bounds, spacing)
        assert raw["raw_mass"] < 1.
        errors.append(np.linalg.norm(moment_vector(raw)-moment_vector(reference)))
    assert errors[2] < errors[1] < errors[0]
    assert reference["raw_mass"] < .95
    assert reference["full_R2"]["mass"] == 1.
    assert not np.allclose(reference["raw_second_about_station_km2"], reference["conditional"]["covariance_km2"])


@pytest.fixture
def fixtures():
    spec = template_config(100.)
    # Один искусственный профиль под шестью ID проверяет контракт, а не физику источников.
    sources = {name: ScaledSource(FiniteRelease(.5, 1., 100.), 1.) for name in direct.SOURCE_IDS}
    return spec, sources


class FakeBackend:
    instances = []
    weights_cache = {}
    fail_source = None
    fail_grid = None
    bad_weights = False
    bad_signal = False
    residual = 1e-12
    domain_shift = .01

    def __init__(self, spec, source, *, source_id):
        self.spec, self.source_id = deepcopy(spec), source_id
        self.calls = self.attempts = 0
        self.primed = set()
        self.projection_calls = 0
        self.events = []
        type(self).instances.append(self)

    def observation_weights(self, observation, *, grid_name):
        self.events.append(("weights", grid_name, observation))
        mesh = self.spec["grids"][grid_name]
        key = grid_name, observation
        if key not in self.weights_cache:
            kernel = (GaussianKernel if observation.spatial_kind == "gaussian" else CompactKernel)(observation.width_km)
            nodes = direct.interior_nodes(mesh["bounds_km"], mesh["spacing_km"])
            spatial = spatial_weights(kernel, nodes, self.spec["observations"]["centers_km"], mesh["spacing_km"]**2)
            temporal = temporal_weights(np.linspace(0., 3.5, 337), FULL_TIMES_HOURS,
                origin_hours=-.5, kind=observation.temporal_kind, window_hours=observation.window_hours)
            self.weights_cache[key] = spatial, temporal
        spatial, temporal = (array.copy() for array in self.weights_cache[key])
        if self.bad_weights:
            spatial[0, 0] += 1.
        return spatial, temporal

    def prime_truth(self, *, grid_name, observations, include_log_width_derivatives):
        assert observations == DEFAULT_OBSERVATIONS and include_log_width_derivatives is True
        assert grid_name not in self.primed, "An attempted field was retried"
        self.primed.add(grid_name)
        self.events.append(("prime", grid_name))
        self.attempts += 1
        if self.source_id == self.fail_source and grid_name == self.fail_grid:
            raise RuntimeError("Deliberate one-case failure")
        self.calls += 1

    def project_truth(self, observation, *, grid_name, log_width_derivative=False):
        assert grid_name in self.primed
        self.projection_calls += 1
        self.events.append(("project", grid_name, observation, log_width_derivative))
        shift = self.domain_shift if grid_name == direct.D1_GRID else 0.
        value = np.full(288, observation.width_km+shift+(2. if log_width_derivative else 0.))
        if self.bad_signal:
            value[0] = np.nan
        return value, self.residual


@pytest.fixture
def fake(monkeypatch):
    for name, value in dict(instances=[], weights_cache={}, fail_source=None, fail_grid=None,
        bad_weights=False, bad_signal=False,
        residual=1e-12, domain_shift=.01).items():
        monkeypatch.setattr(FakeBackend, name, value)
    # В тестах порядка вызовов квадратура заменена заглушкой; её формулы проверяются выше.
    monkeypatch.setattr(direct, "_quadrature_report", lambda spec, direct: {"test_fixture": "quadrature tested separately"})
    return FakeBackend


def test_exact_12_fields_all_five_H_and_derivatives_with_owned_spec(fixtures, fake):
    spec, sources = fixtures
    before = JSONRecord(spec).to_bytes()
    result = direct.collect_direct(spec, sources, backend_factory=fake)
    assert JSONRecord(result).to_dict() == result
    assert JSONRecord(spec).to_bytes() == before and direct.D1_GRID not in spec["grids"]
    assert result["status"] == "complete"
    assert result["version"] == 2
    assert result["expected_fields"] == result["complete_fields"] == 12
    assert not {"rss_note", "actual_forward_attempts", "actual_forward_calls"} & result.keys()
    assert len(fake.instances) == 6 and all(x.calls == 2 for x in fake.instances)
    assert all(x.projection_calls == 20 for x in fake.instances)
    expected_order = []
    for grid in ("G0", direct.D1_GRID):
        expected_order.extend(("weights", grid, observation) for observation in DEFAULT_OBSERVATIONS)
        expected_order.append(("prime", grid))
        expected_order.extend(("project", grid, observation, derivative)
                              for observation in DEFAULT_OBSERVATIONS for derivative in (False, True))
    assert all(instance.events == expected_order for instance in fake.instances)
    assert result["domain_sensitive_count"] == 0
    for name, cases in result["fields"].items():
        assert result["sources"][name]["sha256"] == source_record_hash(sources[name])
        for domain, row in cases.items():
            assert row["status"] == "complete"
            assert len(row["projections"]) == 5
            assert not {"seconds", "rss_before_bytes", "rss_after_bytes",
                        "counters_before", "counters_after"} & row.keys()
            for projection in row["projections"].values():
                assert len(projection["signal"]) == len(projection["log_width_derivative"]) == 288
        for comparison in result["comparisons"][name].values():
            for rows in ("primary36", "dense288"):
                assert comparison[rows]["signal"]["rms"] == pytest.approx(.01)
                assert comparison[rows]["signal"]["within_thresholds"] is True


def test_domain_failure_is_retained_without_changing_threshold_or_completion(fixtures, fake):
    fake.domain_shift = .08
    result = direct.collect_direct(*fixtures, backend_factory=fake)
    assert result["status"] == "complete" and result["domain_sensitive_count"] == 30
    assert result["thresholds"]["rms"] == .02 and result["thresholds"]["maximum_absolute"] == .05
    assert result["comparisons"]["PG10"]["C1_snapshot"]["status"] == "domain_sensitive"


def test_failed_field_is_not_retried_and_other_cases_are_not_dropped(fixtures, fake):
    fake.fail_source, fake.fail_grid = "EC04", "G0"
    result = direct.collect_direct(*fixtures, backend_factory=fake)
    assert result["status"] == "incomplete" and result["complete_fields"] == 11
    assert sum(instance.attempts for instance in fake.instances) == 12
    assert sum(instance.calls for instance in fake.instances) == 11
    assert result["fields"]["EC04"]["D0"]["error_type"] == "RuntimeError"
    assert result["fields"]["EC04"]["D1"]["status"] == "complete"
    assert result["comparisons"]["EC04"]["G1_snapshot"]["status"] == "unavailable"
    assert len(fake.instances) == 6 and all(x.attempts == 2 for x in fake.instances)


def test_geometry_contract_stops_before_solving_and_retains_partial_evidence(fixtures, fake):
    fake.bad_weights = True
    with pytest.raises(direct.DirectContractError) as caught:
        direct.collect_direct(*fixtures, backend_factory=fake)
    assert len(fake.instances) == 1 and fake.instances[0].calls == 0
    partial = caught.value.partial_record
    assert JSONRecord(partial).to_dict() == partial
    assert partial["status"] == "contract_failure"
    row = partial["fields"]["PG10"]["D0"]
    assert row["status"] == "contract_failure" and row["projections"] == {}


def test_diagnostic_memory_error_stops_without_attempting_later_fields(fixtures, fake, monkeypatch):
    original = direct._array
    def injected(values, shape, name):
        if name == 'true signal':
            raise MemoryError('diagnostic conversion exhausted memory')
        return original(values, shape, name)
    monkeypatch.setattr(direct, '_array', injected)
    with pytest.raises(MemoryError):
        direct.collect_direct(*fixtures, backend_factory=fake)
    assert len(fake.instances) == 1 and fake.instances[0].calls == 1


@pytest.mark.parametrize("defect", ["bad_signal", "residual"])
def test_nonfinite_or_rejected_residual_is_incomplete_not_robustness_success(fixtures, fake, defect):
    setattr(fake, defect, True if defect == "bad_signal" else .1)
    result = direct.collect_direct(*fixtures, backend_factory=fake)
    assert result["status"] == "incomplete" and result["complete_fields"] == 0
    assert sum(instance.calls for instance in fake.instances) == 12
    assert all(row["status"] == "unavailable" for group in result["comparisons"].values() for row in group.values())


@pytest.mark.parametrize("defect", ["source_ids", "source_mass", "source_type", "grid", "d1", "centers"])
def test_all_metadata_admitted_before_any_backend_or_solve(fixtures, fake, defect):
    spec, sources = fixtures
    if defect == "source_ids":
        sources.pop("NEW-S2")
    elif defect == "source_mass":
        sources["NEW-S2"] = ScaledSource(FiniteRelease(.5, 1., 99.), 1.)
    elif defect == "source_type":
        sources["NEW-S2"] = object()
    elif defect == "grid":
        spec["grids"]["G0"]["steps"] = 0
    elif defect == "d1":
        spec["grids"][direct.D1_GRID] = {"unrelated": True}
    else:
        spec["observations"]["centers_km"] = [[0., 0.]]
    with pytest.raises((ValueError, TypeError)):
        direct.collect_direct(spec, sources, backend_factory=fake)
    assert not fake.instances



def test_direct_scope_uses_configured_grids_without_replacing_kernels(fixtures, fake):
    from experiments.observation_sensitivity.design import resolve_direct_spec
    from tests.experiments.observation_sensitivity.test_design import selected_configuration
    spec, sources = fixtures
    spec["grids"]["G0"].update(spacing_km=1., steps=168)
    plan = resolve_direct_spec(spec, selected_configuration()["direct"])
    class ConfiguredBackend(fake):
        def observation_weights(self, observation, *, grid_name):
            mesh = self.spec["grids"][grid_name]
            kernel = (GaussianKernel if observation.spatial_kind == "gaussian" else CompactKernel)(observation.width_km)
            space = spatial_weights(kernel, direct.interior_nodes(mesh["bounds_km"], mesh["spacing_km"]),
                self.spec["observations"]["centers_km"], mesh["spacing_km"]**2)
            time = temporal_weights(np.linspace(0., self.spec["model"]["horizon"], mesh["steps"]+1), FULL_TIMES_HOURS,
                origin_hours=self.spec["model"]["origin_hours"], kind=observation.temporal_kind, window_hours=observation.window_hours)
            self.events.append(("weights", grid_name, observation))
            return space, time
    before = JSONRecord(spec).to_bytes()
    result = direct.collect_direct(spec, sources, direct=plan, backend_factory=ConfiguredBackend)
    assert JSONRecord(spec).to_bytes() == before
    assert result["expected_fields"] == result["complete_fields"] == 2 and result["status"] == "complete"
    assert set(result["fields"]) == {"PG10"} and len(fake.instances) == 1
    assert fake.instances[0].projection_calls == 20 and fake.instances[0].calls == 2
    assert all(p["temporal_shape"] == [72, 169] for row in result["fields"]["PG10"].values() for p in row["projections"].values())
    assert {p["spatial_shape"][0] for row in result["fields"]["PG10"].values() for p in row["projections"].values()} == {4}
    assert result["direct_spec"]["grids"]["G0"]["spacing_km"] == 1.
    assert result["direct_spec"]["grids"]["E06_D1_direct"]["steps"] == 168
