"""Прямые расчёты сигналов и пространственных моментов ядер E06."""
from __future__ import annotations

from dataclasses import asdict
import json
import math

import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import ndtr

from adrkit.config.validation import JSONRecord
from experiments.source_comparison.calibration import array_hash
from experiments.source_recovery.run import source_record_hash
from experiments.source_recovery.sources import source_record

from .backend import DEFAULT_OBSERVATIONS, PRIMARY_ROWS, ProductionBackend
from .design import (FULL_TIMES_HOURS, DIRECT_SOURCES, DIRECT_OBSERVATIONS,
    DIRECT_DOMAINS, resolve_direct_spec)
from .kernels import CompactKernel, GaussianKernel, spatial_weights
from .operators import temporal_weights


SOURCE_IDS = DIRECT_SOURCES
OBSERVATION_IDS = DIRECT_OBSERVATIONS
D1_GRID = DIRECT_DOMAINS["D1"]
REFERENCE_ATOL = 2e-11
REFERENCE_RTOL = 2e-11
RMS_LIMIT = .02
MAX_LIMIT = .05


class DirectContractError(RuntimeError):
    """Нарушение геометрии наблюдений с частичным результатом.

    Parameters
    ----------
    message : str
        Описание нарушения.
    partial_record : dict
        Уже полученные результаты; сохраняются как отдельная JSON-копия.
    """
    def __init__(self, message, partial_record):
        super().__init__(message)
        self.partial_record = JSONRecord(partial_record).to_dict()


def _array(value, shape, name):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array of shape {shape}")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _rectangle(bounds):
    result = _array(bounds, (4,), "bounds")
    if result[0] >= result[1] or result[2] >= result[3]:
        raise ValueError("Rectangle bounds must be strictly ordered")
    return result


def interior_nodes(bounds, spacing):
    """Построить внутренние узлы равномерной прямоугольной сетки.

    Parameters
    ----------
    bounds : array_like, shape (4,)
        Границы xmin, xmax, ymin, ymax; стороны прямоугольника кратны
        spacing.
    spacing : float
        Положительный шаг в тех же единицах длины, что и bounds.

    Returns
    -------
    ndarray, shape (n_nodes, 2)
        Координаты без граничных узлов; сначала меняется x, затем y.
    """
    bounds = _rectangle(bounds)
    if isinstance(spacing, (bool, np.bool_)) or not np.isscalar(spacing) or np.iscomplexobj(spacing):
        raise ValueError("spacing must be a positive finite real scalar")
    spacing = float(spacing)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be a positive finite real scalar")
    counts = np.diff(bounds.reshape(2, 2), axis=1).ravel()/spacing
    if not np.allclose(counts, np.rint(counts), rtol=0, atol=1e-11) or np.any(counts < 2):
        raise ValueError("Rectangle must align with spacing and contain interior nodes")
    nx, ny = np.rint(counts).astype(int)-1
    xx, yy = np.meshgrid(bounds[0]+spacing*np.arange(1, nx+1),
                         bounds[2]+spacing*np.arange(1, ny+1))
    return np.column_stack((xx.ravel(), yy.ravel()))


def _moments(vector, center):
    mass, first_x, first_y, second_xx, second_xy, second_yy = map(float, vector)
    values = np.asarray([mass, first_x, first_y, second_xx, second_xy, second_yy])
    if not np.isfinite(values).all() or mass < 0:
        raise ValueError("Nonfinite or negative-mass kernel moments")
    first = np.array([first_x, first_y])
    second = np.array([[second_xx, second_xy], [second_xy, second_yy]])
    if mass > 0:
        shift = first/mass
        conditional = dict(centroid_shift_km=shift.tolist(),
            centroid_km=(np.asarray(center)+shift).tolist(),
            covariance_km2=(second/mass-np.outer(shift, shift)).tolist())
    else:
        conditional = None
    return dict(raw_mass=mass, raw_first_about_station_km=first.tolist(),
                raw_second_about_station_km2=second.tolist(), conditional=conditional)


def kernel_moments(kernel, center, bounds, spacing):
    """Вычислить сеточную квадратуру пространственных моментов ядра.

    Сумма по внутренним узлам использует площадь квадратной ячейки.
    Ограничение областью не сопровождается нормировкой; схема записи
    приведена в README.
    Условные центр и ковариация определены только при положительной массе;
    при нулевой массе поле conditional равно None.

    Parameters
    ----------
    kernel : GaussianKernel or CompactKernel
        Нормированное ядро с шириной в километрах.
    center : array_like, shape (2,)
        Координаты поста в километрах.
    bounds : array_like, shape (4,)
        Границы xmin, xmax, ymin, ymax в километрах.
    spacing : float
        Положительный шаг сетки в километрах.

    Returns
    -------
    dict
        Сырые и условные моменты, шаг, площадь ячейки и число узлов. Масса
        безразмерна, первые моменты в км, вторые — в км².
    """
    if not isinstance(kernel, (GaussianKernel, CompactKernel)):
        raise TypeError("Explicit GaussianKernel or CompactKernel required")
    center = _array(center, (2,), "center")
    points = interior_nodes(bounds, spacing)
    weights = spatial_weights(kernel, points, [center], float(spacing)**2)[0]
    dx, dy = (points-center).T
    vector = [weights.sum(), weights@dx, weights@dy,
              weights@(dx*dx), weights@(dx*dy), weights@(dy*dy)]
    result = _moments(vector, center)
    result.update(spacing_km=float(spacing), interior_nodes=int(len(points)),
                  cell_area_km2=float(spacing)**2,
                  quadrature="raw interior-node rectangle sum; boundary values omitted")
    return result


def _gaussian_axis(left, right, width):
    a, b = left/width, right/width
    # В положительном хвосте не вычитаются две CDF, близкие к единице.
    mass = float(ndtr(-a)-ndtr(-b) if a >= 0 else ndtr(b)-ndtr(a))
    phi_a = math.exp(-a*a/2)/math.sqrt(2*math.pi)
    phi_b = math.exp(-b*b/2)/math.sqrt(2*math.pi)
    return mass, width*(phi_a-phi_b), width**2*(mass+a*phi_a-b*phi_b)


def reference_moments(kernel, center, bounds):
    """Вычислить эталонные моменты ядра на прямоугольнике.

    Для GaussianKernel используются разделимые аналитические интегралы, для
    CompactKernel — независимая адаптивная квадратура. Оценки её ошибки не
    являются строгими границами. Схема записи приведена в README.
    Условные центр и ковариация определены только при положительной массе;
    при нулевой массе поле conditional равно None.

    Parameters
    ----------
    kernel : GaussianKernel or CompactKernel
        Нормированное ядро с шириной в километрах.
    center : array_like, shape (2,)
        Координаты поста в километрах.
    bounds : array_like, shape (4,)
        Границы xmin, xmax, ymin, ymax в километрах.

    Returns
    -------
    dict
        Сырые и условные моменты конечной области и полные аналитические
        моменты full_R2. Единицы: масса безразмерна, первые моменты в км,
        вторые в км².
    """
    if not isinstance(kernel, (GaussianKernel, CompactKernel)):
        raise TypeError("Explicit GaussianKernel or CompactKernel required")
    center = _array(center, (2,), "center")
    bounds = _rectangle(bounds)
    relative = bounds-np.repeat(center, 2)
    if isinstance(kernel, GaussianKernel):
        x0, x1, x2 = _gaussian_axis(*relative[:2], kernel.width)
        y0, y1, y2 = _gaussian_axis(*relative[2:], kernel.width)
        vector = [x0*y0, x1*y0, x0*y1, x2*y0, x1*y1, x0*y2]
        method = dict(method="analytic separable truncated Gaussian moments",
                      numerical_error_estimate=None)
    else:
        def bump(u):
            return 0. if u >= 1. else math.exp(-1./(1.-u))
        i0, i0_error = quad(bump, 0., 1., epsabs=REFERENCE_ATOL, epsrel=REFERENCE_RTOL)
        i1, i1_error = quad(lambda u: u*bump(u), 0., 1., epsabs=REFERENCE_ATOL, epsrel=REFERENCE_RTOL)
        radius = kernel.width*math.sqrt(2*i0/i1)
        norm = math.pi*radius**2*i0
        lo, hi = max(relative[0], -radius), min(relative[1], radius)
        inner_errors = []

        def outer(x):
            y_extent = math.sqrt(max(0., radius**2-x*x))
            yl, yh = max(relative[2], -y_extent), min(relative[3], y_extent)
            if yl >= yh:
                return np.zeros(6)
            def inner(y):
                density = bump((x*x+y*y)/radius**2)/norm
                return density*np.array([1., x, y, x*x, x*y, y*y])
            value, error, info = quad_vec(inner, yl, yh, epsabs=REFERENCE_ATOL,
                                         epsrel=REFERENCE_RTOL, full_output=True)
            if not info.success:
                raise ValueError("Compact inner reference quadrature did not converge")
            inner_errors.append(float(error))
            return value

        if lo >= hi:
            vector, outer_error = np.zeros(6), 0.
        else:
            # Деление в точках пересечения диска и прямоугольника устраняет изломы пределов.
            
            cuts = []
            for y in relative[2:]:
                if abs(y) < radius:
                    crossing = math.sqrt(radius**2-y*y)
                    cuts.extend(x for x in (-crossing, crossing) if lo < x < hi)
            vector, outer_error, info = quad_vec(outer, lo, hi, points=sorted(set(cuts)),
                epsabs=REFERENCE_ATOL, epsrel=REFERENCE_RTOL, full_output=True)
            if not info.success:
                raise ValueError("Compact outer reference quadrature did not converge")
        method = dict(method="independent adaptive Cartesian quadrature over rectangle intersect support disk",
            epsabs=REFERENCE_ATOL, epsrel=REFERENCE_RTOL,
            numerical_error_estimate=dict(outer_vector_norm=float(outer_error),
                maximum_sampled_inner_vector_norm=max(inner_errors, default=0.),
                normalizing_integral=float(i0_error), moment_integral=float(i1_error)),
            error_interpretation="adaptive numerical estimates, not rigorous error bounds")
    result = _moments(vector, center)
    result.update(reference=method,
        full_R2=dict(mass=1., first_about_station_km=[0., 0.],
            second_about_station_km2=[[kernel.width**2, 0.], [0., kernel.width**2]]))
    return result


def _moment_errors(raw, reference):
    return dict(mass_absolute=abs(raw["raw_mass"]-reference["raw_mass"]),
        first_max_absolute_km=float(np.max(np.abs(np.asarray(raw["raw_first_about_station_km"])
            -reference["raw_first_about_station_km"]))),
        second_max_absolute_km2=float(np.max(np.abs(np.asarray(raw["raw_second_about_station_km2"])
            -reference["raw_second_about_station_km2"]))))


def _quadrature_report(spec, direct):
    result = {}
    spatial = dict(G1=GaussianKernel(1.), G05=GaussianKernel(.5),
                   G2=GaussianKernel(2.), C1=CompactKernel(1.))
    for domain, grid in direct["domains"].items():
        bounds = direct["grids"][grid]["bounds_km"]
        result[domain] = {}
        for name, kernel in spatial.items():
            posts = []
            for center in spec["observations"]["centers_km"]:
                reference = reference_moments(kernel, center, bounds)
                grids = []
                for spacing in direct["quadrature_spacings_km"]:
                    raw = kernel_moments(kernel, center, bounds, spacing)
                    raw["difference_from_finite_domain_reference"] = _moment_errors(raw, reference)
                    grids.append(raw)
                posts.append(dict(center_km=list(center), reference=reference, quadrature=grids))
            result[domain][name] = posts
    return result


def _contract_failure(message, report, row):
    """Отметить нарушение геометрии в частичном результате."""
    row.update(status="contract_failure", error=message)
    report["status"] = "contract_failure"
    return DirectContractError(message, report)


def _validate_weights(backend, spec, grid, observation, reference_cache):
    # Геометрия и временные эталоны независимы от источника; строки средних за 20 минут
    # интегрируются один раз на сетку.
    key = grid, observation
    if key not in reference_cache:
        mesh = spec["grids"][grid]
        nodes = interior_nodes(mesh["bounds_km"], mesh["spacing_km"])
        kernel = (GaussianKernel if observation.spatial_kind == "gaussian" else CompactKernel)(observation.width_km)
        expected_s = spatial_weights(kernel, nodes, spec["observations"]["centers_km"], mesh["spacing_km"]**2)
        expected_w = temporal_weights(np.linspace(0., spec["model"]["horizon"], mesh["steps"]+1),
            FULL_TIMES_HOURS, origin_hours=spec["model"]["origin_hours"],
            kind=observation.temporal_kind, window_hours=observation.window_hours)
        reference_cache[key] = expected_s, expected_w
    expected_s, expected_w = reference_cache[key]
    actual_s, actual_w = backend.observation_weights(observation, grid_name=grid)
    if not np.array_equal(actual_s, expected_s) or not np.array_equal(actual_w, expected_w):
        raise ValueError("Backend observation matrices do not match declared interior-node geometry/clock")
    return dict(space_sha256=array_hash(actual_s), time_sha256=array_hash(actual_w),
                spatial_shape=list(actual_s.shape), temporal_shape=list(actual_w.shape))


def _differences(left, right):
    difference = np.asarray(left)-np.asarray(right)
    if not np.isfinite(difference).all():
        raise ValueError("Domain difference is not representable")
    maximum = float(np.max(np.abs(difference)))
    rms = 0. if maximum == 0 else maximum*float(np.sqrt(np.mean((difference/maximum)**2)))
    return dict(rms=rms, maximum_absolute=maximum)


def _comparisons(fields, direct):
    result = {}
    rows = np.array(PRIMARY_ROWS)
    for source, cases in fields.items():
        result[source] = {}
        for observation in direct["observation_ids"]:
            if any(cases[d]["status"] != "complete" for d in direct["domains"]):
                result[source][observation] = dict(status="unavailable", reason="one or both domain fields unavailable")
                continue
            a, b = (cases[d]["projections"][observation] for d in direct["domains"])
            values = {}
            for name, indices in (("primary36", rows), ("dense288", np.arange(288))):
                signal = _differences(np.asarray(a["signal"])[indices], np.asarray(b["signal"])[indices])
                signal["within_thresholds"] = signal["rms"] <= RMS_LIMIT and signal["maximum_absolute"] <= MAX_LIMIT
                derivative = _differences(np.asarray(a["log_width_derivative"])[indices],
                                          np.asarray(b["log_width_derivative"])[indices])
                values[name] = dict(signal=signal, log_width_derivative=derivative)
            passes = all(v["signal"]["within_thresholds"] for v in values.values())
            result[source][observation] = dict(status="within_thresholds" if passes else "domain_sensitive", **values)
    return result


def collect_direct(spec, sources, *, direct=None, backend_factory=ProductionBackend):
    """Вычислить прямые поля, проекции наблюдений и моменты ядер.

    Нарушение геометрии прекращает серию с DirectContractError и частичным
    результатом. Уточнение квадратуры ядра проверяется отдельно от сходимости
    PDE.

    Parameters
    ----------
    spec : dict
        Научная конфигурация E06; исходный объект не изменяется.
    sources : dict of str to object
        Объявленные профили источника с известными записями генераторов.
    direct : dict or None, optional
        Прямой план допуска; None использует полный стандартный план.
    backend_factory : callable, optional
        Фабрика backend_factory(spec_copy, source, source_id=...),
        предоставляющая observation_weights, prime_truth и project_truth.

    Returns
    -------
    dict
        Сигналы и производные в мкг/м³, моменты ядер и сравнения областей;
        отказы отдельных полей сохраняются в записи.
    """
    original = JSONRecord(spec)
    settings = original.to_dict()
    direct = resolve_direct_spec(settings, direct)
    if (not callable(backend_factory) or type(sources) is not dict
            or not set(direct["source_ids"]) <= set(sources)):
        raise ValueError("Explicit backend factory and all declared direct sources required")
    if settings["model"]["origin_hours"] != -.5 or settings["model"]["horizon"] != 3.5:
        raise ValueError("Direct series requires the extended [-.5,3] hour clock")
    _array(settings["observations"]["centers_km"], (4, 2), "four fixed posts")
    for grid, mesh in direct["grids"].items():
        if grid in settings["grids"] and settings["grids"][grid] != mesh:
            raise ValueError("A declared direct grid name is already bound differently")
        settings["grids"][grid] = mesh
    observations = [(name, h) for name, h in zip(OBSERVATION_IDS, DEFAULT_OBSERVATIONS)
                    if name in direct["observation_ids"]]
    bound = JSONRecord(settings)
    # Метаданные всех генераторов проверяются до создания любой численной модели.
    source_records = {}
    for name in direct["source_ids"]:
        source = sources[name]
        sha = source_record_hash(source)
        record = json.loads(json.dumps(source_record(source), allow_nan=False))
        if (source.start_hours != -.5 or source.end_hours != 3.
                or not math.isclose(record["unknown_mass"], settings["source_mass"], rel_tol=1e-12, abs_tol=0.)):
            raise ValueError("Source clock/mass differs from admitted direct-screen specification")
        source_records[name] = dict(sha256=sha, record=record)
    report = dict(schema="ym2026.observation_sensitivity.direct", version=2,
        status="running", expected_fields=len(direct["source_ids"])*len(direct["domains"]),
        original_spec_sha256=original.sha256, direct_spec_sha256=bound.sha256,
        direct_spec=bound.to_dict(), sources=source_records,
        observations={name: asdict(h) for name, h in observations},
        units=dict(time="hour", position="km", signal="ug/m^3", log_width_derivative="ug/m^3 per unit log(width)"),
        thresholds=dict(rms=RMS_LIMIT, maximum_absolute=MAX_LIMIT, units="ug/m^3",
            rows="both primary36 and dense288 reported; both required for within_thresholds",
            derivative_threshold=None),
        primary_rows=list(PRIMARY_ROWS), quadrature={}, fields={}, comparisons={},
        scope="domain sensitivity on declared PDE grids; kernel refinement is not PDE convergence")
    report["quadrature"] = _quadrature_report(settings, direct)
    reference_weights = {}
    for name in direct["source_ids"]:
        report["fields"][name] = {}
        backend = backend_factory(bound.to_dict(), sources[name], source_id=name)
        for domain, grid in direct["domains"].items():
            row = dict(status="running", grid=grid, bounds_km=list(direct["grids"][grid]["bounds_km"]),
                       projections={})
            report["fields"][name][domain] = row
            try:
                try:
                    hashes = {label: _validate_weights(backend, settings, grid, observation, reference_weights)
                              for label, observation in observations}
                except ValueError as error:
                    raise DirectContractError(str(error), report) from error
                backend.prime_truth(grid_name=grid, observations=tuple(h for _, h in observations),
                                    include_log_width_derivatives=True)
                for label, observation in observations:
                    signal, residual = backend.project_truth(observation, grid_name=grid)
                    derivative, derivative_residual = backend.project_truth(observation, grid_name=grid,
                                                                           log_width_derivative=True)
                    signal = _array(signal, (288,), "true signal")
                    derivative = _array(derivative, (288,), "log-width derivative signal")
                    if (not isinstance(residual, (int, float)) or isinstance(residual, bool)
                            or not math.isfinite(residual) or residual < 0
                            or residual != derivative_residual):
                        raise ValueError("Projections must share the same finite nonnegative residual certificate")
                    row["projections"][label] = dict(signal=signal.tolist(), log_width_derivative=derivative.tolist(),
                        residual=float(residual), residual_within_tolerance=residual <= settings["solver"]["forward_tolerance"],
                        signal_sha256=array_hash(signal), derivative_sha256=array_hash(derivative), **hashes[label])
                accepted = all(x["residual_within_tolerance"] for x in row["projections"].values())
                row.update(status="complete" if accepted else "residual_rejected",
                    residuals_accepted=accepted)
            except DirectContractError as error:
                raise _contract_failure(str(error), report, row) from error
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                row.update(status="unavailable", error_type=type(error).__name__, error=str(error))
        # Проекции сохранены в отчёте; модель следующего источника создаётся отдельно.
        
        del backend
    report["comparisons"] = _comparisons(report["fields"], direct)
    rows = [row for cases in report["fields"].values() for row in cases.values()]
    complete = all(row["status"] == "complete" and row.get("residuals_accepted") for row in rows)
    report.update(status="complete" if complete else "incomplete",
        expected_fields=len(direct["source_ids"])*len(direct["domains"]), complete_fields=sum(row["status"] == "complete" for row in rows),
        domain_sensitive_count=sum(row["status"] == "domain_sensitive"
            for comparisons in report["comparisons"].values() for row in comparisons.values()))
    return JSONRecord(report).to_dict()
