"""Диагностика сохранённых прямых расчётов E06.

Сравниваются проекции наблюдений, моменты пространственных ядер
и результаты на двух расчётных областях.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math

import numpy as np

from adrkit.config.validation import canonical_bytes, digest, strict_json
from experiments.observation_sensitivity.admission import scientific_configuration
from experiments.observation_sensitivity.design import FULL_TIMES_HOURS, resolve_plan, default_direct_spec, DIRECT_OBSERVATIONS
from experiments.observation_sensitivity.kernels import GaussianKernel, CompactKernel, spatial_weights
from experiments.observation_sensitivity.operators import temporal_weights
from experiments.observation_sensitivity.backend import DEFAULT_OBSERVATIONS
from .paired_effects import _freeze, _direct, PRIMARY_ROWS


WIDTHS = {"G1": 1., "G05": .5, "G2": 2., "C1": 1.}
OBSERVATIONS = dict(zip(DIRECT_OBSERVATIONS, DEFAULT_OBSERVATIONS))
TOL = dict(rtol=1e-10, atol=1e-12)


class DirectAnalysisError(ValueError):
    """Отсутствующие, несогласованные или незавершённые прямые записи."""


def require(value, message):
    """Проверить условие и при нарушении вызвать DirectAnalysisError.

    Parameters
    ----------
    value : bool
        Проверяемое условие.
    message : str
        Сообщение при нарушении.
    """

    if not value:
        raise DirectAnalysisError(message)


def number(value, name, *, nonnegative=True):
    """Проверить конечное вещественное число без булевых значений.

    Parameters
    ----------
    value : int or float
        Проверяемое число.
    name : str
        Название поля для сообщения об ошибке.
    nonnegative : bool, optional
        Требовать неотрицательность, по умолчанию True.

    Returns
    -------
    int or float
        Исходное число без изменения типа.
    """

    require(type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0), f"Invalid numeric {name}")
    return value


def array(value, shape, name):
    """Проверить форму и конечность вещественного массива и вернуть копию.

    Parameters
    ----------
    value : array_like
        Числовые целые или вещественные элементы.
    shape : tuple of int
        Требуемая форма массива.
    name : str
        Название поля для сообщения об ошибке.

    Returns
    -------
    ndarray of float
        Отдельная вещественная копия заданной формы; единицы элементов
        сохраняются.
    """

    a = np.asarray(value)
    require(a.shape == shape and a.dtype.kind in "iuf" and np.isfinite(a).all(), f"Invalid array {name}")
    return np.array(a, dtype=float, copy=True)


def close(actual, expected, name):
    """Проверить численную согласованность с допусками TOL.

    Parameters
    ----------
    actual, expected : array_like
        Сравниваемые значения в одинаковых единицах.
    name : str
        Название сравнения для сообщения об ошибке.
    """

    require(np.allclose(actual, expected, **TOL), f"Inconsistent {name}")


def array_sha(value):
    """Вычислить SHA-256 формы и байтов массива float64.

    Parameters
    ----------
    value : array_like
        Значения, приводимые к float64 с little-endian и C-порядком.

    Returns
    -------
    str
        Хеш привязки `shape`/`dtype` и байтов массива; единицы в привязку не
        входят.
    """

    a = np.asarray(value, dtype="<f8", order="C")
    return hashlib.sha256(digest({"shape": list(a.shape), "dtype": "<f8"}).encode("ascii")
                          + a.tobytes(order="C")).hexdigest()


def rms_max(values):
    """Вычислить среднеквадратичное и максимальное абсолютное значение.

    Parameters
    ----------
    values : array_like
        Непустой массив конечных вещественных значений.

    Returns
    -------
    dict
        `rms` и `maximum_absolute` в единицах входа. Все элементы имеют равный
        вес.
    """
    a = np.asarray(values, dtype=float)
    require(a.size > 0 and np.isfinite(a).all(), "Nonfinite/empty diagnostic difference")
    maximum = float(np.max(np.abs(a)))
    return dict(rms=0. if maximum == 0 else maximum*float(np.sqrt(np.mean((a/maximum)**2))),
                maximum_absolute=maximum)


def nodes(bounds, spacing):
    """Построить внутренние узлы равномерной прямоугольной сетки.

    Parameters
    ----------
    bounds : sequence of float, length 4
        Левая и правая границы по x, затем нижняя и верхняя по y в км;
        протяжённости кратны spacing.
    spacing : float
        Положительный шаг сетки в км.

    Returns
    -------
    ndarray, shape (n_nodes, 2)
        Координаты внутренних узлов в км; x меняется быстрее y.
    """

    nx, ny = (round((bounds[1]-bounds[0])/spacing)-1, round((bounds[3]-bounds[2])/spacing)-1)
    xx, yy = np.meshgrid(bounds[0]+spacing*np.arange(1, nx+1), bounds[2]+spacing*np.arange(1, ny+1))
    return np.column_stack((xx.ravel(), yy.ravel()))


def operator_bindings(spec, *, direct=None):
    """Восстановить формы и хеши матриц наблюдений для плана E06.

    Parameters
    ----------
    spec : dict
        Научная конфигурация с четырьмя центрами постов
        `observations.centers_km` в км.
    direct : dict or None, optional
        Нормализованный прямой план; None использует стандартные области и операторы.

    Returns
    -------
    dict
        Привязки пространственной и временной матриц по паре область–оператор
        на объявленных сетках E06; ключи содержат форму и SHA-256 каждой
        матрицы.
    """
    centers = array(spec["observations"]["centers_km"], (4, 2), "centers")
    direct = default_direct_spec(spec) if direct is None else direct
    result = {}
    for domain, grid in direct["domains"].items():
        mesh = direct["grids"][grid]
        points = nodes(mesh["bounds_km"], mesh["spacing_km"])
        clock = np.linspace(0., spec["model"]["horizon"], mesh["steps"]+1)
        for label in direct["observation_ids"]:
            obs = OBSERVATIONS[label]
            kernel = (CompactKernel if obs.spatial_kind == "compact" else GaussianKernel)(obs.width_km)
            space = spatial_weights(kernel, points, centers, mesh["spacing_km"]**2)
            time = temporal_weights(clock, FULL_TIMES_HOURS, origin_hours=spec["model"]["origin_hours"],
                                    kind=obs.temporal_kind, window_hours=obs.window_hours)
            close(time.sum(axis=1), np.ones(72), "temporal row normalization")
            result[domain, label] = dict(space_sha256=array_sha(space), time_sha256=array_sha(time),
                spatial_shape=list(space.shape), temporal_shape=list(time.shape))
    return result


def moment_vector(record, center):
    """Проверить моменты ядра и вернуть вектор сырых моментов.

    Parameters
    ----------
    record : dict
        Запись с `raw_mass`, `raw_first_about_station_km`,
        `raw_second_about_station_km2` и `conditional`.
    center : ndarray, shape (2,)
        Координаты поста в км, относительно которого заданы сырые моменты.

    Returns
    -------
    ndarray, shape (6,)
        Масса, две компоненты первого момента и три независимые компоненты
        симметричного второго момента. Их единицы — 1, км и км²
        соответственно; порядок компонентов: xx, xy, yy.

    Notes
    -----
    Условные моменты проверяются относительно `center`. При нулевой массе
    сырые моменты должны быть нулевыми, а `conditional` — None. Полные
    определения сырых и условных моментов приведены в README эксперимента.
    """
    mass = number(record["raw_mass"], "raw mass")
    first = array(record["raw_first_about_station_km"], (2,), "raw first moment")
    second = array(record["raw_second_about_station_km2"], (2, 2), "raw second moment")
    close(second, second.T, "moment symmetry")
    conditional = record["conditional"]
    if mass == 0:
        require(conditional is None, "Zero mass cannot define a conditional average")
        close(first, np.zeros(2), "zero-mass first moment")
        close(second, np.zeros((2, 2)), "zero-mass second moment")
    else:
        require(type(conditional) is dict, "Positive mass needs conditional moments")
        shift = first/mass
        covariance = second/mass-np.outer(shift, shift)
        for key, expected in (("centroid_shift_km", shift), ("centroid_km", center+shift),
                              ("covariance_km2", covariance)):
            close(array(conditional[key], expected.shape, key), expected, key)
        require(float(np.min(np.linalg.eigvalsh(covariance))) >= -1e-10*max(1., float(np.max(np.abs(covariance)))),
                "Conditional covariance is not positive semidefinite within arithmetic tolerance")
    return np.array([mass, *first, second[0, 0], second[0, 1], second[1, 1]])


def quadrature_rows(report, spec, *, direct=None):
    """Сопоставить сохранённые квадратуры с эталоном на конечных областях.

    Parameters
    ----------
    report : dict
        Сохранённые квадратуры обеих расчётных областей по семействам ядер и постам.
    spec : dict
        Научная конфигурация с четырьмя центрами постов в км.
    direct : dict or None, optional
        Нормализованный прямой план; None использует стандартные области и квадратуру.

    Returns
    -------
    list of dict
        Строки выбранных областей, четырёх ядер, четырёх постов и шагов квадратуры.
        Ошибки массы, первого и второго моментов имеют единицы 1, км и км².

    Notes
    -----
    Недостающая масса относительно всей плоскости отделена от ошибки
    квадратуры относительно конечной области. Новые интегралы здесь
    не вычисляются.
    """
    all_rows = []
    direct = default_direct_spec(spec) if direct is None else direct
    require(set(report) == set(direct["domains"]), "Quadrature needs all declared domains")
    centers = array(spec["observations"]["centers_km"], (4, 2), "centers")
    for domain, grid_name in direct["domains"].items():
        bounds = direct["grids"][grid_name]["bounds_km"]
        require(set(report[domain]) == set(WIDTHS), "Quadrature kernel family changed")
        for kernel, width in WIDTHS.items():
            posts = report[domain][kernel]
            require(type(posts) is list and len(posts) == 4, "Quadrature needs four posts")
            for station, (post, center) in enumerate(zip(posts, centers)):
                close(array(post["center_km"], (2,), "quadrature center"), center, "quadrature station")
                reference = post["reference"]
                ref = moment_vector(reference, center)
                require(ref[0] <= 1.+1e-10, "Finite-domain reference mass exceeds full mass")
                full = reference["full_R2"]
                close(number(full["mass"], "full mass"), 1., "full mass")
                close(array(full["first_about_station_km"], (2,), "full first"), [0., 0.], "full first")
                close(array(full["second_about_station_km2"], (2, 2), "full second"), np.eye(2)*width**2, "full second")
                grids = post["quadrature"]
                require(type(grids) is list and len(grids) == len(direct["quadrature_spacings_km"]), "All declared quadrature resolutions required")
                for grid, spacing in zip(grids, direct["quadrature_spacings_km"]):
                    require(grid["spacing_km"] == spacing and grid["cell_area_km2"] == spacing**2
                            and type(grid["interior_nodes"]) is int
                            and grid["interior_nodes"] == len(nodes(bounds, spacing)), "Quadrature geometry changed")
                    raw = moment_vector(grid, center)
                    diff = np.abs(raw-ref)
                    expected = dict(mass_absolute=float(diff[0]), first_max_absolute_km=float(max(diff[1:3])),
                                    second_max_absolute_km2=float(max(diff[3:])))
                    stored = grid["difference_from_finite_domain_reference"]
                    require(set(stored) == set(expected), "Missing quadrature errors")
                    for name, value in expected.items():
                        close(number(stored[name], name), value, name)
                    all_rows.append(dict(domain=domain, kernel=kernel, station=station, center_km=center.tolist(),
                        spacing_km=spacing, raw_moments=grid, finite_domain_reference=reference,
                        errors_to_finite_domain=expected, missing_R2_reference_mass=float(1.-ref[0]),
                        raw_mass_minus_R2_mass=float(raw[0]-1.)))
    return all_rows


def analyze_direct(freeze, direct, *, expected_freeze_sha256):
    """Свести завершённые прямые записи E06 к диагностике проекций.

    Parameters
    ----------
    freeze : dict
        Закреплённый план E06.
    direct : dict
        Завершённый прямой журнал E06.
    expected_freeze_sha256 : str
        Независимо записанный SHA-256 канонического ``freeze.json`` с
        завершающим LF.

    Returns
    -------
    dict
        Ошибки квадратуры, различия между областями и метрики сохранённых
        проекций наблюдений с исходными единицами и статусами отказов.
    """
    try:
        freeze, direct = strict_json(canonical_bytes([freeze, direct]))
        return _analyze(freeze, direct, expected_freeze_sha256)
    except DirectAnalysisError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise DirectAnalysisError(f"Invalid direct evidence: {error}") from error


def _analyze(freeze, direct, pin):
    admission = _freeze(freeze, pin)
    _direct(direct, freeze, admission)
    result = direct["result"]
    spec = scientific_configuration(admission["baseline"]["configuration"])
    plan = resolve_plan(admission["configuration"], admission["baseline"]["configuration"])["direct"]
    spec["grids"].update(plan["grids"])
    domains = {label: plan["grids"][grid]["bounds_km"] for label, grid in plan["domains"].items()}
    observations = {label: OBSERVATIONS[label] for label in plan["observation_ids"]}
    require(result["direct_spec"] == spec and result["direct_spec_sha256"] == digest(spec), "Direct extended spec mismatch")
    require(result["primary_rows"] == PRIMARY_ROWS
            and result["observations"] == {k: asdict(v) for k, v in observations.items()}, "Observation design changed")
    require(result["thresholds"]["rms"] == .02 and result["thresholds"]["maximum_absolute"] == .05
            and result["thresholds"]["derivative_threshold"] is None
            and result["thresholds"]["units"] == "ug/m^3", "Domain thresholds changed")
    require(result["units"] == dict(time="hour", position="km", signal="ug/m^3",
            log_width_derivative="ug/m^3 per unit log(width)"), "Diagnostic units changed")
    quadrature = quadrature_rows(result["quadrature"], spec, direct=plan)
    bindings = operator_bindings(spec, direct=plan)
    fields, metrics, comparisons, within_field = [], [], [], []
    arrays = {}
    layouts = {"primary36": np.array(PRIMARY_ROWS), "dense288": np.arange(288)}
    for source in plan["source_ids"]:
        for domain, bounds in domains.items():
            row = result["fields"][source][domain]
            require(row["bounds_km"] == bounds and row["grid"] == plan["domains"][domain],
                    "Field grid/domain mismatch")
            if row["status"] != "unavailable":
                require(set(row["projections"]) == set(observations), "Missing completed projections")
            else:
                require(type(row.get("error")) is str and type(row.get("error_type")) is str, "Missing failure evidence")
                require(set(row["projections"]) <= set(observations), "Unknown partial projection")
            require(len({number(p["residual"], "residual") for p in row["projections"].values()}) <= 1,
                    "One field must have one shared residual certificate")
            accepted = []
            for label, projection in row["projections"].items():
                require(all(projection[k] == v for k, v in bindings[domain, label].items()), "Observation matrix binding mismatch")
                signal = array(projection["signal"], (288,), "signal")
                derivative = array(projection["log_width_derivative"], (288,), "log-width derivative")
                require(array_sha(signal) == projection["signal_sha256"]
                        and array_sha(derivative) == projection["derivative_sha256"], "Projection content hash mismatch")
                good = number(projection["residual"], "residual") <= spec["solver"]["forward_tolerance"]
                require(projection["residual_within_tolerance"] is good, "Residual certificate contradiction")
                accepted.append(good)
                arrays[source, domain, label] = signal, derivative
                for layout, indices in layouts.items():
                    s, d = rms_max(signal[indices]), rms_max(derivative[indices])
                    metrics.append(dict(source=source, domain=domain, observation=label, layout=layout,
                        field_status=row["status"], diagnostic_only=row["status"] != "complete",
                        signal=s, log_width_derivative=d, signal_is_zero=s["rms"] == 0))
            if row["status"] != "unavailable":
                require(row["residuals_accepted"] is all(accepted)
                        and (row["status"] == "complete") == all(accepted), "Field acceptance contradicts residuals")
            fields.append(dict(source=source, domain=domain, status=row["status"],
                projection_count=len(row["projections"]), error=row.get("error")))
    require(set(result["comparisons"]) == set(plan["source_ids"]), "Missing domain comparisons")
    for source in plan["source_ids"]:
        require(set(result["comparisons"][source]) == set(observations), "Missing observation domain comparisons")
        for label in observations:
            stored = result["comparisons"][source][label]
            available = all(result["fields"][source][d]["status"] == "complete" for d in domains)
            if not available:
                require(stored["status"] == "unavailable", "Failed field cannot pass domain thresholds")
                comparisons.append(dict(source=source, observation=label, status="unavailable"))
                continue
            values = {}
            for layout, indices in layouts.items():
                a, b = arrays[source, "D0", label], arrays[source, "D1", label]
                s, d = (rms_max(a[i][indices]-b[i][indices]) for i in range(2))
                good = s["rms"] <= .02 and s["maximum_absolute"] <= .05
                for name, current in (("signal", s), ("log_width_derivative", d)):
                    for key, value in current.items():
                        close(number(stored[layout][name][key], key), value, "domain "+name+" "+key)
                require(stored[layout]["signal"]["within_thresholds"] is good, "Domain threshold flag contradiction")
                values[layout] = dict(signal=dict(s, within_thresholds=good), log_width_derivative=d)
            state = "within_thresholds" if all(v["signal"]["within_thresholds"] for v in values.values()) else "domain_sensitive"
            require(stored["status"] == state, "Overall domain status contradicts both row sets")
            comparisons.append(dict(source=source, observation=label, status=state, **values))
        for domain in domains:
            if result["fields"][source][domain]["status"] != "complete":
                continue
            for label in observations:
                if label == "G1_snapshot" or "G1_snapshot" not in observations:
                    continue
                for layout, indices in layouts.items():
                    diff = arrays[source, domain, label][0][indices]-arrays[source, domain, "G1_snapshot"][0][indices]
                    within_field.append(dict(source=source, domain=domain, baseline="G1_snapshot", variant=label,
                        layout=layout, difference="variant minus baseline", signal_difference=rms_max(diff)))
    sensitive = sum(r["status"] == "domain_sensitive" for r in comparisons)
    require(type(result["domain_sensitive_count"]) is int and result["domain_sensitive_count"] == sensitive,
            "Domain-sensitive count differs")
    return dict(schema="ym2026.observation_sensitivity.direct_analysis", version=2, status="complete",
        inputs=dict(freeze_raw_sha256=pin, direct_record_sha256=digest(direct)),
        units=result["units"], thresholds=result["thresholds"], fields=fields,
        quadrature=quadrature, projection_metrics=metrics, domain_comparisons=comparisons,
        changes_on_same_field=within_field, domain_sensitive_count=sensitive)


def main(argv=None):
    """Прочитать записи E06 и записать прямую и ковариационную диагностику.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Аргументы без имени программы; None использует ``sys.argv[1:]``.

    Notes
    -----
    Парная сводка вычисляется для проверки входов и привязки
    ковариационного анализа. Все анализы используют один снимок записей.
    """
    import argparse
    import json
    from pathlib import Path
    from . import read_results as snapshot
    from . import paired_effects as paired
    from . import covariance

    parser = argparse.ArgumentParser(description=__doc__)
    snapshot.add_read_arguments(parser)
    parser.add_argument("--output", required=True, help="Новый файл JSON характеристик прямых полей и ковариаций")
    args = parser.parse_args(argv)
    protected = snapshot.input_root(args)
    records = snapshot.records_from_arguments(args)
    destination = snapshot.export_destination(args.output, protected_root=protected,
        additional_roots=snapshot.scientific_input_roots(records["freeze"]["admission"]))
    summary = paired.summarize_records(records["freeze"], records["direct"], records["groups"],
                                       expected_freeze_sha256=args.freeze_sha256)
    direct_report = analyze_direct(records["freeze"], records["direct"], expected_freeze_sha256=args.freeze_sha256)
    covariance_report = covariance.summarize_covariances(records["freeze"], summary,
                                                        expected_freeze_sha256=args.freeze_sha256)
    module_paths = dict(snapshot=Path(snapshot.__file__), paired=Path(paired.__file__),
                        direct=Path(__file__), covariance=Path(covariance.__file__))
    result = dict(schema="ym2026.observation_sensitivity.analysis", version=1,
        status="complete", freeze_raw_sha256=args.freeze_sha256,
        raw_input_file_sha256=records["raw_file_sha256"], pair_summary_sha256=digest(summary),
        implementation_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in module_paths.items()},
        direct_analysis=direct_report, covariance_analysis=covariance_report)
    # Все три результата сериализуются до открытия нового файла.
    # Режим x защищает уже существующий файл от перезаписи.
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
    print(json.dumps({"output": str(destination), "input_files": len(records["raw_file_sha256"])}))


if __name__ == "__main__":
    main()
