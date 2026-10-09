"""Запуск сравнения регуляризаций и запись результатов восстановления."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import tempfile
from time import perf_counter

import numpy as np
import scipy
import adrkit

from adrkit.backends.bounded import BoundedSolver
from adrkit.config.validation import strict_json
from experiments import file_locks
from experiments.source_comparison.calibration import draw_panel, fit_covariance
from experiments.source_comparison.config import load_protocol, execution_parameters
from experiments.source_comparison.observations import build_observations
from experiments.source_comparison.inverse import penalty_grams
from experiments.source_comparison.truth import build_truth, interval_controls
from adrkit.inverse.misfit import covariance_metric
from adrkit.loads import IntervalAverageLoad
from adrkit.observations.grid import apply_observation
from adrkit.sources import P1Basis
from .prediction import Prediction, state_solver
from adrkit.inverse.projected import fit, lcurve_corner


def digest(path):
    """Вернуть шестнадцатеричный SHA-256 байтов указанного файла."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, payload):
    """Записать JSON без неконечных чисел с заменой через временный файл."""

    path = Path(path)
    text = json.dumps(payload, indent=2, allow_nan=False)+"\n"
    descriptor, name = tempfile.mkstemp(prefix="."+path.name+".", suffix=".tmp",
                                        dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_source_id(source_id):
    """Проверить, что источник задаёт одно допустимое имя файла результата."""

    if (not isinstance(source_id, str) or not source_id
            or source_id in (".", "..") or source_id.casefold() == "run"
            or source_id[-1] in " ." or any(ord(char) < 32 for char in source_id)
            or any(char in '<>:"/\\|?*' for char in source_id)
            or PureWindowsPath(source_id).is_reserved()):
        raise ValueError("Source identifier must be a safe result filename")


def _configured_sources(settings):
    """Проверить непустой список источников без совпадающих имён файлов."""

    if not isinstance(settings, dict):
        raise ValueError("Configuration must be a JSON object")
    sources = settings.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a nonempty list")
    for source_id in sources:
        _validate_source_id(source_id)
    if len({source_id.casefold() for source_id in sources}) != len(sources):
        raise ValueError("sources must contain unique result filenames")
    return sources


def _existing_run(output, bindings):
    """Прочитать существующую запись запуска и проверить её привязку."""

    path = output/"run.json"
    if not path.exists():
        return None
    record = strict_json(path.read_bytes())
    if not isinstance(record, dict):
        raise ValueError("Run record must be a JSON object")
    if record.get("bindings") != bindings:
        raise ValueError("Existing run has different code/config; use a new output directory")
    return record


def _verify_recorded_outputs(output, record, *, source_id=None):
    """Проверить сохранённые хеши результатов до повторного использования записи."""

    if not isinstance(record, dict):
        raise ValueError("Run record must be a JSON object")
    outputs = record.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("Run outputs must be a JSON object")
    names = {source+".json" for source in _configured_sources(record["configuration"])}
    for name, expected in outputs.items():
        if name not in names:
            raise ValueError("Pinned output must belong to a configured source")
        if source_id is not None and name != source_id+".json":
            continue
        path = output/name
        if not path.is_file():
            raise ValueError("Previously pinned source output is missing")
        if digest(path) != expected:
            raise ValueError("Previously pinned source output has changed")


def _update_run(output, record):
    """Записать состояние и хеши проверенных результатов под общей блокировкой."""

    _verify_recorded_outputs(output, record)
    sources = _configured_sources(record["configuration"])
    pinned = record.get("outputs", {})
    outputs = {}
    for source_id in sources:
        path = output/(source_id+".json")
        if path.exists():
            data = path.read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            if path.name in pinned and actual != pinned[path.name]:
                raise ValueError("Previously pinned source output has changed")
            result = strict_json(data)
            if not isinstance(result, dict):
                raise ValueError("Source output must be a JSON object")
            if result.get("source") != source_id or result.get("bindings") != record["bindings"]:
                raise ValueError("Source output belongs to different source/code/config")
            outputs[path.name] = actual
        elif path.name in pinned:
            raise ValueError("Previously pinned source output is missing")
    record["status"] = "completed" if len(outputs) == len(sources) else "partial"
    record["outputs"] = outputs
    save_json(output/"run.json", record)


def setup(config_path, *, config_bytes=None):
    """Прочитать настройки запуска и определить каталог результатов.

    Parameters
    ----------
    config_path : str or path-like
        Файл JSON с относительной ссылкой на протокол и каталог
        результатов.
    config_bytes : bytes or None, optional
        Прочитанные байты JSON; при None содержимое читается из файла.
        Относительные пути разрешаются относительно config_path.
        Проверку неизменности файла выполняет вызывающий код.

    Returns
    -------
    settings : dict
        Настройки запуска из JSON-файла.
    protocol : ProtocolBinding
        Протокол с проверенным хешем.
    output : Path
        Абсолютный путь каталога результатов, не пересекающий защищённые
        каталоги.
    """

    path = Path(config_path).resolve()
    config = strict_json(path.read_bytes() if config_bytes is None else config_bytes)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    protocol = load_protocol(path.parent/config["protocol"],
                             expected_sha256=config["protocol_sha256"])
    output = (path.parent/config["output"]).resolve()
    protected = [Path(__file__).resolve().parents[2], Path(adrkit.__file__).resolve().parent,
                 *(path.parent/entry for entry in config.get("protected_roots", []))]
    if any(output == root.resolve() or output.is_relative_to(root.resolve())
           or root.resolve().is_relative_to(output) for root in protected):
        raise ValueError("Output must not overlap source or protected input directories")
    _configured_sources(config)
    return config, protocol, output


def forward_truth(protocol, settings, source, spacing, steps):
    """Вычислить прогноз аналитического источника без добавленного шума.

    Parameters
    ----------
    protocol : ProtocolBinding
        Протокол модели и плотного календаря наблюдений.
    settings : dict
        Границы пространственной области в км.
    source : FiniteRelease or BiExponential
        Аналитический профиль с интенсивностью в C·км²/ч.
    spacing : float
        Пространственный шаг в км.
    steps : int
        Число шагов на временном горизонте протокола.

    Returns
    -------
    values : ndarray, shape (n_observations,)
        Концентрации в мкг/м³, упорядоченные по постам и времени.
    residual : float
        Максимальная масштабированная невязка состояния.
    """

    solver = state_solver(protocol, settings, spacing, steps)
    design = build_observations(protocol, solver, regime="dense")
    trajectory = solver.solve_controls(interval_controls(source, solver.times, origin=-.5))
    values = apply_observation(trajectory.states, design.space_weights, design.time_weights)
    return values, float(trajectory.max_scaled_residual)


def difference(left, right):
    """Вычислить RMSE и максимальное отклонение двух прогнозов.

    Parameters
    ----------
    left, right : array_like
        Совместимые по форме концентрации в одной единице.

    Returns
    -------
    errors : dict of str to float
        rmse и maximum в единицах входных концентраций.
    """

    delta = np.asarray(left)-right
    return dict(rmse=float(np.sqrt(np.mean(delta**2))), maximum=float(np.max(np.abs(delta))))


def check_model(prediction, protocol, settings, source, *, solver, load):
    """Проверить производные прогноза и чувствительность к сетке.

    Parameters
    ----------
    prediction : Prediction
        Полулинейный прогноз с производными и оператором наблюдений.
    protocol : ProtocolBinding
        Протокол модели и календаря наблюдений.
    settings : dict
        Обратная, генерирующая и уточнённая сетки.
    source : FiniteRelease or BiExponential
        Заданный аналитический профиль для сравнения прямых решений.
    solver : CachedSolver
        Исходный решатель с параметрами, переданными в prediction.
    load : IntervalAverageLoad
        Исходная нагрузка, переданная в prediction; используется независимым
        решателем по методу Ньютона при сравнении прямых решений.

    Returns
    -------
    truth : ndarray, shape (n_observations,)
        Концентрации в мкг/м³ без добавленного шума на сетке генератора.
    diagnostics : dict
        Относительные отклонения производных и абсолютные различия
        прогнозов в мкг/м³.

    Notes
    -----
    Сравнение конечного набора сеток не устанавливает сходимость к
    непрерывному решению ADR.
    """
    n = prediction.basis.size
    at = np.full(n, .3)
    direction = np.random.default_rng(710).normal(size=n)
    dual = np.random.default_rng(711).normal(size=prediction.observations.space.size)
    prediction.predict(at)
    tangent = prediction.jvp(at, direction)
    adjoint = prediction.vjp(at, dual)
    left, right = float(tangent @ dual), float(direction @ adjoint)
    duality = abs(left-right)/max(1., abs(left), abs(right))
    step = 1e-4
    central = (prediction.predict(at+step*direction)-prediction.predict(at-step*direction))/(2*step)
    derivative = float(np.linalg.norm(central-tangent)/max(1., np.linalg.norm(tangent)))
    cached = prediction.predict(at)
    independent = BoundedSolver(solver.model, interior_points=(solver.nx, solver.ny),
                                time_steps=solver.nt, domain=solver.bounds)
    newton = prediction.observe(independent.solve_controls(load.predict(at)).states)
    kernel = difference(cached, newton)
    if duality > 1e-8 or derivative > 1e-6 or kernel["maximum"] > 1e-8:
        raise RuntimeError(f"Numerical implementation check failed: {duality}, {derivative}, {kernel}")
    inverse, r_inverse = forward_truth(protocol, settings, source,
        settings["inverse_spacing_km"], settings["inverse_steps"])
    truth, r_truth = forward_truth(protocol, settings, source,
        settings["truth_spacing_km"], settings["truth_steps"])
    refined, r_refined = forward_truth(protocol, settings, source,
        settings["check_spacing_km"], settings["check_steps"])
    expanded_settings = dict(settings, bounds_km=[-20, 10, -10, 10])
    expanded, _ = forward_truth(protocol, expanded_settings, source,
        settings["inverse_spacing_km"], settings["inverse_steps"])
    return truth, dict(duality_relative=duality, directional_derivative_relative=derivative,
        cached_vs_newton=kernel, inverse_vs_truth=difference(inverse, truth),
        truth_vs_refined=difference(truth, refined), compact_vs_original_domain=difference(inverse, expanded),
        max_forward_residual=max(r_inverse, r_truth, r_refined),
        continuum_convergence_certified=False)


def source_metrics(source, basis, coefficients, *, q_reference=100.):
    # Квадратура разбивается в событиях источника и узлах P1; значение в скачке не размывается
    # трапецией.
    """Вычислить относительные ошибки восстановленного источника.

    Parameters
    ----------
    source : FiniteRelease or BiExponential
        Аналитический профиль с положительными интегралом и нормой L².
    basis : P1Basis
        Базис на интервале от 0 до 3 часов.
    coefficients : array_like, shape (basis.size,)
        Безразмерные узловые значения.
    q_reference : float, optional
        Положительный конечный масштаб интенсивности в C·км²/ч.
        По умолчанию 100; научный расчёт явно передаёт Qref
        из проверенного протокола.

    Returns
    -------
    metrics : dict of str to float
        relative_L2 — ошибка относительно нормы источника;
        relative_mass_error — модуль ошибки интеграла относительно
        истинного интеграла. estimated_mass и true_mass заданы в C·км².

    Notes
    -----
    Квадратура разбивается по узлам P1 и событиям источника; интеграл
    восстановленного профиля вычисляется по его кусочно-линейным узлам.
    """

    if (type(q_reference) not in (int, float) or not np.isfinite(q_reference)
            or q_reference <= 0):
        raise ValueError("q_reference must be a finite positive real number")
    edges = np.unique(np.r_[basis.knots, [v for v in source.events if 0 < v < 3]])
    z, weights = np.polynomial.legendre.leggauss(8)
    times = ((edges[:-1, None]+edges[1:, None])/2+
             np.diff(edges)[:, None]*z/2).ravel()
    weights = (np.diff(edges)[:, None]*weights/2).ravel()
    actual = source.value(times)
    recovered = np.interp(times, basis.knots, q_reference*np.asarray(coefficients))
    error = recovered-actual
    true_mass = source.integral(0, 3)
    recovered_mass = float(np.trapezoid(q_reference*np.asarray(coefficients), basis.knots))
    return dict(relative_L2=float(np.sqrt(np.sum(weights*error**2)/np.sum(weights*actual**2))),
        relative_mass_error=float(abs(recovered_mass-true_mass)/true_mass),
        estimated_mass=recovered_mass, true_mass=true_mass)


def _compute_source(settings, protocol, output, source_id, bindings):
    """Выполнить сравнение регуляризаций для одного источника.

    Подгонка и выбор используют разные шумовые панели. Истинный источник
    используется при последующей оценке ошибок.

    Parameters
    ----------
    settings : dict
        Прочитанные настройки запуска.
    protocol : ProtocolBinding
        Протокол с проверенным хешем.
    output : Path
        Каталог результатов, защищённый вызывающим кодом.
    source_id : str
        Зарегистрированный источник из настроек.
    bindings : dict
        Хеши кода, конфигурации и протокола для результатов.

    Returns
    -------
    source_id : str
        Идентификатор обработанного источника.
    status : {'completed'}
        Новый результат записан в JSON.
    """

    started = perf_counter()
    destination = output/(source_id+".json")
    print(f"Начало расчёта: источник {source_id}", flush=True)
    source = build_truth(protocol, source_id).source
    spec = protocol.document.to_dict()["execution_config"]
    parameters = execution_parameters(protocol)
    solver = state_solver(protocol, settings,
        settings["inverse_spacing_km"], settings["inverse_steps"])
    basis = P1Basis(np.array(spec["basis"]["knots_hours"]), time_unit="h")
    load = IntervalAverageLoad(basis, solver.times,
        origin=spec["model"]["extended_start_hours"],
        q_reference=parameters["q_reference"], history_integral=source.integral,
        history_initial=source.value(-.5), source_unit="C*km^2/hour")
    model = Prediction(protocol, solver, basis, load)
    truth_y, checks = check_model(model, protocol, settings, source,
                                solver=solver, load=load)
    zero = np.zeros(model.basis.size)
    offset = model.predict(zero).copy()
    jacobian = model.jacobian(zero)
    grams = penalty_grams(model.basis, load.q_reference, tau_hours=parameters["tau_hours"])
    forward_tolerance = protocol.document.to_dict()["solver"]["forward"]["scaled_residual_tolerance"]
    cases = []
    for replicate in settings["replicates"]:
        panels = {key: draw_panel(protocol, noise_id=settings["noise"], replicate=replicate,
            panel=key).values.ravel() for key in ("fit_obs", "select_obs", "test_obs")}
        calibration = np.stack([draw_panel(protocol, noise_id=settings["noise"],
            replicate=replicate, panel="calib_noise", index=i).values for i in range(parameters["calibration_panels"])])
        fit_y, select_y = truth_y+panels["fit_obs"], truth_y+panels["select_obs"]
        for weight in settings["weights"]:
            calibrated = fit_covariance(calibration, spec["observations"]["full_times_hours"],
                                         family=weight, specification=spec["calibration"])
            metric = covariance_metric(calibrated.covariance, layout=model.observations.space)
            information = metric.whiten_matrix(jacobian).T @ metric.whiten_matrix(jacobian)
            for arm, gram in grams.items():
                reference = float(np.trace(np.linalg.solve(gram, information))/model.basis.size)
                path = []
                for exponent in settings["alpha_exponents"]:
                    result = fit(model, fit_y, metric, gram, reference*10.**exponent,
                        jacobian, offset, tolerance=settings["kkt_tolerance"],
                        max_iterations=settings["max_iterations"])
                    result["forward_residual"] = float(model.trajectory.max_scaled_residual)
                    result["accepted"] = bool(result["accepted"] and
                        result["forward_residual"] <= forward_tolerance)
                    result["coefficients"] = result["coefficients"].tolist()
                    result["prediction"] = result["prediction"].tolist()
                    result["exponent"] = exponent
                    result["validation_mse"] = float(np.mean((np.array(result["prediction"])-select_y)**2))
                    path.append(result)
                eligible = [i for i, row in enumerate(path) if row["accepted"]]
                selected = min(eligible, key=lambda i: (path[i]["validation_mse"], -path[i]["alpha"])) if eligible else None
                case = dict(replicate=replicate, weight=weight, regularization=arm,
                    calibration=calibrated.provenance.to_dict(), alpha_reference=reference,
                    path=path, selected=selected, lcurve_diagnostic=lcurve_corner(path),
                    complete_path=len(eligible) == len(path))
                if selected is not None:
                    chosen = path[selected]
                    case["selected_at_boundary"] = selected in (0, len(path)-1)
                    case["metrics"] = source_metrics(source, model.basis, chosen["coefficients"],
                        q_reference=load.q_reference)
                    case["metrics"].update(noiseless_prediction_rmse=float(np.sqrt(np.mean(
                        (np.array(chosen["prediction"])-truth_y)**2))),
                        test_rmse=float(np.sqrt(np.mean((np.array(chosen["prediction"])-truth_y-panels["test_obs"])**2))))
                cases.append(case)
                print(f"Восстановление: {source_id}, повтор {replicate}, вес {weight}, штраф {arm}; принято {len(eligible)}/{len(path)}, выбран индекс {selected}", flush=True)
    times = np.linspace(0, 3, 1801)
    result = dict(source=source_id, bindings=bindings, settings=settings, checks=checks,
        q_reference=load.q_reference,
        cases=cases, truth_observations=truth_y.tolist(),
        source_times=times.tolist(), source_values=source.value(times).tolist(),
        knots=model.basis.knots.tolist(), seconds=perf_counter()-started)
    save_json(destination, result)
    return source_id, "completed"


def _source_context(config_path, bindings):
    """Прочитать настройки источника из тех же байтов, что связаны с результатом."""

    if "config_sha256" in bindings:
        config_bytes = Path(config_path).read_bytes()
        if hashlib.sha256(config_bytes).hexdigest() != bindings["config_sha256"]:
            raise ValueError("Configuration changed after the run was prepared")
        context = setup(config_path, config_bytes=config_bytes)
    else:
        context = setup(config_path)
    if "protocol_sha256" in bindings and context[1].full_sha256 != bindings["protocol_sha256"]:
        raise ValueError("Protocol differs from the binding supplied for this source")
    return context


def _run_source_worker(config_path, source_id, bindings, *, context=None, expected_output=None):
    """Обработать источник под его блокировкой; общей блокировкой владеет родитель."""

    _validate_source_id(source_id)
    if context is None:
        context = _source_context(config_path, bindings)
    settings, protocol, output = context
    if expected_output is not None and output != Path(expected_output):
        raise ValueError("Worker output differs from the directory locked by its parent")
    if source_id not in _configured_sources(settings):
        raise ValueError("Source must be listed in configuration")
    output.mkdir(parents=True, exist_ok=True)
    with file_locks.FileLock(output/(source_id+".json.lock")):
        record = _existing_run(output, bindings)
        if record is not None:
            _verify_recorded_outputs(output, record, source_id=source_id)
        destination = output/(source_id+".json")
        pinned = record.get("outputs", {}) if record is not None else {}
        if destination.exists():
            data = destination.read_bytes()
            if (destination.name in pinned
                    and hashlib.sha256(data).hexdigest() != pinned[destination.name]):
                raise ValueError("Previously pinned source output has changed")
            previous = strict_json(data)
            if not isinstance(previous, dict):
                raise ValueError("Source output must be a JSON object")
            if previous.get("source") != source_id or previous.get("bindings") != bindings:
                raise ValueError("Output belongs to different code/config; use a new output directory")
            return source_id, "reused"
        if destination.name in pinned:
            raise ValueError("Previously pinned source output is missing")
        return _compute_source(settings, protocol, output, source_id, bindings)


def run_source(config_path, source_id, bindings):
    """Рассчитать или прочитать результат одного источника из конфигурации.

    Parameters
    ----------
    config_path : str or path-like
        Файл настроек запуска; относительные пути протокола и результатов
        разрешаются относительно этого файла.
    source_id : str
        Идентификатор источника, объявленный в sources конфигурации.
    bindings : dict
        Привязки кода, конфигурации и протокола для записи результата.
        При наличии config_sha256 и protocol_sha256 эти значения сверяются
        с прочитанными входными файлами.

    Returns
    -------
    source_id : str
        Идентификатор обработанного источника.
    status : {'reused', 'completed'}
        Прочитан согласованный сохранённый результат либо записан новый.

    Notes
    -----
    Общая блокировка запуска удерживается до завершения операции. Если
    run.json существует, его закреплённые результаты проверяются и запись
    обновляется. Без run.json сохраняется только результат источника.
    """

    _validate_source_id(source_id)
    context = _source_context(config_path, bindings)
    output = context[2]
    output.mkdir(parents=True, exist_ok=True)
    with file_locks.FileLock(output/".run.lock"):
        record = _existing_run(output, bindings)
        if record is not None:
            _verify_recorded_outputs(output, record)
        result = _run_source_worker(config_path, source_id, bindings, context=context)
        if record is not None:
            _update_run(output, record)
        return result


def main(argv=None):
    """Запустить сравнение источников из командной строки.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Аргументы командной строки; None читает аргументы процесса.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON научной конфигурации сравнения регуляризаций")
    parser.add_argument("--source", help="Выполнить расчёт для одного источника из конфигурации; незавершённый расчёт можно продолжить")
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    config_bytes = config_path.read_bytes()
    settings, protocol, output = setup(config_path, config_bytes=config_bytes)
    roots = {"adrkit": Path(adrkit.__file__).resolve().parent,
             "experiments/source_comparison": Path(__file__).resolve().parent}
    files = {name+"/"+path.relative_to(root).as_posix(): digest(path)
             for name, root in roots.items() for path in sorted(root.rglob("*.py"))
             if path.name != "figures.py"}
    files["experiments/file_locks.py"] = digest(file_locks.__file__)
    bindings = dict(config_sha256=hashlib.sha256(config_bytes).hexdigest(), protocol_sha256=protocol.full_sha256,
                    code_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
    configured = _configured_sources(settings)
    sources = [args.source] if args.source else configured
    if not set(sources) <= set(configured):
        parser.error("Источник должен быть указан в конфигурации")
    output.mkdir(parents=True, exist_ok=True)
    with file_locks.FileLock(output/".run.lock"):
        previous = _existing_run(output, bindings)
        if previous is not None:
            _verify_recorded_outputs(output, previous)
        record = dict(bindings=bindings, configuration=settings, code_files=files,
            versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
            threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
            status="running", outputs=previous.get("outputs", {}) if previous is not None else {})
        save_json(output/"run.json", record)
        with ProcessPoolExecutor(max_workers=min(settings["workers"], len(sources))) as pool:
            pending = [pool.submit(_run_source_worker, str(config_path), source, bindings,
                                   expected_output=str(output)) for source in sources]
            for future in as_completed(pending):
                print("Расчёт завершён:", *future.result(), flush=True)
        _update_run(output, record)


if __name__ == "__main__":
    main()
