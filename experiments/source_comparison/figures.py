"""Таблицы CSV и рисунки по сохранённым численным результатам."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

from experiments.source_comparison.calibration import draw_panel
from .run import setup, digest, save_json
from .config import execution_parameters


COLORS = {"L2": "#246ca6", "H1": "#d05c32"}
REGULARIZATIONS = {"L2": r"$L^2$", "H1": r"$H^1$"}
WEIGHTS = {"W01": "Общая дисперсия", "W02": "Дисперсии постов", "W03": "Временная ковариация"}
STATIONS = {"KrAZ": "КрАЗ", "Severny": "Северный", "Peschanka": "Песчанка", "Soloncy": "Солонцы"}


def csv_rows(path, rows):
    """Записать непустую последовательность словарей в CSV с общим заголовком."""

    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finish(figure, output, name):
    """Сохранить рисунок в PDF и PNG, затем закрыть его."""

    for suffix in ("pdf", "png"):
        figure.savefig(output/f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def selected_case(record, weight, arm, replicate=None):
    """Вернуть единственный случай; без replicate взять первый объявленный."""

    if replicate is None:
        replicate = record["settings"]["replicates"][0]
    cases = [c for c in record["cases"] if
        (c["weight"], c["regularization"], c["replicate"]) == (weight, arm, replicate)]
    if len(cases) != 1:
        raise ValueError("Exactly one configured source case is required")
    return cases[0]


def make_tables(records, output, *, q_reference):
    """Сохранить таблицы путей регуляризации и оценок источника.

    Медианы и диапазоны по реализациям — описательные показатели, а не
    доверительные интервалы.

    Parameters
    ----------
    records : sequence of dict
        Сохранённые результаты run_source по источникам.
    output : Path
        Каталог для CSV и TABLES.md.
    q_reference : float
        Проверенный масштаб интенсивности из протокола, в C·км²/ч.

    Returns
    -------
    results : list of dict
        Выбранные результаты с метриками и отметками границы и неполного
        пути.
    """

    paths, results, coefficients, truths, grids, covariance = [], [], [], [], [], []
    for record in records:
        source = record["source"]
        truths.extend(dict(source=source, time_h=t, q_C_km2_per_h=q)
                      for t, q in zip(record["source_times"], record["source_values"]))
        for name, result in record["checks"].items():
            if isinstance(result, dict):
                grids.append(dict(source=source, comparison=name, **result))
        for case in record["cases"]:
            key = dict(source=source, replicate=case["replicate"], weight=case["weight"],
                       regularization=case["regularization"])
            for i, row in enumerate(case["path"]):
                paths.append(dict(**key, **{k: v for k, v in row.items() if
                    k not in ("coefficients", "prediction")},
                    selected=i == case["selected"], lcurve_diagnostic=i == case["lcurve_diagnostic"]))
            if case["selected"] is None:
                results.append(dict(**key, status="no_converged_candidate"))
                continue
            chosen = case["path"][case["selected"]]
            corner = case["lcurve_diagnostic"]
            results.append(dict(**key, alpha_validation=chosen["alpha"],
                alpha_lcurve_diagnostic=None if corner is None else case["path"][corner]["alpha"],
                boundary=case["selected_at_boundary"], complete_path=case["complete_path"],
                status="boundary_unresolved" if case["selected_at_boundary"] else
                    "selected" if case["complete_path"] else "incomplete_path", **case["metrics"]))
            coefficients.extend(dict(**key, time_h=t, q_C_km2_per_h=q_reference*a)
                for t, a in zip(record["knots"], chosen["coefficients"]))
            if case["regularization"] == "L2":
                calibrated = case["calibration"]
                covariance.extend(dict(source=source, replicate=case["replicate"],
                    weight=case["weight"], station=i, variance_C2=v,
                    correlation_h=calibrated["ell_hours"][i] if calibrated["ell_hours"] else None)
                    for i, v in enumerate(calibrated["station_variances"]))
    for name, rows in [("alpha_paths", paths), ("selected_results", results),
                       ("reconstructions", coefficients), ("model_sources", truths),
                       ("grid_checks", grids), ("calibration", covariance)]:
        csv_rows(output/f"{name}.csv", rows)
    lines = ["# Результаты дополнительных расчётов", "",
        "Синтетическая полулинейная ADR. Медиана [min; max] относительной L2-ошибки q.",
        "Запланированное число реализаций: " + "; ".join(
            f"{r['source']}: {len(r['settings']['replicates'])}" for r in records) + ".",
        "Это описательные числа, не доверительные интервалы и не универсальное ранжирование методов.",
        "Граничный выбор α и неполная траектория отмечены отдельно в selected_results.csv.", "",
        "| Источник | Вес | L2 | H1 | Граничных α / всего |", "|---|---|---:|---:|---:|"]
    for record in records:
        for weight in record["settings"]["weights"]:
            group = [r for r in results if r["source"] == record["source"] and r["weight"] == weight]
            formatted = []
            for arm in COLORS:
                values = [r["relative_L2"] for r in group if r["regularization"] == arm and "relative_L2" in r]
                formatted.append(f"{np.median(values):.3f} [{min(values):.3f}; {max(values):.3f}]; n={len(values)}" if values else "нет решения; n=0")
            lines.append(f"| {record['source']} | {weight} | {' | '.join(formatted)} | {sum(r.get('boundary', False) for r in group)}/{len(group)} |")
    lines += ["", "α_L — только диагностический максимум дискретной кривизны; устойчивость угла не установлена.",
              "Параметр α выбирается по отдельной проверочной реализации наблюдений. Истинный q используется только при оценке ошибки."]
    (output/"TABLES.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return results


def source_panels(records, output):
    """Сохранить рисунок аналитических профилей источника."""

    figure, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for axis, record in zip(axes.flat, records):
        axis.plot(record["source_times"], record["source_values"], color="#243b45",
                  label=f"Заданный профиль: {record['source']}")
        axis.set(xlabel=r"$t$, ч", ylabel=r"$q(t)$, C·км²/ч", xlim=(0, 3))
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    finish(figure, output, "model_sources")


def reconstruction_and_paths(record, output, *, q_reference):
    """Сохранить реконструкции с заданным Qref для первой объявленной реализации."""

    weights = record["settings"]["weights"]
    source = record["source"]
    figure, axes = plt.subplots(1, len(weights), figsize=(12, 3.6), layout="constrained", squeeze=False, sharey=True)
    lc, lc_axes = plt.subplots(2, len(weights), figsize=(12, 7), layout="constrained", squeeze=False)
    val, val_axes = plt.subplots(2, len(weights), figsize=(12, 7), layout="constrained", squeeze=False)
    for col, weight in enumerate(weights):
        axis = axes[0, col]
        axis.plot(record["source_times"], record["source_values"], color="#263640", lw=1.8,
                  label=f"Заданный профиль: {source}")
        for row, (arm, color) in enumerate(COLORS.items()):
            case = selected_case(record, weight, arm)
            path = case["path"]
            if case["selected"] is not None:
                chosen = path[case["selected"]]
                axis.plot(record["knots"], q_reference*np.array(chosen["coefficients"]),
                          color=color, label=REGULARIZATIONS[arm])
            a = lc_axes[row, col]
            a.loglog([p["residual_norm"] for p in path], [p["penalty_norm"] for p in path],
                     ".-", color=color, label=REGULARIZATIONS[arm])
            for ax, key in [(a.xaxis, "residual_norm"), (a.yaxis, "penalty_norm")]:
                ticks = np.geomspace(min(p[key] for p in path), max(p[key] for p in path), 4)
                ax.set_major_locator(FixedLocator(ticks))
                ax.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
                ax.set_minor_locator(NullLocator())
            b = val_axes[row, col]
            b.semilogx([p["alpha"] for p in path], [p["validation_mse"] for p in path],
                       ".-", color=color, label=REGULARIZATIONS[arm])
            for index, marker, label in [(case["selected"], "*", r"$\alpha$ по проверочным данным"),
                                         (case["lcurve_diagnostic"], "D", r"$\alpha_L$: по L-кривой")]:
                if index is not None:
                    point = path[index]
                    a.plot(point["residual_norm"], point["penalty_norm"], marker, ms=10, label=label)
                    b.plot(point["alpha"], point["validation_mse"], marker, ms=10, label=label)
            for ax in (a, b):
                ax.grid(alpha=.25)
                ax.legend(title=f"{source}\n{weight}: {WEIGHTS[weight]}",
                          fontsize=7, title_fontsize=8)
            a.set(xlabel=r"$\|R^{-1/2}(F(q)-y)\|$", ylabel=r"$\sqrt{a^T G a}$")
            b.set(xlabel=r"$\alpha$", ylabel="Среднеквадратичная ошибка, C²")
        axis.set(xlabel=r"$t$, ч", ylabel=r"$q(t)$, C·км²/ч", xlim=(0, 3))
        axis.grid(alpha=.25)
        axis.legend(title=f"{weight}: {WEIGHTS[weight]}", fontsize=7, title_fontsize=8)
    finish(figure, output, f"reconstruction_{source}")
    finish(lc, output, f"lcurves_{source}")
    finish(val, output, f"alpha_validation_{source}")


def concentration_panels(record, protocol, output):
    """Сохранить рисунок и CSV наблюдений и прогнозов на постах."""

    spec = protocol.document.to_dict()["execution_config"]["observations"]
    times, stations = spec["full_times_hours"], spec["station_names"]
    truth = np.array(record["truth_observations"]).reshape(len(stations), -1)
    replicate = record["settings"]["replicates"][0]
    weights = record["settings"]["weights"]
    weight = "W03" if "W03" in weights else weights[0]
    noise = draw_panel(protocol, noise_id=record["settings"]["noise"], replicate=replicate, panel="fit_obs").values
    figure, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    rows = []
    for i, (station, axis) in enumerate(zip(stations, axes.flat)):
        axis.scatter(times, truth[i]+noise[i], s=8, color=".65", label="Наблюдения")
        axis.plot(times, truth[i], color=".15", label="Без шума")
        arrays = {}
        for arm, color in COLORS.items():
            case = selected_case(record, weight, arm, replicate)
            if case["selected"] is not None:
                values = np.array(case["path"][case["selected"]]["prediction"]).reshape(len(stations), -1)[i]
                arrays[arm] = values
                axis.plot(times, values, color=color, label=REGULARIZATIONS[arm])
        rows.extend(dict(source=record["source"], station=station, time_h=t, noiseless_C=truth[i, j],
            observed_C=truth[i, j]+noise[i, j], **{arm: v[j] for arm, v in arrays.items()}) for j, t in enumerate(times))
        axis.set(xlabel=r"$t$, ч", ylabel="Концентрация, мкг/м³")
        axis.grid(alpha=.25)
        axis.legend(title=f"{record['source']}\n{STATIONS.get(station, station)}; {weight}",
                    fontsize=7, title_fontsize=8)
    finish(figure, output, f"concentrations_{record['source']}")
    csv_rows(output/f"concentrations_{record['source']}.csv", rows)


def weight_and_error_panels(records, results, output):
    """Сохранить рисунки весов наблюдения и ошибок восстановления."""

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    radius = np.linspace(-4, 4, 401)
    lag = np.linspace(0, 1, 401)
    axes[0].plot(radius, np.exp(-radius**2/2)/(2*np.pi), label="Гауссово ядро")
    axes[0].set(xlabel="Расстояние, км", ylabel=r"$H$, км⁻²")
    for weight in records[0]["settings"]["weights"]:
        c = selected_case(records[0], weight, "L2")["calibration"]
        axes[1].plot(range(4), 1/np.array(c["station_variances"]), "o-",
                     label=f"{weight}: {WEIGHTS[weight]}")
        if c["ell_hours"]:
            for i, length in enumerate(c["ell_hours"]):
                axes[2].plot(lag, np.exp(-lag/length),
                             label=f"{weight}, пост {i}; ℓ = {length:.2f} ч")
    axes[1].set(xlabel="Номер поста", ylabel=r"$\sigma^{-2}$, C⁻²")
    axes[2].set(xlabel=r"$\Delta t$, ч", ylabel="Корреляция")
    for axis in axes:
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend(fontsize=7)
    finish(figure, output, "observation_weights")
    figure, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for record, axis in zip(records, axes.flat):
        for i, weight in enumerate(record["settings"]["weights"]):
            for j, (arm, color) in enumerate(COLORS.items()):
                values = [r["relative_L2"] for r in results if r["source"] == record["source"] and
                          r["weight"] == weight and r["regularization"] == arm and "relative_L2" in r]
                x = i+(j-.5)*.22
                axis.scatter([x]*len(values), values, color=color, s=25,
                             label=f"Оценки, {REGULARIZATIONS[arm]}" if i == 0 else None)
                if values:
                    axis.plot([x-.05, x+.05], [np.median(values)]*2, color=color,
                              label=f"Медиана, {REGULARIZATIONS[arm]}" if i == 0 else None)
        axis.set(xticks=range(len(record["settings"]["weights"])),
                 xticklabels=record["settings"]["weights"], xlabel="Весовая матрица",
                 ylabel=r"$\|\widehat q-q\|_{L^2}/\|q\|_{L^2}$")
        axis.grid(alpha=.25)
        axis.legend(title=record["source"], fontsize=8, title_fontsize=8)
    finish(figure, output, "comparison_errors")


def grid_check_panels(records, output):
    """Сохранить отклонения прогнозов при изменении сетки и области."""

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    comparisons = {"inverse_vs_truth": "Рабочая и мелкая сетки",
                   "truth_vs_refined": "Мелкая и уточнённая сетки",
                   "compact_vs_original_domain": "Исходная и расширенная области"}
    for j, metric in enumerate(("rmse", "maximum")):
        axis = axes[j]
        for name, label in comparisons.items():
            axis.plot(range(len(records)), [r["checks"][name][metric] for r in records],
                      "o-", label=label)
        axis.axhline(.02 if j == 0 else .05, color=".4", ls="--", label="Порог отклонения")
        axis.set(yscale="log", ylabel=("Среднеквадратичное отклонение" if j == 0 else "Максимальное отклонение")+", мкг/м³",
                 xlabel="Профиль источника",
                 xticks=range(len(records)), xticklabels=[r["source"] for r in records])
        axis.tick_params(axis="x", labelrotation=20)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    finish(figure, output, "grid_checks")


def _validate_records(records, settings, parameters):
    """Проверить состав рисунков и масштаб до создания каталога вывода."""
    q_reference = parameters["q_reference"]
    replicates, weights = settings["replicates"], settings["weights"]
    if (type(replicates) is not list or not replicates
            or any(type(r) is not int or r < 0 for r in replicates)
            or len(set(replicates)) != len(replicates)
            or type(weights) is not list or not weights
            or any(w not in WEIGHTS for w in weights)
            or len(set(weights)) != len(weights) or not records):
        raise ValueError("Figures require nonempty unique configured replicates and registered weights")
    expected = {(w, arm, r) for w in weights for arm in COLORS for r in replicates}
    for record in records:
        if record["settings"] != settings:
            raise ValueError("Saved source settings differ from the rendering configuration")
        scale = record.get("q_reference")
        if type(scale) not in (int, float) or scale != q_reference:
            raise ValueError("Saved source q_reference differs from the rendering protocol")
        actual = [(c["weight"], c["regularization"], c["replicate"]) for c in record["cases"]]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Saved source cases must match all configured replicate/weight/penalty combinations")


def render(config, input_dir, output):
    """Сохранить рисунки и таблицы CSV по результатам сравнения источников.

    Parameters
    ----------
    config : path-like
        Файл конфигурации, соответствующий сохранённому расчёту.
    input_dir : path-like
        Каталог с описанием расчёта и результатами для каждого источника.
    output : path-like
        Новый каталог для рисунков, таблиц и файла с их контрольными суммами.
    """
    settings, protocol, _ = setup(config)
    input_dir, output = Path(input_dir), Path(output)
    if output.exists():
        raise ValueError('Use a new figure directory')
    record = json.loads((input_dir/'run.json').read_text())
    if record['status'] != 'completed':
        raise ValueError('Finish all configured sources before creating the report')
    if record['bindings']['config_sha256'] != digest(config):
        raise ValueError('Saved solutions and rendering configuration differ')
    if record['bindings']['protocol_sha256'] != protocol.full_sha256:
        raise ValueError('Saved solutions and rendering protocol differ')
    records = []
    for source in settings['sources']:
        path = input_dir/(source+'.json')
        if digest(path) != record['outputs'][path.name]:
            raise ValueError('Source output hash mismatch')
        result = json.loads(path.read_text())
        if result['bindings'] != record['bindings']:
            raise ValueError('Source/config/code binding mismatch')
        records.append(result)
    parameters = execution_parameters(protocol)
    _validate_records(records, settings, parameters)
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                         'axes.spines.top':False,'axes.spines.right':False})
    results = make_tables(records, output, q_reference=parameters["q_reference"])
    source_panels(records, output)
    for result in records:
        reconstruction_and_paths(result, output, q_reference=parameters["q_reference"])
        concentration_panels(result, protocol, output)
    weight_and_error_panels(records, results, output)
    grid_check_panels(records, output)
    save_json(output/'manifest.json', dict(renderer_sha256=digest(__file__),
        numerical_bindings=record['bindings'], files={p.name:digest(p) for p in sorted(output.iterdir())
        if p.is_file() and p.name != 'manifest.json'}))
    print(f'Рисунки и таблицы сохранены: {output}')
