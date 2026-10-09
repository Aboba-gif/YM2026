"""Сопоставление заданных полулинейных откликов с непересекающимися окнами измерений КрАЗ."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import adrkit
import timeseries
from timeseries.metrics import build_metrics

from experiments.source_comparison.observations import build_observations
from experiments.source_comparison.truth import build_truth, interval_controls
from adrkit.observations.grid import apply_observation
from experiments.source_comparison.prediction import state_solver
from experiments.source_comparison.run import digest, save_json, setup
from experiments.pm25_imputation.data import load_network
from .plots import plot


def template_bank(settings, protocol, factors):
    """Решить полулинейное уравнение для множителей профиля источника.

    Parameters
    ----------
    settings : dict
        Настройки сетки и список заданных источников.
    protocol : ProtocolBinding
        Протокол модели и геометрии наблюдений.
    factors : sequence of float
        Непустая последовательность безразмерных множителей
        интенсивности источника.

    Returns
    -------
    times : ndarray, shape (72,)
        Физические моменты наблюдений от начала окна, в часах.
    coarse : ndarray, shape (9,)
        Индексы наблюдений через каждые 20 минут.
    curves : dict of str to ndarray
        Концентрации на посту КрАЗ в единицах модели; массив каждого
        источника имеет форму (n_factors, 72).
    records : pandas.DataFrame
        Таблица множителей, времён, интенсивностей источника и концентраций.
    max_residual : float
        Максимальная масштабированная невязка решателя; безразмерная.
    """
    solver = state_solver(protocol, settings, settings["truth_spacing_km"],
                          settings["truth_steps"])
    design = build_observations(protocol, solver, regime="dense")
    spec = protocol.document.to_dict()["execution_config"]["observations"]
    station = spec["station_names"].index("KrAZ")
    times = np.asarray(design.physical_times)
    coarse = np.arange(7, 72, 8)
    np.testing.assert_allclose(times[coarse], np.arange(1, 10)/3)
    records, curves, residuals = [], {}, []
    for source_id in settings["sources"]:
        source = build_truth(protocol, source_id).source
        controls = interval_controls(source, solver.times, origin=-.5)
        responses = []
        for factor in factors:
            print(f"Прямой расчёт: источник {source_id}, множитель {factor}", flush=True)
            trajectory = solver.solve_controls(factor*controls)
            predicted = apply_observation(trajectory.states, design.space_weights,
                                          design.time_weights).reshape(4, -1)[station]
            residuals.append(trajectory.max_scaled_residual)
            responses.append(predicted)
            records.extend(dict(source=source_id, amplitude_factor=factor,
                relative_hours=t, q=factor*source.value(t), concentration=c)
                for t, c in zip(times, predicted))
        curves[source_id] = np.asarray(responses)
    return times, coarse, curves, pd.DataFrame(records), max(residuals)


def choose_templates(values, templates, fit_points):
    """Выбрать шаблон и неотрицательную постоянную добавку.

    Добавка и шаблон минимизируют сумму квадратов невязок на участке
    подгонки. Остальные точки не влияют на выбор. Для постоянного участка
    подгонки нормирующий знаменатель равен нулю.

    Parameters
    ----------
    values : ndarray, shape (n_windows, n_times)
        Значения концентрации в окнах.
    templates : ndarray, shape (n_templates, n_times)
        Прогнозы в тех же временах и единицах концентрации.
    fit_points : int
        Положительное число первых точек для подгонки и выбора,
        не больше n_times.

    Returns
    -------
    choices : ndarray, shape (n_windows,)
        Индексы выбранных шаблонов; при равенстве выбирается первый.
    background : ndarray, shape (n_windows,)
        Неотрицательные добавки в единицах концентрации.
    prediction : ndarray, shape (n_windows, n_times)
        Полные прогнозы выбранных шаблонов с добавкой.
    scores : ndarray, shape (n_windows,)
        Безразмерная сумма квадратов невязок, делённая на центрированную
        сумму квадратов первых fit_points наблюдений.
    """
    y = values[:, :fit_points]
    offsets = np.maximum(0., y.mean(axis=1)[:, None]
                         - templates[:, :fit_points].mean(axis=1)[None, :])
    residual = y[:, None, :] - templates[None, :, :fit_points] - offsets[:, :, None]
    losses = np.sum(residual**2, axis=2)
    choices = np.argmin(losses, axis=1)
    rows = np.arange(len(y))
    background = offsets[rows, choices]
    prediction = templates[choices] + background[:, None]
    denominator = np.sum((y-y.mean(axis=1)[:, None])**2, axis=1)
    scores = losses[rows, choices]/denominator
    return choices, background, prediction, scores


def run(config_path):
    """Сохранить результаты сопоставления модельных откликов с окнами КрАЗ.

    Первые шесть значений каждого трёхчасового окна используются для
    подгонки, последние три — для прогноза. Результаты записываются в
    таблицы, рисунки и manifest.json.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Путь к конфигурации входов, окон и нового выходного каталога.

    Raises
    ------
    ValueError
        Выходной каталог пересекает защищённые пути или уже существует,
        настройки не соответствуют постановке либо конечная проверка
        обнаружила изменение файлов настроек или этого модуля.
    """

    config_path = Path(config_path).resolve()
    runner_sha256 = digest(__file__)
    config_bytes = config_path.read_bytes()
    cfg = json.loads(config_bytes)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    simulation = (config_path.parent/cfg["simulation_config"]).resolve()
    simulation_bytes = simulation.read_bytes()
    simulation_sha256 = hashlib.sha256(simulation_bytes).hexdigest()
    settings, protocol, _ = setup(simulation, config_bytes=simulation_bytes)
    output = (config_path.parent/cfg["output"]).resolve()
    data_root = (config_path.parent/cfg["data_root"]).resolve()
    protected = [Path(__file__).resolve().parents[2], data_root,
                 Path(adrkit.__file__).resolve().parent,Path(timeseries.__file__).resolve().parent,
                 *(config_path.parent/entry for entry in cfg.get("protected_roots", []))]
    if any(output.is_relative_to(root.resolve()) or root.resolve().is_relative_to(output)
           for root in protected):
        raise ValueError("Output must not overlap source or protected input directories")
    if output.exists():
        raise ValueError("Use a new output directory; completed evidence is immutable")
    factors = np.array(cfg["amplitude_factors"])
    nf = cfg["fit_points"]
    if cfg["window_hours"] != 3 or nf != 6 or 1.0 not in factors:
        raise ValueError("This study requires 3h windows,6 fit points and amplitude1 baseline")
    times, coarse, curves, templates, max_residual = template_bank(settings, protocol, factors)
    index, frames, raw_manifest = load_network(data_root)
    starts = np.arange(0, len(index)-9, 9)
    windows = frames["KrAZ"].pm25.to_numpy()[starts[:, None]+np.arange(1, 10)]
    complete = np.isfinite(windows).all(axis=1)
    fit_range = np.ptp(windows[:, :nf], axis=1)
    eligible = complete & (fit_range >= cfg["minimum_fit_range"])
    window_clock = index[starts]
    output.mkdir(parents=True)
    templates.to_csv(output/"templates.csv", index=False)
    all_tables, selected, overlays, summaries = [], [], [], []
    holdout_metrics = build_metrics([{"kind": "rmse", "parameters": {}}])
    for source_id, dense in curves.items():
        for split, bounds in cfg["splits"].items():
            period = (window_clock >= bounds[0]) & (index[starts+9] < bounds[1])
            ids = np.flatnonzero(period & eligible)
            y = windows[ids]
            if not len(y):
                raise ValueError(f"No eligible raw windows in {split}")
            choice, background, predicted, score = choose_templates(y, dense[:, coarse], nf)
            baseline = y[:, :nf].mean(axis=1)
            holdout_rmse = np.sqrt(np.mean((predicted[:, nf:]-y[:, nf:])**2, axis=1))
            baseline_rmse = np.sqrt(np.mean((baseline[:, None]-y[:, nf:])**2, axis=1))
            table = pd.DataFrame(dict(source=source_id, split=split,
                window_start=window_clock[ids].astype(str),
                window_end=index[starts[ids]+9].astype(str),
                amplitude_factor=factors[choice], background=background,
                amplitude_grid_edge=(choice == 0) | (choice == len(factors)-1),
                fit_normalized_sse=score, fit_rmse=np.sqrt(score*np.var(y[:, :nf], axis=1)),
                holdout_rmse=holdout_rmse, constant_holdout_rmse=baseline_rmse,
                all_rmse=np.sqrt(np.mean((predicted-y)**2, axis=1)),
                observed_peak_minutes=20*(np.argmax(y, axis=1)+1),
                predicted_peak_minutes=20*(np.argmax(predicted, axis=1)+1)))
            table["peak_difference_minutes"] = table.predicted_peak_minutes-table.observed_peak_minutes
            best = int(np.argmin(score))  # Минимум ошибки подгонки среди полных окон; при равенстве — первое по времени.
            record = table.iloc[best].to_dict()
            record["selection_rule"] = cfg["window_selection"]
            record["candidates"] = len(ids)
            base_id = int(np.flatnonzero(factors == 1.)[0])
            base_curve = dense[base_id, coarse]
            base_offset = max(0., np.mean(y[best, :nf]-base_curve[:nf]))
            record["unscaled_holdout_rmse"] = holdout_metrics.score(
                y[best, nf:], base_curve[nf:]+base_offset)["rmse"]
            selected.append(record)
            all_tables.append(table)
            summaries.append(dict(source=source_id, split=split,
                calendar_windows=int(period.sum()), complete_windows=int((period & complete).sum()),
                eligible_windows=len(ids), median_holdout_rmse=float(np.median(holdout_rmse)),
                median_constant_holdout_rmse=float(np.median(baseline_rmse)),
                fraction_beating_constant=float(np.mean(holdout_rmse < baseline_rmse)),
                fraction_amplitude_grid_edge=float(table.amplitude_grid_edge.mean())))
            origin = window_clock[ids[best]]
            for j, t in enumerate(times[coarse]):
                overlays.append(dict(source=source_id, split=split,
                    timestamp=str(origin+pd.Timedelta(minutes=20*(j+1))),
                    relative_hours=t, used_for_fit=j < nf, raw_pm25=y[best, j],
                    prediction=predicted[best, j], unscaled_prediction=base_curve[j]+base_offset,
                    constant_prediction=baseline[best]))
    pd.concat(all_tables, ignore_index=True).to_csv(output/"all_windows.csv", index=False)
    pd.DataFrame(selected).to_csv(output/"selected_windows.csv", index=False)
    pd.DataFrame(overlays).to_csv(output/"overlays.csv", index=False)
    pd.DataFrame(summaries).to_csv(output/"summary.csv", index=False)
    plot(templates, pd.DataFrame(selected), pd.DataFrame(overlays), protocol, output)
    payload = dict(status="completed", scope=cfg["scope"], configuration=cfg,
        config_sha256=config_sha256, runner_sha256=runner_sha256,
        simulation_config_sha256=simulation_sha256, protocol_sha256=protocol.full_sha256,
        raw=raw_manifest, max_scaled_pde_residual=float(max_residual),
        selected=selected, summary=summaries)
    payload["artifacts"] = {p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    for path, expected in ((config_path, config_sha256),
                           (simulation, simulation_sha256),
                           (Path(__file__), runner_sha256)):
        if digest(path) != expected:
            raise ValueError(f"Input or runner changed during KrAZ run: {path}")
    save_json(output/"manifest.json", payload)
    print(json.dumps(dict(selected=selected, summary=summaries), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON данных, модели, периодов поиска и нового каталога результата")
    run(parser.parse_args().config)
