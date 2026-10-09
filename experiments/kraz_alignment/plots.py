"""Рисунки заданного источника и прогноза концентрации на посту КрАЗ."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.source_comparison.run import setup
from experiments.source_comparison.truth import build_truth


def plot(templates, selected, overlays, protocol, output):
    """Построить профили источника и сохранённые прогнозы концентрации.

    Parameters
    ----------
    templates : pandas.DataFrame
        Плотные отклики в столбцах source, amplitude_factor,
        relative_hours и concentration.
    selected : pandas.DataFrame
        Выбранные окна с источником, периодом, множителем и добавкой.
    overlays : pandas.DataFrame
        Девять измерений каждого окна, их времена, признак подгонки
        и постоянный прогноз.
    protocol : ProtocolBinding
        Протокол аналитических профилей источника.
    output : path-like
        Существующий каталог для PNG и PDF.

    Returns
    -------
    paths : tuple of pathlib.Path
        Имена созданных рисунков в порядке источников и форматов.
    """
    output = Path(output)
    source_times = np.linspace(0, 3, 361)
    paths = []
    for source_id, records in selected.groupby("source", sort=False):
        source = build_truth(protocol, source_id).source
        bank = templates[templates.source == source_id]
        baseline = bank[bank.amplitude_factor == 1.].sort_values("relative_hours")
        baseline_values = baseline.set_index("relative_hours").concentration
        figure, axes = plt.subplots(2, len(records), squeeze=False,
                                   figsize=(6.5*len(records), 6.8), sharex="col")
        try:
            for column, record in enumerate(records.itertuples(index=False)):
                points = overlays[(overlays.source == source_id)
                                  & (overlays.split == record.split)].sort_values("relative_hours")
                dense = bank[bank.amplitude_factor == record.amplitude_factor].sort_values("relative_hours")
                fitted = points.used_for_fit.to_numpy(dtype=bool)
                baseline_at_points = baseline_values.loc[points.relative_hours].to_numpy()
                baseline_offset = max(0., np.mean(points.raw_pm25.to_numpy()[fitted]
                                                 - baseline_at_points[fitted]))
                axes[0, column].plot(source_times,
                    record.amplitude_factor*source.value(source_times), color="C1",
                    label="Заданный профиль")
                axes[0, column].set_ylabel(r"$q(t)$, $C\cdot$км²/ч")
                axis = axes[1, column]
                axis.plot(dense.relative_hours, dense.concentration+record.background,
                          label="Модельный прогноз")
                axis.plot(baseline.relative_hours, baseline.concentration+baseline_offset,
                          "--", color=".5", label="Прогноз при исходной интенсивности")
                axis.scatter(points.relative_hours[fitted], points.raw_pm25[fitted],
                             color="black", s=25, label="Измерения для подбора")
                axis.scatter(points.relative_hours[~fitted], points.raw_pm25[~fitted],
                             color="red", marker="x", s=45, label="Отложенные измерения")
                axis.axhline(points.constant_prediction.iloc[0], color=".6", ls=":",
                             label="Постоянный прогноз")
                axis.set(xlabel=r"$t$, ч", ylabel=r"$\mathrm{PM}_{2.5}$, единицы исходного ряда")
                for row in range(2):
                    axes[row, column].axvline(2, color=".6", ls=":", label="Граница подбора")
                    axes[row, column].legend(fontsize=7)
                    axes[row, column].grid(alpha=.2)
            figure.tight_layout()
            for extension in ("png", "pdf"):
                path = output/f"{source_id}.{extension}"
                figure.savefig(path, dpi=180)
                paths.append(path)
        finally:
            plt.close(figure)
    return tuple(paths)


def export(config_path, input_path, output_path):
    """Построить рисунки завершённого расчёта без решения уравнения.

    Parameters
    ----------
    config_path : path-like
        Исходная конфигурация расчёта, закреплённая в manifest.json.
    input_path : path-like
        Каталог manifest.json и сохранённых CSV.
    output_path : path-like
        Новый каталог рисунков и копий исходных CSV вне исходников
        и входных данных.

    Returns
    -------
    paths : tuple of pathlib.Path
        Имена созданных PNG и PDF.

    Raises
    ------
    ValueError
        Расчёт не завершён, входы не соответствуют манифесту либо
        выходной каталог пересекает защищённые пути или уже существует.
    """
    config_path, input_path, output_path = (
        Path(path).resolve() for path in (config_path, input_path, output_path))
    manifest = json.loads((input_path/"manifest.json").read_bytes())
    if manifest["status"] != "completed":
        raise ValueError("Требуется завершённый расчёт КрАЗ")
    config_bytes = config_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != manifest["config_sha256"]:
        raise ValueError("Конфигурация не соответствует сохранённому расчёту")
    config = json.loads(config_bytes)
    simulation_path = (config_path.parent/config["simulation_config"]).resolve()
    simulation_bytes = simulation_path.read_bytes()
    if hashlib.sha256(simulation_bytes).hexdigest() != manifest["simulation_config_sha256"]:
        raise ValueError("Настройки модели не соответствуют сохранённому расчёту")
    settings, protocol, _ = setup(simulation_path, config_bytes=simulation_bytes)
    if protocol.full_sha256 != manifest["protocol_sha256"]:
        raise ValueError("Протокол не соответствует сохранённому расчёту")
    protected = [Path(__file__).resolve().parents[2], input_path, config_path,
                 simulation_path, simulation_path.parent/settings["protocol"],
                 config_path.parent/config["data_root"],
                 *(config_path.parent/entry for entry in config.get("protected_roots", []))]
    if any(output_path.is_relative_to(path.resolve())
           or path.resolve().is_relative_to(output_path) for path in protected):
        raise ValueError("Каталог рисунков пересекает исходники или входные данные")
    if output_path.exists():
        raise ValueError("Для рисунков требуется новый каталог")
    table_names = ("templates.csv", "selected_windows.csv", "overlays.csv")
    buffers = {}
    for name in (*table_names, "all_windows.csv", "summary.csv"):
        raw = (input_path/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["artifacts"][name]:
            raise ValueError(f"Таблица не соответствует манифесту: {name}")
        buffers[name] = raw
    output_path.mkdir(parents=True)
    for name, raw in buffers.items():
        (output_path/name).write_bytes(raw)
    tables = [pd.read_csv(BytesIO(buffers[name]), float_precision="round_trip")
              for name in table_names]
    return plot(*tables, protocol, output_path)


def main():
    """Прочитать пути входов и построить рисунки завершённого расчёта."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Исходная конфигурация завершённого расчёта")
    parser.add_argument("--input", type=Path, required=True, help="Каталог завершённого расчёта с манифестом и пятью CSV")
    parser.add_argument("--output", type=Path, required=True, help="Новый каталог рисунков и копий таблиц")
    args = parser.parse_args()
    export(args.config, args.input, args.output)


if __name__ == "__main__":
    main()
