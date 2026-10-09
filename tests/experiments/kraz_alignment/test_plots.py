"""Проверки состава рисунков и экспорта сохранённых данных КрАЗ."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from experiments.kraz_alignment import plots, run as kraz
from experiments.source_comparison.run import setup
from experiments.source_comparison.truth import build_truth


@pytest.fixture
def plot_inputs(short_inputs):
    config, simulation, output = short_inputs
    _, protocol, _ = setup(simulation)
    source_id = "SF01-PG10"
    times = np.arange(1, 73)/24
    coarse = np.arange(7, 72, 8)
    records, selected, overlays = [], [], []
    for factor in (0.5, 1.0):
        records.extend(dict(source=source_id, amplitude_factor=factor,
            relative_hours=t, concentration=c) for t, c in zip(times, factor*times**2))
    for split, factor, background in (("exploration", 0.5, 7.25), ("later_period", 1., 2.125)):
        values = factor*times[coarse]**2+background+np.arange(9)*0.03125
        baseline_offset = max(0., np.mean(values[:6]-times[coarse][:6]**2))
        selected.append(dict(source=source_id, split=split,
                             amplitude_factor=factor, background=background))
        overlays.extend(dict(source=source_id, split=split, relative_hours=t,
            used_for_fit=i < 6, raw_pm25=values[i],
            unscaled_prediction=times[coarse][i]**2+baseline_offset,
            constant_prediction=values[:6].mean()) for i, t in enumerate(times[coarse]))
    return config, simulation, output, protocol, (
        pd.DataFrame(records), pd.DataFrame(selected), pd.DataFrame(overlays))


def _snapshot(figure):
    return [dict(lines=[np.array(line.get_xydata(), copy=True) for line in axis.lines],
                 points=[np.array(collection.get_offsets(), copy=True)
                         for collection in axis.collections]) for axis in figure.axes]


def _save_tables(config, simulation, folder, protocol, tables):
    folder.mkdir()
    for name, table in zip(("templates.csv", "selected_windows.csv", "overlays.csv"), tables):
        table.to_csv(folder/name, index=False)
    for name in ("all_windows.csv", "summary.csv"):
        tables[1].to_csv(folder/name, index=False)
    manifest = dict(status="completed", config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        simulation_config_sha256=hashlib.sha256(simulation.read_bytes()).hexdigest(),
        protocol_sha256=protocol.full_sha256,
        artifacts={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in folder.iterdir()})
    (folder/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_figure_explains_original_curves_with_brief_legends(plot_inputs, monkeypatch):
    _, _, output, protocol, tables = plot_inputs
    output.mkdir()
    figures = []
    original = plt.Figure.savefig

    def save(figure, path, **kwargs):
        if Path(path).suffix == ".png":
            assert not figure.texts
            for index, axis in enumerate(figure.axes):
                assert axis.get_title() == "" and not axis.texts
                legend = axis.get_legend()
                assert legend is not None
                if index < len(tables[1]):
                    expected_labels = ["Заданный профиль", "Граница подбора"]
                    assert axis.lines[0].get_label() == "Заданный профиль"
                else:
                    expected_labels = ["Модельный прогноз", "Прогноз при исходной интенсивности",
                                       "Измерения для подбора", "Отложенные измерения",
                                       "Постоянный прогноз", "Граница подбора"]
                    assert [line.get_label() for line in axis.lines] == [
                        "Модельный прогноз", "Прогноз при исходной интенсивности",
                        "Постоянный прогноз", "Граница подбора"]
                    assert [collection.get_label() for collection in axis.collections] == [
                        "Измерения для подбора", "Отложенные измерения"]
                assert [label.get_text() for label in legend.get_texts()] == expected_labels
            figures.append(_snapshot(figure))
            assert figure.axes[0].get_ylabel() == r"$q(t)$, $C\cdot$км²/ч"
            assert figure.axes[2].get_xlabel() == r"$t$, ч"
            assert figure.axes[2].get_ylabel() == r"$\mathrm{PM}_{2.5}$, единицы исходного ряда"
        return original(figure, path, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", save)
    paths = plots.plot(*tables, protocol, output)
    assert {path.suffix for path in paths} == {".png", ".pdf"}
    assert all(path.is_file() for path in paths)
    assert len(figures) == 1
    source = build_truth(protocol, "SF01-PG10").source
    templates, selected, overlays = tables
    for column, record in enumerate(selected.itertuples(index=False)):
        profile = figures[0][column]["lines"][0]
        times = np.linspace(0, 3, 361)
        expected = record.amplitude_factor*source.amplitude*(
            (times >= source.events[0]) & (times < source.events[1]))
        np.testing.assert_array_equal(profile, np.column_stack((times, expected)))
        dense = templates[templates.amplitude_factor == record.amplitude_factor]
        points = overlays[overlays.split == record.split]
        concentration = figures[0][2+column]
        np.testing.assert_array_equal(concentration["lines"][0],
            np.column_stack((dense.relative_hours, dense.concentration+record.background)))
        assert len(concentration["lines"][0]) == len(concentration["lines"][1]) == 72
        for collection, mask in zip(concentration["points"],
                                    (points.used_for_fit, ~points.used_for_fit)):
            np.testing.assert_array_equal(collection,
                np.column_stack((points.relative_hours[mask], points.raw_pm25[mask])))
        assert [len(collection) for collection in concentration["points"]] == [6, 3]
        np.testing.assert_array_equal(concentration["lines"][2][:, 1],
                                     np.repeat(points.constant_prediction.iloc[0], 2))


def test_saved_export_preserves_all_artist_arrays_without_solving(plot_inputs, monkeypatch):
    config, simulation, output, protocol, tables = plot_inputs
    source = config.parent/"saved"
    _save_tables(config, simulation, source, protocol, tables)
    snapshots = []

    def save(figure, path, **kwargs):
        if Path(path).suffix == ".png":
            snapshots.append(_snapshot(figure))

    def solve(*args, **kwargs):
        raise AssertionError("Экспорт не должен решать уравнение или выбирать шаблон")

    monkeypatch.setattr(plt.Figure, "savefig", save)
    monkeypatch.setattr(kraz, "template_bank", solve)
    monkeypatch.setattr(kraz, "choose_templates", solve)
    plots.plot(*tables, protocol, output)
    plots.export(config, source, output)
    assert len(snapshots) == 2
    for path in source.glob("*.csv"):
        assert (output/path.name).read_bytes() == path.read_bytes()
    for direct, saved in zip(*snapshots):
        for kind in ("lines", "points"):
            assert len(direct[kind]) == len(saved[kind])
            for before, after in zip(direct[kind], saved[kind]):
                np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("changed", ["config", "simulation", "templates.csv", "all_windows.csv"])
def test_saved_export_rejects_changed_inputs(plot_inputs, changed):
    config, simulation, output, protocol, tables = plot_inputs
    source = config.parent/"saved"
    _save_tables(config, simulation, source, protocol, tables)
    path = {"config": config, "simulation": simulation}.get(changed, source/changed)
    path.write_bytes(path.read_bytes()+b"\n")
    with pytest.raises(ValueError, match="не соответствует|не соответствуют"):
        plots.export(config, source, output)
    assert not output.exists()


def test_saved_export_rejects_incomplete_run(plot_inputs):
    config, simulation, output, protocol, tables = plot_inputs
    source = config.parent/"saved"
    _save_tables(config, simulation, source, protocol, tables)
    manifest_path = source/"manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["status"] = "partial"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="завершённый расчёт"):
        plots.export(config, source, output)
    assert not output.exists()


def test_saved_export_rejects_output_inside_saved_input(plot_inputs):
    config, simulation, _, protocol, tables = plot_inputs
    source = config.parent/"saved"
    _save_tables(config, simulation, source, protocol, tables)
    with pytest.raises(ValueError, match="пересекает"):
        plots.export(config, source, source/"figures")
    assert not (source/"figures").exists()
