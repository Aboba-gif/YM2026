"""Рисунки читают масштаб и состав опыта из проверенных сохранений; PDE не запускается."""
from copy import deepcopy
from types import SimpleNamespace
import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments.source_comparison import figures


def saved_run(project_root, tmp_path, *, q_reference, replicates, weights):
    original = project_root / "experiments/source_comparison/configs"
    protocol = json.loads((original / "protocol.json").read_bytes())
    protocol["execution_config"]["basis"]["Qref"] = q_reference
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, allow_nan=False), encoding="utf-8")
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    settings = json.loads((original / "experiment.json").read_bytes())
    settings.update(protocol="protocol.json", protocol_sha256=protocol_sha,
                    output="numerical-results", protected_roots=[],
                    sources=["SF01-PG10"], replicates=replicates, weights=weights)
    config = tmp_path / "experiment.json"
    config.write_text(json.dumps(settings, allow_nan=False), encoding="utf-8")
    bindings = dict(config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                    protocol_sha256=protocol_sha, code_sha256="a" * 64)
    observations = protocol["execution_config"]["observations"]
    size = len(observations["station_names"]) * len(observations["full_times_hours"])
    record = dict(source="SF01-PG10", bindings=bindings, settings=settings,
        checks={name: dict(rmse=.001, maximum=.002) for name in
                ("inverse_vs_truth", "truth_vs_refined", "compact_vs_original_domain")},
        source_times=[0., 1.5, 3.], source_values=[q_reference] * 3,
        knots=[0., 1.5, 3.], truth_observations=[1.] * size, cases=[])
    record["q_reference"] = q_reference
    for replicate in replicates:
        for weight in weights:
            for arm in ("L2", "H1"):
                prediction = float(10 * replicate + weights.index(weight) + (arm == "H1"))
                path = [dict(alpha=alpha, coefficients=[1., 1., 1.], prediction=[prediction] * size,
                    residual_norm=alpha, penalty_norm=4. - alpha, validation_mse=alpha)
                    for alpha in (1., 2., 3.)]
                record["cases"].append(dict(replicate=replicate, weight=weight,
                    regularization=arm, path=path, selected=1, lcurve_diagnostic=0,
                    selected_at_boundary=False, complete_path=True,
                    calibration=dict(station_variances=[1.] * len(observations["station_names"]), ell_hours=[]),
                    metrics=dict(relative_L2=0., relative_mass_error=0.,
                                 estimated_mass=3. * q_reference, true_mass=3. * q_reference)))
    inputs = tmp_path / "numerical-results"
    inputs.mkdir()
    source_file = inputs / "SF01-PG10.json"
    source_file.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
    manifest = dict(status="completed", bindings=bindings,
                    outputs={source_file.name: hashlib.sha256(source_file.read_bytes()).hexdigest()})
    (inputs / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    return config, inputs, record


@pytest.mark.parametrize("q_reference,replicates,weights", [
    (200., [1], ["W01"]),
    (100., [2, 1], ["W02", "W01"]),
    (100., [0, 1, 2], ["W01", "W02", "W03"]),
])
def test_render_uses_configured_scale_replicate_weights_and_counts(
        project_root, tmp_path, monkeypatch, q_reference, replicates, weights):
    config, inputs, record = saved_run(project_root, tmp_path,
        q_reference=q_reference, replicates=replicates, weights=weights)
    before = {p.name: p.read_bytes() for p in inputs.iterdir()}
    captured, noise_replicates = {}, []

    def finish(figure, output, name):
        assert not figure.texts
        for axis in figure.axes:
            assert not axis.get_title()
            if axis.lines or axis.collections:
                legend = axis.get_legend()
                assert legend is not None
                assert [text.get_text() for text in legend.get_texts()] == axis.get_legend_handles_labels()[1]
            assert not axis.texts
            assert not axis.tables
        captured[name] = [dict(
            lines=[(np.array(line.get_xdata(), copy=True), np.array(line.get_ydata(), copy=True))
                   for line in axis.lines],
            ticks=[tick.get_text() for tick in axis.get_xticklabels()],
            xlabel=axis.get_xlabel(), ylabel=axis.get_ylabel(),
            legend=axis.get_legend_handles_labels()[1],
            legend_title=axis.get_legend().get_title().get_text() if axis.get_legend() else "",
            xscale=axis.get_xscale(), yscale=axis.get_yscale())
            for axis in figure.axes]
        figures.plt.close(figure)

    def panel(protocol, *, noise_id, replicate, panel):
        assert noise_id == record["settings"]["noise"] and panel == "fit_obs"
        noise_replicates.append(replicate)
        observation = protocol.document.to_dict()["execution_config"]["observations"]
        return SimpleNamespace(values=np.zeros((len(observation["station_names"]),
                                               len(observation["full_times_hours"]))))

    monkeypatch.setattr(figures, "finish", finish)
    monkeypatch.setattr(figures, "draw_panel", panel)
    output = tmp_path / "figures"
    figures.render(config, inputs, output)
    assert noise_replicates == [replicates[0]]
    selected_weight = "W03" if "W03" in weights else weights[0]
    assert set(captured) == {
        "model_sources", "reconstruction_SF01-PG10", "lcurves_SF01-PG10",
        "alpha_validation_SF01-PG10", "concentrations_SF01-PG10",
        "observation_weights", "comparison_errors", "grid_checks"}
    for axis, weight in zip(captured["reconstruction_SF01-PG10"], weights):
        assert axis["legend"] == ["Заданный профиль: SF01-PG10", r"$L^2$", r"$H^1$"]
        assert axis["legend_title"].startswith(weight + ":")
        assert len(axis["lines"]) == 3
        for _, values in axis["lines"]:
            np.testing.assert_array_equal(values, np.full(3, q_reference))
    expected = [next(case for case in record["cases"]
                     if (case["weight"], case["regularization"], case["replicate"])
                     == (selected_weight, arm, replicates[0]))["path"][1]["prediction"]
                for arm in ("L2", "H1")]
    for axis, station in zip(captured["concentrations_SF01-PG10"],
                             ("Северный", "Песчанка", "Солонцы", "КрАЗ")):
        assert axis["legend"] == ["Наблюдения", "Без шума", r"$L^2$", r"$H^1$"]
        assert axis["legend_title"] == f"SF01-PG10\n{station}; {selected_weight}"
        assert len(axis["lines"]) == 3
        for (_, values), prediction in zip(axis["lines"][1:], expected):
            np.testing.assert_array_equal(values, np.full(values.size, prediction[0]))
    for axis in captured["lcurves_SF01-PG10"]:
        assert (axis["xscale"], axis["yscale"]) == ("log", "log")
        np.testing.assert_array_equal(axis["lines"][0][0], [1., 2., 3.])
        np.testing.assert_array_equal(axis["lines"][0][1], [3., 2., 1.])
        assert len(axis["lines"]) == 3
        assert axis["legend"][1:] == [r"$\alpha$ по проверочным данным", r"$\alpha_L$: по L-кривой"]
        np.testing.assert_array_equal(axis["lines"][1][0], [2.])
        np.testing.assert_array_equal(axis["lines"][2][0], [1.])
    for axis in captured["alpha_validation_SF01-PG10"]:
        assert (axis["xscale"], axis["yscale"]) == ("log", "linear")
        assert axis["ylabel"] == "Среднеквадратичная ошибка, C²"
    assert captured["comparison_errors"][0]["ticks"] == weights
    assert captured["comparison_errors"][0]["legend"] == [
        r"Оценки, $L^2$", r"Медиана, $L^2$", r"Оценки, $H^1$", r"Медиана, $H^1$"]
    for axis, threshold in zip(captured["grid_checks"], (.02, .05)):
        assert axis["yscale"] == "log"
        assert len(axis["lines"]) == 4
        assert axis["legend"] == ["Рабочая и мелкая сетки", "Мелкая и уточнённая сетки",
                                   "Исходная и расширенная области", "Порог отклонения"]
        np.testing.assert_array_equal(axis["lines"][-1][1], [threshold, threshold])
    with (output / "reconstructions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {int(r["replicate"]) for r in rows} == set(replicates)
    assert {r["weight"] for r in rows} == set(weights)
    assert {float(r["q_C_km2_per_h"]) for r in rows} == {q_reference}
    tables = (output / "TABLES.md").read_text(encoding="utf-8")
    assert f"SF01-PG10: {len(replicates)}" in tables
    assert f"n={len(replicates)}" in tables
    assert {p.name: p.read_bytes() for p in inputs.iterdir()} == before


@pytest.mark.parametrize("change,message", [
    ("missing_scale", "q_reference"), ("wrong_scale", "q_reference"),
    ("missing_case", "cases"), ("duplicate_case", "cases"),
])
def test_inconsistent_saved_results_are_rejected_before_output_creation(
        project_root, tmp_path, change, message):
    config, inputs, record = saved_run(project_root, tmp_path,
        q_reference=200., replicates=[1], weights=["W01"])
    if change == "missing_scale":
        record.pop("q_reference")
    elif change == "wrong_scale":
        record["q_reference"] = 100.
    elif change == "missing_case":
        record["cases"].pop()
    else:
        record["cases"].append(deepcopy(record["cases"][0]))
    source_file = inputs / "SF01-PG10.json"
    source_file.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
    manifest_path = inputs / "run.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["outputs"][source_file.name] = hashlib.sha256(source_file.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in inputs.iterdir()}
    output = tmp_path / "figures"
    with pytest.raises(ValueError, match=message):
        figures.render(config, inputs, output)
    assert not output.exists()
    assert {p.name: p.read_bytes() for p in inputs.iterdir()} == before


@pytest.mark.skipif(os.name != "nt", reason="The native Windows text encoding is required")
def test_native_windows_renderer_publishes_utf8_tables(project_root, tmp_path):
    config, inputs, _ = saved_run(project_root, tmp_path,
        q_reference=200., replicates=[1], weights=["W01"])
    before = {p.name: p.read_bytes() for p in inputs.iterdir()}
    output = tmp_path / "native-figures"
    script = r"""
import csv, hashlib, json, locale, sys
from pathlib import Path
from experiments.source_comparison import figures, run

config, inputs, output = map(Path, sys.argv[1:])
assert sys.flags.utf8_mode == 0

def forbidden(*args, **kwargs):
    raise AssertionError("The renderer must not solve ADR in this storage test")
for name in ("state_solver", "check_model", "build_truth", "fit"):
    setattr(run, name, forbidden)
figures.finish = lambda figure, *_: figures.plt.close(figure)
figures.render(config, inputs, output)
table = (output / "TABLES.md").read_bytes().decode("utf-8")
assert "Граничный выбор α" in table and "n=1" in table
manifest = json.loads((output / "manifest.json").read_bytes())
assert manifest["files"]
for name, expected in manifest["files"].items():
    raw = (output / name).read_bytes()
    raw.decode("utf-8")
    assert hashlib.sha256(raw).hexdigest() == expected
with (output / "reconstructions.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
assert rows and {float(row["q_C_km2_per_h"]) for row in rows} == {200.}
unicode_csv = output.parent / "unicode-table.csv"
figures.csv_rows(unicode_csv, [{"Пост": "КрАЗ", "Параметр": "α"}])
with unicode_csv.open(encoding="utf-8", newline="") as stream:
    assert list(csv.DictReader(stream)) == [{"Пост": "КрАЗ", "Параметр": "α"}]
print(json.dumps(dict(native_encoding=locale.getencoding(), utf8_mode=sys.flags.utf8_mode)))
"""
    environment = dict(os.environ, PYTHONUTF8="0", PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, "-X", "utf8=0", "-B", "-c", script,
        str(config), str(inputs), str(output)], cwd=project_root,
        env=environment, text=True, encoding="utf-8", capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout.splitlines()[-1])
    assert observed["utf8_mode"] == 0
    assert observed["native_encoding"]
    assert {p.name: p.read_bytes() for p in inputs.iterdir()} == before
