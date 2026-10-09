"""Проверки рисунков, таблиц и исключений на синтетических записях E06."""
from copy import deepcopy
import csv
import hashlib

import pytest

from adrkit.config.validation import canonical_bytes
from experiments.observation_sensitivity.analysis import plots as plot
from experiments.observation_sensitivity.analysis import paired_effects as analyzer
from experiments.observation_sensitivity.design import DEFAULT_FIGURES


FIGURES = {f["id"]: tuple(f["contrast_keys"]) for f in DEFAULT_FIGURES}
from tests.experiments.observation_sensitivity.fixtures.paired_records import make_records, alter, pin, reseal, source_score


def make_plot_records():
    records = make_records()
    for group_name, group in records[2].items():
        source, rep = group_name.split("/r")
        r = int(rep)
        for identifier, score in group["scores"].items():
            condition, _, _, penalty = identifier.split("/")
            change = 0.
            if condition.startswith("spatial_") and condition.endswith("_assumed_G1"):
                tag = condition.split("_")[1]
                change = {"G05": -.035, "G2": .055, "C1": .015}[tag]*(r-1.5)
            if condition == "covariance_exp_population":
                change = .025*(r-1.8)
            if condition == "covariance_exp_estimated":
                change = -.035*(r-1.7)
            if source == "EC04":
                change *= -.65
            if penalty == "H1":
                change *= 1.3
            score["source"] = source_score(score["source"]["E_q"]+change)
        reseal(group)
    for r in range(1, 5):
        alter(records, r, kind="scoring", penalty="L2", condition="spatial_G05_assumed_G1")
    for r in range(2, 5):
        alter(records, r, kind="scoring", penalty="H1", condition="spatial_G05_assumed_G1")
    alter(records, 1, kind="lower", penalty="H1", condition="spatial_G2_assumed_G1")
    for r in (3, 4):
        group = records[2][f"EC04/r{r}"]
        identifier = f"covariance_exp_estimated/EC04/r{r}/H1"
        group["scores"][identifier] = dict(procedure_accepted=True, diagnostic_only=False,
            score_row_count=36, status="unavailable", reason="scoring_failure", error="artificial fixture")
        reseal(group)
    return records


def make_plot_summary():
    records = make_plot_records()
    return analyzer.summarize_records(*records, expected_freeze_sha256=pin(records[0]))


def build_visual_fixture(destination, *, project_root):
    """Сохранить рисунки синтетического примера для ручной проверки макета."""
    project = project_root
    return plot.render(make_plot_summary(), destination, protected_root=project, fixture=True)


@pytest.fixture(scope="module")
def template():
    return make_plot_summary()


@pytest.fixture
def summary(template):
    return deepcopy(template)


@pytest.fixture
def export_root(tmp_path, project_root):
    # Путь вывода соответствует контракту render: вне каталога проекта.
    project = project_root
    assert not tmp_path.resolve().is_relative_to(project)
    return tmp_path


def cell(summary, key="assumed_spatial/G05", source="PG10", penalty="L2"):
    return next(r for r in summary["contrast_summaries"] if (r["key"], r["source"], r["penalty"]) == (key, source, penalty))


def test_validation_is_owned_complete_and_recomputes_all_not_only_shown_pairs(summary):
    before = canonical_bytes(summary)
    result = plot.validate_summary(summary)
    assert len(result["paired_outcomes"]) == 224
    assert len(result["contrast_summaries"]) == 60
    result["counts"]["paths"] = 0
    assert canonical_bytes(summary) == before
    pairs, rows, failures = plot._tables(summary)
    assert len(pairs) == 80 and len(rows) == 20
    assert set(r["contrast"] for r in pairs) == {key for keys in FIGURES.values() for key in keys}
    assert failures


@pytest.mark.parametrize("kind", ["schema", "partial", "counts", "missing_pair", "duplicate_pair", "missing_raw", "missing_cell",
    "mean", "mcse", "n", "retained", "reason", "sign", "raw_metric", "raw_flag", "nan", "bool"])
def test_tampering_rejected_before_figures_or_output_creation(summary, export_root, kind, project_root):
    row = cell(summary, key="assumed_spatial/G2", penalty="L2")
    pair = next(r for r in summary["paired_outcomes"] if r["contrast_id"] == row["contrast_ids"][0])
    if kind == "schema": summary["schema"] = "arbitrary"
    elif kind == "partial": summary["status"] = "partial"
    elif kind == "counts": summary["counts"]["paths"] = 175
    elif kind == "missing_pair": summary["paired_outcomes"].pop(0)  # Проверяется и пара штрафов, отсутствующая на рисунке.
                                                                    
    elif kind == "duplicate_pair": summary["paired_outcomes"].append(deepcopy(pair))
    elif kind == "missing_raw": summary["raw_outcomes"].pop()
    elif kind == "missing_cell": summary["contrast_summaries"].pop()
    elif kind == "mean": row["statistics"]["E_q"]["mean"] += .1
    elif kind == "mcse": cell(summary, penalty="H1")["statistics"]["E_q"]["mcse"] = 0.
    elif kind == "n": row["n_eligible"] -= 1
    elif kind == "retained": row["retained_replicates"].reverse()
    elif kind == "reason": cell(summary)["exclusion_counts"] = {}
    elif kind == "sign": pair["difference_right_minus_left"]["E_q"] *= -1
    elif kind == "raw_metric": summary["raw_outcomes"][0]["metrics"]["E_q"] += .01
    elif kind == "raw_flag": summary["raw_outcomes"][0]["eligible"] = False
    elif kind == "nan": row["statistics"]["E_q"]["mean"] = float("nan")
    elif kind == "bool": row["statistics"]["E_q"]["n"] = True
    output = export_root/"must-not-exist"
    with pytest.raises(plot.PlotError):
        plot.render(summary, output, protected_root=project_root, fixture=True)
    assert not output.exists()


def test_n0_n1_n2_and_n4_artists_match_individual_pairs_and_shared_scale(summary):
    import matplotlib.pyplot as plt
    before = canonical_bytes(summary)
    for kind in ("spatial", "covariance"):
        fig = plot.make_figure(summary, kind)
        try:
            fig.canvas.draw()
            assert len(fig.axes) == 2
            assert fig.axes[0].get_xlim() == fig.axes[1].get_xlim()
            assert fig.axes[0].get_xlim()[0] < 0 < fig.axes[0].get_xlim()[1]
            assert not fig.texts and len(fig.legends) == 1
            assert all(not axis.texts and not axis.get_title() and axis.get_legend() is None
                       for axis in fig.axes)
            assert [axis.get_ylabel() for axis in fig.axes] == list(plot._figure_rows(summary, kind))
            assert [text.get_text() for text in fig.legends[0].get_texts()] == ["L²", "Полная H¹", "Отдельная пара", "Среднее ± стандартная ошибка"]
            markers = [line for axis in fig.axes for line in axis.lines if line.get_gid()]
            keys = set(FIGURES[kind])
            expected_pairs = [p for p in summary["paired_outcomes"] if p["key"] in keys and p["eligible"]]
            points = [line for line in markers if line.get_gid().startswith("pair:")]
            means = [line for line in markers if line.get_gid().startswith("mean:")]
            assert {line.get_gid()[5:] for line in points} == {p["contrast_id"] for p in expected_pairs}
            expected_means = [r for r in summary["contrast_summaries"] if r["key"] in keys and r["n_eligible"] >= 2]
            assert len(means) == len(expected_means)
            for line in points:
                pair = next(p for p in expected_pairs if p["contrast_id"] == line.get_gid()[5:])
                assert list(line.get_xdata()) == [pair["difference_right_minus_left"]["E_q"]]
            if kind == "spatial":
                assert not any("PG10:assumed_spatial/G05" in line.get_gid() for line in means)
            assert all(line.get_markerfacecolor() == "white" for line in means)
            if kind == "spatial":
                assert all("Ядро генератора" in text.get_text() for text in fig.axes[0].get_yticklabels())
            renderer = fig.canvas.get_renderer()
            for legend in fig.legends:
                box = legend.get_window_extent(renderer)
                assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
            texts = list(fig.texts) + [t for axis in fig.axes for t in (list(axis.texts)+list(axis.get_yticklabels())+[axis.title, axis.xaxis.label, axis.yaxis.label])]
            for text in texts:
                if text.get_text() and text.get_visible():
                    box = text.get_window_extent(renderer)
                    assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, text.get_text()
        finally:
            plt.close(fig)
    assert canonical_bytes(summary) == before


def test_render_exports_exact_scope_hashes_and_csv_missing_semantics(summary, export_root, project_root):
    project = project_root
    output = export_root/"two-figures"
    manifest = plot.render(summary, output, protected_root=project, fixture=True)
    expected = {"e06-spatial.png", "e06-spatial.svg", "e06-covariance.png", "e06-covariance.svg",
                "pairs.csv", "summary.csv", "failures.csv", "figures.json"}
    assert {p.name for p in output.iterdir()} == expected
    assert manifest["fixture"] is True and manifest["shown_pair_count"] == 80 and manifest["shown_cell_count"] == 20
    for filename, sha in manifest["files"].items():
        assert hashlib.sha256((output/filename).read_bytes()).hexdigest() == sha
    for filename in ("e06-spatial.svg", "e06-covariance.svg"):
        text = (output/filename).read_text(encoding="utf-8")
        assert "Тестовые данные — проверка макета" not in text
        assert "Среднее ± стандартная ошибка" in text
        assert "n = " not in text
    def rows(name):
        with (output/name).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    assert len(rows("pairs.csv")) == 80 and len(rows("summary.csv")) == 20
    zero = next(r for r in rows("summary.csv") if (r["contrast"], r["source"], r["penalty"]) == ("assumed_spatial/G05", "PG10", "L2"))
    assert zero["n_expected"] == zero["n_terminal"] == "4" and zero["n_eligible"] == "0"
    assert zero["mean_E_q_difference"] == zero["sd"] == zero["mcse"] == ""
    one = next(r for r in rows("summary.csv") if (r["contrast"], r["source"], r["penalty"]) == ("assumed_spatial/G05", "PG10", "H1"))
    assert one["n_eligible"] == "1" and one["mean_E_q_difference"] and one["mcse"] == ""
    failures = rows("failures.csv")
    assert any("ковариации" not in r["reason_ru"] and "ошибку" in r["reason_ru"] for r in failures)
    boundary = [r for r in failures if r["contrast"] == "assumed_spatial/G2" and r["source"] == "PG10" and r["penalty"] == "H1"]
    assert len(boundary) == 2 and all(r["counts_overlap"] == "True" for r in boundary)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(plot.PlotError, match="new output"):
        plot.render(summary, output, protected_root=project, fixture=True)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    repeated = export_root/"repeated-two-figures"
    second = plot.render(summary, repeated, protected_root=project, fixture=True)
    assert second == manifest
    assert before == {p.name: p.read_bytes() for p in repeated.iterdir()}


def test_outputs_inside_project_or_ancestor_are_rejected_without_write(summary, export_root, project_root):
    project = project_root
    for destination in (project/"new-figure", project, project.parent):
        with pytest.raises(plot.PlotError, match="outside"):
            plot.render(summary, destination, protected_root=project, fixture=True)


def test_ordinary_relative_parent_segments_are_not_misidentified_as_symlinks(export_root, project_root):
    project = project_root
    expected = export_root/"new-output"
    assert plot._destination(export_root/"unused"/".."/"new-output", project) == expected


def test_cli_uses_safe_loader_and_fresh_aggregator_not_arbitrary_summary(export_root, monkeypatch, project_root):
    from experiments.observation_sensitivity.analysis import read_results as snapshot
    records = make_plot_records()
    expected_pin = pin(records[0])
    calls = []
    def read(run, *, expected_freeze_sha256):
        calls.append((run, expected_freeze_sha256))
        return dict(freeze=records[0], direct=records[1], groups=records[2], raw_file_sha256={"fixture": "f"*64})
    monkeypatch.setattr(snapshot, "load_records", read)
    def export(summary, destination, *, protected_root, additional_roots=(), fixture=False):
        assert fixture is False
        assert additional_roots == snapshot.scientific_input_roots(records[0]["admission"])
        assert len(summary["paired_outcomes"]) == 224 and len(summary["contrast_summaries"]) == 60
        assert summary["inputs"]["freeze_file_sha256"] == expected_pin
        assert summary["raw_input_file_sha256"] == {"fixture": "f"*64}
        calls.append("render")
        return dict(figure_count=2)
    monkeypatch.setattr(plot, "render", export)
    project = project_root
    result = plot.main(["--run", "artificial-unused-run", "--freeze-sha256", expected_pin,
                        "--output-dir", str(export_root/"cli")])
    assert result == 0 and calls == [("artificial-unused-run", expected_pin), "render"]


def test_cli_never_reaches_renderer_after_locked_or_missing_results(export_root, monkeypatch, project_root):
    from experiments.observation_sensitivity.analysis import read_results as snapshot
    def blocked(*args, **kwargs):
        raise snapshot.SnapshotError("result lock is held")
    monkeypatch.setattr(snapshot, "load_records", blocked)
    monkeypatch.setattr(plot, "render", lambda *a, **kw: pytest.fail("must not render"))
    output = export_root/"blocked"
    with pytest.raises(snapshot.SnapshotError, match="lock is held"):
        plot.main(["--run", "not-opened", "--freeze-sha256", "f"*64,
                   "--output-dir", str(output)])
    assert not output.exists()



def selected_summary(*, empty=False):
    from tests.experiments.observation_sensitivity.fixtures.paired_records import select_records
    ids = ["spatial_G1_matched/PG10/r1/L2", "spatial_G1_matched/PG10/r1/H1",
           "spatial_G2_matched/PG10/r1/L2", "spatial_G2_matched/PG10/r1/H1"]
    records = select_records(make_records(), ids, contrast_ids=[] if empty else None,
        figures=[] if empty else [dict(id="matched", contrast_keys=["matched_spatial/G2"])])
    return analyzer.summarize_records(*records, expected_freeze_sha256=pin(records[0]))


def test_declared_single_figure_exports_actual_scope_without_changing_inputs(export_root, project_root):
    summary = selected_summary()
    before = canonical_bytes(summary)
    output = export_root / "selected"
    manifest = plot.render(summary, output, protected_root=project_root)
    assert canonical_bytes(summary) == before
    assert manifest["figure_count"] == 1 and manifest["shown_pair_count"] == manifest["shown_cell_count"] == 2
    assert manifest["signs"] == {"matched_spatial/G2": "right minus left"}
    assert set(p.name for p in output.iterdir()) == {"e06-matched.png", "e06-matched.svg", "pairs.csv", "summary.csv", "failures.csv", "figures.json"}
    with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and all(r["n_expected"] == "1" and r["mcse"] == r["sd"] == "" for r in rows)
    with pytest.raises(plot.PlotError): plot.render(summary, output, protected_root=project_root)


def test_matched_spatial_comparison_keeps_actual_axis_categories():
    summary = selected_summary()
    summary["figures"][0]["id"] = "spatial"
    before = canonical_bytes(summary)
    figure = plot.make_figure(summary, "spatial")
    try:
        assert not figure.texts
        assert all("Разность нормированных ошибок" in axis.get_xlabel() for axis in figure.axes)
        assert all("изменённое ядро − исходное ядро (в данных и модели)" in axis.get_xlabel() for axis in figure.axes)
        assert all("σ = 2 км" in t.get_text() and "в данных и модели" in t.get_text()
                   for t in figure.axes[0].get_yticklabels())
        assert canonical_bytes(summary) == before
    finally:
        figure.clear()


def test_selected_covariance_keeps_actual_repeat_count_in_csv(export_root, project_root):
    from tests.experiments.observation_sensitivity.fixtures.paired_records import select_records
    ids = [f"covariance_{condition}/PG10/r1/{penalty}"
           for condition in ("mix_oracle", "exp_population", "exp_estimated") for penalty in ("L2", "H1")]
    records = select_records(make_records(), ids,
        figures=[dict(id="custom_covariance", contrast_keys=["covariance/family", "covariance/estimation"])])
    summary = analyzer.summarize_records(*records, expected_freeze_sha256=pin(records[0]))
    before = canonical_bytes(summary)
    figure = plot.make_figure(summary, "custom_covariance")
    try:
        assert not figure.texts
        assert all(not axis.texts and not axis.get_title() for axis in figure.axes)
    finally:
        figure.clear()
    manifest = plot.render(summary, export_root / "one-repeat-covariance", protected_root=project_root)
    assert manifest["signs"] == {"covariance_family": "population minus oracle", "covariance_estimation": "estimated minus population"}
    with (export_root / "one-repeat-covariance/summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert all(row["n_expected"] == "1" and row["mcse"] == "" for row in csv.DictReader(stream))
    assert canonical_bytes(summary) == before


def test_full_default_figure_axes_and_csv_signs_are_preserved(summary, export_root, project_root):
    for kind in ("spatial", "covariance"):
        figure = plot.make_figure(summary, kind)
        try:
            assert not figure.texts and len(figure.legends) == 1
            assert [axis.get_ylabel() for axis in figure.axes] == list(plot._figure_rows(summary, kind))
            if kind == "spatial":
                assert all(r"E_q(H_{\mathrm{inv}}) - E_q(H_{\mathrm{true}})" in axis.get_xlabel() for axis in figure.axes)
                assert all("Принятое ядро" in axis.get_xlabel() and "гауссово, σ = 1 км" in axis.get_xlabel() for axis in figure.axes)
            else:
                assert all("Разность нормированных ошибок" in axis.get_xlabel() for axis in figure.axes)
                labels = [text.get_text() for text in figure.axes[0].get_yticklabels()]
                assert any("Популяционная экспонента −\nизвестная смесь" in label for label in labels)
                assert any("Оценённая экспонента −\nпопуляционная экспонента" in label for label in labels)
        finally:
            figure.clear()
    manifest = plot.render(summary, export_root / "full-captions", protected_root=project_root)
    assert manifest["signs"] == {"spatial": "assumed minus matched", "covariance_family": "population minus oracle", "covariance_estimation": "estimated minus population"}


def test_all_registered_contrasts_have_non_overlapping_row_labels(summary):
    from experiments.observation_sensitivity.design import build_design
    keys = list(dict.fromkeys(c.key for c in build_design().contrasts))
    summary["figures"] = [dict(id="all_comparisons", contrast_keys=keys)]
    before = canonical_bytes(summary)
    figure = plot.make_figure(summary, "all_comparisons")
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        cells = plot._figure_rows(summary, "all_comparisons")
        assert len(keys) == 21 and all(len(rows) == 30 for rows in cells.values())
        labels = figure.axes[0].get_yticklabels()
        assert len({label.get_text() for label in labels}) == 30
        bounds = [label.get_window_extent(renderer) for label in labels]
        assert all(a.y0 > b.y1 for a, b in zip(bounds, bounds[1:]))
        assert not figure.texts and len(figure.legends) == 1
        assert all(not axis.texts and not axis.get_title() and axis.get_legend() is None
                   for axis in figure.axes)
        texts = list(figure.texts) + [text for axis in figure.axes
            for text in list(axis.texts) + list(axis.get_yticklabels()) + [axis.title]]
        for text in texts:
            if text.get_text() and text.get_visible():
                box = text.get_window_extent(renderer)
                assert box.x0 >= -1 and box.y0 >= -1
                assert box.x1 <= figure.bbox.width+1 and box.y1 <= figure.bbox.height+1, text.get_text()
        assert canonical_bytes(summary) == before
    finally:
        figure.clear()


def test_empty_figure_and_contrast_plan_exports_only_tables(export_root, project_root):
    summary = selected_summary(empty=True)
    output = export_root / "no-contrasts"
    manifest = plot.render(summary, output, protected_root=project_root)
    assert manifest["figure_count"] == manifest["shown_pair_count"] == manifest["shown_cell_count"] == 0
    assert {p.name for p in output.iterdir()} == {"pairs.csv", "summary.csv", "failures.csv", "figures.json"}
    with (output / "pairs.csv").open(encoding="utf-8-sig", newline="") as stream: assert list(csv.DictReader(stream)) == []


@pytest.mark.parametrize("defect", ["unknown_figure_key", "duplicate_figure", "missing_pair", "reordered_design"])
def test_selected_renderer_rejects_inconsistent_scope_before_creating_output(export_root, project_root, defect):
    summary = selected_summary()
    if defect == "unknown_figure_key": summary["figures"][0]["contrast_keys"] = ["assumed_spatial/G05"]
    elif defect == "duplicate_figure": summary["figures"].append(deepcopy(summary["figures"][0]))
    elif defect == "missing_pair": summary["paired_outcomes"].pop()
    else: summary["design"]["paths"].reverse()
    output = export_root / "invalid-selected"
    with pytest.raises(plot.PlotError): plot.render(summary, output, protected_root=project_root)
    assert not output.exists()
