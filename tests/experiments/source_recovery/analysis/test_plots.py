"""Проверки сводок и рисунков на искусственных контрольных записях."""
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest
import pytest

from experiments.source_recovery.analysis import plots as p
from experiments.source_recovery.analysis import summary as s


def make_fixture(root, project_root):
    project = project_root
    spec = json.loads((project/"experiments/source_recovery/configs/experiment.json").read_text(encoding="utf-8"))
    spec["sources"] = ["PG10", "EC04", "NEW-J2"]
    spec["replicates"] = [1, 2, 3]
    spec["single_fits"] = []
    for condition in spec["conditions"]:
        condition["sources"] = spec["sources"]
        condition["replicates"] = spec["replicates"]
    def write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    write(root/"run.json", dict(configuration=spec, bindings={"fixture": "NO_SCIENTIFIC_RESULTS"}))
    for source in spec["sources"]:
        for replicate in spec["replicates"]:
            paths, scores = {}, {}
            for i, c in enumerate(spec["conditions"]):
                for arm in c["penalties"]:
                    pid = c["id"]+"/"+arm
                    accepted = not (source == "PG10" or (source == "EC04" and replicate != 1))
                    paths[pid] = dict(finalized=True, complete_path=True, procedure_accepted=accepted,
                        tuning_unresolved=not accepted, final_certificate={"accepted": True}, candidates={})
                    value = .2 + i*.002 + (.04*replicate if arm == "H1" else 0)
                    scores[pid] = dict(E_q=value, relative_L2=value*2, full_primary_test_rmse=value,
                        full_primary_noiseless_rmse=value, score_row_count=36, diagnostic_only=not accepted)
            write(root/source/f"replicate_{replicate}.json", dict(bindings={"fixture": "NO_SCIENTIFIC_RESULTS"},
                paths=paths, scores=scores, expected_paths=list(paths), exponents=[0], stage="scored",
                selection_seal=s.selection_hash(paths)))
    return s.summarize(root, "main"), spec


@pytest.fixture(scope="class")
def plot_configuration(request, project_root):
    request.cls.project_root = project_root


@pytest.mark.usefixtures("plot_configuration")
class PlotContract(unittest.TestCase):
    """Контракты полноты, статистик и рисунков на искусственных записях."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.report, self.spec = make_fixture(self.root/"run", self.project_root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_fixture_is_accepted_without_input_mutation(self):
        before = {str(x): p.digest(x) for x in self.root.rglob("*.json")}
        self.assertEqual(len(p.validate_summary(self.report)), 22)
        self.assertEqual(p.verify_inputs(self.report), self.spec)
        self.assertEqual(before, {str(x): p.digest(x) for x in self.root.rglob("*.json")})

    def test_partial_checkpoints_refuse_before_outputs(self):
        key = next(iter(self.report["checkpoints"]))
        for stage in ("pending", "running", "sealed", None):
            self.report["checkpoints"][key]["stage"] = stage
            with self.assertRaisesRegex(ValueError, "ALL checkpoints"):
                p.render(self.report, self.spec, self.root/"must_not_exist")
            self.assertFalse((self.root/"must_not_exist").exists())

    def test_zero_one_and_three_pairs_have_correct_uncertainty(self):
        zero, one, three = self.report["primary_paired_results"]
        self.assertEqual(p.validate_pair(zero), [])
        self.assertIsNone(zero["statistics"]["E_q"]["mean"])
        self.assertEqual(one["nactual"], 1)
        self.assertIsNone(one["statistics"]["E_q"]["mcse"])
        self.assertAlmostEqual(three["statistics"]["E_q"]["mcse"], .04/(3**.5))
        self.assertEqual(zero["exclusion_counts"]["right:alpha_boundary_unresolved"], 3)
        self.assertEqual(zero["exclusion_counts"]["right:diagnostic_only"], 3)

    def test_zero_effect_is_not_substituted_for_missing_pairs(self):
        self.report["primary_paired_results"][0]["statistics"]["E_q"]["mean"] = 0.
        with self.assertRaisesRegex(ValueError, "must be null"):
            p.validate_summary(self.report)

    def test_n1_mcse_and_wrong_difference_are_rejected(self):
        one = self.report["primary_paired_results"][1]
        one["statistics"]["E_q"]["mcse"] = 0.
        with self.assertRaisesRegex(ValueError, "must be null"):
            p.validate_pair(one)
        one["statistics"]["E_q"]["mcse"] = None
        one["strict_pairs"][0]["difference_right_minus_left"]["E_q"] *= -1
        with self.assertRaisesRegex(ValueError, "difference/sign"):
            p.validate_pair(one)

    def test_deleted_checkpoint_even_with_scored_flags_is_rejected(self):
        self.report["checkpoints"].pop(next(iter(self.report["checkpoints"])))
        with self.assertRaises(ValueError):
            p.verify_inputs(self.report)

    def test_changed_bound_input_and_coverage_rejected(self):
        path = self.root/"run"/"PG10"/"replicate_1.json"
        path.write_text(path.read_text()+"\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Input changed"):
            p.verify_inputs(self.report)
        self.report["coverage"]["candidate_attempted"] += 1
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            p.validate_summary(self.report)

    def test_missing_contrast_cell_and_pair_denominator_rejected(self):
        row = self.report["contrasts"].pop()
        with self.assertRaisesRegex(ValueError, "summary differs"):
            p.verify_inputs(self.report)
        self.report["contrasts"].append(row)
        cell = self.report["primary_paired_results"][0]
        cell["nplanned"] += 1
        cell["nexcluded"] += 1
        with self.assertRaisesRegex(ValueError, "summary differs"):
            p.verify_inputs(self.report)

    def test_consistently_tampered_summary_cannot_replace_checkpoint_scores(self):
        from unittest.mock import patch
        report = copy.deepcopy(self.report)
        # Пересчитать сводки по изменённым ошибкам, сохранив исходные файлы.
        # Проверка сравнивает подменённую сводку с исходными записями.
        real_snapshot = s.snapshot
        def altered_snapshot(path):
            payload, sha = real_snapshot(path)
            if "scores" in payload:
                for score in payload["scores"].values():
                    score["E_q"] += .5
            return payload, sha
        with patch.object(s, "snapshot", side_effect=altered_snapshot):
            forged = s.summarize(self.root/"run", "main")
        p.validate_summary(forged)
        with self.assertRaisesRegex(ValueError, "summary differs"):
            p.verify_inputs(forged)
        self.assertEqual(report, self.report)

    def test_duplicate_and_nonfinite_json_rejected(self):
        path = self.root/"bad.json"
        for contents in ('{"a":1,"a":2}', '{"a":NaN}'):
            path.write_text(contents, encoding="utf-8")
            with self.assertRaises(ValueError):
                p.load_json(path)

    def test_fixture_render_has_no_artist_for_n0_and_22_pdf_pages(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        output = self.root/"plots"
        result = p.render(self.report, self.spec, output)
        self.assertEqual(len(result), 27)
        # Matplotlib записывает словари страниц без сжатия, хотя потоки шрифтов могут сжиматься;
        # проверка относится к выходу этого средства построения.
        self.assertEqual(len(re.findall(rb"/Type /Page\b", (output/"contrast-series.pdf").read_bytes())), 22)
        fig = p.pair_figure(self.report["primary_paired_results"], xlabel=r"$E_q(H^1) - E_q(L^2)$")
        ax = fig.axes[0]
        self.assertFalse(fig.texts)
        self.assertFalse(fig.legends)
        self.assertFalse(ax.texts)
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()], ["Отдельная пара", "Среднее ± стандартная ошибка"])
        self.assertEqual(ax.get_title(), "")
        # Все маркеры данных лежат на строках n=1 или n=3; строка n=0 содержит только нулевую линию.
        points = [line for line in ax.lines if line.get_marker() in ("o", "D")]
        self.assertTrue(points)
        self.assertTrue(all(all(float(y) > .5 for y in line.get_ydata()) for line in points))
        plt.close(fig)
        coverage = p.coverage_figure(self.report)
        try:
            self.assertEqual([text.get_text() for text in coverage.axes[0].get_legend().get_texts()],
                             ["Допущены к сравнению", "Исключены из сравнения"])
            self.assertFalse(coverage.texts)
            self.assertTrue(all(not axis.texts and not axis.get_title() for axis in coverage.axes))
        finally:
            plt.close(coverage)


def declared_contrast_fixture(root, project_root, contrasts):
    _, spec = make_fixture(root, project_root)
    spec["analysis_contrasts"] = contrasts
    spec["sources"] = ["PG10"]
    spec["replicates"] = [1]
    ids = {"main"}
    for plan in contrasts:
        ids.update((plan["baseline"], plan["variant"]))
    spec["conditions"] = [c for c in spec["conditions"] if c["id"] in ids]
    for condition in spec["conditions"]:
        condition["sources"] = ["PG10"]
        condition["replicates"] = [1]
    manifest = json.loads((root / "run.json").read_bytes())
    manifest["configuration"] = spec
    (root / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    for checkpoint in root.glob("*/replicate_*.json"):
        if checkpoint.parent.name != "PG10" or checkpoint.name != "replicate_1.json":
            checkpoint.unlink()
            continue
        record = json.loads(checkpoint.read_bytes())
        record["paths"] = {key: value for key, value in record["paths"].items() if key.split("/")[0] in ids}
        record["scores"] = {key: value for key, value in record["scores"].items() if key in record["paths"]}
        for path in record["paths"].values():
            path["procedure_accepted"] = True
            path["tuning_unresolved"] = False
        for score in record["scores"].values():
            score["diagnostic_only"] = False
        record["expected_paths"] = list(record["paths"])
        record["selection_seal"] = s.selection_hash(record["paths"])
        checkpoint.write_text(json.dumps(record), encoding="utf-8")
    return s.summarize(root, "main"), spec


def test_declared_single_contrast_renders_one_real_pdf_page(tmp_path, project_root):
    plans = [{"id": "weight_W01", "baseline": "main", "variant": "weight_W01", "factor": "weight"}]
    report, spec = declared_contrast_fixture(tmp_path / "run", project_root, plans)
    before = {str(path): p.digest(path) for path in (tmp_path / "run").rglob("*.json")}
    assert p.verify_inputs(report) == spec
    assert list(report["checkpoints"]) == ["PG10/1"]
    assert report["primary_paired_results"][0]["nplanned"] == 1
    assert report["primary_paired_results"][0]["nactual"] == 1
    assert report["primary_paired_results"][0]["statistics"]["E_q"]["mcse"] is None
    assert set(p.validate_summary(report, ["weight_W01"])) == {"weight_W01"}
    output = tmp_path / "plots"
    files = p.render(report, spec, output)
    assert set(files) == {"main-paired.pdf", "main-paired.png", "coverage.pdf", "coverage.png",
        "contrast-series.pdf", str(Path("contrast-pages") / "contrast-01.png")}
    assert len(re.findall(rb"/Type /Page\b", (output / "contrast-series.pdf").read_bytes())) == 1
    assert before == {str(path): p.digest(path) for path in (tmp_path / "run").rglob("*.json")}


def test_renderer_matches_declared_ids_and_rejects_duplicate_plan(tmp_path, project_root):
    plan = {"id": "weight_W01", "baseline": "main", "variant": "weight_W01", "factor": "weight"}
    report, spec = declared_contrast_fixture(tmp_path / "run", project_root, [plan])
    for ids in (["weight_W02"], ["weight_W01", "weight_W01"]):
        with pytest.raises(ValueError):
            p.validate_summary(report, ids)
    duplicate_spec = copy.deepcopy(spec)
    duplicate_spec["analysis_contrasts"].append(plan)
    with pytest.raises(ValueError, match="Unique"):
        p.render(report, duplicate_spec, tmp_path / "must_not_exist")
    assert not (tmp_path / "must_not_exist").exists()
    report["contrasts"] = []
    with pytest.raises(ValueError, match="declared plan"):
        p.validate_summary(report, ["weight_W01"])


def test_cli_manifest_and_rendered_rows_use_declared_plan(tmp_path, project_root, monkeypatch, capsys):
    import sys

    plans = [{"id": "weight_W01", "baseline": "main", "variant": "weight_W01", "factor": "weight"}]
    report, spec = declared_contrast_fixture(tmp_path / "run", project_root, plans)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(report), encoding="utf-8")
    plotted_rows = []
    labels = []
    actual_pair_figure = p.pair_figure

    def capture_rows(rows, *, xlabel):
        plotted_rows.append(rows)
        labels.append(xlabel)
        figure = actual_pair_figure(rows, xlabel=xlabel)
        assert not figure.texts and not figure.legends
        assert all(not axis.texts and not axis.get_title() and axis.get_legend() is not None
                   for axis in figure.axes)
        return figure

    monkeypatch.setattr(p, "pair_figure", capture_rows)
    monkeypatch.setattr(sys, "argv", ["plots", "--summary", str(summary_path), "--output-dir", str(tmp_path / "figures")])
    p.main()
    manifest = json.loads((tmp_path / "figures/plot-manifest.json").read_bytes())
    assert manifest["planned_contrasts"] == 1
    assert manifest["all_checkpoints_scored"] is True
    assert len(manifest["figure_sha256"]) == 6
    assert plotted_rows == [report["primary_paired_results"], report["contrasts"]]
    assert labels[0].startswith("Разность нормированных ошибок восстановления:")
    assert r"$E_q(H^1) - E_q(L^2)$" in labels[0]
    assert labels[1].endswith("Веса W01 − Основной случай")
    assert not any("минус база" in label for label in labels)
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"status": "rendered_complete_summary", "contrasts": 1, "figures": 6}


def test_default_plan_keeps_all_22_ids_and_declared_count(tmp_path, project_root):
    report, spec = make_fixture(tmp_path / "run", project_root)
    ids = [c["id"] for c in spec["analysis_contrasts"]]
    assert len(ids) == 22
    assert set(p.validate_summary(report, ids)) == set(ids)
    assert p.verify_inputs(report) == spec
    with pytest.raises(ValueError, match="order"):
        p.validate_summary(report, list(reversed(ids)))


def test_main_only_plan_renders_four_files_without_input_mutation(tmp_path, project_root, monkeypatch, capsys):
    import sys

    report, spec = declared_contrast_fixture(tmp_path / "run", project_root, [])
    assert report["contrasts"] == []
    assert p.validate_summary(report, []) == {}
    assert p.verify_inputs(report) == spec
    before = {str(path): p.digest(path) for path in (tmp_path / "run").rglob("*.json")}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(report), encoding="utf-8")
    summary_hash = p.digest(summary_path)
    output = tmp_path / "figures"
    monkeypatch.setattr(sys, "argv", ["plots", "--summary", str(summary_path), "--output-dir", str(output)])
    p.main()
    manifest = json.loads((output / "plot-manifest.json").read_bytes())
    assert manifest["planned_contrasts"] == 0
    assert manifest["all_checkpoints_scored"] is True
    assert set(manifest["figure_sha256"]) == {"main-paired.pdf", "main-paired.png", "coverage.pdf", "coverage.png"}
    assert all((output / name).is_file() and (output / name).stat().st_size > 0 for name in manifest["figure_sha256"])
    assert not (output / "contrast-series.pdf").exists()
    assert not (output / "contrast-pages").exists()
    assert before == {str(path): p.digest(path) for path in (tmp_path / "run").rglob("*.json")}
    assert p.digest(summary_path) == summary_hash
    assert json.loads(capsys.readouterr().out) == {"status": "rendered_complete_summary", "contrasts": 0, "figures": 4}


def test_render_manifest_lists_only_current_outputs_in_reused_directory(tmp_path, project_root, monkeypatch, capsys):
    import sys

    output = tmp_path / "figures"
    full_report, full_spec = make_fixture(tmp_path / "full-run", project_root)
    assert len(p.render(full_report, full_spec, output)) == 27
    unrelated = output / "unrelated.png"
    unrelated.write_bytes(b"user-owned-file")
    unrelated_hash = p.digest(unrelated)
    old_page = output / "contrast-pages/contrast-22.png"
    old_page_hash = p.digest(old_page)
    empty, _ = declared_contrast_fixture(tmp_path / "main-run", project_root, [])
    plan = {"id": "weight_W01", "baseline": "main", "variant": "weight_W01", "factor": "weight"}
    single, _ = declared_contrast_fixture(tmp_path / "single-run", project_root, [plan])
    input_hashes = {str(path): p.digest(path) for path in tmp_path.rglob("*.json")}
    main_files = {"main-paired.pdf", "main-paired.png", "coverage.pdf", "coverage.png"}
    for index, (report, count) in enumerate(((empty, 0), (single, 1), (single, 1))):
        summary_path = tmp_path / f"summary-{index}.json"
        summary_path.write_text(json.dumps(report), encoding="utf-8")
        summary_hash = p.digest(summary_path)
        monkeypatch.setattr(sys, "argv", ["plots", "--summary", str(summary_path), "--output-dir", str(output)])
        p.main()
        manifest = json.loads((output / "plot-manifest.json").read_bytes())
        expected = main_files | ({"contrast-series.pdf", str(Path("contrast-pages") / "contrast-01.png")} if count else set())
        assert manifest["planned_contrasts"] == count
        assert set(manifest["figure_sha256"]) == expected
        assert all(manifest["figure_sha256"][name] == p.digest(output / name) for name in expected)
        assert json.loads(capsys.readouterr().out) == {"status": "rendered_complete_summary", "contrasts": count, "figures": len(expected)}
        assert p.digest(summary_path) == summary_hash
        assert p.digest(unrelated) == unrelated_hash
        assert p.digest(old_page) == old_page_hash
        assert all(p.digest(path) == digest for path, digest in input_hashes.items())


if __name__ == "__main__":
    unittest.main()
