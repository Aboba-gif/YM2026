"""Графики парных разностей E06 и экспорт их данных в CSV."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from adrkit.config.validation import canonical_bytes, digest, strict_json
from experiments.observation_sensitivity.design import (
    build_design, design_from_record, plan_counts, resolve_figures,
)
from . import paired_effects as analyzer


ROW_LABELS = {"assumed_spatial/G05": "Гауссово ядро, σ = 0,5 км",
              "assumed_spatial/G2": "Гауссово ядро, σ = 2 км",
              "assumed_spatial/C1": "Компактное ядро,\nмасштаб 1 км",
              "covariance/family": "Популяционная экспонента −\nизвестная смесь",
              "covariance/estimation": "Оценённая экспонента −\nпопуляционная экспонента"}
CONDITION_LABELS = {
    "spatial_G1_matched": "Гауссово ядро, σ = 1 км\nв данных и модели",
    "spatial_G05_matched": "Гауссово ядро, σ = 0,5 км\nв данных и модели",
    "spatial_G2_matched": "Гауссово ядро, σ = 2 км\nв данных и модели",
    "spatial_C1_matched": "Компактное ядро, масштаб 1 км\nв данных и модели",
    "spatial_G05_assumed_G1": "Гауссово ядро:\nσ = 0,5 км в данных,\nσ = 1 км в модели",
    "spatial_G2_assumed_G1": "Гауссово ядро:\nσ = 2 км в данных,\nσ = 1 км в модели",
    "spatial_C1_assumed_G1": "Компактное ядро 1 км в данных,\nгауссово ядро, σ = 1 км в модели",
    "covariance_mix_oracle": "Известная смесь экспонент",
    "covariance_exp_population": "Экспонента:\nфиксированные параметры",
    "covariance_exp_estimated": "Экспонента:\nоценённые параметры",
    "temporal_average_matched": "Среднее за 20 мин\nв данных и модели",
    "temporal_average_assumed_snapshot": "Среднее за 20 мин в данных,\nмгновенное наблюдение в модели",
}
ROW_LABELS.update({
    "matched_spatial/G05": ROW_LABELS["assumed_spatial/G05"]+"\nв данных и модели",
    "matched_spatial/G2": ROW_LABELS["assumed_spatial/G2"]+"\nв данных и модели",
    "matched_spatial/C1": ROW_LABELS["assumed_spatial/C1"]+"\nв данных и модели",
    "assumed_temporal/average20": "Мгновенное наблюдение\nвместо среднего за 20 мин",
    **{"penalty/"+condition: label for condition, label in CONDITION_LABELS.items()},
})
PENALTY_LABELS = {"L2": "L²", "H1": "Полная H¹", "L2_to_H1": "H¹ − L²"}
COLORS = {"L2": "#2d5985", "H1": "#b45e28", "L2_to_H1": "#627b43"}
CAUSES = {
    "calibration_failure": "Не выполнены условия оценки ковариации",
    "candidate_path_rejected": "Не принят хотя бы один кандидат α",
    "solver_failure": "Численный отказ при оценивании",
    "no_accepted_candidate": "Нет принятого кандидата α",
    "alpha_boundary_unresolved": "Выбрана граница диапазона α",
    "alpha_lower_boundary": "Выбрана нижняя граница α",
    "alpha_upper_boundary": "Выбрана верхняя граница α",
    "final_certificate_rejected": "Не пройден итоговый численный контроль",
    "scoring_failure": "Не удалось вычислить итоговую ошибку",
}


class PlotError(ValueError):
    """Неполная или несогласованная сводка либо недопустимый каталог вывода."""


def _require(condition, message):
    if not condition:
        raise PlotError(message)


def _same(left, right):
    return canonical_bytes(left) == canonical_bytes(right)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _moments(values):
    n = len(values)
    sd = statistics.stdev(values) if n > 1 else None
    return dict(n=n, mean=statistics.mean(values) if n else None, sd=sd,
                mcse=sd/math.sqrt(n) if sd is not None else None)


def _statistics(actual, expected):
    _require(type(actual) is dict and set(actual) == set(expected), "Statistics fields differ")
    for name, value in expected.items():
        old = actual[name]
        if value is None:
            _require(old is None, "Undefined mean/SD/MCSE must be null")
        elif name == "n":
            _require(type(old) is int and old == value, "Statistical sample count differs")
        else:
            _require(_finite(old) and math.isclose(old, value, rel_tol=1e-12, abs_tol=1e-14),
                     "Statistics disagree with individual paired differences")


def _index(rows, key, expected, label):
    _require(type(rows) is list and all(type(row) is dict for row in rows), f"Invalid {label}")
    indexed = {key(row): row for row in rows}
    _require(len(indexed) == len(rows) and set(indexed) == set(expected), f"Missing, duplicate, or unexpected {label}")
    return indexed


def validate_summary(summary):
    """Проверить сводку E06 и вернуть её отдельную JSON-копию.

    Parameters
    ----------
    summary : dict
        Завершённая сводка `summarize_records` со всеми объявленными исходами, парами и
        контрастами.

    Returns
    -------
    dict
        Копия с проверенными планом, парными разностями, числом случаев и
        статистикой.

    Notes
    -----
    Привязку сводки к исходным файлам проверяет вызывающий код.
    """
    try:
        result = strict_json(canonical_bytes(summary))
        _validate(result)
        return result
    except PlotError:
        raise
    except (KeyError, TypeError, ValueError, ArithmeticError) as error:
        raise PlotError(f"Malformed E06 summary: {error}") from error


def _validate(summary):
    metrics_in_scope = analyzer.METRICS
    _require(summary["schema"] == analyzer.SCHEMA and type(summary["version"]) is int and summary["version"] == 1
             and summary["status"] == "complete_records", "Final figures require a complete E06 summary")
    _require(summary["difference"] == "right minus left"
             and summary["exclusion_counts_are_not_mutually_exclusive"] is True, "Sign/exclusion convention differs")
    record = summary.get("design", json.loads(json.dumps(asdict(build_design()), allow_nan=False)))
    design = design_from_record(record)
    sources = set(summary["direct"]["result"]["sources"]) | {p.source for p in design["paths"]}
    _require(_same(summary["counts"], plan_counts(design["paths"], design["contrasts"], sources)), "Plan counts differ")
    by_id = {p.id: p for p in design["paths"]}
    outcomes = _index(summary["raw_outcomes"], lambda r: r["path_id"], by_id, "raw outcomes")
    for pid, row in outcomes.items():
        path, score, estimate = by_id[pid], row["score"], row["estimate"]
        _require((row["condition"], row["source"], row["replicate"], row["penalty"], row["group"]) ==
                 (path.condition, path.source, path.replicate, path.penalty, f"{path.source}/r{path.replicate}")
                 and type(row["replicate"]) is int, "Raw outcome identity differs from the canonical path")
        _require(row["origin"] == ("reused" if path.reuse else "new")
                 and _same(row["true_h"], asdict(path.true_h)) and _same(row["inverse_h"], asdict(path.inverse_h)),
                 "Raw outcome provenance or observation design differs")
        accepted = estimate["procedure_accepted"]
        available = score["status"] == "available"
        _require(type(accepted) is bool and score["status"] in ("available", "unavailable")
                 and score["procedure_accepted"] is accepted and score["diagnostic_only"] is (not accepted)
                 and row["procedure_accepted"] is accepted and row["score_available"] is available
                 and row["eligible"] is (accepted and available), "Raw score eligibility differs from its procedure")
        causes = []
        if "calibration_failure" in estimate:
            causes.append("calibration_failure")
        else:
            if estimate["complete_path"] is False:
                causes.append("candidate_path_rejected")
            if any(c.get("status") == "numerical_failure" for c in estimate["candidates"].values()):
                causes.append("solver_failure")
            if estimate["selected_exponent"] is None:
                causes.append("no_accepted_candidate")
            if estimate["tuning_unresolved"]:
                causes.extend(["alpha_boundary_unresolved", "alpha_lower_boundary" if estimate["selected_exponent"] == -8.
                               else "alpha_upper_boundary"])
            if estimate.get("final_certificate", {}).get("accepted") is False:
                causes.append("final_certificate_rejected")
        if not available:
            causes.append(score["reason"])
        _require(row["exclusion_causes"] == list(dict.fromkeys(causes))
                 and (not row["eligible"] or not causes), "Raw failure reasons differ")
        expected = ({**{k: score["source"][k] for k in analyzer.SOURCE_METRICS},
                     **{k: score[k] for k in analyzer.OBSERVATION_METRICS}} if available else {})
        _require(_same(row["metrics"], expected) and all(_finite(v) for v in expected.values()),
                 "Raw metric values differ from the committed score")
        if available:
            _require(expected["E_q"] >= 0, "Negative source error")
    planned_pairs = {c.id: c for c in design["contrasts"]}
    pairs = _index(summary["paired_outcomes"], lambda r: r["contrast_id"], planned_pairs, "planned pairs")
    cells = defaultdict(list)
    for contrast in design["contrasts"]:
        row = pairs[contrast.id]
        path, other = by_id[contrast.left_path], by_id[contrast.right_path]
        left, right = outcomes[path.id], outcomes[other.id]
        penalty = "L2_to_H1" if contrast.factor == "penalty" else path.penalty
        _require((row["key"], row["factor"], row["source"], row["replicate"], row["penalty"], row["left_path"], row["right_path"]) ==
                 (contrast.key, contrast.factor, path.source, path.replicate, penalty, path.id, other.id)
                 and type(row["replicate"]) is int, "Pair identity or sign orientation differs")
        available = left["score_available"] and right["score_available"]
        accepted = left["procedure_accepted"] and right["procedure_accepted"]
        same_target = path.data_design_key == other.data_design_key
        reasons = [side+":"+cause for side, arm in (("left", left), ("right", right)) for cause in arm["exclusion_causes"]]
        _require(row["available"] is available and row["both_accepted"] is accepted
                 and row["eligible"] is (accepted and available) and row["observation_same_target"] is same_target
                 and row["same_data"] is same_target and row["exclusion_causes"] == reasons,
                 "Pair eligibility, target, or exclusions differ from raw outcomes")
        _require(_same(row["left_metrics"], left["metrics"]) and _same(row["right_metrics"], right["metrics"]),
                 "Pair metrics differ from raw outcomes")
        difference = ({k: right["metrics"][k]-left["metrics"][k] for k in metrics_in_scope
                       if same_target or k in analyzer.SOURCE_METRICS} if available else None)
        _require(_same(row["difference_right_minus_left"], difference), "Pair difference/sign differs from raw scores")
        cells[(contrast.key, path.source, penalty)].append(row)
    summaries = _index(summary["contrast_summaries"], lambda r: (r["key"], r["source"], r["penalty"]), cells, "contrast cells")
    for key, rows in cells.items():
        row = summaries[key]
        eligible = [r for r in rows if r["eligible"]]
        for name, value in dict(n_expected=len(rows), n_available=sum(r["available"] for r in rows),
            n_both_accepted=sum(r["both_accepted"] for r in rows), n_eligible=len(eligible)).items():
            _require(type(row[name]) is int and row[name] == value, "Aggregate denominator differs from planned pairs")
        reasons = dict(sorted(Counter(c for r in rows for c in set(r["exclusion_causes"])).items()))
        same_target = rows[0]["observation_same_target"]
        _require(row["planned_replicates"] == [r["replicate"] for r in rows]
                 and row["retained_replicates"] == [r["replicate"] for r in eligible]
                 and row["contrast_ids"] == [r["contrast_id"] for r in rows]
                 and row["factor"] == rows[0]["factor"] and row["observation_same_target"] is same_target
                 and _same(row["exclusion_counts"], reasons), "Aggregate membership/reasons differ")
        metrics = metrics_in_scope if same_target else analyzer.SOURCE_METRICS
        _require(set(row["statistics"]) == set(metrics), "Statistical metric set differs")
        for metric in metrics:
            _statistics(row["statistics"][metric], _moments([p["difference_right_minus_left"][metric] for p in eligible]))
    _figure_plan(summary)


def _figure_plan(summary):
    record = summary.get("design", json.loads(json.dumps(asdict(build_design()), allow_nan=False)))
    design = design_from_record(record)
    return {f["id"]: f["contrast_keys"] for f in resolve_figures(design["contrasts"], summary.get("figures"))}


def _figure_rows(summary, kind):
    figures = _figure_plan(summary)
    _require(kind in figures, "Unknown declared figure")
    rows = [row for key in figures[kind] for row in summary["contrast_summaries"] if row["key"] == key]
    sources = dict.fromkeys(row["source"] for row in rows)
    return {source: [row for row in rows if row["source"] == source] for source in sources}


def make_figure(summary, kind):
    """Построить рисунок парных эффектов E06.

    Parameters
    ----------
    summary : dict
        Завершённая сводка `summarize_records`.
    kind : str
        Идентификатор рисунка из закреплённого плана.

    Returns
    -------
    matplotlib.figure.Figure
        Рисунок отдельных пригодных пар и средних безразмерных разностей E_q.

    Notes
    -----
    При n >= 2 отрезки показывают среднее плюс-минус s / sqrt(n), где s —
    выборочное стандартное отклонение парных разностей. Это стандартная
    ошибка среднего, а не доверительный интервал. При одной паре
    показана её разность ошибок; при отсутствии пар оценка не отображается.
    """
    return _make_figure(validate_summary(summary), kind)


def _make_figure(summary, kind):
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from matplotlib.lines import Line2D
    cells = _figure_rows(summary, kind)
    factors = {row["factor"] for rows in cells.values() for row in rows}
    xlabel = r"Разность нормированных ошибок $E_q$ (безразмерная)"
    if factors == {"assumed_spatial_H"}:
        xlabel = r"$E_q(H_{\mathrm{inv}}) - E_q(H_{\mathrm{true}})$ (безразмерная разность)"
    elif factors == {"matched_spatial_design"}:
        xlabel += "\nизменённое ядро − исходное ядро (в данных и модели)"
    row_labels = {source: [(r"Ядро генератора $H_{\mathrm{true}}$:" + "\n" if row["factor"] == "assumed_spatial_H" else "")
                           + ROW_LABELS[row["key"]] + "\n" + PENALTY_LABELS[row["penalty"]]
                           for row in rows] for source, rows in cells.items()}
    pairs = {r["contrast_id"]: r for r in summary["paired_outcomes"]}
    if factors == {"assumed_spatial_H"}:
        outcomes = {row["path_id"]: row for row in summary["raw_outcomes"]}
        kernels = {(kernel["spatial_kind"], kernel["width_km"])
                   for rows in cells.values() for row in rows for identifier in row["contrast_ids"]
                   for kernel in (outcomes[pairs[identifier]["right_path"]]["inverse_h"],)}
        if len(kernels) == 1:
            family, width = next(iter(kernels))
            name = "гауссово, σ" if family == "gaussian" else "компактное, масштаб"
            xlabel += "\n" + r"Принятое ядро $H_{\mathrm{inv}}$: " + name + f" = {width:g} км".replace(".", ",")
    data = [p["difference_right_minus_left"]["E_q"] for rows in cells.values() for r in rows
            for identifier in r["contrast_ids"] if (p := pairs[identifier])["eligible"]]
    extents = data + [0.]
    for rows in cells.values():
        for row in rows:
            stats = row["statistics"]["E_q"]
            if row["n_eligible"] >= 2:
                extents.extend([stats["mean"]-stats["mcse"], stats["mean"]+stats["mcse"]])
    low, high = min(extents), max(extents)
    padding = .15*(high-low) if high > low else .025
    with mpl.rc_context({"font.family": "DejaVu Sans", "font.size": 14, "mathtext.fontset": "dejavusans",
                         "svg.fonttype": "path", "svg.hashsalt": "YM2026-E06-two-main-figures-v1"}):
        label_count = max(sum(label.count("\n") + 1 for label in labels)
                          for labels in row_labels.values())
        fig = Figure(figsize=(9*len(cells), max(7., .4*label_count)), facecolor="white")
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, len(cells), sharex=True, squeeze=False)[0]
        for axis, source in zip(axes, cells):
            rows = cells[source]
            for y, row in enumerate(rows):
                color = COLORS[row["penalty"]]
                chosen = [pairs[k] for k in row["contrast_ids"] if pairs[k]["eligible"]]
                for pair in chosen:
                    jitter = .09*(pair["replicate"]-statistics.mean(row["planned_replicates"]))
                    line, = axis.plot(pair["difference_right_minus_left"]["E_q"], y+jitter,
                                      "o", color=color, markersize=5.5, alpha=.80, zorder=3)
                    line.set_gid(f"pair:{pair['contrast_id']}")
                n = row["n_eligible"]
                stats = row["statistics"]["E_q"]
                if n >= 2:
                    bar = axis.errorbar(stats["mean"], y, xerr=stats["mcse"], fmt="D", markersize=7,
                        markerfacecolor="white", markeredgewidth=1.7, color=color, capsize=4, linewidth=1.3, zorder=4)
                    bar.lines[0].set_gid(f"mean:{source}:{row['key']}:{row['penalty']}")
            axis.set_yticks(range(len(rows)), row_labels[source] if source == next(iter(cells)) else [""]*len(rows))
            axis.tick_params(axis="both", labelsize=14, length=3)
            axis.set_ylim(len(rows)-.48, -.55)
            axis.set_xlim(low-padding, high+padding)
            axis.axvline(0., color=".45", linewidth=1., linestyle="--", zorder=1)
            axis.grid(axis="x", color=".9", zorder=0)
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.3g}".replace(".", ",")))
            axis.set_ylabel(source, fontsize=16)
            axis.set_xlabel(xlabel, fontsize=14)
        penalties = dict.fromkeys(row["penalty"] for rows in cells.values() for row in rows)
        handles = [Line2D([], [], color=COLORS[penalty], linewidth=2, label=PENALTY_LABELS[penalty])
                   for penalty in penalties]
        handles.extend([
            Line2D([], [], marker="o", linestyle="none", color=".35", label="Отдельная пара"),
            Line2D([], [], marker="D", markerfacecolor="white", color=".35", label="Среднее ± стандартная ошибка"),
        ])
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=11)
        fig.tight_layout(rect=(0, .07, 1, 1))
    return fig


def reason_label(reason):
    """Перевести зарегистрированную причину исключения пары на русский язык.

    Parameters
    ----------
    reason : str
        Причина в формате left:code или right:code, где code входит в
        `CAUSES`.

    Returns
    -------
    str
        Текст с указанием левой или правой процедуры и причины исключения.
    """

    side, _, code = reason.partition(":")
    _require(side in ("left", "right") and code in CAUSES, "Unregistered exclusion reason")
    return ("Левая процедура: " if side == "left" else "Правая процедура: ") + CAUSES[code]


def _tables(summary):
    cells = list({(row["key"], row["source"], row["penalty"]): row
                  for kind in _figure_plan(summary) for rows in _figure_rows(summary, kind).values() for row in rows}.values())
    identifiers = {identifier for row in cells for identifier in row["contrast_ids"]}
    pairs = [row for row in summary["paired_outcomes"] if row["contrast_id"] in identifiers]
    pair_rows = []
    for row in pairs:
        pair_rows.append(dict(contrast=row["key"], source=row["source"], penalty=row["penalty"], replicate=row["replicate"],
            left_path=row["left_path"], right_path=row["right_path"], available=row["available"],
            both_accepted=row["both_accepted"], eligible=row["eligible"],
            left_E_q=row["left_metrics"].get("E_q"), right_E_q=row["right_metrics"].get("E_q"),
            difference_right_minus_left_E_q=None if row["difference_right_minus_left"] is None else row["difference_right_minus_left"]["E_q"],
            exclusion_reasons=" | ".join(row["exclusion_causes"]),
            exclusion_reasons_ru=" | ".join(reason_label(x) for x in row["exclusion_causes"])))
    summaries, failures = [], []
    for row in cells:
        summaries.append(dict(contrast=row["key"], source=row["source"], penalty=row["penalty"],
            n_expected=row["n_expected"], n_terminal=row["n_expected"], n_available=row["n_available"],
            n_both_accepted=row["n_both_accepted"], n_eligible=row["n_eligible"],
            n_excluded=row["n_expected"]-row["n_eligible"], retained_replicates=";".join(map(str, row["retained_replicates"])),
            mean_E_q_difference=row["statistics"]["E_q"]["mean"], sd=row["statistics"]["E_q"]["sd"],
            mcse=row["statistics"]["E_q"]["mcse"], units="dimensionless", sign="right minus left"))
        for reason, count in sorted(row["exclusion_counts"].items()):
            failures.append(dict(contrast=row["key"], source=row["source"], penalty=row["penalty"],
                reason=reason, reason_ru=reason_label(reason), count=count, n_expected=row["n_expected"],
                fraction=count/row["n_expected"], counts_overlap=True))
    return pair_rows, summaries, failures


def _destination(output_dir, protected_root, additional_roots=()):
    from .read_results import export_destination, SnapshotError
    try:
        return export_destination(output_dir, protected_root=protected_root, additional_roots=additional_roots)
    except SnapshotError as error:
        raise PlotError(str(error)) from error


def _csv(path, rows, fields):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render(summary, output_dir, *, protected_root, additional_roots=(), fixture=False):
    """Записать объявленные рисунки PNG/SVG, три CSV и figures.json.

    Parameters
    ----------
    summary : dict
        Завершённая сводка `summarize_records`.
    output_dir : str or Path
        Новый каталог вывода вне входов и загруженных исходников.
    protected_root : str or Path
        Корень входных результатов, защищённый от записи.
    additional_roots : iterable of str or Path, optional
        Пути научных входов из проверенного допуска.
    fixture : bool, optional
        Записать в манифест признак синтетического примера.

    Returns
    -------
    dict
        Манифест экспорта с составом и SHA-256 созданных файлов.

    Notes
    -----
    При ошибке частично записанные файлы остаются в каталоге вывода.
    """
    _require(type(fixture) is bool, "fixture must be an explicit boolean")
    report = validate_summary(summary)
    destination = _destination(output_dir, protected_root, additional_roots)
    pair_rows, summaries, failures = _tables(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    import matplotlib as mpl
    exported = []
    for kind in _figure_plan(report):
        fig = _make_figure(report, kind)
        try:
            for format_name in ("png", "svg"):
                target = destination/f"e06-{kind}.{format_name}"
                with target.open("xb") as stream, mpl.rc_context({"svg.hashsalt": "YM2026-E06-two-main-figures-v1", "svg.fonttype": "path"}):
                    metadata = {"Software": "YM2026 E06 paired renderer"} if format_name == "png" else {"Date": None, "Creator": "YM2026 E06 paired renderer"}
                    fig.savefig(stream, format=format_name, dpi=170, metadata=metadata)
                exported.append(target)
        finally:
            fig.clear()
    for filename, rows, fields in (
        ("pairs.csv", pair_rows, list(pair_rows[0]) if pair_rows else
         ["contrast", "source", "penalty", "replicate", "left_path", "right_path", "available", "both_accepted",
          "eligible", "left_E_q", "right_E_q", "difference_right_minus_left_E_q", "exclusion_reasons", "exclusion_reasons_ru"]),
        ("summary.csv", summaries, list(summaries[0]) if summaries else
         ["contrast", "source", "penalty", "n_expected", "n_terminal", "n_available", "n_both_accepted", "n_eligible",
          "n_excluded", "retained_replicates", "mean_E_q_difference", "sd", "mcse", "units", "sign"]),
        ("failures.csv", failures, ["contrast", "source", "penalty", "reason", "reason_ru", "count", "n_expected", "fraction", "counts_overlap"]),
    ):
        target = destination/filename
        _csv(target, rows, fields)
        exported.append(target)
    shown_keys = {row["contrast"] for row in summaries}
    sign_labels = {"assumed_spatial_H": ("spatial", "assumed minus matched"),
                   "covariance_family": ("covariance_family", "population minus oracle"),
                   "covariance_estimation": ("covariance_estimation", "estimated minus population")}
    signs = dict(sign_labels.get(row["factor"], (row["key"], "right minus left"))
                 for row in report["contrast_summaries"] if row["key"] in shown_keys)
    manifest = dict(schema="ym2026.observation_sensitivity.figures", version=1, status="complete",
        fixture=fixture, figure_count=len(_figure_plan(report)), shown_pair_count=len(pair_rows), shown_cell_count=len(summaries),
        summary_sha256=digest(report), inputs=report["inputs"],
        raw_input_file_sha256=report.get("raw_input_file_sha256"),
        snapshot_reader_sha256=report.get("snapshot_reader_sha256"),
        analyzer_sha256=hashlib.sha256(Path(analyzer.__file__).read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matplotlib_version=mpl.__version__, signs=signs,
        metrics=dict(name="E_q", units="dimensionless", multiplier=1, uncertainty="one MCSE; not a confidence interval"),
        metric_scope=list(analyzer.METRICS),
        denominators=dict(terminal="all planned pairs; completeness required before rendering",
            available="both source scores present, including diagnostics; concentration availability may differ", both_accepted="both original procedures accepted",
            eligible="both original procedures accepted AND both source scores available; statistics use this count"),
        exclusion_counts_overlap=True, csv_missing_value="empty field means undefined/unavailable, never zero",
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in exported})
    with (destination/"figures.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    return manifest


def main(argv=None):
    """Прочитать завершённый расчёт E06 и экспортировать рисунки парных эффектов.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Аргументы без имени программы; None использует ``sys.argv[1:]``.

    Returns
    -------
    int
        0 после успешной записи рисунков и таблиц.
    """

    from . import read_results as snapshot
    parser = argparse.ArgumentParser(description=__doc__)
    snapshot.add_read_arguments(parser)
    parser.add_argument("--output-dir", required=True, help="Новый каталог рисунков и таблиц CSV")
    args = parser.parse_args(argv)
    protected = snapshot.input_root(args)
    records = snapshot.records_from_arguments(args)
    roots = snapshot.scientific_input_roots(records["freeze"]["admission"])
    snapshot.export_destination(args.output_dir, protected_root=protected, additional_roots=roots)
    summary = analyzer.summarize_records(records["freeze"], records["direct"], records["groups"],
                                        expected_freeze_sha256=args.freeze_sha256)
    summary["raw_input_file_sha256"] = records["raw_file_sha256"]
    summary["snapshot_reader_sha256"] = hashlib.sha256(Path(snapshot.__file__).read_bytes()).hexdigest()
    manifest = render(summary, args.output_dir, protected_root=protected, additional_roots=roots)
    print(json.dumps(dict(output=str(Path(args.output_dir).absolute()), figures=manifest["figure_count"], fixture=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
