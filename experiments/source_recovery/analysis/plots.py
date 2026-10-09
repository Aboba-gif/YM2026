"""Графики парных разностей и полноты расчётов восстановления."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import textwrap

from adrkit.config.validation import strict_json


SCHEMA = "ym2026.research_validation.summary.v1"
SOURCE_LABELS = {"NEW-J2": "J2", "NEW-S2": "S2"}
CONDITIONS = {
    "main": "Основной случай", "mismatch_Gxt": "Генератор Gxt",
    "weight_W01": "Веса W01", "weight_W02": "Веса W02", "weight_Woracle": "Известная ковариация",
    "availability_onlyKrAZ": "Только пост КрАЗ", "availability_withoutKrAZ": "Без поста КрАЗ",
    "mask_random2": "Случайные пропуски", "mask_block2": "Блок пропусков",
    "noise_iid_high": "Независимые ошибки", "noise_iid_low": "Уменьшенный шум",
    "tau_5": "τ = 5 мин", "tau_60": "τ = 60 мин",
    "linear_matched": "Линейная истинная модель", "linear_mismatch": "Линейная обратная модель",
    "basis_145": "Базис: 145 узлов", "basis_289": "Базис: 289 узлов",
    "grid_time": "Обратная сетка Gt", "grid_space": "Обратная сетка Gx",
    "dense": "Наблюдения через 2,5 мин", "relocation_downwind": "Сдвиг поста по ветру",
    "relocation_upwind": "Сдвиг поста против ветра", "temporal_average": "Усреднение за 20 мин",
    "generator_grid": "Сетка генератора: G0 → Gxt",
}
CAUSES = {
    "path_pending": "Расчёт не начат", "path_not_finalized": "Процедура не завершена",
    "calibration_failure": "Не удалось оценить ковариацию", "incomplete_candidate_path": "Не все значения α допустимы",
    "alpha_boundary_unresolved": "Выбранное α на границе диапазона",
    "final_certificate_rejected": "Итоговая проверка не пройдена",
    "single_fit_rejected": "Одиночный расчёт отклонён",
    "baseline_procedure_rejected": "Базовая процедура отклонена",
    "procedure_not_accepted": "Процедура не принята", "score_pending": "Оценка отсутствует",
    "diagnostic_only": "Только диагностическая оценка", "not_planned_or_missing": "Нет предусмотренной пары",
}


def digest(path):
    """Вернуть шестнадцатеричный SHA-256 байтов указанного файла."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    """Прочитать JSON с конечными значениями и уникальными ключами.

    Parameters
    ----------
    path : str or path-like
        Файл JSON для построения рисунков.

    Returns
    -------
    record : object
        Десериализованное значение JSON; ожидаемую структуру проверяет вызывающий код.

    Raises
    ------
    ValueError
        Есть повторный ключ, неконечное число либо некорректный JSON.
    """

    return strict_json(Path(path).read_bytes())


def finite(value):
    """Проверить, что значение — конечный встроенный int или float."""

    return type(value) in (int, float) and math.isfinite(value)


def validate_pair(row):
    """Проверить статистику сохранённых парных ошибок источника.

    Parameters
    ----------
    row : dict
        Сводка пары с числом реализаций, strict_pairs, статистиками и
        причинами исключения.

    Returns
    -------
    differences : list of float
        Безразмерные разности E_q справа минус слева для включённых пар.

    Raises
    ------
    ValueError
        Неверны пары, знак разности, знаменатели, статистики или счётчики
        исключения.
    """
    for field in ("nplanned", "nactual", "nexcluded"):
        if type(row[field]) is not int or row[field] < 0:
            raise ValueError(f"Invalid {field}")
    n, planned = row["nactual"], row["nplanned"]
    if planned < 1 or n > planned or row["nexcluded"] != planned - n:
        raise ValueError("Invalid pair denominator")
    pairs = row["strict_pairs"]
    if len(pairs) != n or len({p["replicate"] for p in pairs}) != n:
        raise ValueError("Duplicate/missing strict pairs")
    values = []
    for pair in pairs:
        if pair.get("strict_eligible") is not True or pair.get("exclusion_causes"):
            raise ValueError("Excluded pair included in strict statistics")
        value = pair["difference_right_minus_left"]["E_q"]
        a, b = pair["left_scores"]["E_q"], pair["right_scores"]["E_q"]
        if not all(finite(x) for x in (value, a, b)) or not math.isclose(value, b-a, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("Invalid paired difference/sign")
        values.append(value)
    sd = statistics.stdev(values) if n > 1 else None
    expected = {"n": n, "mean": statistics.mean(values) if n else None,
                "sd": sd, "mcse": sd/math.sqrt(n) if sd is not None else None,
                "median": statistics.median(values) if n else None}
    actual = row["statistics"]["E_q"]
    for key, value in expected.items():
        old = actual[key]
        if value is None:
            if old is not None:
                raise ValueError(f"Undefined {key} must be null")
        elif not finite(old) or not math.isclose(value, old, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError(f"Incorrect paired {key}")
    if row.get("exclusion_counts_are_not_mutually_exclusive") is not True:
        raise ValueError("Exclusion overlap must be declared")
    for count in row["exclusion_counts"].values():
        if type(count) is not int or not 0 <= count <= row["nexcluded"]:
            raise ValueError("Invalid exclusion count")
    return values


def validate_summary(report, expected_contrasts=None):
    """Проверить стадии процедур, полноту их учёта и парные статистики.

    Parameters
    ----------
    report : dict
        Сводка summarize с файлами и процедурами на стадии scored.
    expected_contrasts : sequence of str or None, optional
        Идентификаторы контрастов в требуемом порядке. При None соответствие
        сводки плану контрастов не проверяется.

    Returns
    -------
    groups : dict of str to list of dict
        Парные результаты, сгруппированные по идентификатору контраста.

    Raises
    ------
    ValueError
        Не согласованы стадии, идентификаторы, полнота, парные статистики
        или план контрастов.
    """

    if report.get("schema") != SCHEMA:
        raise ValueError("Unsupported summary schema")
    checkpoints = report.get("checkpoints", {})
    if not checkpoints or any(c.get("stage") != "scored" for c in checkpoints.values()):
        raise ValueError("Final plots require ALL checkpoints stage=scored")
    procedures = report.get("procedures", [])
    if not procedures or any(row.get("stage") != "scored" for row in procedures):
        raise ValueError("Final plots require ALL procedures stage=scored")
    identities = [(r["source"], r["replicate"], r["path_id"]) for r in procedures]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate procedure identity")
    if report["coverage"]["planned_procedures"] != len(procedures):
        raise ValueError("Coverage denominator mismatch")
    for key in ("candidate_attempted", "candidate_accepted", "candidate_rejected", "nominal_candidates"):
        values = [row["coverage"][key] for row in procedures]
        if any(type(x) is not int or x < 0 for x in values) or sum(values) != report["coverage"][key]:
            raise ValueError("Candidate coverage mismatch")
    groups = defaultdict(list)
    for row in report["primary_paired_results"] + report["contrasts"] + report["single_fit_comparisons"]:
        validate_pair(row)
    if not report["primary_paired_results"]:
        raise ValueError("Missing primary pairs")
    for row in report["contrasts"]:
        contrast, arm = row["label"].rsplit("/", 1)
        if arm not in ("L2", "H1"):
            raise ValueError("Unexpected penalty")
        groups[contrast].append(row)
    if expected_contrasts is not None:
        ids = tuple(expected_contrasts)
        if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
            raise ValueError("Unique nonempty contrast IDs required")
        if tuple(groups) != ids:
            raise ValueError("Summary contrast IDs or order differ from the declared plan")
    return groups


def verify_inputs(report):
    """Сопоставить сводку с исходными файлами расчёта.

    Parameters
    ----------
    report : dict
        Сводка с каталогом расчёта, хешем анализатора и идентификатором
        основного условия.

    Returns
    -------
    spec : dict
        Конфигурация расчёта с явным планом контрастов.

    Raises
    ------
    ValueError
        Хеш анализатора, исходные файлы, повторная агрегация или план не
        совпадают.
    """
    from .summary import summarize
    root = Path(report["run_directory"]).resolve()
    if report["analyzer_sha256"] != digest(Path(__file__).with_name("summary.py")):
        raise ValueError("Analyzer hash differs; regenerate the summary explicitly")
    fresh = summarize(root, report["main_condition"])
    if fresh != report:
        raise ValueError("Input changed or summary differs from read-only checkpoint aggregation")
    spec = load_json(root / "run.json")["configuration"]
    plans = spec.get("analysis_contrasts")
    if not isinstance(plans, list):
        raise ValueError("Explicit contrast plan required")
    validate_summary(fresh, [c["id"] for c in plans])
    return spec


def reason_label(reason):
    """Преобразовать код причины исключения в подпись оси."""

    side, sep, code = reason.partition(":")
    if sep:
        prefix = {"left": "База", "right": "Вариант"}.get(side, side)
        return prefix + ": " + CAUSES.get(code, code)
    return CAUSES.get(reason, reason)


def pair_figure(rows, *, xlabel):
    """Построить график парных разностей ошибок источника.

    Parameters
    ----------
    rows : sequence of dict
        Парные сводки с ошибками E_q и их статистиками.
    xlabel : str
        Подпись оси с явным порядком вычитания условий.

    Returns
    -------
    matplotlib.figure.Figure
        График отдельных пар и средних с отрезками плюс-минус одну MCSE.

    Notes
    -----
    Стандартная ошибка среднего равна s / sqrt(n), где s — выборочное
    стандартное отклонение парных разностей. При n < 2 она не определена.
    Отрезок не является доверительным интервалом; при отсутствии
    включённых пар оценка среднего не изображается.
    """

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, max(5., .6*len(rows))))
    labels = []
    for y, row in enumerate(rows):
        values = validate_pair(row)
        source = SOURCE_LABELS.get(row["source"], row["source"])
        arm = row["label"].rsplit("/", 1)[-1] if "/" in row["label"] else ""
        labels.append(source + (" / " + {"L2": "L²", "H1": "H¹"}[arm] if arm else ""))
        n = row["nactual"]
        stats = row["statistics"]["E_q"]
        if n:
            for i, value in enumerate(values):
                jitter = .26 * ((i / max(n-1, 1))-.5)
                ax.plot(value, y+jitter, "o", color="#92a8c5", markersize=5)
            ax.errorbar(stats["mean"], y, xerr=stats["mcse"], fmt="D", color="#17386b", capsize=4, markersize=5)
    ax.axvline(0, color="0.5", linewidth=.8, linestyle="--")
    ax.set_yticks(range(len(rows)), labels)
    ax.set_ylim(len(rows)-.5, -.5)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.tick_params(axis="both", labelsize=13)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, position: f"{x:g}".replace(".", ",")))
    ax.grid(axis="x", color=".9")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color="#92a8c5", label="Отдельная пара"),
        Line2D([], [], marker="D", color="#17386b", label="Среднее ± стандартная ошибка"),
    ], loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False, fontsize=11)
    fig.tight_layout()
    return fig


def coverage_figure(report):
    """Построить рисунок полноты процедур и причин исключения.

    Parameters
    ----------
    report : dict
        Сводка процедур и числа попыток регуляризации.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Несохранённый рисунок числа допустимых и исключённых процедур и
        пересекающихся причин.
    """

    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.3, 1]})
    groups = defaultdict(list)
    for row in report["procedures"]:
        key = "Одиночные контроли" if row["path_id"].startswith("single/") else CONDITIONS.get(row["condition"], row["condition"])
        groups[key].append(row)
    labels, ok, excluded = [], [], []
    for key, rows in groups.items():
        labels.append(key)
        ok.append(sum(not r["exclusion_causes"] for r in rows))
        excluded.append(len(rows)-ok[-1])
    ax.barh(labels, ok, color="#426898", label="Допущены к сравнению")
    ax.barh(labels, excluded, left=ok, color="#d4a6a0", label="Исключены из сравнения")
    ax.set_xlim(0, max(a+b for a,b in zip(ok,excluded))*1.18)
    ax.invert_yaxis()
    ax.set_xlabel("Число процедур")
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False, fontsize=9)
    reasons = Counter()
    for row in report["procedures"]:
        reasons.update(set(row["exclusion_causes"]))
    if reasons:
        names, counts = zip(*sorted(reasons.items()))
        bx.barh([textwrap.fill(reason_label(x), 35) for x in names], counts, color="#a95548")
        bx.set_xlim(0, max(counts)*1.16)
        bx.invert_yaxis()
    else:
        bx.set_yticks([])
    bx.set_xlabel("Число процедур")
    bx.tick_params(axis="y", labelsize=8)
    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.18, right=.98, top=.94, bottom=.08, wspace=.72)
    return fig


def render(report, spec, destination):
    """Сохранить рисунки полноты и всех запланированных контрастов.

    Parameters
    ----------
    report : dict
        Полная проверяемая сводка с файлами на стадии scored.
    spec : dict
        Конфигурация с analysis_contrasts для порядка рисунков.
    destination : str or path-like
        Каталог PDF и PNG вне каталога исходных расчётов.

    Returns
    -------
    files : dict of str to str
        Относительные имена PDF/PNG, сохранённых этим вызовом, и SHA-256 их байтов.
    """

    plans = spec["analysis_contrasts"]
    groups = validate_summary(report, [c["id"] for c in plans])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.unicode_minus": True, "figure.facecolor": "white", "savefig.facecolor": "white"})
    destination = Path(destination).resolve()
    if destination.is_relative_to(Path(report["run_directory"]).resolve()):
        raise ValueError("Graphs must stay outside the immutable run directory")
    destination.mkdir(parents=True, exist_ok=True)
    saved_files = []
    for name, figure in (("main-paired", pair_figure(report["primary_paired_results"],
                         xlabel="Разность нормированных ошибок восстановления:\n"
                         + r"$E_q(H^1) - E_q(L^2)$ (безразмерная)")),
                         ("coverage", coverage_figure(report))):
        pdf_path = destination/(name+".pdf")
        png_path = destination/(name+".png")
        figure.savefig(pdf_path)
        figure.savefig(png_path, dpi=160)
        saved_files.extend((pdf_path, png_path))
        plt.close(figure)
    if plans:
        pages = destination / "contrast-pages"
        pages.mkdir(exist_ok=True)
        series_path = destination/"contrast-series.pdf"
        with PdfPages(series_path) as pdf:
            for i, plan in enumerate(plans, 1):
                key = plan["id"]
                baseline = CONDITIONS.get(plan["baseline"], plan["baseline"])
                variant = CONDITIONS.get(plan["variant"], plan["variant"])
                figure = pair_figure(groups[key], xlabel=r"Разность $E_q$ (безразмерная):" + "\n" + variant + " − " + baseline)
                pdf.savefig(figure)
                page_path = pages/f"contrast-{i:02d}.png"
                figure.savefig(page_path, dpi=160)
                saved_files.append(page_path)
                plt.close(figure)
        saved_files.append(series_path)
    return {str(path.relative_to(destination)): digest(path) for path in saved_files}


def main():
    """Проверить сводку и входы, сохранить рисунки и описание файлов."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Сводка, созданная текущим анализатором по завершённому E05")
    parser.add_argument("--output-dir", required=True, help="Новый каталог рисунков PDF и PNG")
    args = parser.parse_args()
    path = Path(args.summary).resolve()
    original_hash = digest(path)
    report = load_json(path)
    spec = verify_inputs(report)
    rendered = render(report, spec, args.output_dir)
    if digest(path) != original_hash:
        raise ValueError("Summary changed during rendering")
    verify_inputs(report)
    evidence = dict(schema="ym2026.experiment_plots.v1", summary_sha256=original_hash,
                    renderer_sha256=digest(__file__), all_checkpoints_scored=True,
                    planned_contrasts=len(spec["analysis_contrasts"]), figure_sha256=rendered,
                    statistics="Accepted paired differences only; ±MCSE, not confidence intervals; no fit/scoring/selection invoked.")
    (Path(args.output_dir)/"plot-manifest.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "rendered_complete_summary", "contrasts": len(spec["analysis_contrasts"]), "figures": len(rendered)}))


if __name__ == "__main__":
    main()
