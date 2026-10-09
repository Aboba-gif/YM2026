"""Парные различия и полнота расчётов по файлам восстановления источника.

Повтором считается реализация шума. Средние и стандартные ошибки
рассчитываются по парам с двумя принятыми процедурами восстановления.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from adrkit.config.validation import digest as json_digest, strict_json


METRICS = ("E_q", "relative_L2", "full_primary_test_rmse", "full_primary_noiseless_rmse")
FACTORS = ("weight", "noise", "availability", "mask", "nodes", "grid", "tau_hours",
           "temporal_H", "relocation_km", "fit_regime", "reaction_regime", "truth_grid")


def digest(path):
    """Вернуть шестнадцатеричный SHA-256 байтов указанного файла."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(path):
    """Прочитать строгую запись JSON и хеш тех же байтов.

    Parameters
    ----------
    path : str or path-like
        Файл JSON со стандартными конечными значениями и уникальными
        ключами.

    Returns
    -------
    record : object
        Десериализованное значение JSON; ожидаемую структуру проверяет вызывающий код.
    digest : str
        SHA-256 прочитанных байтов.

    Raises
    ------
    ValueError
        Есть повторный ключ, неконечное число либо некорректный JSON.
    """

    raw=Path(path).read_bytes()
    return strict_json(raw),hashlib.sha256(raw).hexdigest()


def selection_hash(value):
    """Вернуть хеш канонического JSON выбора с нормализацией отрицательного нуля."""

    return json_digest(value)


def moments(values):
    r"""Вычислить описательные статистики последовательности оценок.

    Parameters
    ----------
    values : sequence of float
        Конечные значения одной метрики, например разности по реализациям
        шума.

    Returns
    -------
    statistics : dict
        n, mean, median, выборочное sd и mcse в единицах входа. При пустом
        входе оценки None; sd и mcse также None при одном значении.

    Notes
    -----
    Для :math:`n>1` возвращается :math:`\mathrm{MCSE}=s/\sqrt{n}`,
    где :math:`s` — выборочное стандартное отклонение с делителем
    :math:`n-1`. Интерпретация как ошибки среднего Монте-Карло предполагает
    независимые одинаково распределённые реализации. Здесь функция только
    вычисляет статистики, не проверяя эту предпосылку.
    """

    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
        raise ValueError("Nonfinite/missing score in an explicitly included pair")
    n=len(values)
    sd=statistics.stdev(values) if n>1 else None
    return dict(n=n,mean=statistics.mean(values) if n else None,sd=sd,
        mcse=sd/math.sqrt(n) if sd is not None else None,
        median=statistics.median(values) if n else None)


def exclusion_causes(path, score, stage):
    """Определить причины исключения процедуры из парного анализа.

    Parameters
    ----------
    path : dict or None
        Запись оценки либо None, если путь ещё отсутствует.
    score : dict or None
        Проверочные ошибки либо None, если они ещё не вычислены.
    stage : str
        Стадия файла состояния расчёта.

    Returns
    -------
    causes : list of str
        Коды причин; пустой список означает допустимую процедуру. Причины
        могут пересекаться.
    """

    causes=[]
    if path is None:
        return ["path_pending"]
    if not path.get("finalized"):
        causes.append("path_not_finalized")
    if path.get("calibration_failure"):
        causes.append("calibration_failure")
    if path.get("complete_path") is False:
        causes.append("incomplete_candidate_path")
    if path.get("tuning_unresolved"):
        causes.append("alpha_boundary_unresolved")
    certificate=path.get("final_certificate")
    if certificate is not None and not certificate.get("accepted"):
        causes.append("final_certificate_rejected")
    if path.get("candidate",{}).get("accepted") is False:
        causes.append("single_fit_rejected")
    if path.get("baseline_procedure_accepted") is False:
        causes.append("baseline_procedure_rejected")
    if not path.get("procedure_accepted") and not causes:
        causes.append("procedure_not_accepted")
    if score is None or stage!="scored":
        causes.append("score_pending")
    elif score.get("diagnostic_only"):
        causes.append("diagnostic_only")
    return causes


def factor_signature(condition,default_truth_grid):
    """Выделить параметры научного условия для сравнения факторов.

    Parameters
    ----------
    condition : dict
        Условие модели, наблюдений, шума, штрафа и сетки.
    default_truth_grid : str
        Сетка генератора, если она не переопределена в условии.

    Returns
    -------
    signature : dict
        Значения факторов с явно указанными календарём наблюдений,
        коэффициентами реакции и сеткой генератора.
    """

    result={k:condition[k] for k in FACTORS if k not in ("fit_regime","reaction_regime","truth_grid")}
    result["fit_regime"]=condition.get("fit_regime","primary")
    result["reaction_regime"]=[condition["truth_gamma"],condition["inverse_gamma"]]
    result["truth_grid"]=condition.get("truth_grid",default_truth_grid)
    return result


def contrast_plan(spec,main_condition):
    """Получить план сравнений научных условий.

    Parameters
    ----------
    spec : dict
        Конфигурация опыта с условиями и необязательным analysis_contrasts.
    main_condition : str
        Идентификатор основного условия для порядка автоматически созданных
        пар.

    Returns
    -------
    contrasts : list of dict
        Явные сравнения либо пары условий, отличающиеся одним фактором.
    origin : str
        Описание явного или автоматически созданного плана.

    Notes
    -----
    Автоматические пары формируются по условиям, без обращения к
    результатам. Сравнения остаются описательными.
    """

    if "analysis_contrasts" in spec:
        return spec["analysis_contrasts"],"explicit frozen manifest"
    # При отсутствии явного плана сравниваются
    # пары условий, различающиеся одним фактором.
    conditions=spec["conditions"]
    order=sorted(conditions,key=lambda c:(c["id"]!=main_condition,c["id"]))
    contrasts=[]
    for i,base in enumerate(order):
        for variant in order[i+1:]:
            a,b=factor_signature(base,spec.get("truth_grid")),factor_signature(variant,spec.get("truth_grid"))
            changed=[key for key in FACTORS if a[key]!=b[key]]
            if len(changed)==1:
                contrasts.append(dict(id=base["id"]+"__"+variant["id"],
                    baseline=base["id"],variant=variant["id"],factor=changed[0]))
    return contrasts,"outcome-blind one-factor conditions, descriptive only"


def pair_summary(rows,label,source,planned_replicates,left,right):
    """Составить сводку разностей двух процедур по реализациям.

    Parameters
    ----------
    rows : dict
        Записи по ключам (источник, реализация, путь) с причинами
        исключения и ошибками.
    label : str
        Подпись сравнения.
    source : str
        Идентификатор источника.
    planned_replicates : sequence of int
        Запланированные реализации шума.
    left, right : str
        Идентификаторы базовой и сравниваемой процедур.

    Returns
    -------
    summary : dict
        Статистики разностей справа минус слева для пар без причин
        исключения, диагностические пары и пересекающиеся счётчики причин.

    Notes
    -----
    Средние и MCSE рассчитаны только по парам, в которых обе процедуры приняты.
    Совпадение целей ошибки прогноза отмечается в каждой паре;
    несовпадение целей само по себе здесь не исключает пару.
    """

    strict=[]
    diagnostic=[]
    reasons=Counter()
    for replicate in planned_replicates:
        a=rows.get((source,replicate,left))
        b=rows.get((source,replicate,right))
        causes=[]
        for side,row in (("left",a),("right",b)):
            causes += [side+":"+x for x in (row["exclusion_causes"] if row else ["not_planned_or_missing"])]
        if causes:
            reasons.update(set(causes))
        if a and b and a["score"] and b["score"]:
            delta={name:b["score"][name]-a["score"][name] for name in METRICS}
            record=dict(replicate=replicate,left=left,right=right,
                left_scores={k:a["score"][k] for k in METRICS},
                right_scores={k:b["score"][k] for k in METRICS},difference_right_minus_left=delta,
                strict_eligible=not causes,exclusion_causes=causes,
                prediction_score_same_target=a["score"].get("score_design")==b["score"].get("score_design"),
                left_score_design=a["score"].get("score_design"),right_score_design=b["score"].get("score_design"))
            diagnostic.append(record)
            if not causes:
                strict.append(record)
    return dict(label=label,source=source,difference="right minus left",
        nplanned=len(planned_replicates),nactual=len(strict),nexcluded=len(planned_replicates)-len(strict),
        exclusion_counts=dict(sorted(reasons.items())),exclusion_counts_are_not_mutually_exclusive=True,
        statistics={name:moments([p["difference_right_minus_left"][name] for p in strict]) for name in METRICS},
        strict_pairs=strict,all_scored_pairs_with_diagnostic_flags=diagnostic,
        )


def summarize(run,main_condition):
    """Свести сохранённые расчёты в парные сравнения и полноту.

    Parameters
    ----------
    run : str or path-like
        Каталог run.json и файлов состояния по источникам и реализациям.
    main_condition : str
        Зарегистрированное основное условие для сравнения H¹ и L².

    Returns
    -------
    report : dict
        Полнота процедур, причины исключения, пары и статистики, включая
        незавершённые расчёты.

    Notes
    -----
    Файлы читаются без повторного восстановления или выбора. Строгие
    статистики относятся к подмножеству пар без причин исключения;
    источники и условия, использующие общие потоки, не считаются
    независимыми реализациями.
    """

    run=Path(run).resolve()
    manifest_path=run/"run.json"
    manifest,manifest_digest=snapshot(manifest_path)
    spec=manifest["configuration"]
    conditions={c["id"]:c for c in spec["conditions"]}
    if main_condition not in conditions:
        raise ValueError("Explicit main condition does not exist")
    rows={}
    file_bindings={"run.json":manifest_digest}
    checkpoints={}
    global_coverage=Counter()
    for source in spec["sources"]:
        for replicate in spec["replicates"]:
            conditions_here=[c for c in spec["conditions"] if source in c["sources"] and replicate in c["replicates"]]
            singles=[s for s in spec.get("single_fits",[]) if source in s["sources"] and replicate in s["replicates"]]
            expected=[(c["id"]+"/"+arm,c["id"],arm,"path") for c in conditions_here for arm in c["penalties"]]
            expected += [("single/"+s["id"],s["condition"],s["penalty"],s["kind"]) for s in singles]
            if not expected:
                continue
            path=run/source/f"replicate_{replicate}.json"
            cp,checkpoint_digest=snapshot(path) if path.exists() else (None,None)
            if cp is not None:
                if any(cp["bindings"].get(k)!=v for k,v in manifest["bindings"].items()):
                    raise ValueError(f"Checkpoint/manifest binding disagreement: {path}")
                if set(cp["expected_paths"])!={r[0] for r in expected}:
                    raise ValueError(f"Checkpoint expected-path disagreement: {path}")
                if cp["stage"] in ("sealed","scored") and selection_hash(cp["paths"])!=cp.get("selection_seal"):
                    raise ValueError(f"Selection seal mismatch: {path}")
                file_bindings[str(path.relative_to(run))]=checkpoint_digest
            stage=cp["stage"] if cp else "pending"
            checkpoints[f"{source}/{replicate}"]=dict(stage=stage,selection_seal=cp.get("selection_seal") if cp else None)
            for pid,condition_id,arm,kind in expected:
                record=cp.get("paths",{}).get(pid) if cp else None
                score=cp.get("scores",{}).get(pid) if cp and stage=="scored" else None
                if score is not None and score.get("score_row_count")!=36:
                    raise ValueError("Common-score comparison requires all36 primary rows")
                candidates=list(record.get("candidates",{}).values()) if record else []
                if record and "candidate" in record:
                    candidates.append(record["candidate"])
                accepted=sum(bool(c.get("accepted")) for c in candidates)
                failures=Counter(c.get("status","unknown") for c in candidates if not c.get("accepted"))
                causes=exclusion_causes(record,score,stage)
                coverage=dict(candidate_attempted=len(candidates),candidate_accepted=accepted,
                    candidate_rejected=len(candidates)-accepted,
                    nominal_candidates=25 if kind=="path" else 1,
                    maximum_candidates=25 if kind=="path" else 1,
                    final_grid_candidates=(len(cp["exponents"]) if cp else 25) if kind=="path" else 1,
                    candidate_failure_statuses=dict(failures),
                    complete_path=bool(record and record.get("complete_path")),
                    finalized=bool(record and record.get("finalized")),
                    procedure_accepted=bool(record and record.get("procedure_accepted")),
                    final_certificate_accepted=bool(record and record.get("final_certificate",{}).get("accepted")),
                    alpha_boundary_unresolved=bool(record and record.get("tuning_unresolved")),
                    calibration_failed=bool(record and record.get("calibration_failure")))
                for key,value in coverage.items():
                    if type(value) in (bool,int):
                        global_coverage[key]+=int(value)
                global_coverage["planned_procedures"]+=1
                rows[(source,replicate,pid)]=dict(source=source,replicate=replicate,path_id=pid,
                    condition=condition_id,penalty=arm,kind=kind,stage=stage,coverage=coverage,
                    selected_exponent=record.get("selected_exponent") if record else None,
                    exclusion_causes=causes,score=score)
    main=conditions[main_condition]
    primary=[pair_summary(rows,"main H1 minus L2",source,main["replicates"],
                          main_condition+"/L2",main_condition+"/H1") for source in main["sources"]]
    contrasts,contrast_origin=contrast_plan(spec,main_condition)
    comparisons=[]
    for contrast in contrasts:
        base,variant=conditions[contrast["baseline"]],conditions[contrast["variant"]]
        sources=[s for s in spec["sources"] if s in base["sources"] and s in variant["sources"]]
        replicates=sorted(set(base["replicates"])&set(variant["replicates"]))
        for arm in sorted(set(base["penalties"])&set(variant["penalties"])):
            for source in sources:
                report=pair_summary(rows,contrast["id"]+"/"+arm,source,replicates,
                    base["id"]+"/"+arm,variant["id"]+"/"+arm)
                report["factor"]=contrast["factor"]
                comparisons.append(report)
    single_comparisons=[]
    for single in spec.get("single_fits",[]):
        for source in single["sources"]:
            targets=[("baseline_physical_alpha",single["baseline_condition"]+"/"+single["penalty"])]
            if single["kind"]=="fixed_alpha":
                targets.append(("independently_tuned_target",single["condition"]+"/"+single["penalty"]))
            for label,target in targets:
                single_comparisons.append(pair_summary(rows,single["id"]+"/"+label,source,single["replicates"],
                    target,"single/"+single["id"]))
    return dict(schema="ym2026.research_validation.summary.v1",run_directory=str(run),
        analyzer_sha256=digest(__file__),input_file_sha256=file_bindings,main_condition=main_condition,
        checkpoints=checkpoints,coverage=dict(global_coverage),procedures=list(rows.values()),
        primary_metric="E_q=||qhat-q||L2/(Qref*sqrt(3h)); relative_L2 is secondary",
        primary_paired_results=primary,contrasts=comparisons,contrast_plan_origin=contrast_origin,
        single_fit_comparisons=single_comparisons,
        statistical_limits=["Replicates share random streams across arms and sources; cells are not independent.",
            "Only pairs with both accepted complete procedures enter strict means/SD/MCSE.",
            "Excluded, failed, boundary and pending cases remain explicit; strict estimates are conditional on this selection.",
            "n=8 does not prove adequate power; n=1 has no empirical SD/MCSE.",
            "All factor results are casewise descriptive; no simultaneous coverage, interactions or field-emission validation is claimed.",
            "Partial checkpoints are read only; no scorer, fit, selection or checkpoint mutation occurs."])


def main():
    """Сохранить сводку парных сравнений вне каталога исходных расчётов."""

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True, help="Каталог сохранённых результатов E05")
    parser.add_argument("--main-condition",required=True, help="Идентификатор основного условия, например main")
    parser.add_argument("--output",required=True, help="Новый файл JSON парных различий и полноты расчёта")
    args=parser.parse_args()
    destination=Path(args.output).resolve()
    result=summarize(args.run,args.main_condition)
    if destination.is_relative_to(Path(args.run).resolve()):
        raise ValueError("Write summary outside immutable experiment checkpoints")
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result["coverage"],ensure_ascii=False))


if __name__=="__main__":
    main()
