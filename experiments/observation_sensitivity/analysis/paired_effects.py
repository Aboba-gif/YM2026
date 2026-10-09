"""Парные разности и сводка плана E06 по сохранённым записям.

Разность вычисляется как результат правой процедуры минус результат левой.
В статистику входят пары с двумя принятыми процедурами и доступными
ошибками; число запланированных повторов учитывает также исключения.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import statistics
from uuid import UUID

from adrkit.config.validation import canonical_bytes, digest, strict_json
from experiments.observation_sensitivity.design import ALPHA_EXPONENTS, resolve_plan
from experiments.observation_sensitivity.admission import scientific_configuration
from experiments.observation_sensitivity.input_binding import validate_input_binding
from experiments.observation_sensitivity.lifecycle import _validate as validate_checkpoint


SCHEMA = "ym2026.observation_sensitivity.summary"
PRIMARY_ROWS = [station * 72 + tick for station in range(4) for tick in range(7, 72, 8)]
PRIMARY_DESIGN = dict(removed_primary_tick_indices=[], rows=PRIMARY_ROWS, retained_count=36, mask="none")
IDENTITY_ATOL = 1e-12  # ug/m³; допуск согласованности для алгебраически тождественных H.
IDENTITY_RTOL = 1e-12
SOURCE_SQUARED_ROUNDOFF_FACTOR = 1e-10
SOURCE_METRICS = ("E_q", "relative_L2", "signed_mass_error", "absolute_mass_error")
OBSERVATION_METRICS = (
    "assumed_H_test_rmse", "prediction_under_true_H_test_rmse",
    "assumed_H_noiseless_rmse", "prediction_under_true_H_noiseless_rmse",
    "observation_model_discrepancy_rmse",
)
METRICS = SOURCE_METRICS + OBSERVATION_METRICS



class AggregationError(ValueError):
    """Неполные или несогласованные записи заданного плана E06."""


def _require(condition, message):
    if not condition:
        raise AggregationError(message)


def _same(left, right):
    return canonical_bytes(left) == canonical_bytes(right)


def _number(value, name, *, nonnegative=False):
    _require(type(value) in (int, float) and math.isfinite(value)
             and (not nonnegative or value >= 0), f"Invalid finite numeric {name}")
    return value


def _is_sha256(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _sealed(record, label):
    _require(type(record) is dict and _is_sha256(record.get("content_sha256")), f"Invalid {label} digest")
    _require(record["content_sha256"] == digest({k: v for k, v in record.items()
                                               if k != "content_sha256"}), f"{label} content digest mismatch")


def _file_sha(record):
    """Вычислить SHA-256 полного канонического JSON с завершающим LF."""
    return hashlib.sha256(canonical_bytes(record) + b"\n").hexdigest()


def _time(value, name):
    _require(type(value) is str, f"Invalid {name}")
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(moment.utcoffset() == timezone.utc.utcoffset(None), f"{name} needs UTC timezone")
    return moment


def _freeze(freeze, expected):
    _require(_is_sha256(expected), "An independently trusted expected_freeze_sha256 is required")
    _sealed(freeze, "freeze")
    _require(_file_sha(freeze) == expected, "Freeze differs from the trusted raw-file digest")
    _require(set(freeze) == {"schema", "version", "admission", "admission_sha256", "frozen_at", "content_sha256"}
             and freeze["schema"] == "ym2026.observation_sensitivity.freeze"
             and type(freeze["version"]) is int and freeze["version"] in (3, 4), "Unsupported freeze envelope")
    _time(freeze["frozen_at"], "frozen_at")
    admission = freeze["admission"]
    _require(type(admission) is dict and digest(admission) == freeze["admission_sha256"], "Admission digest mismatch")
    _require(admission["schema"] == "ym2026.observation_sensitivity.admission"
             and type(admission["version"]) is int and admission["version"] == freeze["version"]
             and admission["status"] == "prepared_not_frozen", "Unsupported admission")
    validate_input_binding(admission)
    plan = resolve_plan(admission["configuration"], admission["baseline"]["configuration"])
    _require(_same(admission["design"], plan["design"]) and admission["design_sha256"] == digest(plan["design"])
             and _same(admission["counts"], plan["counts"]), "Incomplete or changed E06 design")
    for source in admission["sources"].values():
        _require(set(source) == {"record", "sha256"} and digest(source["record"]) == source["sha256"],
                 "Source record digest mismatch")
    return admission


def _direct(direct, freeze, admission):
    _sealed(direct, "direct journal")
    keys = {"schema", "version", "admission_sha256", "freeze_sha256", "run_id", "started_at",
             "status", "finished_at", "result", "content_sha256"}
    _require(set(direct) == keys and direct["schema"] == "ym2026.observation_sensitivity.direct_journal"
             and type(direct["version"]) is int and direct["version"] == freeze["version"]
             and direct["status"] == "completed", "Direct journal must be committed completed, not partial/started")
    _require(direct["admission_sha256"] == freeze["admission_sha256"]
             and direct["freeze_sha256"] == freeze["content_sha256"], "Direct freeze/admission mismatch")
    _require(type(direct["run_id"]) is str and str(UUID(direct["run_id"])) == direct["run_id"],
             "Invalid direct invocation identity")
    start = _time(direct["started_at"], "direct start")
    _require(_time(direct["finished_at"], "direct finish") >= start, "Direct finishes before its start")
    result = direct["result"]
    plan = resolve_plan(admission["configuration"], admission["baseline"]["configuration"])["direct"]
    expected_fields = len(plan["source_ids"]) * len(plan["domains"])
    _require(result["schema"] == "ym2026.observation_sensitivity.direct" and type(result["version"]) is int
             and result["version"] == 2 and result["status"] in ("complete", "incomplete")
             and type(result["expected_fields"]) is int and result["expected_fields"] == expected_fields,
             "Invalid terminal direct report")
    _require(result["original_spec_sha256"] == admission["science_spec_sha256"]
             and _same(result["sources"], {s: admission["sources"][s] for s in plan["source_ids"]}), "Direct scientific binding mismatch")
    fields = result["fields"]
    _require(set(fields) == set(plan["source_ids"]), "Missing direct source fields")
    complete = 0
    for cases in fields.values():
        _require(set(cases) == set(plan["domains"]), "Missing direct domain fields")
        for row in cases.values():
            _require(row.get("status") in ("complete", "residual_rejected", "unavailable"), "Nonterminal direct field")
            complete += row["status"] == "complete"
    _require(type(result["complete_fields"]) is int and result["complete_fields"] == complete
             and (result["status"] == "complete") == (complete == expected_fields), "Direct completion count mismatch")


def _reuse(record, path, admission):
    reuse = record.get("reuse")
    if path.reuse is None:
        _require(reuse is None and record["condition"] == path.condition, "New path falsely reused or relabelled")
        return
    _require(type(reuse) is dict, "Planned baseline path lacks reuse provenance")
    expected = dict(admission["baseline"]["bindings"], source=path.source, replicate=path.replicate,
                    source_record_sha256=admission["sources"][path.source]["sha256"])
    _require(reuse["study"] == "research_validation_v2" and reuse["new_path_id"] == path.id
             and reuse["original_path_id"] == f"{path.reuse.condition}/{path.penalty}"
             and record["condition"] == path.reuse.condition and _same(reuse["bindings"], expected),
             "Exact reused path locator/bindings mismatch")
    original = {k: v for k, v in record.items() if k not in ("reuse", "driver_binding", "e06_provenance")}
    _require(reuse["original_path_sha256"] == digest(original)
             and _is_sha256(reuse["source_file_sha256"]) and _is_sha256(reuse["selection_seal"]), "Reused original estimate digest mismatch")


def _panel(record, path, name):
    codes = {"fit": 211, "selection": 307, "test": 401}
    _require(type(record) is dict and record.get("generator") == "PCG64" and record.get("panel") == name
             and _same(record.get("seed"), [path.stream.seed, path.stream.version, path.replicate, codes[name], 0]),
             "Panel seed/stream/purpose differs from the canonical path")
    for field in ("standard_normal_sha256", "covariance_sha256", "residual_sha256"):
        _require(_is_sha256(record.get(field)), "Missing residual panel identity")


def _provenance(record, path):
    provenance = record["e06_provenance"] if path.reuse else record.get("provenance")
    if "calibration_failure" in record:
        if path.reuse:
            _require(_same(provenance, {"calibration_failure": record["calibration_failure"]}),
                     "Reused calibration failure provenance mismatch")
        return
    _require(type(provenance) is dict and type(provenance.get("calibration")) is dict
             and provenance["calibration"], "Missing estimator calibration provenance")
    _require(_same(provenance.get("design"), PRIMARY_DESIGN), "Estimator observation design differs from primary36")
    keys = ("fit_y_sha256", "selection_y_sha256", "selected_covariance_sha256", "jacobian_sha256", "offset_sha256")
    for field in keys:
        _require(_is_sha256(provenance.get(field)), "Missing estimator input hash")
    _require(set(provenance["panel_records"]) == {"fit", "selection"}, "Estimator panel set differs")
    for name in ("fit", "selection"):
        _panel(provenance["panel_records"][name], path, name)
    if path.reuse:
        for field in (*keys, "panel_records", "calibration", "design"):
            _require(_same(provenance[field], record["provenance"][field]), "Reused estimator provenance mismatch")


def _path(record, path, admission):
    spec = scientific_configuration(admission["baseline"]["configuration"])
    _require(_same(record["driver_binding"], dict(spec_sha256=digest(spec),
             path_sha256=digest(json.loads(json.dumps(asdict(path)))))), "Driver path/spec binding mismatch")
    _require(record["penalty"] == path.penalty, "Path penalty mismatch")
    _reuse(record, path, admission)
    _provenance(record, path)
    candidates = list(record["candidates"].values())
    if "calibration_failure" in record:
        _require(record.get("selected_exponent") is None, "Calibration failure cannot select a candidate")
        return ["calibration_failure"], None
    accepted = [c for c in candidates if c["accepted"]]
    for candidate in accepted:
        _require(candidate["status"] == "accepted", "Accepted candidate has rejected status")
        _number(candidate["selection_mse"], "selection MSE", nonnegative=True)
        _require(0 <= _number(candidate["forward_residual"], "candidate residual")
                 <= spec["solver"]["forward_tolerance"], "Accepted candidate exceeds forward tolerance")
    selected = None
    if accepted:
        best = min(c["selection_mse"] for c in accepted)
        threshold = spec["alpha"]["tie_atol"] + spec["alpha"]["tie_rtol"] * abs(best)
        selected = max((c for c in accepted if c["selection_mse"] - best <= threshold), key=lambda c: c["alpha"])
    boundary = selected is not None and selected["exponent"] in (ALPHA_EXPONENTS[0], ALPHA_EXPONENTS[-1])
    full = len(accepted) == 25
    _require(type(record["candidate_count"]) is int and record["candidate_count"] == 25
             and type(record["accepted_count"]) is int and record["accepted_count"] == len(accepted)
             and record["complete_path"] is full and record["tuning_unresolved"] is boundary
             and _same(record["selected_exponent"], None if selected is None else selected["exponent"]),
             "Path count, selection, or alpha boundary differs from committed candidates")
    certificate = record.get("final_certificate")
    if selected is not None:
        _require(type(certificate) is dict and type(certificate.get("accepted")) is bool, "Missing final certificate")
        if certificate["accepted"]:
            _require(certificate.get("finite") is True and certificate.get("optimizer_success") is True,
                     "Accepted final certificate contradicts its finite/optimizer status")
            _require(0 <= _number(certificate["forward_residual"], "certificate residual") <= spec["solver"]["forward_tolerance"],
                     "Accepted final certificate exceeds forward tolerance")
            for name in ("primal", "dual", "stationarity", "complementarity", "free_coordinate"):
                limit = 1e-8 if name in ("primal", "dual") else 1e-6
                _require(0 <= _number(certificate["norms"][name], f"certificate {name}") <= limit,
                         "Accepted final certificate exceeds its KKT tolerance")
    else:
        _require(certificate is None, "Unselected path has a final certificate")
    procedure = bool(full and certificate is not None and certificate["accepted"] and not boundary)
    _require(record.get("procedure_accepted") is procedure, "Procedure acceptance contradicts its path/certificate")
    causes = []
    if not full:
        causes.append("candidate_path_rejected")
    if any(c.get("status") == "numerical_failure" for c in candidates):
        causes.append("solver_failure")
    if selected is None:
        causes.append("no_accepted_candidate")
    if boundary:
        causes.extend(["alpha_boundary_unresolved", "alpha_lower_boundary" if selected["exponent"] == ALPHA_EXPONENTS[0]
                       else "alpha_upper_boundary"])
    if certificate is not None and not certificate["accepted"]:
        causes.append("final_certificate_rejected")
    return causes, selected


def _score(score, record, selected, path, admission):
    accepted = record["procedure_accepted"]
    _require(score.get("procedure_accepted") is accepted and score.get("diagnostic_only") is (not accepted)
             and type(score.get("score_row_count")) is int and score["score_row_count"] == 36,
             "Score eligibility differs from the sealed procedure")
    if score.get("status") == "unavailable":
        reason = score.get("reason")
        expected = ("calibration_failure" if "calibration_failure" in record else "no_accepted_candidate") if selected is None else "scoring_failure"
        _require(reason == expected and not any(k in score for k in ("source", *OBSERVATION_METRICS)),
                 "Unavailable score has an inconsistent reason or fabricated values")
        return {}, [reason]
    _require(score.get("status") == "available" and selected is not None, "Unknown score status or absent selected candidate")
    values = {}
    for name in SOURCE_METRICS:
        values[name] = _number(score["source"][name], name, nonnegative=name != "signed_mass_error")
    raw = score["source"]
    for name in ("estimated_mass", "true_mass", "zero_prior_E_q"):
        _number(raw[name], name, nonnegative=True)
    _require(math.isclose(raw["true_mass"], admission["sources"][path.source]["record"]["unknown_mass"], rel_tol=1e-12, abs_tol=1e-12),
             "Source score has a different shared analytic truth mass")
    qref = admission["baseline"]["configuration"]["Qref"]
    _require(math.isclose(values["signed_mass_error"], (raw["estimated_mass"]-raw["true_mass"])/(qref*3.), rel_tol=1e-10, abs_tol=1e-12)
             and math.isclose(values["E_q"], values["relative_L2"]*raw["zero_prior_E_q"], rel_tol=1e-10, abs_tol=1e-12),
             "Source normalization or shared E_q target mismatch")
    # Допуск учитывает потерю точности при вычитании квадратов норм в source_scores.
    # Норму P1-профиля считаем по его линейным участкам.
    # Это проверка согласованности, не доказанная граница ошибки округления.
    point = selected["coefficients"]
    _require(type(point) is list and len(point) == 73, "Selected source must have 73 P1 coefficients")
    for coefficient in point:
        _number(coefficient, "selected source coefficient", nonnegative=True)
    norm2 = math.fsum(a*a+a*b+b*b for a, b in zip(point[:-1], point[1:])) / (3.*72.)
    _number(norm2, "normalized P1 squared norm", nonnegative=True)
    tolerance2 = SOURCE_SQUARED_ROUNDOFF_FACTOR * max(1./(qref*qref*3.), raw["zero_prior_E_q"]**2, norm2)
    _require(values["absolute_mass_error"]**2 <= values["E_q"]**2 + tolerance2,
             "Source error violates the Cauchy mass bound beyond cancellation allowance")
    _require(raw["zero_prior_E_q"] > 0 and raw["true_mass"]/(qref*3.) <= raw["zero_prior_E_q"] + IDENTITY_ATOL,
             "True source norm violates its positive-mass Cauchy bound")
    _require(math.isclose(values["absolute_mass_error"], abs(values["signed_mass_error"]), rel_tol=1e-12, abs_tol=1e-14),
             "Signed/absolute source mass errors disagree")
    for name in OBSERVATION_METRICS:
        values[name] = _number(score[name], name, nonnegative=True)
    _require(0 <= _number(score["forward_residual"], "score residual")
             <= admission["baseline"]["configuration"]["solver"]["forward_tolerance"], "Score exceeds forward tolerance")
    _require(score["score_design"]["primary_rows"] == PRIMARY_ROWS
             and _same(score["test_provenance"].get("design"), PRIMARY_DESIGN)
             and _is_sha256(score["test_provenance"]["test_y_sha256"]), "Score target/primary rows or test provenance missing")
    _panel(score["test_provenance"]["panel_record"], path, "test")
    if path.true_h == path.inverse_h:
        _require(values["observation_model_discrepancy_rmse"] <= IDENTITY_ATOL
                 and all(math.isclose(values[f"assumed_H_{target}_rmse"], values[f"prediction_under_true_H_{target}_rmse"],
                                      rel_tol=IDENTITY_RTOL, abs_tol=IDENTITY_ATOL) for target in ("test", "noiseless")),
                 "Matched-H projections violate their numerical identity")
    return values, []


def _paired_inputs(a, b, same_target):
    """Проверить привязку к общему шуму и, при same_target, совпадение хешей y.

    При разных истинных H совпадение шумовых панелей допускает разные средние
    сигнала; оно не означает равенства полных наблюдений.
    """
    estimates = [r["estimate"] for r in (a, b)]
    if all("calibration_failure" not in e for e in estimates):
        left, right = [e.get("e06_provenance", e.get("provenance")) for e in estimates]
        _require(_same(left["panel_records"], right["panel_records"]), "Paired residual panels differ")
        if same_target:
            _require(all(left[name] == right[name] for name in ("fit_y_sha256", "selection_y_sha256")),
                     "Same-target paired fit/selection inputs differ")
    if a["score_available"] and b["score_available"]:
        left, right = (r["score"]["test_provenance"] for r in (a, b))
        _require(_same(left["panel_record"], right["panel_record"]), "Paired test residual panels differ")
        if same_target:
            _require(left["test_y_sha256"] == right["test_y_sha256"], "Same-target paired test observations differ")


def _moments(values):
    r"""Вычислить среднее, выборочный разброс и стандартную ошибку среднего.

    Parameters
    ----------
    values : sequence of float
        Значения по включённым повторам в единицах соответствующей метрики.

    Returns
    -------
    dict
        `n`, `mean`, `sd` и `mcse`. При пустой выборке `mean` равен None; при
        менее чем двух значениях `sd` и `mcse` равны None.

    Notes
    -----
    При независимых одинаково распределённых повторах и `n` не меньше двух
    используются выборочное стандартное отклонение s со знаменателем `n` − 1
    и оценка стандартной ошибки среднего

    .. math::

        \widehat{\mathrm{MCSE}} = \frac{s}{\sqrt{n}}.

    MCSE характеризует среднее по включённым повторам, а не доверительный
    интервал или вероятность принятия процедуры.
    """
    n = len(values)
    sd = statistics.stdev(values) if n >= 2 else None
    return dict(n=n, mean=statistics.mean(values) if n else None, sd=sd,
                mcse=sd / math.sqrt(n) if sd is not None else None)


def _counts(rows):
    counts = Counter(cause for row in rows for cause in set(row["exclusion_causes"]))
    return dict(sorted(counts.items()))


def summarize_records(freeze, direct, groups, *, expected_freeze_sha256):
    """Свести завершённые записи E06 к парным разностям и статистике.
    
    Parameters
    ----------
    freeze, direct : dict
        Закреплённый план и завершённый прямой журнал E06.
    groups : dict
        Все завершённые группы с ключами source/rN из допуска.
    expected_freeze_sha256 : str
        Независимо записанный SHA-256 канонического ``freeze.json`` с конечным LF.
    
    Returns
    -------
    dict
        Все запланированные исходы, причины исключений и сводка разностей
        ``right - left``. Среднее, SD и MCSE относятся только к парам с двумя
        принятыми процедурами и доступными ошибками.
        Разности ошибок наблюдений доступны только для общей цели наблюдений;
        при разных целях сравниваются ошибки источника.
    """
    try:
        # Каноническая JSON-копия отделяет результат от входных объектов.
        # Допустимые числовые типы проверяются при разборе отдельных полей.
        freeze, direct, groups = strict_json(canonical_bytes([freeze, direct, groups]))
        return _summarize(freeze, direct, groups, expected_freeze_sha256)
    except AggregationError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, ArithmeticError) as error:
        raise AggregationError(f"Malformed or inconsistent E06 records: {error}") from error


def _summarize(freeze, direct, groups, expected):
    admission = _freeze(freeze, expected)
    plan = resolve_plan(admission["configuration"], admission["baseline"]["configuration"])
    _direct(direct, freeze, admission)
    group_ids = tuple(plan["expected_paths"])
    _require(type(groups) is dict and set(groups) == set(group_ids), "Exactly all admitted groups are required")
    paths = {p.id: p for p in plan["paths"]}
    outcomes = {}
    for group_id in group_ids:
        source, replicate_text = group_id.split("/r")
        replicate = int(replicate_text)
        planned = plan["expected_paths"][group_id]
        bindings = dict(source=source, replicate=replicate, admission_sha256=freeze["admission_sha256"],
            source_record_sha256=admission["sources"][source]["sha256"], direct_record_sha256=digest(direct))
        cp = groups[group_id]
        validate_checkpoint(cp, bindings, planned)
        _require(cp["stage"] == "scored", f"Group {group_id} is not scored; no partial effect summary")
        for pid in planned:
            path, estimate, score = paths[pid], cp["paths"][pid], cp["scores"][pid]
            causes, selected = _path(estimate, path, admission)
            values, score_causes = _score(score, estimate, selected, path, admission)
            causes = list(dict.fromkeys(causes + score_causes))
            outcomes[pid] = dict(path_id=pid, group=group_id, source=source, replicate=replicate,
                condition=path.condition, penalty=path.penalty, origin="new" if path.reuse is None else "reused",
                procedure_accepted=estimate["procedure_accepted"], score_available=score["status"] == "available",
                eligible=estimate["procedure_accepted"] and score["status"] == "available",
                exclusion_causes=causes, metrics=values, selected_alpha=None if selected is None else selected["alpha"],
                true_h=asdict(path.true_h), inverse_h=asdict(path.inverse_h),
                noise=json.loads(json.dumps(asdict(path.noise))), stream=asdict(path.stream), estimate=estimate, score=score)
    for source in dict.fromkeys(p.source for p in plan["paths"]):
        norms = [r["score"]["source"]["zero_prior_E_q"] for r in outcomes.values()
                 if r["source"] == source and r["score_available"]]
        _require(not norms or all(math.isclose(z, norms[0], rel_tol=IDENTITY_RTOL, abs_tol=IDENTITY_ATOL) for z in norms),
                 "One source has inconsistent analytic truth norms across paths")
    pairs, buckets = [], defaultdict(list)
    for contrast in plan["contrasts"]:
        a, b = outcomes[contrast.left_path], outcomes[contrast.right_path]
        left, right = paths[contrast.left_path], paths[contrast.right_path]
        same_target = left.data_design_key == right.data_design_key
        _paired_inputs(a, b, same_target)
        available = a["score_available"] and b["score_available"]
        accepted = a["procedure_accepted"] and b["procedure_accepted"]
        eligible = accepted and available
        reasons = [side + ":" + cause for side, row in (("left", a), ("right", b)) for cause in row["exclusion_causes"]]
        difference = ({name: b["metrics"][name] - a["metrics"][name] for name in METRICS
                       if same_target or name in SOURCE_METRICS} if available else None)
        if difference is not None:
            for name, value in difference.items():
                _number(value, f"paired {name}")
        penalty = "L2_to_H1" if contrast.factor == "penalty" else left.penalty
        row = dict(contrast_id=contrast.id, key=contrast.key, factor=contrast.factor,
            source=left.source, replicate=left.replicate, penalty=penalty,
            left_path=left.id, right_path=right.id, available=available, both_accepted=accepted, eligible=eligible,
            exclusion_causes=reasons, left_metrics=a["metrics"], right_metrics=b["metrics"],
            difference_right_minus_left=difference, observation_same_target=same_target, same_data=same_target,
            left_true_h=a["true_h"], right_true_h=b["true_h"])
        pairs.append(row)
        buckets[(contrast.key, left.source, penalty)].append(row)
    summaries = []
    for (key, source, penalty), rows in buckets.items():
        same_target = all(row["observation_same_target"] for row in rows)
        metrics = METRICS if same_target else SOURCE_METRICS
        included = [row for row in rows if row["eligible"]]
        summaries.append(dict(key=key, factor=rows[0]["factor"], source=source, penalty=penalty,
            planned_replicates=[r["replicate"] for r in rows], n_expected=len(rows),
            n_available=sum(r["available"] for r in rows), n_both_accepted=sum(r["both_accepted"] for r in rows),
            n_eligible=len(included), exclusion_counts=_counts(rows), observation_same_target=same_target,
            retained_replicates=[r["replicate"] for r in included],
            statistics={name: _moments([r["difference_right_minus_left"][name] for r in included]) for name in metrics},
            contrast_ids=[r["contrast_id"] for r in rows]))
    cells = defaultdict(list)
    for row in outcomes.values():
        cells[(row["condition"], row["source"], row["penalty"])].append(row)
    coverage = [dict(condition=c, source=s, penalty=p, n_expected=len(rows),
        n_available=sum(r["score_available"] for r in rows), n_accepted=sum(r["procedure_accepted"] for r in rows),
        n_eligible=sum(r["eligible"] for r in rows), exclusion_counts=_counts(rows))
        for (c, s, p), rows in cells.items()]
    attempt_counts = {}
    for origin in ("new", "reused"):
        rows = [r for r in outcomes.values() if r["origin"] == origin]
        attempt_counts[origin] = dict(planned_paths=len(rows), nominal_candidates=25*len(rows),
            recorded_candidates=sum(len(r["estimate"]["candidates"]) for r in rows),
            accepted_candidates=sum(c["accepted"] for r in rows for c in r["estimate"]["candidates"].values()),
            calibration_failures=sum("calibration_failure" in r["estimate"] for r in rows))
    result = dict(schema=SCHEMA, version=1, status="complete_records", counts=plan["counts"],
        inputs=dict(freeze_file_sha256=expected, freeze_content_sha256=freeze["content_sha256"],
            admission_sha256=freeze["admission_sha256"], direct_file_sha256=_file_sha(direct),
            direct_content_sha256=direct["content_sha256"], direct_record_sha256=digest(direct),
            group_content_sha256={k: groups[k]["content_sha256"] for k in group_ids}),
        difference="right minus left", sampling_unit="one source-specific replicate; no pooling sources, penalties or alpha candidates",
        observation_target_note="Across distinct true H, own-H RMSE has distinct targets; no observation-metric difference or aggregate is computed for those pairs.",
        matched_H_identity_tolerance=dict(atol_ug_m3=IDENTITY_ATOL, rtol=IDENTITY_RTOL,
            purpose="consistency of identical recorded projections, not a scientific acceptance threshold"),
        source_mass_consistency=dict(squared_roundoff_factor=SOURCE_SQUARED_ROUNDOFF_FACTOR,
            formula="mass_error^2 <= E_q^2 + 1e-10*max(1/(Qref^2*T), zero_prior_E_q^2, ||qhat||_2^2/(Qref^2*T))",
            purpose="consistency allowance matching source_scores cancellation scale, not a certified numerical error bound"),
        exclusion_counts_are_not_mutually_exclusive=True, attempt_counts=attempt_counts,
        direct=direct, raw_outcomes=list(outcomes.values()), outcome_coverage=coverage,
        paired_outcomes=pairs, contrast_summaries=summaries)
    if admission["version"] == 4:
        result.update(design=plan["design"], figures=plan["figures"])
    return result
