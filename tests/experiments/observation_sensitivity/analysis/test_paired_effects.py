"""Проверки парных разностей, знаменателей и исключений на синтетическом плане E06."""

from tests.experiments.observation_sensitivity.fixtures.paired_records import seal, pin, reseal, source_score, make_records, path_id, alter, summary_cell
from copy import deepcopy
import math

import pytest

from adrkit.config.validation import canonical_bytes, digest
from experiments.observation_sensitivity.design import build_design
from experiments.observation_sensitivity.analysis import paired_effects as module


def test_scientific_input_binding_rejects_resealed_specification_change():
    freeze, direct, groups = make_records()
    original = module.summarize_records(freeze, direct, groups, expected_freeze_sha256=pin(freeze))
    assert original["attempt_counts"]["reused"]["recorded_candidates"] == 600
    freeze["admission"]["science_spec_sha256"] = "b" * 64
    freeze["admission_sha256"] = digest(freeze["admission"])
    seal(freeze)
    with pytest.raises(module.AggregationError, match="specification digest"):
        module.summarize_records(freeze, direct, groups, expected_freeze_sha256=pin(freeze))


@pytest.fixture(scope="module")
def template():
    return make_records()


@pytest.fixture
def records(template):
    return deepcopy(template)


def summarize(records):
    return module.summarize_records(*records, expected_freeze_sha256=pin(records[0]))


def test_complete_plan_counts_sign_and_replication_unit(records):
    report = summarize(records)
    assert len(report["raw_outcomes"]) == 176
    assert len(report["paired_outcomes"]) == 224
    assert len(report["contrast_summaries"]) == 60
    assert len(report["outcome_coverage"]) == 48
    assert sum(row["n_expected"] == 4 for row in report["contrast_summaries"]) == 52
    assert sum(row["n_expected"] == 2 for row in report["contrast_summaries"]) == 8
    cell = summary_cell(report)
    assert (cell["n_expected"], cell["n_available"], cell["n_both_accepted"], cell["n_eligible"]) == (4, 4, 4, 4)
    assert cell["statistics"]["E_q"]["mean"] == -.0625
    diffs = [.125, -.25, .375, -.5]
    sd = math.sqrt(sum((v+.0625)**2 for v in diffs)/3)
    assert cell["statistics"]["E_q"]["sd"] == pytest.approx(sd)
    assert cell["statistics"]["E_q"]["mcse"] == pytest.approx(sd/2)
    assert report["attempt_counts"]["new"]["recorded_candidates"] == 3800
    assert report["attempt_counts"]["reused"]["recorded_candidates"] == 600
    assert report["inputs"]["freeze_file_sha256"] == pin(records[0])
    assert report["inputs"]["freeze_file_sha256"] != report["inputs"]["freeze_content_sha256"]


def test_owned_output_and_input_order_do_not_change_results(records):
    before = canonical_bytes(list(records))
    expected = summarize(records)
    reordered = module.summarize_records(records[0], records[1], dict(reversed(list(records[2].items()))), expected_freeze_sha256=pin(records[0]))
    assert reordered == expected
    expected["raw_outcomes"][0]["estimate"]["candidates"]["-8.0"]["alpha"] = 0.
    assert canonical_bytes(list(records)) == before


@pytest.mark.parametrize("kind,reason", [("calibration", "calibration_failure"), ("candidate", "candidate_path_rejected"),
    ("certificate", "final_certificate_rejected"), ("lower", "alpha_lower_boundary"), ("upper", "alpha_upper_boundary"),
    ("all_candidates", "no_accepted_candidate")])
def test_terminal_rejections_remain_in_denominators(records, kind, reason):
    alter(records, kind=kind)
    cell = summary_cell(summarize(records))
    assert cell["n_expected"] == 4 and cell["n_both_accepted"] == cell["n_eligible"] == 3
    assert cell["exclusion_counts"]["right:"+reason] == 1
    assert cell["retained_replicates"] == [2, 3, 4]
    if kind in ("lower", "upper"):
        assert cell["exclusion_counts"]["right:alpha_boundary_unresolved"] == 1


def test_scoring_failure_is_separate_from_estimation_acceptance(records):
    alter(records, kind="scoring")
    cell = summary_cell(summarize(records))
    assert (cell["n_expected"], cell["n_available"], cell["n_both_accepted"], cell["n_eligible"]) == (4, 3, 4, 3)
    assert cell["exclusion_counts"]["right:scoring_failure"] == 1


@pytest.mark.parametrize("n", [0, 1, 2])
def test_small_accepted_samples_have_no_fabricated_statistics(records, n):
    for r in range(n+1, 5):
        alter(records, r, kind="scoring")
    cell = summary_cell(summarize(records))
    assert cell["n_eligible"] == n
    stats = cell["statistics"]["E_q"]
    assert stats["n"] == n
    if n == 0:
        assert stats == dict(n=0, mean=None, sd=None, mcse=None)
    elif n == 1:
        assert stats == dict(n=1, mean=.125, sd=None, mcse=None)
    else:
        assert stats["mcse"] == pytest.approx(.1875)


def test_zero_spread_is_zero_mcse_and_signed_mass_is_not_clipped(records):
    for r in range(1, 5):
        cp = records[2][f"PG10/r{r}"]
        cp["scores"][path_id(r)]["source"] = source_score(1., -.02)
        reseal(cp)
    stats = summary_cell(summarize(records))["statistics"]
    assert stats["E_q"] == dict(n=4, mean=0., sd=0., mcse=0.)
    assert stats["signed_mass_error"]["mean"] == pytest.approx(-.03)


def test_distinct_true_h_keeps_own_targets_and_excludes_rmse_differences(records):
    report = summarize(records)
    cell = summary_cell(report, "matched_spatial/G05", penalty="L2")
    assert not cell["observation_same_target"]
    assert set(cell["statistics"]) == set(module.SOURCE_METRICS)
    pair = next(r for r in report["paired_outcomes"] if r["contrast_id"] == cell["contrast_ids"][0])
    assert set(pair["difference_right_minus_left"]) == set(module.SOURCE_METRICS)
    assert pair["left_true_h"] != pair["right_true_h"]
    assert "assumed_H_test_rmse" in pair["left_metrics"]


def test_diagnostic_extreme_score_never_changes_primary_mean(records):
    alter(records, kind="candidate")
    cp = records[2]["PG10/r1"]
    cp["scores"][path_id()]["source"] = source_score(1e9)
    reseal(cp)
    cell = summary_cell(summarize(records))
    assert cell["n_available"] == 4
    assert cell["statistics"]["E_q"]["mean"] == pytest.approx((-.25+.375-.5)/3)


def test_reused_failure_preserved_without_fabricated_attempts(records):
    alter(records, kind="calibration", condition="spatial_G1_matched")
    report = summarize(records)
    assert report["attempt_counts"]["reused"]["recorded_candidates"] == 575
    assert report["attempt_counts"]["reused"]["nominal_candidates"] == 600
    assert report["attempt_counts"]["reused"]["calibration_failures"] == 1


@pytest.mark.parametrize("kind", ["missing_group", "extra_group", "missing_path", "missing_score", "extra_path", "reordered_paths",
    "estimating", "sealed", "unknown_stage", "changed_binding", "changed_exponents", "selection_seal", "content_digest"])
def test_incomplete_unknown_or_tampered_group_rejected(records, kind):
    cp = records[2]["PG10/r1"]
    if kind == "missing_group": records[2].pop("EC04/r4")
    elif kind == "extra_group": records[2]["PG10/r5"] = deepcopy(cp)
    elif kind == "missing_path": cp["paths"].pop(path_id())
    elif kind == "missing_score": cp["scores"].pop(path_id())
    elif kind == "extra_path": cp["paths"]["unplanned"] = deepcopy(cp["paths"][path_id()])
    elif kind == "reordered_paths": cp["expected_paths"].reverse()
    elif kind in ("estimating", "sealed", "unknown_stage"): cp["stage"] = kind
    elif kind == "changed_binding": cp["bindings"]["source"] = "EC04"
    elif kind == "changed_exponents": cp["exponents"][0] = -9.
    elif kind == "selection_seal": cp["selection_seal"] = "0"*64
    elif kind == "content_digest": cp["content_sha256"] = "0"*64
    if kind not in ("selection_seal", "content_digest"):
        reseal(cp)
    elif kind == "selection_seal": seal(cp)
    with pytest.raises(module.AggregationError): summarize(records)


@pytest.mark.parametrize("kind", ["flag", "selected", "count", "alpha", "boundary", "certificate", "driver", "penalty", "reuse"])
def test_self_consistently_resealed_scientific_contradiction_rejected(records, kind):
    cp = records[2]["PG10/r1"]
    row = cp["paths"][path_id()]
    if kind == "flag": row["procedure_accepted"] = False
    elif kind == "selected": row["selected_exponent"] = -1.
    elif kind == "count": row["accepted_count"] = 24
    elif kind == "alpha": row["candidates"]["-8.0"]["alpha"] *= 2
    elif kind == "boundary": row["tuning_unresolved"] = True
    elif kind == "certificate": row["final_certificate"]["norms"]["stationarity"] = .1
    elif kind == "driver": row["driver_binding"]["path_sha256"] = "0"*64
    elif kind == "penalty": row["penalty"] = "L2"
    elif kind == "reuse": row["reuse"] = {"unplanned": True}
    reseal(cp)
    with pytest.raises(module.AggregationError): summarize(records)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1., None])
def test_invalid_scores_never_turn_into_zero_or_missing_effect(records, value):
    cp = records[2]["PG10/r1"]
    cp["scores"][path_id()]["source"]["E_q"] = value
    if type(value) is not float or math.isfinite(value):
        reseal(cp)
    with pytest.raises(module.AggregationError): summarize(records)


@pytest.mark.parametrize("kind", ["flag", "rows", "unknown", "mass", "target", "seed", "fit_pair", "test_pair", "residual_pair"])
def test_score_or_pair_semantics_fail_closed(records, kind):
    cp = records[2]["PG10/r1"]
    score = cp["scores"][path_id()]
    record = cp["paths"][path_id()]
    if kind == "flag": score["diagnostic_only"] = True
    elif kind == "rows": score["score_row_count"] = 288
    elif kind == "unknown": score["status"] = "pending"
    elif kind == "mass": score["source"]["absolute_mass_error"] = 9.
    elif kind == "target": score["source"]["true_mass"] = 99.
    elif kind == "seed": score["test_provenance"]["panel_record"]["seed"][1] = 3
    elif kind == "fit_pair": record["provenance"]["fit_y_sha256"] = "0"*64
    elif kind == "test_pair": score["test_provenance"]["test_y_sha256"] = "0"*64
    elif kind == "residual_pair": record["provenance"]["panel_records"]["selection"]["residual_sha256"] = "0"*64
    reseal(cp)
    with pytest.raises(module.AggregationError): summarize(records)


@pytest.mark.parametrize("status", ["started", "partial", "invented"])
def test_unfinished_direct_rejects_effect_summary(records, status):
    records[1]["status"] = status
    seal(records[1])
    with pytest.raises(module.AggregationError): summarize(records)


def test_completed_direct_numerical_failure_remains_a_visible_outcome(records):
    direct = records[1]
    direct["result"]["fields"]["PG10"]["D1"]["status"] = "unavailable"
    direct["result"].update(status="incomplete", complete_fields=11)
    seal(direct)
    for cp in records[2].values():
        cp["bindings"]["direct_record_sha256"] = digest(direct)
        reseal(cp)
    report = summarize(records)
    assert report["direct"]["result"]["status"] == "incomplete"
    assert len(report["paired_outcomes"]) == 224


def test_external_raw_freeze_pin_and_canonical_plan_are_required(records):
    with pytest.raises(module.AggregationError, match="raw-file"):
        module.summarize_records(*records, expected_freeze_sha256=records[0]["content_sha256"])
    freeze = records[0]
    freeze["admission"]["design"]["paths"][0]["tau_hours"] = .5
    freeze["admission"]["design_sha256"] = digest(freeze["admission"]["design"])
    freeze["admission_sha256"] = digest(freeze["admission"])
    seal(freeze)
    with pytest.raises(module.AggregationError, match="Ordered design"):
        summarize(records)


def test_function_does_not_access_files_or_scientific_loader(records, monkeypatch):
    import builtins
    import pathlib
    def forbidden(*args, **kwargs):
        pytest.fail("pure aggregation tried to open a file")
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(pathlib.Path, "open", forbidden)
    assert summarize(records)["status"] == "complete_records"


@pytest.mark.parametrize("where", ["estimator", "test"])
def test_self_resealed_wrong_primary_layout_is_rejected(records, where):
    cp = records[2]["PG10/r1"]
    design = (cp["paths"][path_id()]["provenance"]["design"] if where == "estimator"
              else cp["scores"][path_id()]["test_provenance"]["design"])
    design["rows"].reverse()
    reseal(cp)
    with pytest.raises(module.AggregationError, match="design|primary"):
        summarize(records)


@pytest.mark.parametrize("field", ["observation_model_discrepancy_rmse", "assumed_H_test_rmse", "prediction_under_true_H_noiseless_rmse"])
def test_matched_h_projection_contradiction_is_rejected(records, field):
    cp = records[2]["PG10/r1"]
    cp["scores"][path_id()][field] = .1
    reseal(cp)
    with pytest.raises(module.AggregationError, match="Matched-H"):
        summarize(records)


def test_matched_h_roundoff_allowance_does_not_become_a_scientific_effect(records):
    cp = records[2]["PG10/r1"]
    cp["scores"][path_id()]["observation_model_discrepancy_rmse"] = 1e-13
    cp["scores"][path_id()]["assumed_H_test_rmse"] += 1e-13
    reseal(cp)
    assert summarize(records)["status"] == "complete_records"


def test_output_counts_do_not_alias_the_module_or_break_future_calls(records):
    first = summarize(records)
    first["counts"]["paths"] = 0
    assert len(build_design().paths) == summarize(records)["counts"]["paths"] == 176


@pytest.mark.parametrize("signed,allowed", [(1e-7, True), (.1, False)])
def test_cauchy_mass_bound_accounts_for_squared_norm_cancellation(records, signed, allowed):
    cp = records[2]["PG10/r1"]
    cp["scores"][path_id()]["source"] = source_score(0., signed)
    reseal(cp)
    if allowed:
        assert summarize(records)["status"] == "complete_records"
    else:
        with pytest.raises(module.AggregationError, match="Cauchy mass"):
            summarize(records)


@pytest.mark.parametrize("kind", ["zero", "too_small", "different", "invalid_point"])
def test_source_norm_or_basis_contradictions_rejected(records, kind):
    cp = records[2]["PG10/r1"]
    score = cp["scores"][path_id()]["source"]
    if kind == "invalid_point":
        cp["paths"][path_id()]["candidates"]["-2.0"]["coefficients"][0] = True
    else:
        score["zero_prior_E_q"] = {"zero": 0., "too_small": .1, "different": .75}[kind]
        score["E_q"] = score["relative_L2"]*score["zero_prior_E_q"]
        if kind == "zero":
            score.update(signed_mass_error=0., absolute_mass_error=0., estimated_mass=100.)
    reseal(cp)
    with pytest.raises(module.AggregationError, match="source|Source|coefficient"):
        summarize(records)



def test_selected_summary_retains_all_four_pairs_and_single_replicate_semantics():
    from tests.experiments.observation_sensitivity.fixtures.paired_records import select_records
    ids = ["spatial_G1_matched/PG10/r1/L2", "spatial_G1_matched/PG10/r1/H1",
           "spatial_G2_matched/PG10/r1/L2", "spatial_G2_matched/PG10/r1/H1"]
    records = select_records(make_records(), ids, figures=[dict(id="matched", contrast_keys=["matched_spatial/G2"])])
    before = canonical_bytes(list(records))
    report = summarize(records)
    assert canonical_bytes(list(records)) == before
    assert len(report["raw_outcomes"]) == len(report["paired_outcomes"]) == 4
    assert report["counts"]["new_fit_attempts"] == report["counts"]["reused_fit_attempts"] == 50
    assert tuple(report["inputs"]["group_content_sha256"]) == ("PG10/r1",)
    assert all(r["n_expected"] == 1 and r["statistics"]["E_q"]["sd"] is None
               and r["statistics"]["E_q"]["mcse"] is None for r in report["contrast_summaries"])
    matched = [r for r in report["paired_outcomes"] if r["key"] == "matched_spatial/G2"]
    assert all(r["observation_same_target"] is False and set(r["difference_right_minus_left"]) == set(module.SOURCE_METRICS) for r in matched)
    records[2]["PG10/r1"]["expected_paths"].pop()
    with pytest.raises(module.AggregationError): summarize(records)


def test_selected_summary_accepts_an_explicit_empty_contrast_plan():
    from tests.experiments.observation_sensitivity.fixtures.paired_records import select_records
    ids = ["spatial_G1_matched/PG10/r1/L2", "spatial_G1_matched/PG10/r1/H1"]
    records = select_records(make_records(), ids, contrast_ids=[], figures=[])
    report = summarize(records)
    assert len(report["raw_outcomes"]) == 2 and report["paired_outcomes"] == report["contrast_summaries"] == []
    assert report["counts"]["contrasts"] == 0 and report["figures"] == []
