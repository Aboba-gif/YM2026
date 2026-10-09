"""Синтетические журналы E06 для проверки парных разностей и исключений."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

from adrkit.config.validation import canonical_bytes, digest
from experiments.observation_sensitivity.input_binding import scientific_configuration
from experiments.observation_sensitivity.design import ALPHA_EXPONENTS, build_design, resolve_plan, DIRECT_SOURCES
from experiments.observation_sensitivity.analysis import paired_effects as module


def seal(record):
    record["content_sha256"] = digest({k: v for k, v in record.items() if k != "content_sha256"})
    return record

def pin(record):
    return hashlib.sha256(canonical_bytes(record) + b"\n").hexdigest()

def reseal(group):
    group["selection_seal"] = digest({k: group[k] for k in
        ("schema", "version", "bindings", "expected_paths", "exponents", "paths")})
    seal(group)

def attach_reuse(estimate, path, admission):
    if path.reuse:
        original = {k: v for k, v in estimate.items() if k not in ("reuse", "driver_binding", "e06_provenance")}
        estimate["reuse"] = dict(study="research_validation_v2", new_path_id=path.id,
            original_path_id=f"{path.reuse.condition}/{path.penalty}", original_path_sha256=digest(original),
            source_file_sha256=digest([path.source, path.replicate, "v2 file"]), selection_seal=digest("v2 seal"),
            bindings=dict(admission["baseline"]["bindings"], source=path.source, replicate=path.replicate,
                source_record_sha256=admission["sources"][path.source]["sha256"]))
        estimate["e06_provenance"] = deepcopy(estimate.get("provenance", {"calibration_failure": estimate.get("calibration_failure")}))

def panel(path, name):
    seed = [path.stream.seed, path.stream.version, path.replicate,
            {"fit": 211, "selection": 307, "test": 401}[name], 0]
    noise = json.loads(json.dumps(asdict(path.noise)))
    return dict(seed=seed, panel=name, generator="PCG64", standard_normal_sha256=digest(seed),
                covariance_sha256=digest(noise), residual_sha256=digest([seed, noise]))

def observation_hash(path, name):
    return digest([path.source, asdict(path.true_h), panel(path, name)])

def provenance(path):
    return dict(fit_y_sha256=observation_hash(path, "fit"), selection_y_sha256=observation_hash(path, "selection"),
        selected_covariance_sha256=digest([json.loads(json.dumps(asdict(path.noise))), path.weight]),
        jacobian_sha256=digest([asdict(path.inverse_h), "J"]), offset_sha256=digest([asdict(path.inverse_h), "offset"]),
        panel_records={n: panel(path, n) for n in ("fit", "selection")},
        calibration=dict(weight=path.weight, replicate=path.replicate, fixture=True),
        design=deepcopy(module.PRIMARY_DESIGN))

def source_score(error, signed=.01):
    return dict(E_q=error, relative_L2=error / .5, zero_prior_E_q=.5,
                estimated_mass=100. + 300. * signed, true_mass=100.,
                signed_mass_error=signed, absolute_mass_error=abs(signed))

def baseline_spec():
    """Дополнить параметры шаблона действующим каталогом условий E05."""
    from experiments.source_recovery.config import template_config
    from experiments.observation_sensitivity.admission import _registered_config_path
    spec = template_config(100.)
    declared = json.loads(_registered_config_path().read_text(encoding="utf-8"))
    spec.update(conditions=declared["conditions"], frozen=True, study=declared["study"])
    return spec


def make_admission(spec, sources, design, *, output=None):
    """Собрать метаданные синтетического допуска с полным планом E06."""
    if output is None:
        temporary = Path(tempfile.gettempdir())
        project = Path(__file__).resolve().parents[4]
        if temporary.is_relative_to(project):
            temporary = project.parent
        output = temporary / "ym2026-analysis-fixture/observations"
    output = Path(output)
    root = output.parent
    protocol = str(root / "inputs/protocol.json")
    complete = baseline_spec()
    complete.pop("source_protocol", None)
    complete.pop("input_files", None)
    for key, value in deepcopy(spec).items():
        if type(value) is dict and type(complete.get(key)) is dict:
            complete[key].update(value)
        elif key != "conditions" or value:
            complete[key] = value
    spec = complete
    spec.setdefault("source_protocol", dict(path=protocol, sha256=digest("protocol")))
    spec.setdefault("input_files", [])
    base_path, run_path = str(root / "baseline.json"), str(root / "baseline/run.json")
    declared = json.loads(json.dumps(asdict(design)))
    configuration = dict(schema="ym2026.observation_sensitivity.config", version=3,
        study="YM2026-E06-20260927-v1", output=str(output), design_sha256=digest(declared),
        baseline=dict(kind="completed_run", config_path=base_path, run_manifest_path=run_path))
    plan = resolve_plan(configuration, spec)
    code_files = {"adrkit/baseline.py": digest("recorded source")}
    bindings = dict(config_sha256=digest(spec), code_files=code_files, code_sha256=digest(code_files),
        versions=dict(python="baseline-python", numpy="baseline-numpy", scipy="baseline-scipy"),
        threads=dict(OPENBLAS_NUM_THREADS="8", OMP_NUM_THREADS="4", MKL_NUM_THREADS=None),
        input_files={protocol: spec["source_protocol"]["sha256"]})
    return dict(schema="ym2026.observation_sensitivity.admission", version=3,
        status="prepared_not_frozen", study=configuration["study"], output=str(output),
        config_path=str(root / "observation_sensitivity.json"), config_sha256=digest(configuration),
        config_canonical_sha256=digest(configuration), configuration=configuration,
        static_input_paths={protocol: protocol}, source_protocol_path=protocol,
        sources=deepcopy(sources), design=declared, design_sha256=digest(declared),
        science_spec_sha256=digest(scientific_configuration(spec)), counts=plan["counts"],
        exponents=list(ALPHA_EXPONENTS), expected_paths=plan["expected_paths"],
        baseline=dict(kind="completed_run", configuration=spec, bindings=bindings,
            config_path=base_path, config_sha256=bindings["config_sha256"], run_manifest_path=run_path,
            run_manifest_sha256=digest("baseline run"), group_files={
                str(Path(run_path).parent / source / f"replicate_{r}.json"): digest([source, r, "v2 file"])
                for source, r in plan["baseline_groups"]}))


def make_records():
    """Построить полный синтетический план E06 с заданными ошибками."""
    design = build_design()
    spec = dict(Qref=100., solver=dict(forward_tolerance=1e-11),
                alpha=dict(tie_atol=1e-12, tie_rtol=1e-10))
    sources = {}
    for name in DIRECT_SOURCES:
        record = dict(name=name, unknown_mass=100.)
        sources[name] = dict(record=record, sha256=digest(record))
    admission = make_admission(spec, sources, design)
    spec = scientific_configuration(admission["baseline"]["configuration"])
    freeze = seal(dict(schema="ym2026.observation_sensitivity.freeze", version=3,
        admission=admission, admission_sha256=digest(admission), frozen_at="2026-09-27T00:00:00+00:00"))
    result = dict(schema="ym2026.observation_sensitivity.direct", version=2, status="complete", expected_fields=12,
        complete_fields=12, original_spec_sha256=digest(spec), sources=deepcopy(sources),
        fields={name: {domain: dict(status="complete") for domain in ("D0", "D1")} for name in sources})
    direct = seal(dict(schema="ym2026.observation_sensitivity.direct_journal", version=3, status="completed",
        admission_sha256=digest(admission), freeze_sha256=freeze["content_sha256"],
        run_id="12345678-1234-1234-1234-123456789012",
        started_at="2026-09-27T00:01:00+00:00", finished_at="2026-09-27T00:02:00+00:00", result=result))
    groups = {}
    for group_id in admission["expected_paths"]:
        source, r = group_id.split("/r")
        replicate = int(r)
        paths = [p for p in design.paths if (p.source, p.replicate) == (source, replicate)]
        cp = dict(schema="ym2026.observation_sensitivity.checkpoint", version=1, revision=1, stage="scored",
            bindings=dict(source=source, replicate=replicate, admission_sha256=digest(admission),
                          source_record_sha256=sources[source]["sha256"], direct_record_sha256=digest(direct)),
            expected_paths=[p.id for p in paths], exponents=list(ALPHA_EXPONENTS), paths={}, scores={})
        for path in paths:
            candidates = {str(e): dict(alpha=10.**e, exponent=e, accepted=True, status="accepted",
                                      selection_mse=(e + 2.)**2, forward_residual=0.) for e in ALPHA_EXPONENTS}
            # Коэффициенты заданы только для выбранного кандидата.
            # Остальные попытки содержат скалярные диагностические показатели.
            candidates["-2.0"]["coefficients"] = [103./300.]*73
            estimate = dict(condition=path.reuse.condition if path.reuse else path.condition, penalty=path.penalty,
                alpha_reference=1., provenance=provenance(path), candidates=candidates, finalized=True,
                candidate_count=25, accepted_count=25, complete_path=True, tuning_unresolved=False,
                selected_exponent=-2., procedure_accepted=True, lcurve_diagnostic=None,
                final_certificate=dict(accepted=True, finite=True, optimizer_success=True, forward_residual=0.,
                    norms={name: 0. for name in ("primal", "dual", "stationarity", "complementarity", "free_coordinate")}),
                driver_binding=dict(spec_sha256=digest(spec), path_sha256=digest(json.loads(json.dumps(asdict(path))))))
            attach_reuse(estimate, path, admission)
            cp["paths"][path.id] = estimate
            # Двоично представимые разности выявляют ошибки n, ddof и знака.
            error = 1. if path.penalty == "L2" else 1. + (.125, -.25, .375, -.5)[replicate-1]
            cp["scores"][path.id] = dict(procedure_accepted=True, diagnostic_only=False, score_row_count=36,
                status="available", source=source_score(error), forward_residual=0.,
                **{name: .25 for name in module.OBSERVATION_METRICS},
                score_design=dict(primary_rows=module.PRIMARY_ROWS),
                test_provenance=dict(test_y_sha256=observation_hash(path, "test"), panel_record=panel(path, "test"),
                                     design=deepcopy(module.PRIMARY_DESIGN)))
            if path.true_h == path.inverse_h:
                cp["scores"][path.id]["observation_model_discrepancy_rmse"] = 0.
        reseal(cp)
        groups[group_id] = cp
    return freeze, direct, groups

def path_id(r=1, penalty="H1", condition="spatial_G05_matched", source="PG10"):
    return f"{condition}/{source}/r{r}/{penalty}"

def alter(records, r=1, *, kind, penalty="H1", condition="spatial_G05_matched"):
    freeze, _, groups = records
    cp = groups[f"PG10/r{r}"]
    pid = path_id(r, penalty, condition)
    path = next(p for p in build_design().paths if p.id == pid)
    estimate, score = cp["paths"][pid], cp["scores"][pid]
    if kind == "calibration":
        estimate = dict(condition=estimate["condition"], penalty=penalty, candidates={}, finalized=True,
            procedure_accepted=False, calibration_failure="fixture: covariance not estimable",
            driver_binding=estimate["driver_binding"])
        cp["paths"][pid] = estimate
        score = dict(procedure_accepted=False, diagnostic_only=True, score_row_count=36,
                     status="unavailable", reason="calibration_failure")
        cp["scores"][pid] = score
    elif kind == "scoring":
        cp["scores"][pid] = dict(procedure_accepted=True, diagnostic_only=False, score_row_count=36,
                                status="unavailable", reason="scoring_failure", error="fixture: scorer failed")
    else:
        if kind in ("lower", "upper"):
            e = -8. if kind == "lower" else 4.
            estimate["candidates"][str(e)]["selection_mse"] = -0.0
            estimate["candidates"][str(e)]["coefficients"] = estimate["candidates"]["-2.0"]["coefficients"][:]
            estimate["candidates"]["-2.0"]["selection_mse"] = 1.
            estimate.update(selected_exponent=e, tuning_unresolved=True)
        elif kind == "candidate":
            estimate["candidates"]["-8.0"].update(accepted=False, status="numerical_failure")
            estimate.update(accepted_count=24, complete_path=False)
        elif kind == "certificate":
            estimate["final_certificate"] = dict(accepted=False, error="fixture: fresh forward failure")
        elif kind == "all_candidates":
            for candidate in estimate["candidates"].values():
                candidate.update(accepted=False, status="numerical_failure")
            estimate.update(accepted_count=0, complete_path=False, selected_exponent=None)
            estimate.pop("final_certificate")
            cp["scores"][pid] = dict(procedure_accepted=False, diagnostic_only=True, score_row_count=36,
                                    status="unavailable", reason="no_accepted_candidate")
        else:
            raise AssertionError(kind)
        estimate["procedure_accepted"] = False
        score.update(procedure_accepted=False, diagnostic_only=True)
    attach_reuse(estimate, path, freeze["admission"])
    reseal(cp)

def summary_cell(report, key="penalty/spatial_G05_matched", source="PG10", penalty="L2_to_H1"):
    return next(r for r in report["contrast_summaries"] if (r["key"], r["source"], r["penalty"]) == (key, source, penalty))



def select_admission(admission, path_ids, *, contrast_ids=None, figures=None):
    """Выбрать часть синтетического журнала с теми же научными настройками."""
    from experiments.observation_sensitivity.design import resolve_plan, DIRECT_OBSERVATIONS, DIRECT_SOURCES
    result = deepcopy(admission)
    config = result["configuration"]
    config["version"] = result["version"] = 4
    config["selection"] = dict(path_ids=list(path_ids))
    if contrast_ids is not None:
        config["selection"]["contrast_ids"] = list(contrast_ids)
    sources = [name for name in DIRECT_SOURCES if any(f"/{name}/" in pid for pid in path_ids)]
    config["direct"] = dict(source_ids=sources, domains=dict(D0="G0", D1="E06_D1_direct"),
        grids=dict(E06_D1_direct=dict(bounds_km=[-18., 12., -10., 10.], spacing_km=.25, steps=336)),
        observation_ids=list(DIRECT_OBSERVATIONS), quadrature_spacings_km=[.25, .125, .0625])
    if figures is not None:
        config["analysis"] = dict(figures=deepcopy(figures))
    plan = resolve_plan(config, result["baseline"]["configuration"])
    result.update(config_sha256=digest(config), config_canonical_sha256=digest(config),
        design=plan["design"], design_sha256=digest(plan["design"]), expected_paths=plan["expected_paths"],
        counts=plan["counts"], direct=plan["direct"], figures=plan["figures"],
        sources={s: result["sources"][s] for s in plan["source_ids"]})
    root = Path(result["baseline"]["run_manifest_path"]).parent
    group_paths = {str(root / source / f"replicate_{replicate}.json") for source, replicate in plan["baseline_groups"]}
    result["baseline"]["group_files"] = {p: h for p, h in result["baseline"]["group_files"].items() if p in group_paths}
    return result


def select_records(records, path_ids, *, contrast_ids=None, figures=None):
    """Составить согласованные синтетические записи выбранных условий."""
    freeze, direct, groups = deepcopy(records)
    admitted = select_admission(freeze["admission"], path_ids, contrast_ids=contrast_ids, figures=figures)
    freeze.update(version=4, admission=admitted, admission_sha256=digest(admitted))
    seal(freeze)
    direct.update(version=4, admission_sha256=digest(admitted), freeze_sha256=freeze["content_sha256"])
    result = direct["result"]
    sources = admitted["direct"]["source_ids"]
    result["sources"] = {s: result["sources"][s] for s in sources}
    result["fields"] = {s: result["fields"][s] for s in sources}
    result["expected_fields"] = len(sources)*2
    result["complete_fields"] = sum(r["status"] == "complete" for rows in result["fields"].values() for r in rows.values())
    if "comparisons" in result:
        result["comparisons"] = {s: result["comparisons"][s] for s in sources}
        result["domain_sensitive_count"] = sum(r["status"] == "domain_sensitive" for rows in result["comparisons"].values() for r in rows.values())
    seal(direct)
    selected = {}
    for key, expected in admitted["expected_paths"].items():
        group = groups[key]
        group["bindings"].update(admission_sha256=digest(admitted), direct_record_sha256=digest(direct))
        group["expected_paths"] = expected
        group["paths"] = {p: group["paths"][p] for p in expected}
        group["scores"] = {p: group["scores"][p] for p in expected}
        reseal(group)
        selected[key] = group
    return freeze, direct, selected
