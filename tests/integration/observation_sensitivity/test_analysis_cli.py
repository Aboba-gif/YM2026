"""Проверки диагностического CLI на синтетических записях E06.

Три анализатора используют подставленные записи с вычисленными
ковариациями и фиктивными хешами выборок невязок.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from adrkit.config.validation import canonical_bytes, digest
from experiments.observation_sensitivity.analysis import covariance as covariance
from experiments.observation_sensitivity.analysis import prediction_error as direct_analysis
from experiments.observation_sensitivity.analysis import read_results as snapshot
from experiments.observation_sensitivity.analysis import paired_effects as paired
from experiments.observation_sensitivity.design import build_design
from experiments.observation_sensitivity.input_binding import scientific_configuration

from tests.experiments.observation_sensitivity.fixtures.direct_records import direct_template as direct_template
from tests.experiments.observation_sensitivity.fixtures.covariance_records import make_records as covariance_records
from tests.experiments.observation_sensitivity.fixtures.paired_records import attach_reuse, make_records as paired_records, pin, reseal


def build_integration_records(direct_records):
    """Связать синтетические группы оценок с прямыми записями и ковариациями."""
    freeze, journal = deepcopy(direct_records)
    admission = freeze["admission"]
    spec = scientific_configuration(admission["baseline"]["configuration"])
    _, _, groups = paired_records()
    covariance_freeze, covariance_summary, _ = covariance_records()
    assert spec["calibration"] == covariance_freeze["admission"]["baseline"]["configuration"]["calibration"]
    cov_by_id = {r["path_id"]: r for r in covariance_summary["raw_outcomes"]}
    paths = {p.id: p for p in build_design().paths}
    for group in groups.values():
        source = group["bindings"]["source"]
        replicate = group["bindings"]["replicate"]
        source_record = admission["sources"][source]["record"]
        assert source_record["type"] == "FiniteRelease" and source_record["scale"] == 1.
        assert source_record["unknown_mass"] == 100.
        assert source_record["definition"]["onset_hours"] == .5
        assert source_record["definition"]["duration_hours"] == 1.
        assert source_record["definition"]["amplitude"] == 100.
        group["bindings"] = dict(source=source, replicate=replicate,
            admission_sha256=freeze["admission_sha256"],
            source_record_sha256=admission["sources"][source]["sha256"],
            direct_record_sha256=digest(journal))
        for pid, estimate in group["paths"].items():
            path, cov = paths[pid], cov_by_id[pid]
            estimate["driver_binding"]["spec_sha256"] = digest(spec)
            provenance = estimate["provenance"]
            provenance.update(deepcopy(cov["estimate"]["provenance"]))
            for purpose in ("fit", "selection"):
                # Фиктивные хеши наблюдений сохраняют привязку к общему шуму.
                
                provenance[purpose+"_y_sha256"] = digest([
                    source, asdict(path.true_h), provenance["panel_records"][purpose]])
            score = group["scores"][pid]
            test = deepcopy(cov["score"]["test_provenance"])
            test["test_y_sha256"] = digest([source, asdict(path.true_h), test["panel_record"]])
            score["test_provenance"] = test
            concentration = 35. + .2 * replicate + (.3 if path.penalty == "H1" else 0.)
            qref, duration, mass, truth_squared_norm = spec["Qref"], 3., 100., 10000.
            estimate["candidates"]["-2.0"]["coefficients"] = [concentration/qref]*73
            error = math.sqrt(duration*concentration**2 - 2*concentration*mass + truth_squared_norm) / (qref*math.sqrt(duration))
            zero_prior = math.sqrt(truth_squared_norm)/(qref*math.sqrt(duration))
            signed = (duration*concentration-mass)/(qref*duration)
            score["source"] = dict(E_q=error, relative_L2=error/zero_prior, zero_prior_E_q=zero_prior,
                estimated_mass=duration*concentration, true_mass=mass,
                signed_mass_error=signed, absolute_mass_error=abs(signed))
            # Обновляем привязку повторно используемой оценки к этой фикстуре.
            
            attach_reuse(estimate, path, admission)
        reseal(group)
    raw = {"freeze": pin(freeze), "direct": pin(journal)}
    raw.update({name: pin(group) for name, group in groups.items()})
    return dict(freeze=freeze, direct=journal, groups=groups, raw_file_sha256=raw)


def test_cli_safe_snapshot_to_three_real_analyzers(direct_template, tmp_path, monkeypatch, capsys, project_root):
    project = project_root
    assert not tmp_path.resolve().is_relative_to(project)
    records = build_integration_records(direct_template)
    expected_pin = pin(records["freeze"])
    original_bytes = canonical_bytes(records)
    expected_summary = paired.summarize_records(records["freeze"], records["direct"], records["groups"],
                                                expected_freeze_sha256=expected_pin)
    expected_direct = direct_analysis.analyze_direct(records["freeze"], records["direct"],
                                                    expected_freeze_sha256=expected_pin)
    expected_covariance = covariance.summarize_covariances(records["freeze"], expected_summary,
                                                          expected_freeze_sha256=expected_pin)
    assert expected_summary["counts"]["paths"] == 176
    assert expected_covariance["counts"]["available_sets"] == 16
    calls = []
    def read(run_arg, *, expected_freeze_sha256):
        calls.append((run_arg, expected_freeze_sha256))
        return records
    monkeypatch.setattr(snapshot, "load_records", read)
    # Подмены отклоняют повторный расчёт полей, калибровку и генерацию шума.
    
    import experiments.source_comparison.calibration as calibration_module
    from experiments.observation_sensitivity import direct as direct_producer
    def forbidden(*args, **kwargs):
        pytest.fail("The diagnostics CLI must not generate fields, noise or fitted covariance")
    monkeypatch.setattr(direct_producer, "collect_direct", forbidden)
    monkeypatch.setattr(calibration_module, "fit_covariance", forbidden)
    monkeypatch.setattr(np.random, "Generator", forbidden)
    output = tmp_path/"diagnostics.json"
    run = "artificial-not-opened-run"
    direct_analysis.main(["--run", run,
        "--freeze-sha256", expected_pin, "--output", str(output)])
    assert calls == [(run, expected_pin)]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "ym2026.observation_sensitivity.analysis"
    assert payload["status"] == "complete"
    assert payload["direct_analysis"] == expected_direct
    assert payload["covariance_analysis"] == expected_covariance
    assert payload["pair_summary_sha256"] == digest(expected_summary)
    assert payload["raw_input_file_sha256"] == records["raw_file_sha256"]
    assert payload["freeze_raw_sha256"] == expected_pin
    expected_modules = dict(snapshot=snapshot, paired=paired, direct=direct_analysis, covariance=covariance)
    assert payload["implementation_sha256"] == {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in expected_modules.items()}
    assert len(payload["raw_input_file_sha256"]) == 10
    assert canonical_bytes(records) == original_bytes
    assert output.read_bytes().endswith(b"\n") and b"\r\n" not in output.read_bytes()
    assert json.loads(capsys.readouterr().out) == {"output": str(output), "input_files": 10}
