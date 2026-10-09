"""Проверки метрик сохранённых профилей и их сериализации в JSON."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from adrkit.config.validation import canonical_bytes, digest, strict_json
from experiments.source_comparison.truth import BiExponential, FiniteRelease
from adrkit.sources import P1Basis
from experiments.source_recovery import backend as module
from experiments.source_recovery.sources import ScaledSource, unknown_L2_squared


FIXTURE = Path(__file__).with_name("fixtures") / "selected_source_scores.json"
METRIC_NAMES = {
    "E_q", "relative_L2", "estimated_mass", "true_mass", "signed_mass_error",
    "absolute_mass_error", "zero_prior_E_q",
}


@pytest.mark.parametrize("source_name", ["EC04", "PG10"])
def test_saved_selected_source_scores_are_defined_json_and_unchanged(
        source_name, monkeypatch, tmp_path):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    case = fixture["cases"][source_name]

    def no_solve(*args, **kwargs):
        pytest.fail("Source scoring must not construct or execute a PDE solver")

    monkeypatch.setattr(module, "make_solver", no_solve)
    monkeypatch.setattr(module.CachedSolver, "__init__", no_solve)

    record = case["source_record"]
    assert digest(record) == case["provenance"]["source_record_sha256"]
    kind = {"BiExponential": BiExponential, "FiniteRelease": FiniteRelease}[record["type"]]
    source = ScaledSource(kind(**record["definition"]), record["scale"])
    basis = P1Basis(case["basis_knots"], time_unit=case["time_unit"])
    coefficients = np.array(case["coefficients"], dtype=float)
    original_coefficients = coefficients.copy()

    # Для нормированных ошибок нужны конечные положительные масштабы.
    # Проверяются норма заданного источника и масштаб его интенсивности.
    norm2 = unknown_L2_squared(source)
    assert norm2 == case["expected_truth_l2_squared"]
    assert math.isfinite(norm2) and norm2 > 0
    assert case["qref"] > 0 and math.isfinite(case["qref"])
    assert basis.knots[0] == 0.0 and basis.knots[-1] == 3.0
    assert np.isfinite(coefficients).all()

    result = module.source_scores(source, basis, coefficients, case["qref"])
    assert set(result) == METRIC_NAMES
    assert all(type(value) is float and math.isfinite(value) for value in result.values())
    assert result == case["expected_metrics"]
    assert np.array_equal(coefficients, original_coefficients)

    # Каноническая сериализация проверяется записью метрик и чтением с диска.
    encoded = canonical_bytes(result)
    output = tmp_path / f"{source_name}-source-scores.json"
    output.write_bytes(encoded + b"\n")
    restored = strict_json(output.read_bytes())
    assert restored == result
    assert canonical_bytes(restored) == encoded
