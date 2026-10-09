"""Научные настройки принимаются только с корректной контрольной суммой и структурой."""
import hashlib
import json

import pytest

from adrkit.errors import ConfigError
from experiments.source_comparison.config import load_protocol
from experiments.source_comparison.run import setup
from experiments.source_comparison.truth import build_truth


def configuration(project_root):
    path = project_root / "experiments/source_comparison/configs/protocol.json"
    settings = json.loads((path.parent / "experiment.json").read_bytes())
    return path, settings["protocol_sha256"]


def test_current_configuration_loads_settings_and_sources(project_root):
    path, sha = configuration(project_root)
    settings, binding, output = setup(path.parent / "experiment.json")
    assert binding.full_sha256 == sha
    assert len(settings["alpha_exponents"]) == 17
    assert settings["max_iterations"] == 80
    assert settings["kkt_tolerance"] == 1e-6
    protocol = binding.document.to_dict()
    assert protocol["selector"]["tau_hours"] == .25
    assert protocol["solver"]["forward"]["scaled_residual_tolerance"] == 1e-11
    assert output.is_absolute()
    for name in ("SF01-PG10", "SF01-SB150", "SF03-EC04", "SF03-EC06"):
        assert build_truth(binding, name).source.integral(0., 3.) == pytest.approx(100.)


def test_changed_file_requires_new_declared_sum(project_root, tmp_path):
    path, sha = configuration(project_root)
    values = json.loads(path.read_bytes())
    values["execution_config"]["basis"]["Qref"] *= 2
    changed = tmp_path / "protocol.json"
    changed.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ConfigError, match="SHA256"):
        load_protocol(changed, expected_sha256=sha)


@pytest.mark.parametrize("section", ["source_strata", "reproducibility", "solver", "selector"])
def test_incomplete_settings_rejected_before_model_construction(project_root, tmp_path, section):
    path, _sha = configuration(project_root)
    values = json.loads(path.read_bytes())
    del values[section]
    changed = tmp_path / "protocol.json"
    changed.write_text(json.dumps(values), encoding="utf-8")
    actual_sha = hashlib.sha256(changed.read_bytes()).hexdigest()
    with pytest.raises(ConfigError, match="schema"):
        load_protocol(changed, expected_sha256=actual_sha)


def test_consistent_changed_execution_parameters_are_used(project_root, tmp_path):
    from experiments.source_comparison.config import execution_parameters
    path, _sha = configuration(project_root)
    values = json.loads(path.read_bytes())
    values["execution_config"]["basis"].update(Qref=200., tau_hours=.5)
    values["selector"]["tau_hours"] = .5
    values["execution_config"]["calibration"]["panels"]["calib_noise"] = 16
    values["calibration_and_splits"]["panels"]["calib_noise"] = 16
    changed = tmp_path / "protocol.json"
    changed.write_text(json.dumps(values, allow_nan=False), encoding="utf-8")
    pin = hashlib.sha256(changed.read_bytes()).hexdigest()
    binding = load_protocol(changed, expected_sha256=pin)
    assert execution_parameters(binding) == dict(q_reference=200., tau_hours=.5,
                                                 calibration_panels=16)
    assert build_truth(binding, "SF01-PG10").source.integral(0., 3.) == pytest.approx(200.)


@pytest.mark.parametrize("field,value", [("tau_hours", .5), ("calib_noise", 16)])
def test_conflicting_execution_declarations_are_rejected(project_root, tmp_path, field, value):
    path, _sha = configuration(project_root)
    values = json.loads(path.read_bytes())
    if field == "tau_hours":
        values["execution_config"]["basis"][field] = value
    else:
        values["execution_config"]["calibration"]["panels"][field] = value
    changed = tmp_path / "protocol.json"
    changed.write_text(json.dumps(values, allow_nan=False), encoding="utf-8")
    pin = hashlib.sha256(changed.read_bytes()).hexdigest()
    with pytest.raises(ConfigError, match="declarations differ"):
        load_protocol(changed, expected_sha256=pin)


@pytest.mark.parametrize("field,value", [("Qref", 0), ("Qref", True),
    ("tau_hours", 0), ("tau_hours", True), ("calib_noise", 0), ("calib_noise", 16.5)])
def test_invalid_execution_parameters_are_rejected(project_root, tmp_path, field, value):
    path, _sha = configuration(project_root)
    values = json.loads(path.read_bytes())
    if field in ("Qref", "tau_hours"):
        values["execution_config"]["basis"][field] = value
    else:
        values["execution_config"]["calibration"]["panels"][field] = value
    changed = tmp_path / "protocol.json"
    changed.write_text(json.dumps(values, allow_nan=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="positive"):
        load_protocol(changed, expected_sha256=hashlib.sha256(changed.read_bytes()).hexdigest())
