"""Проверки происхождения входов короткого расчёта КрАЗ."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.kraz_alignment import run as kraz
from experiments.source_comparison.run import setup


def test_setup_preserves_three_returns_and_uses_supplied_bytes(short_inputs, monkeypatch):
    config, simulation, _ = short_inputs
    raw = simulation.read_bytes()
    usual = setup(simulation)
    original_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == simulation:
            raise AssertionError("setup must parse the supplied simulation buffer")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    buffered = setup(simulation, config_bytes=raw)
    assert len(buffered) == len(usual) == 3
    assert buffered[0] == usual[0] == json.loads(raw)
    assert buffered[1].full_sha256 == usual[1].full_sha256
    assert buffered[2] == usual[2] == simulation.parent / "unused-simulation-output"


def test_run_parses_and_hashes_each_same_input_buffer(short_inputs, monkeypatch):
    config, simulation, output = short_inputs
    buffers = {path: path.read_bytes() for path in (config, simulation)}
    runner_sha = hashlib.sha256(Path(kraz.__file__).read_bytes()).hexdigest()
    reads = []
    original_read_bytes, original_setup = Path.read_bytes, kraz.setup

    def read_bytes(path):
        if path in buffers:
            reads.append(path)
        return original_read_bytes(path)

    def use_setup(path, *, config_bytes):
        assert path == simulation and config_bytes == buffers[simulation]
        result = original_setup(path, config_bytes=config_bytes)
        assert result[0] == json.loads(buffers[simulation])
        return result

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(kraz, "setup", use_setup)
    kraz.run(config)
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["status"] == "completed"
    assert manifest["configuration"] == json.loads(buffers[config])
    assert manifest["config_sha256"] == hashlib.sha256(buffers[config]).hexdigest()
    assert manifest["simulation_config_sha256"] == hashlib.sha256(buffers[simulation]).hexdigest()
    assert manifest["runner_sha256"] == runner_sha
    assert reads.count(config) == reads.count(simulation) == 2
    templates = pd.read_csv(output / "templates.csv")
    assert len(templates) == 2 * 72
    overlays = pd.read_csv(output / "overlays.csv")
    assert overlays.groupby("split").used_for_fit.sum().to_dict() == {
        "exploration": 6, "later_period": 6}
    assert overlays.groupby("split").size().to_dict() == {
        "exploration": 9, "later_period": 9}


@pytest.mark.parametrize("changed", ["config", "simulation"])
def test_temporary_json_replacement_keeps_parsed_buffer_digest(
        short_inputs, monkeypatch, changed):
    config, simulation, output = short_inputs
    path = {"config": config, "simulation": simulation}[changed]
    original = path.read_bytes()
    different = json.loads(original)
    if changed == "config":
        different["amplitude_factors"] = [0.25, 1.0]
        restore_on = simulation
    else:
        different["sources"] = ["SF03-EC04"]
        restore_on = simulation.parent / "protocol.json"
    replacement = json.dumps(different).encode("utf-8")
    original_read_bytes = Path.read_bytes
    injected = restored = False

    def read_bytes(current):
        nonlocal injected, restored
        if current == restore_on and injected and not restored:
            path.write_bytes(original)
            restored = True
        raw = original_read_bytes(current)
        if current == path and not injected:
            path.write_bytes(replacement)
            injected = True
        return raw

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    try:
        kraz.run(config)
        assert injected and restored
        assert path.read_bytes() == original
        manifest = json.loads((output / "manifest.json").read_bytes())
        key = "config_sha256" if changed == "config" else "simulation_config_sha256"
        assert manifest["status"] == "completed"
        assert manifest[key] == hashlib.sha256(original).hexdigest()
        templates = pd.read_csv(output / "templates.csv")
        assert set(templates.source) == {"SF01-PG10"}
        assert set(templates.amplitude_factor) == {0.5, 1.0}
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("changed", ["config", "simulation", "runner"])
def test_run_rejects_input_changed_after_reading(short_inputs, monkeypatch, changed):
    config, simulation, output = short_inputs
    source_path = Path(kraz.__file__).resolve()
    source_bytes = source_path.read_bytes()
    if changed == "runner":
        isolated_root = config.parent / "isolated_app"
        path = isolated_root / "experiments/kraz_alignment/run.py"
        path.parent.mkdir(parents=True)
        assert path.resolve().is_relative_to(config.parent.resolve())
        assert path.resolve().parents[2] == isolated_root.resolve()
        assert path.resolve() != source_path
        path.write_bytes(source_bytes)
        monkeypatch.setattr(kraz, "__file__", str(path))
    else:
        path = {"config": config, "simulation": simulation}[changed]
    assert path.resolve().is_relative_to(config.parent.resolve())
    original = path.read_bytes()
    original_loader = kraz.load_network

    def load_network(data_root):
        result = original_loader(data_root)
        path.write_bytes(original + b"\n")
        return result

    monkeypatch.setattr(kraz, "load_network", load_network)
    try:
        with pytest.raises(ValueError, match="changed during KrAZ run"):
            kraz.run(config)
        assert not (output / "manifest.json").exists()
        assert (output / "all_windows.csv").exists()
    finally:
        path.write_bytes(original)
    assert source_path.read_bytes() == source_bytes
