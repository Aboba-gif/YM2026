"""Проверка постановки E06 и происхождения исходных оценок."""
from dataclasses import asdict
import json
from pathlib import Path
import re

import adrkit

from adrkit.config.validation import canonical_bytes, digest, strict_json
from experiments.source_recovery.config import BASE_EXPONENTS
from .design import build_design, resolve_plan


_BASE_KEYS = {"config_sha256", "input_files", "code_sha256", "code_files", "versions", "threads"}
_ADMISSION_KEYS = {"schema", "version", "status", "study", "output", "static_input_paths",
    "source_protocol_path", "configuration", "config_path", "config_sha256",
    "config_canonical_sha256", "design", "design_sha256", "science_spec_sha256",
    "expected_paths", "exponents", "counts", "baseline", "sources"}


def scientific_configuration(spec):
    """Копировать научные настройки без путей и параметров исполнения.

    Parameters
    ----------
    spec : dict
        Конфигурация с source_protocol и input_files.

    Returns
    -------
    dict
        Новая JSON-копия без output, protected_roots, resources и полей path
        протокола и входов; остальные значения сохранены.
    """
    value = strict_json(canonical_bytes(spec))
    for name in ("output", "protected_roots", "resources"):
        value.pop(name, None)
    value["source_protocol"].pop("path", None)
    for item in value["input_files"]:
        item.pop("path", None)
    return value


def _object(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{label}: exact fields required")
    return value


def _text(value):
    return type(value) is str and bool(value.strip())


def _is_sha256(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _same(actual, expected, label):
    if canonical_bytes(actual) != canonical_bytes(expected):
        raise ValueError(f"{label} mismatch")


def _absolute(value, label):
    if (not _text(value) or "\x00" in value or not Path(value).is_absolute()
            or ".." in Path(value).parts or str(Path(value)) != value):
        raise ValueError(f"{label}: absolute canonical path required")
    return Path(value)


def _resolved(value, parent, label):
    if not _text(value) or "\x00" in value:
        raise ValueError(f"{label}: filesystem path required")
    return (parent / Path(value)).resolve()


def _baseline_bindings(value):
    """Исходные контрольные суммы описывают E05, а не текущий код."""
    bindings = _object(value, _BASE_KEYS, "baseline bindings")
    files = bindings["code_files"]
    if type(files) is not dict or not files or any(not _text(path)
            or "\x00" in path or ":" in path or Path(path).is_absolute()
            or any(part in ("", ".", "..") for part in path.replace("\\", "/").split("/"))
            or not path.lower().endswith(".py") or not _is_sha256(sha) for path, sha in files.items()):
        raise ValueError("Invalid baseline source inventory")
    if not _is_sha256(bindings["code_sha256"]) or digest(files) != bindings["code_sha256"]:
        raise ValueError("Baseline source inventory digest mismatch")
    versions = _object(bindings["versions"], {"python", "numpy", "scipy"}, "baseline versions")
    if any(not _text(value) for value in versions.values()):
        raise ValueError("Recorded baseline versions required")
    threads = _object(bindings["threads"],
        {"OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"}, "baseline threads")
    if any(value is not None and type(value) is not str for value in threads.values()):
        raise ValueError("Invalid recorded baseline thread declarations")
    inputs = bindings["input_files"]
    if type(inputs) is not dict or not inputs or any(not _is_sha256(value) for value in inputs.values()):
        raise ValueError("Baseline static input identities required")
    for path in inputs:
        _absolute(path, "baseline static input")
    return bindings




def validate_configuration(config):
    """Проверить формат конфигурации E06 без чтения входов.

    Parameters
    ----------
    config : dict
        Конфигурация полной серии версии 3 либо конфигурация версии 4.

    Returns
    -------
    dict
        Проверенный исходный объект без копирования и изменения.
    """
    required = {"schema", "version", "study", "output", "baseline", "design_sha256"}
    if type(config) is not dict or not required <= set(config):
        raise ValueError("Required E06 configuration fields are missing")
    version = config["version"]
    if (config["schema"] != "ym2026.observation_sensitivity.config" or type(version) is not int
            or version not in (3, 4) or not _text(config["study"])):
        raise ValueError("Invalid E06 configuration schema/version/study")
    allowed = required | ({"selection", "direct", "analysis"} if version == 4 else set())
    if set(config) - allowed or (version == 3 and config["study"] != "YM2026-E06-20260927-v1"):
        raise ValueError("Unregistered E06 configuration fields or version-3 study")
    return config


def validate_input_binding(admission):
    """Проверить структуру и взаимные привязки допуска E06.

    Сверяются план, контрольные суммы записей и разрешённые пути. Содержимое
    входных файлов здесь не читается; его проверяет build_admission.

    Parameters
    ----------
    admission : dict
        Сохранённая JSON-запись допуска версии 3 или 4.
    """
    if type(admission) is not dict:
        raise ValueError("Admission object required")
    keys = _ADMISSION_KEYS | ({"direct", "figures"} if admission.get("version") == 4 else set())
    _object(admission, keys, "admission")
    canonical_bytes(admission)
    if (admission["schema"] != "ym2026.observation_sensitivity.admission"
            or type(admission["version"]) is not int or admission["version"] not in (3, 4)
            or admission["status"] != "prepared_not_frozen" or not _text(admission["study"])):
        raise ValueError("E06 scientific admission version 3 or 4 required")
    output = _absolute(admission["output"], "output")
    config_path = _absolute(admission["config_path"], "config_path")
    config = validate_configuration(admission["configuration"])
    if config["version"] != admission["version"] or config["study"] != admission["study"]:
        raise ValueError("Configuration and admission version/study differ")
    if (not _is_sha256(admission["config_sha256"])
            or admission["config_canonical_sha256"] != digest(config)):
        raise ValueError("Configuration identity mismatch")
    if _resolved(config["output"], config_path.parent, "configuration output") != output:
        raise ValueError("Configuration output binding mismatch")
    baseline = _object(admission["baseline"],
        {"kind", "config_path", "config_sha256", "run_manifest_path", "run_manifest_sha256",
         "configuration", "bindings", "group_files"}, "baseline")
    if baseline["kind"] != "completed_run":
        raise ValueError("Completed baseline required")
    base_path = _absolute(baseline["config_path"], "baseline config_path")
    run = _absolute(baseline["run_manifest_path"], "baseline run_manifest_path")
    declared_base = _object(config["baseline"],
        {"kind", "config_path", "run_manifest_path"}, "configured baseline")
    if (declared_base["kind"] != baseline["kind"]
            or _resolved(declared_base["config_path"], config_path.parent, "baseline config") != base_path
            or _resolved(declared_base["run_manifest_path"], config_path.parent, "baseline manifest") != run):
        raise ValueError("Configured baseline binding mismatch")
    if run.name != "run.json":
        raise ValueError("Baseline manifest must be run.json")
    if any(not _is_sha256(baseline[name]) for name in ("config_sha256", "run_manifest_sha256")):
        raise ValueError("Baseline raw file identities required")
    bindings = _baseline_bindings(baseline["bindings"])
    if bindings["config_sha256"] != baseline["config_sha256"]:
        raise ValueError("Baseline configuration identity mismatch")
    spec = scientific_configuration(baseline["configuration"])
    if admission["science_spec_sha256"] != digest(spec):
        raise ValueError("Scientific specification digest mismatch")
    full = json.loads(json.dumps(asdict(build_design()), allow_nan=False))
    if config["design_sha256"] != digest(full):
        raise ValueError("Full catalogue digest mismatch")
    plan = resolve_plan(config, baseline["configuration"])
    _same(admission["design"], plan["design"], "Ordered design")
    if admission["design_sha256"] != digest(plan["design"]):
        raise ValueError("Design digest mismatch")
    _same(admission["expected_paths"], plan["expected_paths"], "Complete ordered paths")
    _same(admission["exponents"], list(BASE_EXPONENTS), "Alpha exponents")
    _same(admission["counts"], plan["counts"], "Design counts")
    if admission["version"] == 4:
        _same(admission["direct"], plan["direct"], "Direct plan")
        _same(admission["figures"], plan["figures"], "Figure plan")
    sources = _object(admission["sources"], plan["source_ids"], "sources")
    for source in sources.values():
        _object(source, {"record", "sha256"}, "source")
        if not _is_sha256(source["sha256"]) or digest(source["record"]) != source["sha256"]:
            raise ValueError("Source record digest mismatch")
    groups = baseline["group_files"]
    expected_groups = {str(run.parent / source / f"replicate_{replicate}.json")
                       for source, replicate in plan["baseline_groups"]}
    if (type(groups) is not dict or set(groups) != expected_groups
            or any(not _is_sha256(value) for value in groups.values())):
        raise ValueError("All required completed baseline groups must be pinned")
    static = admission["static_input_paths"]
    if (type(static) is not dict or set(static) != set(bindings["input_files"])
            or any(key != value for key, value in static.items())
            or admission["source_protocol_path"] not in static):
        raise ValueError("Static input path binding mismatch")
    protected = (*scientific_input_roots(admission), Path(__file__).resolve().parents[2],
                 Path(adrkit.__file__).resolve().parent)
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValueError("Output overlaps scientific inputs or installed code")


def scientific_input_roots(admission):
    """Вернуть пути научных входов проверенного допуска.

    Parameters
    ----------
    admission : dict
        Допуск, проверенный validate_input_binding.

    Returns
    -------
    tuple of Path
        Результаты E05, конфигурации, статические входы и явно защищённые корни.
    """
    baseline = admission["baseline"]
    base_path = Path(baseline["config_path"])
    return (Path(baseline["run_manifest_path"]).parent, base_path,
        Path(admission["config_path"]),
        *(Path(value) for value in admission["static_input_paths"].values()),
        *((base_path.parent / value).resolve()
          for value in baseline["configuration"].get("protected_roots", [])))

