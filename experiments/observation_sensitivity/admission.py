"""Проверка конфигурации E06, статических входов и завершённых групп E05."""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

import adrkit
from adrkit.config.validation import canonical_bytes, digest, strict_json
from experiments.source_comparison.config import load_protocol
from experiments.source_recovery.run import source_record_hash
from experiments.source_recovery.config import BASE_EXPONENTS, validate_config
from experiments.source_recovery.sources import make_sources, source_record
from .design import PathSpec, build_design, resolve_plan, DIRECT_SOURCES
from .input_binding import scientific_configuration, validate_input_binding, validate_configuration
from .lifecycle import validate_terminal_v2


CONFIG_SCHEMA = "ym2026.observation_sensitivity.config"
STUDY = "YM2026-E06-20260927-v1"
SOURCE_IDS = DIRECT_SOURCES
BASE_KEYS = {"config_sha256", "input_files", "code_sha256", "code_files", "versions", "threads"}


class AdmissionError(ValueError):
    """Некорректные, изменённые или недоступные входы эксперимента E06."""


def design_record():
    """Вернуть отдельную JSON-запись плана E06.

    Returns
    -------
    dict
        План с упорядоченными последовательностями и парными сравнениями.
    """
    return json.loads(json.dumps(asdict(build_design()), allow_nan=False))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path):
    try:
        return _plain_path(path).read_bytes()
    except OSError as exc:
        raise AdmissionError(f"Cannot read admitted static file: {path}") from exc


def _object(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise AdmissionError(f"{label}: exact keys {sorted(keys)} required")
    return value


def _same(actual, expected, label):
    # Сравнение JSON-байтов различает логические и числовые значения.
    if canonical_bytes(actual) != canonical_bytes(expected):
        raise AdmissionError(f"{label} does not match the approved declaration")


def _registered_config_path():
    return Path(__file__).resolve().parents[1] / "source_recovery/configs/experiment.json"


def _baseline_paths(spec, source, replicate):
    paths = [c["id"] + "/" + arm for c in spec["conditions"]
             if source in c["sources"] and replicate in c["replicates"]
             for arm in c["penalties"]]
    paths += ["single/" + s["id"] for s in spec.get("single_fits", [])
              if source in s["sources"] and replicate in s["replicates"]]
    return paths


def _plain_path(value, *, parent=None, label="path"):
    """Разрешить путь относительно parent и отклонить файловые псевдонимы."""
    if not isinstance(value, (str, Path)) or not str(value) or "\x00" in str(value):
        raise AdmissionError(f"{label}: nonempty filesystem path required")
    path = Path(value)
    if not path.is_absolute():
        if parent is None or path.drive:
            raise AdmissionError(f"{label}: absolute path or explicit parent required")
        path = parent / path
    lexical = Path(os.path.abspath(path))
    for candidate in (lexical, *lexical.parents):
        if candidate.is_symlink() or getattr(candidate, "is_junction", lambda: False)():
            raise AdmissionError(f"{label}: filesystem aliases are forbidden")
    if lexical.resolve() != lexical:
        raise AdmissionError(f"{label}: filesystem aliases are forbidden")
    return lexical


def _overlap(left, right):
    return left.is_relative_to(right) or right.is_relative_to(left)


def _group(source, replicate):
    if source not in ("PG10", "EC04") or type(replicate) is not int or replicate not in range(1, 5):
        raise AdmissionError("E06 inverse groups are exactly PG10/EC04, replicates 1 through 4")


@dataclass(frozen=True)
class Admission:
    """Снимок проверенных входов и плана эксперимента E06.

    Создавайте снимок через build_admission: эта фабрика проверяет
    конфигурации и завершённые исходные группы.

    Parameters
    ----------
    _content : bytes
        Канонический JSON допуска.
    _paths : tuple of PathSpec
        Упорядоченные условия восстановления.
    """

    _content: bytes
    _paths: tuple[PathSpec, ...]

    def to_dict(self):
        """Вернуть отдельную копию записи допуска.

        Returns
        -------
        dict
            Десериализованная запись; её изменение не меняет снимок.
        """

        return strict_json(self._content)

    @property
    def sha256(self):
        """Вернуть SHA-256 канонических байтов допуска.

        Returns
        -------
        str
            Шестнадцатеричная контрольная сумма снимка.
        """

        return _sha(self._content)

    @property
    def spec(self):
        """Вернуть научные настройки исходной постановки.

        Returns
        -------
        dict
            Новая копия конфигурации без путей и параметров исполнения.
        """

        return scientific_configuration(self.to_dict()["baseline"]["configuration"])

    def paths_for(self, source, replicate):
        """Выбрать последовательности одной группы E06.

        Parameters
        ----------
        source : {'PG10', 'EC04'}
            Идентификатор источника.
        replicate : int
            Номер реализации от 1 до 4.

        Returns
        -------
        tuple of PathSpec
            Условия в порядке сохранённого плана.
        """

        _group(source, replicate)
        paths = tuple(p for p in self._paths if p.source == source and p.replicate == replicate)
        if not paths:
            raise AdmissionError("Group is absent from the admitted plan")
        return paths

    def expected_baseline_bindings(self, source, replicate):
        """Вернуть ожидаемые привязки исходной группы E05.

        Parameters
        ----------
        source : {'PG10', 'EC04'}
            Идентификатор источника.
        replicate : int
            Номер реализации от 1 до 4.

        Returns
        -------
        dict
            Привязки исходного запуска, источника, реализации и записи
            генератора.
        """

        _group(source, replicate)
        record = self.to_dict()
        return dict(record["baseline"]["bindings"], source=source, replicate=replicate,
                    source_record_sha256=record["sources"][source]["sha256"])

    def expected_baseline_paths(self, source, replicate):
        """Вернуть полный ожидаемый порядок оценок исходной группы.

        Parameters
        ----------
        source : {'PG10', 'EC04'}
            Идентификатор источника.
        replicate : int
            Номер реализации от 1 до 4.

        Returns
        -------
        tuple of str
            Идентификаторы всех условий и отдельных оценок E05, включая
            неиспользуемые в E06.
        """
        _group(source, replicate)
        return tuple(_baseline_paths(self.spec, source, replicate))


def build_admission(e06_config_path):
    """Проверить входы E05 и сформировать допуск к эксперименту E06.

    Функция читает и проверяет статические входы, но не закрепляет допуск на
    диске. Требования к входам приведены в README.

    Parameters
    ----------
    e06_config_path : str or pathlib.Path
        Путь к конфигурации E06; относительный путь разрешается от текущего
        каталога.

    Returns
    -------
    Admission
        Снимок согласованных конфигурации, плана и завершённых исходных
        групп.
    """
    try:
        return _build(e06_config_path)
    except AdmissionError:
        raise
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise AdmissionError(f"Malformed or unavailable E06 admission input: {exc}") from exc


def _build(e06_config_path):
    path = _plain_path(e06_config_path, parent=Path.cwd(), label="E06 config")
    raw_config = _read(path)
    config = strict_json(raw_config)
    validate_configuration(config)
    base = _object(config["baseline"], {"kind", "config_path", "run_manifest_path"}, "baseline")
    if base["kind"] != "completed_run":
        raise AdmissionError("A completed source-recovery run is required")
    base_path = _plain_path(base["config_path"], parent=path.parent, label="baseline.config_path")
    run_path = _plain_path(base["run_manifest_path"], parent=path.parent, label="baseline.run_manifest_path")
    if run_path.name != "run.json":
        raise AdmissionError("The baseline manifest must be run.json")
    output = _plain_path(config["output"], parent=path.parent, label="output")
    protected = [run_path.parent, path, base_path, Path(__file__).resolve().parents[2],
                 Path(adrkit.__file__).resolve().parent]
    if any(_overlap(output, item) for item in protected):
        raise AdmissionError("Output must not overlap baseline inputs or installed code")
    base_raw, run_raw = _read(base_path), _read(run_path)
    spec, run = strict_json(base_raw), strict_json(run_raw)
    validate_config(spec, require_frozen=True)
    registered_path = _plain_path(_registered_config_path())
    registered_raw = _read(registered_path)
    registered = strict_json(registered_raw)
    if config["version"] == 3:
        _same(scientific_configuration(spec), scientific_configuration(registered),
              "Current source-recovery scientific configuration")
        if tuple(spec["sources"]) != SOURCE_IDS:
            raise AdmissionError("All six source profiles must be admitted in their registered order")
    else:
        for name in ("model", "observations", "noise", "stream", "calibration", "alpha", "solver",
                     "source_mass", "Qref", "truth_grid", "source_protocol"):
            actual = scientific_configuration(spec)[name]
            expected = scientific_configuration(registered)[name]
            _same(actual, expected, f"Source-recovery {name}")
    if _plain_path(spec["output"], parent=base_path.parent, label="baseline output") != run_path.parent:
        raise AdmissionError("Baseline configuration output differs from its run manifest directory")
    _object(run, {"schema", "bindings", "configuration", "initial_state"}, "source-recovery run manifest")
    if run["schema"] != spec["schema"]:
        raise AdmissionError("Baseline run schema mismatch")
    _same(run["configuration"], spec, "Baseline run configuration")
    bindings = _object(run["bindings"], BASE_KEYS, "baseline run bindings")
    design = build_design()
    design_json = json.loads(json.dumps(asdict(design), allow_nan=False))
    if config["design_sha256"] != digest(design_json):
        raise AdmissionError("Complete canonical design SHA256 mismatch")
    plan = resolve_plan(config, spec)
    design_json = plan["design"]
    _same(bindings["config_sha256"], _sha(base_raw), "Baseline configuration identity")
    inputs, input_paths, static_paths = {}, [], {}
    extra_protected = [_plain_path(value, parent=base_path.parent, label="protected input root")
                       for value in spec.get("protected_roots", [])]
    for item in [spec["source_protocol"], *spec["input_files"]]:
        target = _plain_path(item["path"], parent=base_path.parent, label="baseline static input")
        if target.is_relative_to(run_path.parent) or target in input_paths:
            raise AdmissionError("Duplicate input or checkpoint used as a scientific input")
        if _overlap(output, target):
            raise AdmissionError("Output must not overlap static input files")
        actual = _sha(_read(target))
        if actual != item["sha256"]:
            raise AdmissionError(f"Static input bytes changed: {target.name}")
        inputs[str(target)] = actual
        static_paths[str(target)] = str(target)
        input_paths.append(target)
    if any(_overlap(output, item) for item in extra_protected):
        raise AdmissionError("Output must not overlap protected input directories")
    _same(bindings["input_files"], inputs, "Baseline static input identities")
    protocol = load_protocol(input_paths[0], expected_sha256=spec["source_protocol"]["sha256"])
    sources = make_sources(protocol, mass=spec["source_mass"])
    if tuple(sources) != SOURCE_IDS:
        raise AdmissionError("Generator factory must return exactly all six sources")
    source_records = {}
    for name in plan["source_ids"]:
        source = sources[name]
        source_value = json.loads(json.dumps(source_record(source), allow_nan=False))
        source_hash = source_record_hash(source)
        if digest(source_value) != source_hash:
            raise AdmissionError(f"Source serialization contract mismatch: {name}")
        source_records[name] = dict(record=source_value, sha256=source_hash)
    group_files = {}
    for source, replicate in plan["baseline_groups"]:
        group_path = _plain_path(run_path.parent / source / f"replicate_{replicate}.json")
        group_raw = _read(group_path)
        group = strict_json(group_raw)
        validate_terminal_v2(group)
        _same(group["bindings"], dict(bindings, source=source, replicate=replicate,
              source_record_sha256=source_records[source]["sha256"]), "Baseline group bindings")
        _same(group["expected_paths"], _baseline_paths(spec, source, replicate),
              "Complete ordered baseline group paths")
        group_files[str(group_path)] = _sha(group_raw)
    if (_read(path) != raw_config or _read(base_path) != base_raw or _read(run_path) != run_raw
            or _read(registered_path) != registered_raw
            or any(_sha(_read(p)) != inputs[str(p)] for p in input_paths)
            or any(_sha(_read(Path(p))) != expected for p, expected in group_files.items())):
        raise AdmissionError("Admission inputs changed while preparing the declaration")
    record = dict(schema="ym2026.observation_sensitivity.admission", version=config["version"],
        status="prepared_not_frozen", study=config["study"], output=str(output),
        static_input_paths=static_paths, source_protocol_path=str(input_paths[0]),
        configuration=config, config_path=str(path), config_sha256=_sha(raw_config),
        config_canonical_sha256=digest(config), design=design_json, design_sha256=digest(design_json),
        science_spec_sha256=digest(scientific_configuration(spec)),
        expected_paths=plan["expected_paths"], exponents=list(BASE_EXPONENTS), counts=plan["counts"],
        baseline=dict(kind="completed_run", config_path=str(base_path), config_sha256=_sha(base_raw),
                      run_manifest_path=str(run_path), run_manifest_sha256=_sha(run_raw),
                      configuration=spec, bindings=bindings, group_files=group_files),
        sources=source_records)
    if config["version"] == 4:
        record.update(direct=plan["direct"], figures=plan["figures"])
    validate_input_binding(record)
    return Admission(canonical_bytes(record), plan["paths"])
