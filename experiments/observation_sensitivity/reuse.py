"""Повторное использование завершённых оценок E05 в плане E06."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import re

from adrkit.config.validation import canonical_bytes, digest, strict_json
from .design import PathSpec, build_design
from .lifecycle import CheckpointError, read_terminal_v2


_HASH_FIELDS = ("fit_y_sha256", "selection_y_sha256", "selected_covariance_sha256",
                "jacobian_sha256", "offset_sha256")
_MATCH_FIELDS = (*_HASH_FIELDS, "panel_records", "calibration")
_PATH_FIELDS = {"condition", "penalty", "alpha_reference", "provenance", "candidates",
    "finalized", "calibration_failure", "accepted_count", "candidate_count",
    "complete_path", "selected_exponent", "lcurve_diagnostic", "tuning_unresolved",
    "final_certificate", "procedure_accepted"}
_PROVENANCE_FIELDS = {*_MATCH_FIELDS, "design", "observation_diagnostic", "truth_forward_residual"}
# Для повторного использования должны совпасть все условия PathSpec.
_REGISTERED_REUSE = {path.id: path for path in build_design().paths if path.reuse is not None}


def _equal(left, right):
    return canonical_bytes(left) == canonical_bytes(right)


def _bindings(value):
    canonical_bytes(value)
    if (type(value) is not dict or type(value.get("source")) is not str
            or not value["source"] or type(value.get("replicate")) is not int
            or value["replicate"] < 1):
        raise CheckpointError("expected complete bindings with source and integer replicate")


def _expected_paths(value):
    if (type(value) not in (list, tuple) or not value
            or any(type(item) is not str or not item for item in value)
            or len(set(value)) != len(value)):
        raise CheckpointError("expected_paths must be a nonempty ordered list or tuple of unique nonempty strings")
    return list(value)


def _permitted(provenance):
    canonical_bytes(provenance)
    if type(provenance) is not dict or set(_MATCH_FIELDS) - set(provenance):
        raise CheckpointError("all permitted input hashes, panel_records and calibration are required")
    if set(provenance) - _PROVENANCE_FIELDS:
        raise CheckpointError("unexpected provenance field; test/source scores are not estimator inputs")
    for field in _HASH_FIELDS:
        if type(provenance[field]) is not str or re.fullmatch(r"[0-9a-f]{64}", provenance[field]) is None:
            raise CheckpointError(f"invalid input SHA256: {field}")
    if (type(provenance["panel_records"]) is not dict
            or set(provenance["panel_records"]) != {"fit", "selection"}
            or any(type(row) is not dict or not row for row in provenance["panel_records"].values())
            or type(provenance["calibration"]) is not dict or not provenance["calibration"]):
        raise CheckpointError("only nonempty fit/selection panels and calibration provenance are permitted")


@dataclass(frozen=True, init=False)
class VerifiedBaseline:
    """Снимок завершённых оценок E05 для повторного использования в E06.

    Создавайте снимок через from_terminal после проверки ожидаемых привязок,
    порядка последовательностей и SHA-256 файла. Прямой вызов конструктора
    запрещён; проверочные ошибки E05 в снимок не входят.
    """

    _payload: bytes
    _file_sha256: str

    def __init__(self):
        raise TypeError("use VerifiedBaseline.from_terminal with expected bindings, paths and file SHA256")

    @classmethod
    def from_terminal(cls, path, *, expected_bindings, expected_paths, expected_file_sha256):
        """Прочитать исходную группу E05 и проверить её ожидаемые привязки.

        Parameters
        ----------
        path : str or pathlib.Path
            Путь к завершённой группе E05.
        expected_bindings : dict
            Полные ожидаемые привязки исходной группы, включая источник и
            реализацию.
        expected_paths : list or tuple of str
            Полный ожидаемый порядок уникальных идентификаторов, включая
            неиспользуемые оценки.
        expected_file_sha256 : str
            Ожидаемый SHA-256 исходных байтов файла.

        Returns
        -------
        VerifiedBaseline
            Снимок разрешённых оценок main и temporal_average без проверочных
            ошибок и посторонних отдельных оценок.
        """
        _bindings(expected_bindings)
        expected_paths = _expected_paths(expected_paths)
        record, file_sha256 = read_terminal_v2(path,
            expected_file_sha256=expected_file_sha256, include_file_hash=True)
        if not _equal(record["bindings"], expected_bindings):
            raise CheckpointError("v2 full bindings differ from the admitted expected bindings")
        if record["expected_paths"] != expected_paths:
            raise CheckpointError("v2 expected_paths differ from the complete admitted ordered list")
        # Сохраняются только потенциально используемые оценки; ошибки и отдельные несвязанные оценки
        # не входят в этот объект.
        paths = {pid: row for pid, row in record["paths"].items()
                 if pid in ("main/L2", "main/H1", "temporal_average/L2", "temporal_average/H1")}
        payload = dict(bindings=record["bindings"], selection_seal=record["selection_seal"], paths=paths)
        result = object.__new__(cls)
        object.__setattr__(result, "_payload", canonical_bytes(payload))
        object.__setattr__(result, "_file_sha256", file_sha256)
        return result

    def estimate(self, path: PathSpec, *, provenance: dict, alpha_reference: float | None):
        """Вернуть копию сохранённой оценки для совпадающих входов E06.

        Несовпадение зарегистрированных условий или входов вызывает
        CheckpointError. Правила сравнения приведены в README; пересчёт вместо
        отсутствующей оценки не выполняется.

        Parameters
        ----------
        path : PathSpec
            Условия источника, сетки, наблюдений и регуляризации.
        provenance : dict
            Происхождение разрешённых входов оценщика; при отказе калибровки —
            только точно воспроизведённая причина.
        alpha_reference : float or None
            Точно совпадающий положительный масштаб регуляризации; None только
            для терминального отказа калибровки.

        Returns
        -------
        dict
            Отдельная копия исходной оценки и запись reuse с её происхождением.
        """
        if not isinstance(path, PathSpec) or path.reuse is None:
            raise CheckpointError("an explicit reusable PathSpec is required")
        if path.id not in _REGISTERED_REUSE or path != _REGISTERED_REUSE[path.id]:
            raise CheckpointError("PathSpec differs from the registered finite reuse design")
        snapshot = strict_json(self._payload)
        reference = path.reuse
        binding = snapshot["bindings"]
        if (reference.source != binding["source"] or reference.replicate != binding["replicate"]
                or (path.source, path.replicate, path.penalty) !=
                (reference.source, reference.replicate, reference.penalty)):
            raise CheckpointError("v2 source/replicate/penalty locator differs from PathSpec")
        original_id = f"{reference.condition}/{reference.penalty}"
        if original_id not in snapshot["paths"]:
            raise CheckpointError("the exact referenced v2 path is absent; no recomputation fallback")
        original = snapshot["paths"][original_id]
        if (original.get("condition") != reference.condition or original.get("penalty") != reference.penalty
                or original.get("finalized") is not True):
            raise CheckpointError("v2 path body differs from its condition/penalty locator")
        if set(original) - _PATH_FIELDS:
            raise CheckpointError("unexpected v2 path fields; only estimates and permitted provenance can be reused")
        canonical_bytes(provenance)
        if "calibration_failure" in original:
            if (alpha_reference is not None or type(provenance) is not dict
                    or set(provenance) != {"calibration_failure"}
                    or provenance["calibration_failure"] != original["calibration_failure"]):
                raise CheckpointError("terminal calibration failure was not independently reproduced exactly")
            matched = ["bindings", "locator", "calibration_failure"]
            comparison = "terminal calibration failure; estimator arrays/alpha unavailable, not byte-compared"
        else:
            _permitted(provenance)
            _permitted(original.get("provenance"))
            if (type(alpha_reference) not in (int, float) or not math.isfinite(alpha_reference)
                    or alpha_reference <= 0 or not _equal(original.get("alpha_reference"), alpha_reference)):
                raise CheckpointError("exact alpha_reference mismatch")
            for field in _MATCH_FIELDS:
                if not _equal(original["provenance"][field], provenance[field]):
                    raise CheckpointError(f"permitted input mismatch: {field}")
            if "design" in provenance or "design" in original["provenance"]:
                if ("design" not in provenance or "design" not in original["provenance"]
                        or not _equal(provenance["design"], original["provenance"]["design"])):
                    raise CheckpointError("permitted input mismatch: design")
            matched = ["bindings", "locator", "alpha_reference", *_MATCH_FIELDS]
            if "design" in provenance:
                matched.append("design")
            comparison = "exact canonical input hashes, panels, calibration and alpha reference"
        result = deepcopy(original)
        result["reuse"] = dict(study="research_validation_v2", original_path_id=original_id,
            original_path_sha256=digest(original), source_file_sha256=self._file_sha256,
            selection_seal=snapshot["selection_seal"], bindings=deepcopy(binding),
            new_path_id=path.id, matched_fields=matched, comparison=comparison)
        return result
