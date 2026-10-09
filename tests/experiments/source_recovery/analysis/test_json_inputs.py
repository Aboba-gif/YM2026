"""Строгое чтение научных записей и сохранение контрольных сумм."""

import hashlib
import json
import math

import pytest

from experiments.source_recovery.analysis import plots, summary


@pytest.mark.parametrize("reader", [plots.load_json, summary.snapshot])
@pytest.mark.parametrize("contents", [
    b'{"value":1e1000}', b'{"values":[-1e1000]}',
    b'{"value":NaN}', b'{"value":Infinity}',
    b'{"value":1,"value":2}', b'{"nested":{"a":1,"a":2}}',
])
def test_readers_reject_nonfinite_numbers_and_duplicate_keys(tmp_path, reader, contents):
    path = tmp_path / "record.json"
    path.write_bytes(contents)
    with pytest.raises(ValueError):
        reader(path)


def test_snapshot_hashes_original_bytes_without_reformatting(tmp_path):
    raw = '{ "значение" : -0.0, "nested": [2.5, null] }\n'.encode("utf-8")
    path = tmp_path / "record.json"
    path.write_bytes(raw)
    record, checksum = summary.snapshot(path)
    assert record == {"значение": -0.0, "nested": [2.5, None]}
    assert math.copysign(1.0, record["значение"]) == -1.0
    assert checksum == hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw


def test_selection_hash_preserves_existing_normalization():
    canonical = json.dumps({"a": [0.0, 1], "б": "текст"}, sort_keys=True,
                           separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    assert summary.selection_hash({"б": "текст", "a": [-0.0, 1]}) == hashlib.sha256(canonical).hexdigest()
    assert summary.selection_hash({"a": 1}) != summary.selection_hash({"a": 1.0})
