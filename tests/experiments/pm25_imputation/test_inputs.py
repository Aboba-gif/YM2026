"""Чтение исходных байтов и временные границы протокола PM₂.₅."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.pm25_imputation import export, models, data


def raw_csvs(root):
    originals = {}
    for folder in data.FOLDERS.values():
        (root / folder).mkdir(parents=True)
        for year in range(2019, 2023):
            path = root / folder / f'{year}.csv'
            content = f'date;time;t;p;h;ws;wd;pm25\n{year}-01-01;00:00;1;2;3;4;5;1\n'
            if year == 2019:
                content += f'{year}-01-01;00:20;1;2;3;4;5;\n'
            content_bytes = content.encode('utf-8')
            path.write_bytes(content_bytes)
            originals[path] = content_bytes
    return originals


def assert_loaded_sources(index, frames, manifest, originals):
    assert len(manifest) == 16
    for record in manifest:
        path = next(path for path in originals if path.as_posix().endswith('/' + record['file']))
        content = originals[path]
        assert record['bytes'] == len(content)
        assert record['sha256'] == hashlib.sha256(content).hexdigest()
        assert record['rows'] == (2 if path.name == '2019.csv' else 1)
    for frame in frames.values():
        assert frame.index.equals(index)
        assert frame.row_present.sum() == 5
        assert frame.pm25.notna().sum() == 4
        assert frame.loc['2019-01-01 00:20', 'row_present']
        assert pd.isna(frame.loc['2019-01-01 00:20', 'pm25'])
        assert not frame.loc['2019-01-01 00:40', 'row_present']
    matrix = data.pm25_matrix(frames)
    assert matrix.shape == (len(index), len(data.SITES))
    np.testing.assert_array_equal(matrix[0], np.ones(4))


def test_load_network_binds_values_counts_and_hashes_to_read_bytes(tmp_path):
    originals = raw_csvs(tmp_path)
    assert_loaded_sources(*data.load_network(tmp_path), originals)


@pytest.mark.parametrize('after_parse', [1, 16])
def test_csv_changes_during_real_parse_do_not_change_parsed_manifest(tmp_path, monkeypatch, after_parse):
    originals = raw_csvs(tmp_path)
    changed_path = tmp_path / 'sev/2019.csv'
    old = originals[changed_path]
    changed = old.replace(b';5;1\n', b';5;19.25\n') + b'2019-01-01;00:40;1;2;3;4;5;6\n'
    read_csv = pd.read_csv
    calls = []

    def mutate_after_parse(*args, **kwargs):
        frame = read_csv(*args, **kwargs)
        calls.append(args[0])
        if len(calls) == after_parse:
            changed_path.write_bytes(changed)
        return frame

    monkeypatch.setattr(pd, 'read_csv', mutate_after_parse)
    index, frames, manifest = data.load_network(tmp_path)
    assert len(calls) == 16
    assert changed_path.read_bytes() == changed
    assert_loaded_sources(index, frames, manifest, originals)
    record = next(item for item in manifest if item['file'] == 'sev/2019.csv')
    assert record['bytes'] != changed_path.stat().st_size
    assert record['sha256'] != data.file_sha(changed_path)


def panel_protocol(start='2042-01-01', count=1800):
    index = pd.date_range(start, periods=count, freq='20min')
    t = np.arange(count)
    values = np.column_stack([3 + j + np.sin(t / (21 + j))**2 for j in range(4)])
    values[1000:1003, 0] = np.nan
    cfg = json.loads(export.DEFAULT_PROTOCOL.read_bytes())
    cfg.update(period=[str(index[0]), str(index[-1] + pd.Timedelta('20min'))],
               train_end=str(index[600]), frequency='20min',
               splits={'selection': [str(index[600]), str(index[1200])],
                       'test': [str(index[1200]), str(index[-1] + pd.Timedelta('20min'))]})
    cfg['models'].update(common_window_steps=32, common_stride=32,
                         common_lengths=[1, 3], common_max_windows=2)
    cfg['models']['hgb'].update(max_iter=2, min_samples_leaf=3)
    cfg['bank'].update(internal_lengths=[1], boundary_lengths=[1],
                       requested={'selection': 1, 'test': 1},
                       boundary_requested={'selection': 1, 'test': 1})
    cfg['selection']['minimum_blocks'] = 1
    frames = {site: pd.DataFrame({'pm25': values[:, j], 'row_present': True}, index=index)
              for j, site in enumerate(data.SITES)}
    return index, frames, cfg


INVALID_PERIODS = [
    ('train_selection_overlap', {'train_end': '2042-01-10'}),
    ('selection_test_overlap', {'selection': ['2042-01-09 08:00', '2042-01-18']}),
    ('empty_train', {'train_end': '2042-01-01'}),
    ('empty_selection', {'selection': ['2042-01-09 08:00', '2042-01-09 08:00']}),
    ('empty_test', {'test': ['2042-01-17 16:00', '2042-01-17 16:00']}),
    ('reversed_selection', {'selection': ['2042-01-10', '2042-01-09 08:00']}),
    ('test_before_period', {'test': ['2041-12-31', '2042-01-02']}),
    ('test_after_period', {'test': ['2042-01-17 16:00', '2042-02-01']}),
    ('unaligned_train', {'train_end': '2042-01-09 08:01'}),
    ('unaligned_selection', {'selection': ['2042-01-09 08:01', '2042-01-17 16:00']}),
    ('unaligned_test', {'test': ['2042-01-17 16:00', '2042-01-25 23:59']}),
    ('timezone_boundary', {'train_end': '2042-01-09 08:00+00:00'}),
    ('nat_boundary', {'train_end': 'NaT'}),
    ('invalid_date', {'train_end': 'not a date'}),
    ('non_string_boundary', {'train_end': None}),
    ('malformed_selection', {'selection': ['2042-01-09 08:00']}),
    ('malformed_test', {'test': '2042-01-17 16:00'}),
    ('period_start_mismatch', {'period': ['2041-01-01', '2042-01-26']}),
    ('period_end_mismatch', {'period': ['2042-01-01', '2043-01-01']}),
    ('frequency_mismatch', {'frequency': '10min'}),
    ('negative_frequency', {'frequency': '-20min'}),
    ('zero_frequency', {'frequency': '0min'}),
    ('invalid_frequency', {'frequency': 'bad'}),
    ('nat_frequency', {'frequency': 'NaT'}),
]


def changed_protocol(cfg, changes):
    cfg = deepcopy(cfg)
    for name, value in changes.items():
        if name in {'selection', 'test'}:
            cfg['splits'][name] = value
        else:
            cfg[name] = value
    return cfg


@pytest.mark.parametrize('name,changes', INVALID_PERIODS, ids=[row[0] for row in INVALID_PERIODS])
@pytest.mark.parametrize('verify_only', [False, True], ids=['export', 'verify'])
def test_invalid_periods_fail_before_bank_fit_or_output(tmp_path, monkeypatch, name, changes, verify_only):
    index, frames, cfg = panel_protocol()
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps(changed_protocol(cfg, changes)), encoding='utf-8')
    monkeypatch.setattr(export, 'load_network', lambda root: (index, frames, []))

    def forbidden(*args, **kwargs):
        pytest.fail('Invalid temporal protocol reached numerical execution')

    monkeypatch.setattr(export, 'make_bank', forbidden)
    monkeypatch.setattr(models.CoreModels, 'fit', forbidden)
    monkeypatch.setattr(export, 'verify_export', forbidden)
    with pytest.raises(ValueError):
        export.run(tmp_path / 'raw', output_dir=tmp_path / 'output',
                   results_dir=tmp_path / 'metrics', protocol_path=protocol,
                   verify_only=verify_only)
    assert not (tmp_path / 'output').exists()
    assert not (tmp_path / 'metrics').exists()


@pytest.mark.parametrize('start', ['1902-01-01', '2042-01-01'])
@pytest.mark.parametrize('gaps', [False, True], ids=['touching', 'gaps'])
def test_half_open_periods_allow_touching_boundaries_and_gaps(start, gaps):
    index, _, cfg = panel_protocol(start)
    if gaps:
        cfg['splits']['selection'][0] = str(index[601])
        cfg['splits']['test'][0] = str(index[1201])
    export._validate_periods(index, cfg)
    train = index < cfg['train_end']
    selection = (index >= cfg['splits']['selection'][0]) & (index < cfg['splits']['selection'][1])
    test = (index >= cfg['splits']['test'][0]) & (index < cfg['splits']['test'][1])
    assert train.any() and selection.any() and test.any()
    assert not (train & selection).any() and not (selection & test).any()
    assert not train[600] and not selection[1200] and test[1201]


def test_single_sample_nonempty_split_is_not_confused_with_window_eligibility():
    index, _, cfg = panel_protocol()
    cfg['splits']['selection'][1] = str(index[601])
    export._validate_periods(index, cfg)


def short_export_inputs(tmp_path, monkeypatch):
    index, frames, cfg = panel_protocol()
    root = tmp_path / 'raw'
    root.mkdir()
    raw_path = root / 'input.csv'
    raw_bytes = b'original observations\n'
    raw_path.write_bytes(raw_bytes)
    manifest = [dict(file='input.csv', bytes=len(raw_bytes), rows=len(index),
                     sha256=hashlib.sha256(raw_bytes).hexdigest())]
    monkeypatch.setattr(export, 'load_network', lambda root: (index, frames, manifest))
    protocol = tmp_path / 'protocol.json'
    protocol_bytes = json.dumps(cfg, ensure_ascii=False).encode('utf-8')
    protocol.write_bytes(protocol_bytes)
    return index, frames, cfg, root, raw_path, protocol, protocol_bytes


def test_real_short_export_uses_original_protocol_bytes(tmp_path, monkeypatch):
    index, frames, cfg, root, raw_path, protocol, protocol_bytes = short_export_inputs(tmp_path, monkeypatch)
    result = export.run(root, output_dir=tmp_path / 'output', results_dir=tmp_path / 'metrics', protocol_path=protocol)
    assert result['protocol_sha256'] == hashlib.sha256(protocol_bytes).hexdigest()
    assert result['raw'][0]['sha256'] == data.file_sha(raw_path)
    assert result['verification']['columns'] == 33 and result['verification']['rows'] == len(index)
    fits = json.loads((tmp_path / 'metrics/validation_fits.json').read_bytes())
    assert fits['Severny']['training_last_observed_target'] == str(index[599])
    final = pd.read_csv(tmp_path / 'output/pm25_final_2019_2022.csv', float_precision='round_trip')
    for site in data.SITES:
        raw = frames[site].pm25.to_numpy()
        observed = np.isfinite(raw)
        np.testing.assert_array_equal(final[f'{site}_pm25_observed'], raw)
        np.testing.assert_array_equal(final[f'{site}_pm25_filled'][observed], raw[observed])


def test_protocol_mutation_during_read_is_parsed_from_old_bytes_then_rejected(tmp_path, monkeypatch):
    index, frames, cfg, root, raw_path, protocol, protocol_bytes = short_export_inputs(tmp_path, monkeypatch)
    modified = deepcopy(cfg)
    modified['train_end'] = cfg['period'][1]
    changed_bytes = json.dumps(modified).encode('utf-8')
    read_bytes = Path.read_bytes
    calls = []

    def changing_read(path):
        content = read_bytes(path)
        if path == protocol:
            calls.append(content)
            if len(calls) == 1:
                path.write_bytes(changed_bytes)
        return content

    monkeypatch.setattr(Path, 'read_bytes', changing_read)
    with pytest.raises(ValueError, match='Protocol or computation source changed'):
        export.run(root, output_dir=tmp_path / 'output', results_dir=tmp_path / 'metrics', protocol_path=protocol)
    assert calls == [protocol_bytes, changed_bytes]
    fits = json.loads((tmp_path / 'metrics/validation_fits.json').read_bytes())
    assert fits['Severny']['training_last_observed_target'] == str(index[599])
    assert (tmp_path / 'output/pm25_final_2019_2022.csv').exists()
    assert not (tmp_path / 'output/manifest.json').exists()


def test_protocol_changes_during_real_json_parse_are_not_bound_to_a_later_hash(tmp_path, monkeypatch):
    index, frames, cfg, root, raw_path, protocol, protocol_bytes = short_export_inputs(tmp_path, monkeypatch)
    changed = deepcopy(cfg)
    changed['train_end'] = cfg['period'][1]
    changed_bytes = json.dumps(changed).encode('utf-8')
    loads = json.loads
    parsed = []

    def change_after_parse(content, *args, **kwargs):
        result = loads(content, *args, **kwargs)
        if content in (protocol_bytes, protocol_bytes.decode('utf-8')):
            parsed.append(result)
            protocol.write_bytes(changed_bytes)
        return result

    monkeypatch.setattr(json, 'loads', change_after_parse)
    with pytest.raises(ValueError, match='Protocol or computation source changed'):
        export.run(root, output_dir=tmp_path / 'output', results_dir=tmp_path / 'metrics', protocol_path=protocol)
    assert len(parsed) == 1 and parsed[0]['train_end'] == cfg['train_end']
    assert protocol.read_bytes() == changed_bytes
    fits = loads((tmp_path / 'metrics/validation_fits.json').read_bytes())
    assert fits['Severny']['training_last_observed_target'] == str(index[599])
    assert not (tmp_path / 'output/manifest.json').exists()


def test_final_raw_guard_still_rejects_changed_input(tmp_path, monkeypatch):
    index, frames, cfg, root, raw_path, protocol, protocol_bytes = short_export_inputs(tmp_path, monkeypatch)
    build_export = export.build_export

    def change_after_calculation(*args, **kwargs):
        result = build_export(*args, **kwargs)
        raw_path.write_bytes(b'changed observations\n')
        return result

    monkeypatch.setattr(export, 'build_export', change_after_calculation)
    with pytest.raises(ValueError, match='Original CSV changed during export'):
        export.run(root, output_dir=tmp_path / 'output', results_dir=tmp_path / 'metrics', protocol_path=protocol)
    assert (tmp_path / 'output/pm25_final_2019_2022.csv').exists()
    assert not (tmp_path / 'output/manifest.json').exists()
