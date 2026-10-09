"""Проверки прямой диагностики на синтетических полях и эталонных квадратурах ядра."""

from tests.experiments.observation_sensitivity.fixtures.direct_records import direct_template as template
from copy import deepcopy

import pytest

from adrkit.config.validation import digest, canonical_bytes
from experiments.observation_sensitivity import direct
from experiments.observation_sensitivity.analysis import prediction_error as mod
from tests.experiments.observation_sensitivity.fixtures.paired_records import seal, pin


def cli_args(tmp_path, output=None):
    project = tmp_path / 'project'
    project.mkdir(exist_ok=True)
    return ['--run', str(project / 'runs/test'),
            '--freeze-sha256', 'a'*64, '--output', str(output or tmp_path / 'new/report.json')]


@pytest.mark.parametrize('stage', ['loader', 'paired', 'direct', 'covariance'])
def test_diagnostics_cli_rejected_stage_stops_without_export(tmp_path, monkeypatch, stage):
    from experiments.observation_sensitivity.analysis import read_results as loader
    from experiments.observation_sensitivity.analysis import paired_effects as paired
    from experiments.observation_sensitivity.analysis import covariance as covariance
    calls = []
    from tests.experiments.observation_sensitivity.fixtures.paired_records import make_records
    sentinel = dict(freeze=make_records()[0], direct={}, groups={}, raw_file_sha256={})
    def step(name, result):
        def run(*args, **kwargs):
            calls.append(name)
            assert kwargs['expected_freeze_sha256'] == 'a'*64
            if name == stage:
                raise ValueError('fixture: rejected '+name)
            return result
        return run
    for obj, attr, name, result in ((loader, 'load_records', 'loader', sentinel),
            (paired, 'summarize_records', 'paired', {}),
            (mod, 'analyze_direct', 'direct', {}),
            (covariance, 'summarize_covariances', 'covariance', {})):
        monkeypatch.setattr(obj, attr, step(name, result))
    with pytest.raises(ValueError, match='fixture: rejected '+stage):
        mod.main(cli_args(tmp_path))
    assert calls == ['loader', 'paired', 'direct', 'covariance'][:calls.index(stage)+1]
    assert not (tmp_path / 'new').exists()


def test_diagnostics_cli_rejects_output_without_writing(tmp_path, monkeypatch):
    from experiments.observation_sensitivity.analysis import read_results as loader
    from tests.experiments.observation_sensitivity.fixtures.paired_records import make_records
    monkeypatch.setattr(loader, 'load_records', lambda *a, **kw: dict(freeze=make_records()[0]))
    target = tmp_path / 'existing.json'
    target.write_text('retain', encoding='utf-8')
    for path in (target, tmp_path / 'project/runs/test/new.json'):
        with pytest.raises(loader.SnapshotError, match='new output'):
            mod.main(cli_args(tmp_path, path))
    assert target.read_text(encoding='utf-8') == 'retain'


def test_diagnostics_cli_race_cannot_overwrite_file(tmp_path, monkeypatch):
    from experiments.observation_sensitivity.analysis import read_results as loader
    from experiments.observation_sensitivity.analysis import paired_effects as paired
    from experiments.observation_sensitivity.analysis import covariance as covariance
    from tests.experiments.observation_sensitivity.fixtures.paired_records import make_records
    target = tmp_path / 'race.json'
    monkeypatch.setattr(loader, 'load_records', lambda *a, **kw:
                        dict(freeze=make_records()[0], direct={}, groups={}, raw_file_sha256={}))
    monkeypatch.setattr(paired, 'summarize_records', lambda *a, **kw: {})
    monkeypatch.setattr(mod, 'analyze_direct', lambda *a, **kw: {})
    def raced(*args, **kwargs):
        target.write_text('another writer', encoding='utf-8')
        return {}
    monkeypatch.setattr(covariance, 'summarize_covariances', raced)
    with pytest.raises(FileExistsError):
        mod.main(cli_args(tmp_path, target))
    assert target.read_text(encoding='utf-8') == 'another writer'


@pytest.fixture
def records(template):
    return deepcopy(template)


def analyze(records):
    return mod.analyze_direct(*records, expected_freeze_sha256=pin(records[0]))


def test_producer_compatibility_counts_units_and_owned_output(records):
    before = canonical_bytes(list(records))
    result = analyze(records)
    assert 'costs' not in result
    assert len(result['fields']) == 12 and len(result['quadrature']) == 96
    assert len(result['projection_metrics']) == 120 and len(result['domain_comparisons']) == 30
    assert len(result['changes_on_same_field']) == 96 and result['domain_sensitive_count'] == 0
    assert result['domain_comparisons'][0]['primary36']['signal']['rms'] == pytest.approx(.01)
    assert result['domain_comparisons'][0]['dense288']['log_width_derivative']['rms'] == pytest.approx(.02)
    result['quadrature'][0]['raw_moments']['raw_mass'] = 0.
    assert canonical_bytes(list(records)) == before


@pytest.mark.parametrize('kind', ['hash', 'shape', 'operator', 'clock', 'mask', 'threshold', 'units',
                                  'quad_mass', 'quad_conditional', 'quad_error',
                                 'quad_spacing', 'domain_statistic', 'domain_flag', 'derivative', 'residual', 'shared_residual'])
def test_self_resealed_inconsistent_diagnostics_are_rejected(records, kind):
    _, journal = records
    report = journal['result']
    row = report['fields']['PG10']['D0']
    projection = row['projections']['G1_snapshot']
    grid = report['quadrature']['D0']['G1'][0]['quadrature'][0]
    if kind == 'hash': projection['signal'][0] += .1
    elif kind == 'shape': projection['signal'].pop()
    elif kind == 'operator': projection['space_sha256'] = '0'*64
    elif kind == 'clock': projection['temporal_shape'] = [72, 338]
    elif kind == 'mask': report['primary_rows'][0] += 1
    elif kind == 'threshold': report['thresholds']['rms'] = .2
    elif kind == 'units': report['units']['signal'] = 'kg/s'
    elif kind == 'quad_mass': grid['raw_mass'] = -1.
    elif kind == 'quad_conditional': grid['conditional']['centroid_shift_km'][0] += .01
    elif kind == 'quad_error': grid['difference_from_finite_domain_reference']['mass_absolute'] += .01
    elif kind == 'quad_spacing': grid['spacing_km'] = .3
    elif kind == 'domain_statistic': report['comparisons']['PG10']['G1_snapshot']['primary36']['signal']['rms'] += .1
    elif kind == 'domain_flag': report['comparisons']['PG10']['G1_snapshot']['status'] = 'domain_sensitive'
    elif kind == 'derivative': projection['log_width_derivative'][0] += .1
    elif kind == 'residual': projection['residual_within_tolerance'] = False
    elif kind == 'shared_residual': projection['residual'] = 2e-12
    seal(journal)
    with pytest.raises(mod.DirectAnalysisError):
        analyze(records)


def test_domain_thresholds_apply_to_dense_and_primary_not_only_chosen_subset(records):
    _, journal = records
    report = journal['result']
    row = report['fields']['PG10']['D1']['projections']['G1_snapshot']
    row['signal'][0] += 1.  # tick0 не входит в primary36, но сохраняется в dense288.
    row['signal_sha256'] = mod.array_sha(row['signal'])
    report['comparisons'] = direct._comparisons(report['fields'], mod.default_direct_spec(report['direct_spec']))
    report['domain_sensitive_count'] = 1
    seal(journal)
    result = analyze(records)
    contrast = result['domain_comparisons'][0]
    assert contrast['primary36']['signal']['within_thresholds']
    assert not contrast['dense288']['signal']['within_thresholds']
    assert contrast['status'] == 'domain_sensitive'


@pytest.mark.parametrize('status', ['residual_rejected', 'unavailable'])
def test_completed_journal_keeps_terminal_failed_field_without_fake_pass(records, status):
    _, journal = records
    report = journal['result']
    row = report['fields']['PG10']['D0']
    row['status'] = status
    if status == 'residual_rejected':
        for projection in row['projections'].values():
            projection.update(residual=1e-6, residual_within_tolerance=False)
        row['residuals_accepted'] = False
    else:
        row['projections'] = {}
        row.update(error_type='RuntimeError', error='artificial failure after one completed field')
    report.update(status='incomplete', complete_fields=11)
    report['comparisons'] = direct._comparisons(report['fields'], mod.default_direct_spec(report['direct_spec']))
    seal(journal)
    result = analyze(records)
    assert result['fields'][0]['status'] == status
    assert all(r['status'] == 'unavailable' for r in result['domain_comparisons'][:5])
    assert len(result['fields']) == 12


def test_zero_signal_is_reported_without_normalization(records):
    _, journal = records
    report = journal['result']
    for domain in ('D0', 'D1'):
        projection = report['fields']['PG10'][domain]['projections']['G1_snapshot']
        projection['signal'] = [0.]*288
        projection['signal_sha256'] = mod.array_sha(projection['signal'])
    report['comparisons'] = direct._comparisons(report['fields'], mod.default_direct_spec(report['direct_spec']))
    seal(journal)
    result = analyze(records)
    target = [r for r in result['projection_metrics'] if r['source'] == 'PG10' and r['observation'] == 'G1_snapshot']
    assert len(target) == 4 and all(r['signal_is_zero'] for r in target)
    assert all('derivative_rms_over_signal_rms' not in r for r in target)


def test_unavailable_field_keeps_partial_projections_without_claiming_success(records):
    _, journal = records
    report = journal['result']
    row = report['fields']['PG10']['D0']
    row.update(status='unavailable', error_type='RuntimeError', error='failure after first projection')
    row['projections'] = {'G1_snapshot': row['projections']['G1_snapshot']}
    row.pop('residuals_accepted')
    report.update(status='incomplete', complete_fields=11)
    report['comparisons'] = direct._comparisons(report['fields'], mod.default_direct_spec(report['direct_spec']))
    seal(journal)
    result = analyze(records)
    assert result['fields'][0]['status'] == 'unavailable'
    metrics = [r for r in result['projection_metrics'] if r['source'] == 'PG10' and r['domain'] == 'D0']
    assert len(metrics) == 2 and all(r['diagnostic_only'] for r in metrics)
    assert all(r['status'] == 'unavailable' for r in result['domain_comparisons'][:5])


def test_derivative_threshold_is_not_invented(records):
    _, journal = records
    report = journal['result']
    p = report['fields']['PG10']['D1']['projections']['G1_snapshot']
    p['log_width_derivative'] = [v+1000 for v in p['log_width_derivative']]
    p['derivative_sha256'] = mod.array_sha(p['log_width_derivative'])
    report['comparisons'] = direct._comparisons(report['fields'], mod.default_direct_spec(report['direct_spec']))
    seal(journal)
    assert analyze(records)['domain_sensitive_count'] == 0


def test_pin_required_and_no_pde_or_file_calls(records, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(Path, 'open', lambda *a, **kw: pytest.fail('scientific IO'))
    monkeypatch.setattr(direct, 'collect_direct', lambda *a, **kw: pytest.fail('new direct fields'))
    assert analyze(records)['status'] == 'complete'
    with pytest.raises(mod.DirectAnalysisError):
        mod.analyze_direct(*records, expected_freeze_sha256='0'*64)



def test_direct_analysis_uses_the_admitted_source_scope_with_real_operator_bindings(records):
    from tests.experiments.observation_sensitivity.fixtures.paired_records import select_admission
    ids = ["spatial_G1_matched/PG10/r1/L2", "spatial_G1_matched/PG10/r1/H1",
           "spatial_G2_matched/PG10/r1/L2", "spatial_G2_matched/PG10/r1/H1"]
    freeze, journal = records
    admitted = select_admission(freeze["admission"], ids, figures=[])
    freeze.update(version=4, admission=admitted, admission_sha256=digest(admitted))
    seal(freeze)
    journal.update(version=4, admission_sha256=digest(admitted), freeze_sha256=freeze["content_sha256"])
    report = journal["result"]
    for name in ("sources", "fields", "comparisons"): report[name] = {"PG10": report[name]["PG10"]}
    report.update(expected_fields=2, complete_fields=2, domain_sensitive_count=0)
    seal(journal)
    before = canonical_bytes(list(records))
    result = analyze(records)
    assert canonical_bytes(list(records)) == before
    assert len(result["fields"]) == 2 and len(result["projection_metrics"]) == 20
    assert len(result["domain_comparisons"]) == 5 and len(result["quadrature"]) == 96
    assert {r["source"] for r in result["fields"]} == {"PG10"}
    report["expected_fields"] = 12
    seal(journal)
    with pytest.raises(mod.DirectAnalysisError): analyze(records)


def test_direct_producer_and_analyzer_share_the_same_observation_descriptors():
    assert tuple(mod.OBSERVATIONS) == direct.OBSERVATION_IDS
    assert tuple(mod.OBSERVATIONS.values()) == direct.DEFAULT_OBSERVATIONS
    assert all(mod.OBSERVATIONS[label] is observation
               for label, observation in zip(direct.OBSERVATION_IDS, direct.DEFAULT_OBSERVATIONS))
