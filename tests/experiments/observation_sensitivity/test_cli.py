"""Порядок сохранения, повторного чтения и расчёта через CLI."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.observation_sensitivity import cli
from experiments.observation_sensitivity.lifecycle import CheckpointError


class Admission:
    sha256 = 'a' * 64
    spec = {'grids': {'G0': {'bounds_km': [-1, 1, -1, 1], 'spacing_km': .5, 'steps': 2}}}

    def __init__(self, output):
        self.output = str(output)

    def to_dict(self):
        baseline = Path(self.output) / 'baseline'
        return dict(output=self.output, sources={name: {'sha256': 'b' * 64} for name in
            ('PG10', 'SB150', 'EC04', 'EC06', 'NEW-J2', 'NEW-S2')},
            baseline={'run_manifest_path': str(baseline / 'run.json'),
                      'group_files': {str(baseline / 'PG10/replicate_1.json'): 'd' * 64}})

    def paths_for(self, source, replicate):
        if source not in ('PG10', 'EC04') or replicate not in range(1, 5):
            raise ValueError('invalid group')
        return (SimpleNamespace(id=f'matched/{source}/r{replicate}/L2', reuse=True),)

    def expected_baseline_bindings(self, source, replicate):
        return dict(source=source, replicate=replicate, code_sha256='c' * 64)

    def expected_baseline_paths(self, source, replicate):
        return ('main/L2', 'main/H1', 'single/audit')


def forbidden(*args, **kwargs):
    pytest.fail('This operation must not run')


@pytest.fixture
def direct_env(tmp_path, monkeypatch):
    admission = Admission(tmp_path)
    state = dict(record=None, pending=False, starts=0, finishes=0, commits=0, calls=0)

    class Journal:
        def __init__(self, output, admitted, *, read_only=False):
            assert output == admission.output and admitted is admission

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        @property
        def record(self):
            return deepcopy(state['record'])

        @property
        def pending_exists(self):
            return state['pending']

        def start(self):
            assert state['record'] is None and not state['pending']
            state['starts'] += 1
            state['record'] = {'status': 'started'}

        def finish(self, result, *, partial=False):
            assert state['record']['status'] == 'started'
            state['finishes'] += 1
            state['record'] = {'status': 'partial' if partial else 'completed', 'result': result}
            return deepcopy(state['record'])

        def commit_pending(self):
            state['commits'] += 1
            return {'status': 'started'}

    def collect(spec, sources, *, direct=None, backend_factory):
        state['calls'] += 1
        state['direct_plan'] = deepcopy(direct)
        assert state['starts'] == 1 and state['record']['status'] == 'started'
        assert backend_factory is cli.ProductionBackend
        if state.get('error'):
            raise state['error']
        return {'status': 'complete', 'expected_fields': 12}

    monkeypatch.setattr(cli, 'DirectJournal', Journal)
    monkeypatch.setattr(cli, '_require_freeze', lambda _: None)
    monkeypatch.setattr(cli, '_sources', lambda _: {})
    monkeypatch.setattr(cli, 'collect_direct', collect)
    return admission, state


def test_direct_commits_start_before_any_computation(direct_env):
    admission, state = direct_env
    assert cli._direct(admission)['status'] == 'completed'
    assert (state['starts'], state['calls'], state['finishes']) == (1, 1, 1)


@pytest.mark.parametrize('status', ['completed', 'partial'])
def test_direct_terminal_reopen_never_computes(direct_env, status):
    admission, state = direct_env
    state['record'] = {'status': status, 'result': {'old': True}}
    assert cli._direct(admission) == state['record']
    assert state['calls'] == state['starts'] == 0


def test_direct_unresolved_start_forbids_automatic_rerun(direct_env):
    admission, state = direct_env
    state['record'] = {'status': 'started'}
    with pytest.raises(CheckpointError, match='unresolved'):
        cli._direct(admission)
    assert state['calls'] == 0


def test_pending_commit_never_starts_calculation(direct_env):
    admission, state = direct_env
    assert cli._direct(admission, commit_pending=True)['status'] == 'started'
    assert state['commits'] == 1 and state['calls'] == state['starts'] == 0


def test_unresolved_pending_never_returns_stale_result(direct_env):
    admission, state = direct_env
    state['pending'] = True
    state['record'] = {'status': 'completed', 'result': {'old': True}}
    with pytest.raises(CheckpointError, match='Prepared'):
        cli._direct(admission)
    assert state['calls'] == 0


def test_direct_contract_failure_preserves_partial_result(direct_env):
    admission, state = direct_env
    state['error'] = cli.DirectContractError('bad geometry', {'status': 'contract_failure', 'fields': {}})
    with pytest.raises(cli.DirectContractError):
        cli._direct(admission)
    assert state['record']['status'] == 'partial'
    assert state['record']['result'] == state['error'].partial_record
    assert state['calls'] == 1


@pytest.mark.parametrize('error', [MemoryError('allocation failed'), RuntimeError('solver interrupted')])
def test_interrupted_direct_preserves_unresolved_start(direct_env, error):
    admission, state = direct_env
    state['error'] = error
    with pytest.raises(type(error), match=str(error)):
        cli._direct(admission)
    assert state['record'] == {'status': 'started'} and state['finishes'] == 0


@pytest.mark.parametrize('status', [None, 'started', 'partial'])
def test_inverse_requires_terminal_direct_result(direct_env, status):
    admission, state = direct_env
    state['record'] = None if status is None else {'status': status}
    with pytest.raises(CheckpointError):
        cli._require_direct(admission)


def test_inverse_preserves_terminal_numerical_failures(direct_env):
    admission, state = direct_env
    fields = {s: {'D0': {'status': 'complete'}, 'D1': {'status': 'unavailable'}}
              for s in admission.to_dict()['sources']}
    state['record'] = {'status': 'completed', 'result': {'expected_fields': 12, 'fields': fields}}
    assert cli._require_direct(admission) == state['record']
    fields['PG10']['D1']['status'] = 'running'
    with pytest.raises(CheckpointError):
        cli._require_direct(admission)


def test_direct_pending_blocks_inverse(direct_env):
    admission, state = direct_env
    state['pending'] = True
    with pytest.raises(CheckpointError, match='prepared'):
        cli._require_direct(admission)


def test_admit_command_is_read_only(tmp_path, monkeypatch, capsys):
    admission = Admission(tmp_path)
    monkeypatch.setattr(cli, 'build_admission', lambda path: admission)
    for name in ('Freeze', 'DirectJournal', 'ProductionBackend'):
        monkeypatch.setattr(cli, name, forbidden)
    assert cli.main(['--config', 'static.json', 'admit']) == 0
    assert 'admit' in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('args', [
    ['group', '--source', 'SB150', '--replicate', '1'],
    ['group', '--source', 'PG10', '--replicate', '5'],
    ['group', '--source', 'PG10', '--replicate', '1', '--alpha', '1'],
    ['direct', '--retry'],
])
def test_cli_rejects_scientific_overrides_before_admission(monkeypatch, args):
    monkeypatch.setattr(cli, 'build_admission', forbidden)
    with pytest.raises(SystemExit):
        cli.main(['--config', 'ignored', *args])


@pytest.fixture
def group_env(tmp_path, monkeypatch):
    admission = Admission(tmp_path)
    state = dict(events=[], stage='estimating', checkpoint_failure=None, record=None)
    events = state['events']
    monkeypatch.setattr(cli, '_require_freeze', lambda a: events.append('freeze'))
    monkeypatch.setattr(cli, '_require_direct', lambda a: dict(status='completed', direct='sealed'))

    def sources(admitted):
        events.append('sources')
        return {'PG10': object()}

    class CP:
        def __init__(self, path, bindings, expected_paths, *, recover_pending):
            events.append('checkpoint')
            state.update(path=path, bindings=bindings, expected_paths=expected_paths, recover=recover_pending)
            self.record = state['record'] = dict(stage=state['stage'], paths={})

        def __enter__(self):
            if state['checkpoint_failure']:
                raise state['checkpoint_failure']
            return self

        def __exit__(self, *args):
            events.append('checkpoint_exit')

        def require_sealed(self):
            events.append('seal_checked')

    class Baseline:
        @staticmethod
        def from_terminal(path, *, expected_bindings, expected_paths, expected_file_sha256):
            events.append('baseline')
            state.update(baseline_path=path, baseline_bindings=expected_bindings,
                         baseline_expected=expected_paths, baseline_pin=expected_file_sha256)
            return 'verified-baseline'

    def backend(spec, source, *, source_id):
        events.append('backend')
        return SimpleNamespace(cache_info={})

    def run(spec, paths, checkpoint, numerical_backend, panels, baseline):
        events.append('run_group')
        assert baseline == 'verified-baseline'
        return dict(stage='scored', paths={})

    monkeypatch.setattr(cli, '_sources', sources)
    monkeypatch.setattr(cli, 'Checkpoint', CP)
    monkeypatch.setattr(cli, 'VerifiedBaseline', Baseline)
    monkeypatch.setattr(cli, 'ProductionBackend', backend)
    monkeypatch.setattr(cli, 'PanelFactory', lambda spec: object())
    monkeypatch.setattr(cli, 'run_group', run)
    return admission, state


def test_group_pins_complete_baseline_before_backend(group_env):
    admission, state = group_env
    assert cli._group(admission, 'PG10', 1)['stage'] == 'scored'
    events = state['events']
    assert events.index('checkpoint') < events.index('baseline') < events.index('backend') < events.index('run_group')
    assert state['baseline_expected'] == ('main/L2', 'main/H1', 'single/audit')
    assert state['baseline_bindings'] == admission.expected_baseline_bindings('PG10', 1)
    assert state['baseline_pin'] == 'd' * 64
    assert state['bindings']['admission_sha256'] == admission.sha256
    assert state['expected_paths'] == ['matched/PG10/r1/L2']
    assert events[-1] == 'checkpoint_exit'


def test_checkpoint_conflict_precedes_baseline_and_computation(group_env):
    admission, state = group_env
    state['checkpoint_failure'] = CheckpointError('another writer')
    with pytest.raises(CheckpointError, match='writer'):
        cli._group(admission, 'PG10', 1)
    assert not {'sources', 'baseline', 'backend', 'run_group'}.intersection(state['events'])


def test_terminal_group_reopen_only_checks_its_seal(group_env):
    admission, state = group_env
    state['stage'] = 'scored'
    assert cli._group(admission, 'PG10', 1)['stage'] == 'scored'
    assert 'seal_checked' in state['events']
    assert not {'sources', 'baseline', 'backend', 'run_group'}.intersection(state['events'])


def test_explicit_pending_recovery_reaches_checkpoint(group_env):
    admission, state = group_env
    cli._group(admission, 'PG10', 1, recover_pending=True)
    assert state['recover'] is True



def test_cli_passes_admitted_direct_plan_to_the_existing_computation(direct_env, monkeypatch):
    admission, state = direct_env
    original = admission.to_dict
    direct = dict(source_ids=["PG10"], domains=dict(D0="G0", D1="E06_D1_direct"), marker="admitted")
    monkeypatch.setattr(admission, "to_dict", lambda: dict(original(), direct=direct))
    cli._direct(admission)
    assert state["direct_plan"] == direct and state["starts"] == state["calls"] == state["finishes"] == 1


def test_inverse_requires_exact_selected_direct_scope(direct_env, monkeypatch):
    admission, state = direct_env
    original = admission.to_dict
    direct = dict(source_ids=["PG10"], domains=dict(D0="G0", D1="E06_D1_direct"))
    monkeypatch.setattr(admission, "to_dict", lambda: dict(original(), direct=direct))
    state["record"] = dict(status="completed", result=dict(expected_fields=2,
        fields=dict(PG10=dict(D0=dict(status="complete"), D1=dict(status="unavailable")))))
    assert cli._require_direct(admission) == state["record"]
    state["record"]["result"]["fields"]["EC04"] = dict(D0=dict(status="complete"), D1=dict(status="complete"))
    with pytest.raises(CheckpointError, match="admitted fields"): cli._require_direct(admission)
