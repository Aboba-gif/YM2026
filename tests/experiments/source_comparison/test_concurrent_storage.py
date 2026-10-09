"""Реальные процессы и файловые блокировки; научные вычисления подставлены.

Эти проверки устанавливают владение записью, повторное использование сохранённых
байтов и атомарную публикацию JSON. Они не запускают решение ADR или подгонку.
"""
from concurrent.futures import Future, ProcessPoolExecutor
import errno
from functools import partial
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread
from time import monotonic
from types import SimpleNamespace

import pytest

from experiments.file_locks import FileLock, FileLockBusy
from experiments.source_comparison import run


MAIN_PROBE = r'''
from concurrent.futures import Future
import json, sys
from pathlib import Path
from types import SimpleNamespace
from experiments.source_comparison import run

config, mode, selected = sys.argv[1:]
run.load_protocol = lambda *a, **k: SimpleNamespace(full_sha256='a'*64)
settings = json.loads(Path(config).read_text())
output = (Path(config).parent/settings['output']).resolve()
def forbidden(*a, **k):
    raise AssertionError('No scientific computation in the storage test')
for name in ('build_truth', 'state_solver', 'check_model', 'fit'):
    setattr(run, name, forbidden)
def compute(settings, protocol, output, source_id, bindings):
    destination = output/(source_id+'.json')
    if destination.exists():
        recorded = json.loads(destination.read_text())
        assert recorded['bindings'] == bindings
        return source_id, 'reused'
    run.save_json(destination, dict(source=source_id, bindings=bindings,
        settings=settings, cases=[], test_only='No scientific computation'))
    return source_id, 'test_only'
run._compute_source = compute
class TestPool:
    def __init__(self, **options):
        self.options = options
    def __enter__(self):
        if mode == 'hold':
            print('MAIN_HELD', flush=True)
            if sys.stdin.readline().strip() != 'continue':
                raise SystemExit(0)
        return self
    def __exit__(self, *args):
        return False
    def submit(self, function, config_path, source_id, bindings, **options):
        # Execute the submitted ownership wrapper; only numerical work is fake.
        future = Future()
        try:
            future.set_result(function(config_path, source_id, bindings, **options))
        except BaseException as error:
            future.set_exception(error)
        return future
run.ProcessPoolExecutor = TestPool
run.main(['--config', config, '--source', selected])
print('MAIN_DONE', flush=True)
'''


SOURCE_PROBE = r'''
import json, os, sys
from pathlib import Path
from types import SimpleNamespace
from experiments.source_comparison import run

config, source_id, encoded_bindings, mode, entry = sys.argv[1:]
bindings = json.loads(encoded_bindings)
run.load_protocol = lambda *a, **k: SimpleNamespace(full_sha256='a'*64)
settings = json.loads(Path(config).read_text())
destination = (Path(config).parent/settings['output']/
               (source_id+'.json')).resolve()
def forbidden(*a, **k):
    raise AssertionError('Saved source reuse must not compute')
for name in ('build_truth', 'state_solver', 'check_model', 'fit'):
    setattr(run, name, forbidden)
read_bytes = Path.read_bytes
source_read_observed = False
def gated_read(path, *args, **kwargs):
    global source_read_observed
    if path.resolve() == destination and mode == 'hold' and not source_read_observed:
        source_read_observed = True
        print('SOURCE_HELD', flush=True)
        command = sys.stdin.readline().strip()
        if command == 'abrupt':
            os._exit(19)
        if command != 'continue':
            raise SystemExit(0)
    return read_bytes(path, *args, **kwargs)
Path.read_bytes = gated_read
result = getattr(run, entry)(config, source_id, bindings)
assert result == (source_id, 'reused'), result
print('SOURCE_REUSED', flush=True)
'''


SAVE_PROBE = r'''
import json, os, sys
from pathlib import Path
from experiments.source_comparison import run

target, marker = Path(sys.argv[1]), sys.argv[2]
replace, fsync = os.replace, os.fsync
synced = set()
def observe_sync(fd):
    info = os.fstat(fd)
    fsync(fd)
    synced.add((info.st_dev, info.st_ino))
def gated_replace(source, destination, *args, **kwargs):
    prepared = Path(source)
    info = prepared.stat()
    assert (info.st_dev, info.st_ino) in synced, 'Publication before fsync'
    assert prepared.parent == target.parent
    assert json.loads(prepared.read_text())['marker'] == marker
    print('PREPARED '+json.dumps(str(prepared)), flush=True)
    if sys.stdin.readline().strip() != 'continue':
        raise SystemExit(0)
    return replace(source, destination, *args, **kwargs)
os.fsync, os.replace = observe_sync, gated_replace
run.save_json(target, dict(marker=marker, test_only='No computation'))
print('SAVE_DONE', flush=True)
'''


class Child:
    def __init__(self, process):
        self.process = process
        self.events = Queue()
        self.lines = []
        self.collector = Thread(target=self._collect, daemon=True)
        self.collector.start()

    def _collect(self):
        for line in self.process.stdout:
            value = line.rstrip('\r\n')
            self.lines.append(value)
            self.events.put(value)
        self.events.put('<process-ended>')

    def expect(self, prefix, *, timeout=60):
        deadline = monotonic()+timeout
        for _ in range(30):
            remaining = deadline-monotonic()
            if remaining <= 0:
                pytest.fail(f'Timed out waiting for {prefix}: {self.lines}')
            try:
                line = self.events.get(timeout=remaining)
            except Empty:
                pytest.fail(f'Timed out waiting for {prefix}: {self.lines}')
            assert line != '<process-ended>', self.lines
            if line.startswith(prefix):
                return line
        pytest.fail(f'Too many events before {prefix}: {self.lines}')

    def send(self, command):
        self.process.stdin.write(command+'\n')
        self.process.stdin.flush()

    def finish(self, expected=0):
        code = self.process.wait(timeout=60)
        self.collector.join(timeout=10)
        assert not self.collector.is_alive(), 'Child stdout pipe remained open'
        assert code == expected, self.lines
        return self.lines

    def close(self):
        if self.process.poll() is None:
            try:
                self.send('stop')
            except (BrokenPipeError, OSError):
                pass
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        self.collector.join(timeout=10)
        assert not self.collector.is_alive(), 'Child collector survived cleanup'
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except (BrokenPipeError, OSError):
                    pass


@pytest.fixture
def spawn(project_root, tmp_path):
    children = []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
        PYTHONIOENCODING='utf-8', OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    inherited = os.environ.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(project_root)+(os.pathsep+inherited if inherited else '')

    def start(script, *args):
        child = Child(subprocess.Popen([sys.executable, '-B', '-c', script,
            *map(str, args)], cwd=tmp_path, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', bufsize=1))
        children.append(child)
        return child

    yield start
    # Перед ожиданием отправить команду остановки всем дочерним процессам:
    # один процесс может ждать блокировку другого.
    for child in children:
        if child.process.poll() is None:
            try:
                child.send('stop')
            except (BrokenPipeError, OSError):
                pass
    for child in children:
        child.close()


@pytest.fixture
def campaign(tmp_path):
    settings = dict(protocol='unused-protocol.json', protocol_sha256='a'*64,
        output='results', protected_roots=[], workers=2,
        sources=['SF10', 'alternate'])
    config = tmp_path/'experiment.json'
    config.write_text(json.dumps(settings), encoding='utf-8')
    return config, tmp_path/'results', settings


@pytest.fixture
def pinned_campaign(campaign, monkeypatch):
    """Создать частичный манифест настоящим main с явно искусственным расчётом."""
    config, output, settings = campaign
    monkeypatch.setattr(run, 'load_protocol',
        lambda *a, **kw: SimpleNamespace(full_sha256='a'*64))

    def artificial(settings, protocol, output, source_id, bindings):
        run.save_json(output/(source_id+'.json'), dict(source=source_id,
            bindings=bindings, numeric_result=1.25, cases=[],
            test_only='Artificial value; no scientific computation'))
        return source_id, 'test_only'

    class ImmediatePool:
        def __init__(self, **options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, *args, **options):
            future = Future()
            try:
                future.set_result(function(*args, **options))
            except BaseException as error:
                future.set_exception(error)
            return future

    monkeypatch.setattr(run, '_compute_source', artificial)
    monkeypatch.setattr(run, 'ProcessPoolExecutor', ImmediatePool)
    run.main(['--config', str(config), '--source', 'SF10'])
    manifest = (output/'run.json').read_bytes()
    record = json.loads(manifest)
    assert record['status'] == 'partial'
    assert record['outputs'] == {'SF10.json': run.digest(output/'SF10.json')}

    def forbidden(*args, **kwargs):
        pytest.fail('Pinned output admission must fail before pool or computation')
    for name in ('ProcessPoolExecutor', '_compute_source', 'build_truth',
                 'state_solver', 'check_model', 'fit'):
        monkeypatch.setattr(run, name, forbidden)
    return config, output, record, manifest


def test_native_spawn_pool_runs_two_owned_sources_and_pins_their_bytes(campaign, project_root, monkeypatch):
    from tests.experiments.source_comparison import storage_spawn_fixture

    config, output, settings = campaign
    original = project_root/'experiments/source_comparison/configs/experiment.json'
    protocol_settings = json.loads(original.read_bytes())
    settings.update(protocol=str((original.parent/protocol_settings['protocol']).resolve()),
                    protocol_sha256=protocol_settings['protocol_sha256'])
    run.save_json(config, settings)
    # При spawn инициализатор импортируется в настоящем дочернем процессе;
    # численный расчёт заменён.
    monkeypatch.setattr(run, 'ProcessPoolExecutor', partial(ProcessPoolExecutor,
        initializer=storage_spawn_fixture.install_compute_stub,
        mp_context=get_context('spawn')))
    run.main(['--config', str(config)])

    record = json.loads((output/'run.json').read_bytes())
    profiles = {source: json.loads((output/(source+'.json')).read_bytes())
                for source in settings['sources']}
    pids = {profile['pid'] for profile in profiles.values()}
    assert len(pids) == 2 and os.getpid() not in pids
    assert record['status'] == 'completed'
    assert record['bindings']['config_sha256'] == run.digest(config)
    assert record['bindings']['protocol_sha256'] == settings['protocol_sha256']
    assert record['outputs'] == {source+'.json': run.digest(output/(source+'.json'))
                               for source in settings['sources']}
    for source, profile in profiles.items():
        assert profile['source'] == source and profile['bindings'] == record['bindings']
        assert profile['start_method'] == 'spawn'
        assert profile['test_only'] == 'Artificial storage probe; no scientific computation'
        ready = json.loads((output/(source+'.spawn-ready')).read_bytes())
        assert ready == {'source': source, 'pid': profile['pid']}
        with FileLock(output/(source+'.json.lock')):
            pass
    with FileLock(output/'.run.lock'):
        pass


def test_native_main_owner_excludes_contender_and_preserves_manifest(campaign, spawn):
    config, output, settings = campaign
    seed = spawn(MAIN_PROBE, config, 'run', 'SF10')
    seed.finish()
    first = (output/'SF10.json').read_bytes()
    first_pin = hashlib.sha256(first).hexdigest()
    record = json.loads((output/'run.json').read_text())
    assert record['status'] == 'partial'
    assert record['outputs'] == {'SF10.json': first_pin}
    assert record['code_files']['experiments/file_locks.py'] == run.digest(run.file_locks.__file__)
    full_map_pin = hashlib.sha256(json.dumps(record['code_files'], sort_keys=True).encode()).hexdigest()
    assert record['bindings']['code_sha256'] == full_map_pin
    stray = output/'SF_unconfigured.json'
    stray.write_bytes(b'Unrelated data must be retained and excluded')

    owner = spawn(MAIN_PROBE, config, 'hold', 'alternate')
    owner.expect('MAIN_HELD')
    committed = (output/'run.json').read_bytes()
    running = json.loads(committed)
    assert running['status'] == 'running'
    assert running['outputs'] == {'SF10.json': first_pin}
    with pytest.raises(FileLockBusy):
        FileLock(output/'.run.lock')
    contender = spawn(MAIN_PROBE, config, 'run', 'SF10')
    assert contender.process.wait(timeout=60) != 0
    contender.collector.join(timeout=10)
    assert any('FileLockBusy' in line for line in contender.lines), contender.lines
    assert 'MAIN_DONE' not in contender.lines
    assert (output/'run.json').read_bytes() == committed
    assert (output/'SF10.json').read_bytes() == first

    owner.send('continue')
    owner.finish()
    final = json.loads((output/'run.json').read_text())
    assert final['status'] == 'completed'
    assert set(final['outputs']) == {source+'.json' for source in settings['sources']}
    assert final['outputs'] == {source+'.json': run.digest(output/(source+'.json'))
                               for source in settings['sources']}
    assert (output/'SF10.json').read_bytes() == first
    assert stray.read_bytes() == b'Unrelated data must be retained and excluded'
    with FileLock(output/'.run.lock'):
        pass


def test_main_rejects_wrong_binding_in_declared_saved_result(campaign, spawn):
    config, output, _ = campaign
    spawn(MAIN_PROBE, config, 'run', 'SF10').finish()
    recorded = json.loads((output/'SF10.json').read_text())
    recorded['source'] = 'alternate'
    recorded['bindings']['code_sha256'] = 'different_recorded_implementation'
    wrong = output/'alternate.json'
    run.save_json(wrong, recorded)
    raw = wrong.read_bytes()
    child = spawn(MAIN_PROBE, config, 'run', 'SF10')
    assert child.process.wait(timeout=60) != 0
    child.collector.join(timeout=10)
    assert any('ValueError' in line for line in child.lines), child.lines
    assert 'MAIN_DONE' not in child.lines
    assert wrong.read_bytes() == raw
    assert json.loads((output/'run.json').read_text())['status'] != 'completed'
    with FileLock(output/'.run.lock'):
        pass


def test_public_source_owner_excludes_cli_and_other_public_source_and_refreshes_manifest(campaign, spawn):
    config, output, settings = campaign
    spawn(MAIN_PROBE, config, 'run', 'SF10').finish()
    saved = json.loads((output/'SF10.json').read_text())
    bindings = saved['bindings']
    saved['source'] = 'alternate'
    run.save_json(output/'alternate.json', saved)
    profiles = {source: (output/(source+'.json')).read_bytes()
                for source in settings['sources']}
    manifest = (output/'run.json').read_bytes()
    encoded = json.dumps(bindings)
    owner = spawn(SOURCE_PROBE, config, 'SF10', encoded, 'hold', 'run_source')
    owner.expect('SOURCE_HELD')
    with pytest.raises(FileLockBusy):
        FileLock(output/'.run.lock')
    competitors = [spawn(MAIN_PROBE, config, 'run', 'alternate'),
        spawn(SOURCE_PROBE, config, 'alternate', encoded, 'run', 'run_source')]
    for child in competitors:
        assert child.process.wait(timeout=60) != 0
        child.collector.join(timeout=10)
        assert any('FileLockBusy' in line for line in child.lines), child.lines
    assert (output/'run.json').read_bytes() == manifest
    assert profiles == {source: (output/(source+'.json')).read_bytes()
                        for source in settings['sources']}
    owner.send('continue')
    assert 'SOURCE_REUSED' in owner.finish()
    final = json.loads((output/'run.json').read_text())
    assert final['status'] == 'completed'
    assert final['outputs'] == {source+'.json': hashlib.sha256(raw).hexdigest()
                               for source, raw in profiles.items()}
    assert profiles == {source: (output/(source+'.json')).read_bytes()
                        for source in settings['sources']}
    with FileLock(output/'.run.lock'):
        pass


def test_public_source_rejects_foreign_manifest_without_replacing_evidence(campaign, spawn):
    config, output, _ = campaign
    spawn(MAIN_PROBE, config, 'run', 'SF10').finish()
    profile = (output/'SF10.json').read_bytes()
    bindings = json.loads(profile)['bindings']
    foreign = json.loads((output/'run.json').read_text())
    foreign['bindings']['code_sha256'] = 'foreign_implementation'
    run.save_json(output/'run.json', foreign)
    manifest = (output/'run.json').read_bytes()
    child = spawn(SOURCE_PROBE, config, 'SF10', json.dumps(bindings), 'run', 'run_source')
    assert child.process.wait(timeout=60) != 0
    child.collector.join(timeout=10)
    assert any('ValueError' in line for line in child.lines), child.lines
    assert 'SOURCE_REUSED' not in child.lines
    assert (output/'run.json').read_bytes() == manifest
    assert (output/'SF10.json').read_bytes() == profile
    with FileLock(output/'.run.lock'):
        pass


def test_public_profile_only_reuse_keeps_profile_only_contract(campaign, spawn):
    config, output, _ = campaign
    output.mkdir()
    bindings = dict(config_sha256=run.digest(config), protocol_sha256='a'*64, code_sha256='c'*64)
    run.save_json(output/'SF10.json', dict(source='SF10', bindings=bindings,
        test_only='Saved artificial record, no calculation'))
    raw = (output/'SF10.json').read_bytes()
    child = spawn(SOURCE_PROBE, config, 'SF10', json.dumps(bindings), 'run', 'run_source')
    assert 'SOURCE_REUSED' in child.finish()
    assert (output/'SF10.json').read_bytes() == raw
    assert not (output/'run.json').exists()


@pytest.mark.parametrize('interruption', ['continue', 'abrupt'])
def test_native_source_owner_excludes_peer_but_neighbor_reuses(campaign, spawn, interruption):
    config, output, settings = campaign
    output.mkdir()
    bindings = dict(config_sha256=run.digest(config), protocol_sha256='a'*64, code_sha256='c'*64)
    for source in settings['sources']:
        run.save_json(output/(source+'.json'), dict(source=source, bindings=bindings,
            test_only='Saved artificial record, no calculation'))
    original = {source: (output/(source+'.json')).read_bytes() for source in settings['sources']}
    encoded = json.dumps(bindings)
    owner = spawn(SOURCE_PROBE, config, 'SF10', encoded, 'hold', '_run_source_worker')
    owner.expect('SOURCE_HELD')
    with pytest.raises(FileLockBusy):
        FileLock(output/'SF10.json.lock')
    peer = spawn(SOURCE_PROBE, config, 'SF10', encoded, 'run', '_run_source_worker')
    assert peer.process.wait(timeout=60) != 0
    peer.collector.join(timeout=10)
    assert any('FileLockBusy' in line for line in peer.lines), peer.lines
    assert 'SOURCE_REUSED' not in peer.lines
    neighbor = spawn(SOURCE_PROBE, config, 'alternate', encoded, 'hold', '_run_source_worker')
    neighbor.expect('SOURCE_HELD')
    assert owner.process.poll() is None and neighbor.process.poll() is None
    with pytest.raises(FileLockBusy):
        FileLock(output/'alternate.json.lock')
    neighbor.send('continue')
    neighbor.finish()
    owner.send(interruption)
    owner.finish(expected=19 if interruption == 'abrupt' else 0)
    resumed = spawn(SOURCE_PROBE, config, 'SF10', encoded, 'run', '_run_source_worker')
    assert 'SOURCE_REUSED' in resumed.finish()
    assert original == {source: (output/(source+'.json')).read_bytes()
                        for source in settings['sources']}
    for source in settings['sources']:
        with FileLock(output/(source+'.json.lock')):
            pass


def test_native_json_writers_prepare_distinct_synced_files_and_publish_exact_payload(tmp_path, spawn):
    target = tmp_path/'result.json'
    target.write_bytes(b'Committed input remains until publication')
    unrelated = tmp_path/'result.json.tmp'
    unrelated.write_bytes(b'Unrelated temporary file')
    one = spawn(SAVE_PROBE, target, 'one')
    prepared_one = Path(json.loads(one.expect('PREPARED ')[len('PREPARED '):]))
    two = spawn(SAVE_PROBE, target, 'two')
    prepared_two = Path(json.loads(two.expect('PREPARED ')[len('PREPARED '):]))
    assert prepared_one != prepared_two
    assert prepared_one.parent == prepared_two.parent == target.parent
    assert json.loads(prepared_one.read_text())['marker'] == 'one'
    assert json.loads(prepared_two.read_text())['marker'] == 'two'
    assert target.read_bytes() == b'Committed input remains until publication'
    assert unrelated.read_bytes() == b'Unrelated temporary file'
    two.send('continue')
    two.finish()
    assert json.loads(target.read_text())['marker'] == 'two'
    one.send('continue')
    one.finish()
    assert json.loads(target.read_text())['marker'] == 'one'
    assert not prepared_one.exists() and not prepared_two.exists()
    assert set(tmp_path.iterdir()) == {target, unrelated}


@pytest.mark.parametrize('entry,defect', [('_run_source_worker', 'changed_config'),
    ('_run_source_worker', 'output_scope'), ('run_source', 'changed_config'),
    ('_run_source_worker', 'missing_sources')])
def test_source_entry_rejects_changed_configuration_or_parent_output_before_writing(campaign, monkeypatch, entry, defect):
    config, output, settings = campaign
    bindings = dict(config_sha256=run.digest(config), protocol_sha256='a'*64,
                    code_sha256='c'*64)
    locked = config.parent/'locked_by_parent'
    locked.mkdir()
    parent_lock = locked/'.run.lock'
    parent_lock.write_bytes(b'Parent lock metadata')
    held = parent_lock.read_bytes()
    if defect == 'changed_config':
        settings['output'] = 'redirected_results'
        config.write_text(json.dumps(settings), encoding='utf-8')
    monkeypatch.setattr(run, 'load_protocol',
        lambda *a, **kw: SimpleNamespace(full_sha256='a'*64))
    def forbidden(*args, **kwargs):
        pytest.fail('Rejected worker must not begin scientific work')
    monkeypatch.setattr(run, '_compute_source', forbidden)
    with FileLock(parent_lock):
        with pytest.raises(ValueError, match='Configuration changed|Worker output differs|sources'):
            if defect == 'missing_sources':
                incomplete = dict(settings)
                incomplete.pop('sources')
                context = (incomplete, SimpleNamespace(full_sha256='a'*64), locked)
                run._run_source_worker(config, 'SF10', bindings, context=context,
                                       expected_output=str(locked))
            elif entry == '_run_source_worker':
                run._run_source_worker(config, 'SF10', bindings, expected_output=str(locked))
            else:
                run.run_source(config, 'SF10', bindings)
    assert parent_lock.read_bytes() == held
    assert set(locked.iterdir()) == {locked/'.run.lock'}
    assert not output.exists()
    assert not (config.parent/'redirected_results').exists()


@pytest.mark.parametrize('boundary', ['main', 'public', 'worker', 'final'])
@pytest.mark.parametrize('defect', ['numeric_change', 'missing_file'])
def test_pinned_output_rejected_before_reuse_or_manifest_replacement(pinned_campaign, boundary, defect):
    config, output, record, manifest = pinned_campaign
    profile = output/'SF10.json'
    old_pin = record['outputs']['SF10.json']
    if defect == 'numeric_change':
        modified = json.loads(profile.read_bytes())
        modified['numeric_result'] = 1.26
        assert modified['source'] == 'SF10' and modified['bindings'] == record['bindings']
        run.save_json(profile, modified)
        retained = profile.read_bytes()
        assert hashlib.sha256(retained).hexdigest() != old_pin
    else:
        profile.unlink()
        retained = None

    with pytest.raises(ValueError, match='Previously pinned source output'):
        if boundary == 'main':
            # Запрос другого источника не должен обходить проверку ранее сохранённого результата.
            run.main(['--config', str(config), '--source', 'alternate'])
        elif boundary == 'public':
            run.run_source(config, 'alternate', record['bindings'])
        else:
            with FileLock(output/'.run.lock'):
                if boundary == 'worker':
                    run._run_source_worker(config, 'SF10', record['bindings'],
                                           expected_output=str(output))
                else:
                    run._update_run(output, record)

    assert (output/'run.json').read_bytes() == manifest
    assert json.loads(manifest)['outputs'] == record['outputs'] == {'SF10.json': old_pin}
    assert not (output/'alternate.json').exists()
    if retained is None:
        assert not profile.exists()
    else:
        assert profile.read_bytes() == retained
    with FileLock(output/'.run.lock'), FileLock(output/'SF10.json.lock'):
        pass


@pytest.mark.parametrize('boundary', ['worker', 'final'])
@pytest.mark.parametrize('defect', ['changed', 'missing'])
def test_pinned_bytes_checked_again_at_actual_reuse_or_final_scan(pinned_campaign, monkeypatch, boundary, defect):
    config, output, record, manifest = pinned_campaign
    profile = output/'SF10.json'
    read_bytes = Path.read_bytes
    exists = Path.exists
    original = read_bytes(profile)
    modified = json.loads(original)
    modified['numeric_result'] = 1.26
    changed = json.dumps(modified, allow_nan=False).encode('utf-8')
    assert modified['source'] == 'SF10' and modified['bindings'] == record['bindings']
    assert hashlib.sha256(changed).hexdigest() != record['outputs']['SF10.json']
    reads = []

    def drift(path):
        if path.resolve() == profile.resolve():
            reads.append(path)
            # Первичная проверка читает исходные байты; проверка допуска — изменённые,
            # с сохранёнными привязками источника и входов.
            return original if len(reads) == 1 else changed
        return read_bytes(path)

    def disappears(path):
        if path.resolve() == profile.resolve() and reads:
            return False
        return exists(path)

    with monkeypatch.context() as local:
        local.setattr(Path, 'read_bytes', drift)
        if defect == 'missing':
            local.setattr(Path, 'exists', disappears)
        with FileLock(output/'.run.lock'):
            with pytest.raises(ValueError, match='Previously pinned source output (has changed|is missing)'):
                if boundary == 'worker':
                    run._run_source_worker(config, 'SF10', record['bindings'],
                                           expected_output=str(output))
                else:
                    run._update_run(output, record)
    assert len(reads) == (1 if defect == 'missing' else 2)
    assert profile.read_bytes() == original
    assert (output/'run.json').read_bytes() == manifest
    assert record['outputs'] == {'SF10.json': hashlib.sha256(original).hexdigest()}
    assert not (output/'alternate.json').exists()


@pytest.mark.parametrize('entry', ['run_source', '_run_source_worker'])
@pytest.mark.parametrize('include_config_pin', [False, True])
def test_source_entry_rejects_protocol_binding_before_writing(campaign, monkeypatch, entry, include_config_pin):
    config, output, _ = campaign
    raw = config.read_bytes()
    bindings = dict(protocol_sha256='d'*64, code_sha256='c'*64)
    if include_config_pin:
        bindings['config_sha256'] = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(run, 'load_protocol',
        lambda *a, **kw: SimpleNamespace(full_sha256='a'*64))

    def forbidden(*args, **kwargs):
        pytest.fail('Mismatched protocol must fail before computation')
    monkeypatch.setattr(run, '_compute_source', forbidden)
    with pytest.raises(ValueError, match='Protocol differs'):
        getattr(run, entry)(config, 'SF10', bindings)
    assert config.read_bytes() == raw
    assert not output.exists()


def malformed_object(record, defect):
    """Создать некорректную JSON-запись для проверки отказа."""
    if defect == 'non_object':
        return b'[]'
    raw = json.dumps(record, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if defect == 'overflow':
        return raw[:-1]+b',"oversized":1e999}'
    return raw[:-1]+b',"duplicate_probe":1,"duplicate_probe":2}'


@pytest.mark.parametrize('defect', ['non_object', 'overflow', 'duplicate_key'])
def test_malformed_config_rejected_before_output_or_work_and_keeps_input(campaign, monkeypatch, defect):
    config, output, settings = campaign
    raw = malformed_object(settings, defect)
    config.write_bytes(raw)
    monkeypatch.setattr(run, 'load_protocol',
        lambda *a, **kw: SimpleNamespace(full_sha256='a'*64))
    def forbidden(*args, **kwargs):
        pytest.fail('Malformed configuration must fail before pool or computation')
    for name in ('ProcessPoolExecutor', '_compute_source', 'build_truth', 'fit'):
        monkeypatch.setattr(run, name, forbidden)
    with pytest.raises(ValueError):
        run.main(['--config', str(config)])
    assert config.read_bytes() == raw
    assert not output.exists()


@pytest.mark.parametrize('operation', ['manifest', 'source_reuse', 'declared_scan'])
@pytest.mark.parametrize('defect', ['non_object', 'overflow', 'duplicate_key'])
def test_malformed_saved_records_rejected_and_committed_bytes_retained(campaign, monkeypatch, operation, defect):
    config, output, settings = campaign
    output.mkdir()
    bindings = dict(config_sha256=run.digest(config), protocol_sha256='a'*64,
                    code_sha256='c'*64)
    manifest = dict(bindings=bindings, configuration=settings, status='partial', outputs={})
    profile = dict(source='SF10', bindings=bindings, test_only='No computation')
    manifest_path, source_path = output/'run.json', output/'SF10.json'
    run.save_json(manifest_path, manifest)
    run.save_json(source_path, profile)
    target, record = (manifest_path, manifest) if operation == 'manifest' else (source_path, profile)
    target.write_bytes(malformed_object(record, defect))
    retained = {p: p.read_bytes() for p in (config, manifest_path, source_path)}
    monkeypatch.setattr(run, 'load_protocol',
        lambda *a, **kw: SimpleNamespace(full_sha256='a'*64))
    def forbidden(*args, **kwargs):
        pytest.fail('Malformed scientific metadata must not trigger computation')
    for name in ('build_truth', 'state_solver', 'check_model', 'fit'):
        monkeypatch.setattr(run, name, forbidden)
    if operation == 'manifest':
        monkeypatch.setattr(run, '_compute_source', forbidden)
    with FileLock(output/'.run.lock'):
        with pytest.raises(ValueError):
            if operation == 'declared_scan':
                run._update_run(output, manifest)
            else:
                # Эти две проверки допуска не зависят от итогового обхода файлов манифеста.
                run._run_source_worker(config, 'SF10', bindings, expected_output=str(output))
    assert retained == {p: p.read_bytes() for p in retained}
    assert not (output/'alternate.json').exists()


@pytest.mark.parametrize('failure', ['replace', 'fsync', 'serialization'])
def test_json_failure_keeps_committed_and_unrelated_files_and_allows_later_save(tmp_path, monkeypatch, failure):
    target = tmp_path/'result.json'
    target.write_bytes(b'Previously committed result')
    unrelated = tmp_path/'result.json.tmp'
    unrelated.write_bytes(b'Another owner\'s temporary evidence')
    original = {p: p.read_bytes() for p in tmp_path.iterdir()}
    payload = dict(marker='failed')
    with monkeypatch.context() as local:
        if failure == 'serialization':
            payload['value'] = float('nan')
            exception = ValueError
        else:
            def fail(*args, **kwargs):
                raise OSError(errno.EIO, 'Injected publication failure')
            local.setattr(os, failure, fail)
            exception = OSError
        with pytest.raises(exception):
            run.save_json(target, payload)
    assert set(tmp_path.iterdir()) == set(original)
    assert original == {p: p.read_bytes() for p in original}
    run.save_json(target, dict(marker='later_success'))
    assert json.loads(target.read_text()) == {'marker': 'later_success'}
    assert unrelated.read_bytes() == original[unrelated]
    assert set(tmp_path.iterdir()) == set(original)


@pytest.mark.parametrize('sources', [[], ['SF10', 'SF10'], ['../escape'], [True]])
def test_invalid_declared_sources_rejected_before_pool_or_scientific_work(campaign, monkeypatch, sources):
    config, output, settings = campaign
    settings['sources'] = sources
    config.write_text(json.dumps(settings), encoding='utf-8')
    monkeypatch.setattr(run, 'load_protocol',
        lambda *a, **kw: SimpleNamespace(full_sha256='a'*64))
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid source declarations must fail before pool or computation')
    for name in ('ProcessPoolExecutor', 'build_truth', 'state_solver', 'fit'):
        monkeypatch.setattr(run, name, forbidden)
    with pytest.raises(ValueError):
        run.main(['--config', str(config)])
    assert not output.exists()
