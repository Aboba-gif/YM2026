"""Проверки файлов блокировок средствами ОС без научных вычислений."""
import errno
import os
from pathlib import Path
from queue import Empty, Queue
import stat
import subprocess
import sys
from threading import Thread

import pytest

from experiments import file_locks


WAITING_GUARD_PROBE = r'''
import errno, os, sys
from experiments.file_locks import FileLock

check = FileLock._check_open_file
admitted = []

def observed_check(self):
    check(self)
    if not admitted:
        admitted.append(self.fd)
        print('ADMITTED', flush=True)

FileLock._check_open_file = observed_check
try:
    with FileLock(sys.argv[1], blocking=True):
        raise AssertionError('Hardlinked waiting writer must not enter its body')
except ValueError as error:
    assert 'regular' in str(error) or 'unaliased' in str(error), error
    assert len(admitted) == 1
    try:
        os.fstat(admitted[0])
    except OSError as closed:
        assert closed.errno == errno.EBADF, closed
    else:
        raise AssertionError('Rejected waiting writer leaked its descriptor')
    print('REJECTED', flush=True)
'''


@pytest.mark.parametrize('payload', [b'', b'Foreign bytes must remain unchanged'])
@pytest.mark.parametrize('options', [dict(), dict(shared=True, create=False)])
def test_existing_hardlink_rejected_before_open_and_foreign_bytes_retained(tmp_path, monkeypatch, payload, options):
    foreign = tmp_path/'foreign.bin'
    foreign.write_bytes(payload)
    lease = tmp_path/'.second.lock'
    os.link(foreign, lease)
    before = foreign.stat()
    alias = lease.stat()
    assert before.st_nlink == alias.st_nlink == 2
    assert (before.st_dev, before.st_ino) == (alias.st_dev, alias.st_ino)

    def forbidden_open(*args, **kwargs):
        pytest.fail('An existing hardlink must be rejected before opening any descriptor')

    with monkeypatch.context() as local:
        local.setattr(file_locks.os, 'open', forbidden_open)
        with pytest.raises(ValueError):
            file_locks.FileLock(lease, **options)
    assert foreign.read_bytes() == lease.read_bytes() == payload
    assert foreign.stat().st_mtime_ns == before.st_mtime_ns
    assert lease.stat().st_nlink == 2


@pytest.mark.parametrize('kind', ['directory', 'fifo'])
def test_nonregular_lock_rejected_before_open(tmp_path, monkeypatch, kind):
    lease = tmp_path/'.special.lock'
    if kind == 'directory':
        lease.mkdir()
    else:
        if not hasattr(os, 'mkfifo'):
            pytest.skip('This OS does not provide native FIFO creation')
        os.mkfifo(lease)
    before = lease.lstat()

    def forbidden_open(*args, **kwargs):
        pytest.fail('A known nonregular lock must fail before a potentially blocking open')

    with monkeypatch.context() as local:
        local.setattr(file_locks.os, 'open', forbidden_open)
        with pytest.raises(ValueError):
            file_locks.FileLock(lease, create=False)
    after = lease.lstat()
    assert (after.st_dev, after.st_ino, after.st_mode, after.st_size) == (
        before.st_dev, before.st_ino, before.st_mode, before.st_size)


@pytest.mark.parametrize('payload', [b'', b'Opened bytes must remain unchanged'])
def test_hardlink_created_after_open_rejected_without_write_and_descriptor_closed(tmp_path, monkeypatch, payload):
    lease = tmp_path/'.raced.lock'
    lease.write_bytes(payload)
    foreign = tmp_path/'foreign.bin'
    before = lease.stat()
    assert before.st_nlink == 1
    open_file = os.open
    opened = []

    def add_link_after_open(path, flags, *args, **kwargs):
        fd = open_file(path, flags, *args, **kwargs)
        opened.append(fd)
        try:
            assert Path(path) == lease
            os.link(lease, foreign)
            assert os.fstat(fd).st_nlink == 2
        except BaseException:
            os.close(fd)
            raise
        return fd

    with monkeypatch.context() as local:
        local.setattr(file_locks.os, 'open', add_link_after_open)
        with pytest.raises(ValueError):
            file_locks.FileLock(lease)
    assert len(opened) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(opened[0])
    assert closed.value.errno == errno.EBADF
    assert foreign.read_bytes() == lease.read_bytes() == payload
    assert lease.stat().st_mtime_ns == before.st_mtime_ns
    assert lease.stat().st_nlink == foreign.stat().st_nlink == 2

    foreign.unlink()
    with file_locks.FileLock(lease, create=False) as held:
        assert not os.get_inheritable(held.fd)
    assert lease.read_bytes() == payload


@pytest.mark.skipif(os.name != 'posix' or not hasattr(os, 'mkfifo'),
                    reason='A native swapped FIFO requires POSIX')
def test_fifo_swapped_before_readonly_open_is_nonblocking_and_rejected(tmp_path, monkeypatch):
    lease = tmp_path/'.raced.lock'
    lease.write_bytes(b'Regular file retained during the simulated swap')
    retained = tmp_path/'retained.bin'
    original = lease.read_bytes()
    open_file = os.open
    opened = []

    def swap_before_open(path, flags, *args, **kwargs):
        assert Path(path) == lease
        lease.rename(retained)
        os.mkfifo(lease)
        # Наличие O_NONBLOCK проверяется до открытия FIFO, чтобы тест не завис.
        assert flags & os.O_NONBLOCK, 'Opening a swapped FIFO must be nonblocking'
        fd = open_file(path, flags, *args, **kwargs)
        opened.append(fd)
        assert stat.S_ISFIFO(os.fstat(fd).st_mode)
        return fd

    with monkeypatch.context() as local:
        local.setattr(file_locks.os, 'open', swap_before_open)
        with pytest.raises(ValueError):
            file_locks.FileLock(lease, create=False)
    assert len(opened) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(opened[0])
    assert closed.value.errno == errno.EBADF
    assert stat.S_ISFIFO(lease.lstat().st_mode)
    assert retained.read_bytes() == original


def test_symlink_leaf_installed_after_open_rejected_and_foreign_bytes_retained(tmp_path, monkeypatch):
    lease = tmp_path/'.raced.lock'
    lease.write_bytes(b'Original regular leaf must remain intact')
    original = lease.read_bytes()
    foreign = tmp_path/'foreign.bin'
    foreign.write_bytes(b'')
    retained = tmp_path/'retained.bin'
    capability = tmp_path/'symlink-capability'
    try:
        capability.symlink_to(foreign)
    except OSError as error:
        if os.name == 'nt' and (getattr(error, 'winerror', None) in (5, 50, 1314)
                or error.errno in (errno.EACCES, errno.EPERM, errno.ENOTSUP)):
            pytest.skip('Windows symlink privilege or native support is unavailable')
        raise
    capability.unlink()
    assert foreign.stat().st_nlink == lease.stat().st_nlink == 1
    open_file = os.open
    opened = []

    def install_alias_after_open(path, flags, *args, **kwargs):
        assert Path(path) == lease
        fd = open_file(foreign, flags, *args, **kwargs)
        opened.append(fd)
        try:
            lease.rename(retained)
            lease.symlink_to(foreign)
            info = os.fstat(fd)
            assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1
            assert stat.S_ISLNK(lease.lstat().st_mode)
            # По символьной ссылке доступен тот же файл; lstat должен отклонить ссылку
            # даже при совпадении файловых идентификаторов.
            assert os.path.samestat(lease.stat(), info)
        except BaseException:
            os.close(fd)
            raise
        return fd

    with monkeypatch.context() as local:
        local.setattr(file_locks.os, 'open', install_alias_after_open)
        with pytest.raises(ValueError, match='regular|unaliased'):
            file_locks.FileLock(lease)
    assert len(opened) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(opened[0])
    assert closed.value.errno == errno.EBADF
    assert lease.is_symlink()
    assert foreign.read_bytes() == lease.read_bytes() == b''
    assert retained.read_bytes() == original
    assert foreign.stat().st_nlink == retained.stat().st_nlink == 1


def test_foreign_regular_descriptor_rejected_when_original_leaf_is_unchanged(tmp_path, monkeypatch):
    lease = tmp_path/'.raced.lock'
    lease.write_bytes(b'Original path remains a separate regular file')
    original = lease.read_bytes()
    foreign = tmp_path/'foreign.bin'
    foreign.write_bytes(b'')
    before = {path: path.stat().st_mtime_ns for path in (lease, foreign)}
    open_file = os.open
    opened = []

    def open_foreign_descriptor(path, flags, *args, **kwargs):
        assert Path(path) == lease
        fd = open_file(foreign, flags, *args, **kwargs)
        opened.append(fd)
        try:
            leaf, descriptor = lease.lstat(), os.fstat(fd)
            assert stat.S_ISREG(leaf.st_mode) and stat.S_ISREG(descriptor.st_mode)
            assert leaf.st_nlink == descriptor.st_nlink == 1
            assert not os.path.samestat(leaf, descriptor)
        except BaseException:
            os.close(fd)
            raise
        return fd

    with monkeypatch.context() as local:
        local.setattr(file_locks.os, 'open', open_foreign_descriptor)
        with pytest.raises(ValueError, match='regular|unaliased'):
            file_locks.FileLock(lease)
    assert len(opened) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(opened[0])
    assert closed.value.errno == errno.EBADF
    assert lease.read_bytes() == original and foreign.read_bytes() == b''
    assert {path: path.stat().st_mtime_ns for path in before} == before
    assert lease.stat().st_nlink == foreign.stat().st_nlink == 1


def test_native_waiting_writer_rechecks_new_hardlink_before_initializing_empty_file(tmp_path, project_root):
    lease = tmp_path/'.waiting.lock'
    lease.touch()
    foreign = tmp_path/'foreign.bin'
    inherited = os.environ.get('PYTHONPATH', '')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    env['PYTHONPATH'] = str(project_root)+(os.pathsep+inherited if inherited else '')
    events, lines = Queue(), []
    process = collector = None

    def collect():
        for line in process.stdout:
            value = line.rstrip('\r\n')
            lines.append(value)
            events.put(value)
        events.put('<process-ended>')

    def expect(marker):
        try:
            value = events.get(timeout=20)
        except Empty:
            pytest.fail(f'Native writer did not report {marker}: {lines}')
        assert value == marker, lines

    try:
        with file_locks.FileLock(lease, create=False) as held:
            assert os.fstat(held.fd).st_size == 0
            process = subprocess.Popen([sys.executable, '-B', '-X', 'utf8', '-u',
                '-c', WAITING_GUARD_PROBE, str(lease)], cwd=tmp_path, env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1)
            collector = Thread(target=collect, daemon=True)
            collector.start()
            expect('ADMITTED')
            assert process.poll() is None, lines
            os.link(lease, foreign)
            original, alias = lease.stat(), foreign.stat()
            assert os.path.samestat(original, alias)
            assert original.st_nlink == alias.st_nlink == 2
            assert original.st_size == alias.st_size == os.fstat(held.fd).st_size == 0
        expect('REJECTED')
        assert process.wait(timeout=20) == 0, lines
        collector.join(timeout=20)
        assert not collector.is_alive(), 'Native writer collector remained active'
        assert lines == ['ADMITTED', 'REJECTED']
        assert foreign.read_bytes() == lease.read_bytes() == b''
        assert lease.stat().st_nlink == foreign.stat().st_nlink == 2
    finally:
        if process is not None:
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            process.wait(timeout=20)
            if collector is not None and collector.ident is not None:
                collector.join(timeout=20)
                assert not collector.is_alive(), 'Native writer collector survived cleanup'
            if process.stdout is not None:
                process.stdout.close()
