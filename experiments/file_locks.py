"""Совместная или исключительная файловая блокировка ОС.

В Windows используется LockFileEx для байта 0, в Unix — flock.
"""
from __future__ import annotations

import errno
import os
import stat
from pathlib import Path


class FileLockBusy(RuntimeError):
    """Запрошенная блокировка несовместима с блокировкой другого процесса."""


class FileLock:
    """Удерживать файловую блокировку до закрытия.

    По умолчанию занятая блокировка вызывает FileLockBusy. Дескриптор не
    наследуется, файл блокировки после close не удаляется. Экземпляр
    поддерживает контекстный менеджер.

    Parameters
    ----------
    path : str or pathlib.Path
        Путь к обычному файлу блокировки с единственной жёсткой ссылкой,
        без файловых псевдонимов.
    shared : bool, optional
        Совместная блокировка чтения вместо исключительной.
    create : bool, optional
        Разрешить создание файла; False открывает существующий файл без
        записи.
    blocking : bool, optional
        Ждать освобождения несовместимой блокировки; по умолчанию False.
    """

    def __init__(self, path, *, shared=False, create=True, blocking=False):
        self.fd = None
        if any(type(value) is not bool for value in (shared, create, blocking)):
            raise ValueError("shared, create and blocking must be boolean")
        self.path = Path(path).absolute()
        if (self.path.is_symlink() or getattr(self.path, "is_junction", lambda: False)()
                or self.path.resolve() != self.path):
            raise ValueError("File lock path must not be a filesystem alias")
        try:
            try:
                existing = self.path.lstat()
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1:
                    raise ValueError("File lock must be one unaliased regular file")
            flags = os.O_RDWR | os.O_CREAT if create else os.O_RDONLY
            # Подмена файла каналом FIFO не должна приводить к блокирующему открытию.
            flags |= getattr(os, "O_NONBLOCK", 0)
            self.fd = os.open(self.path, flags, 0o600)
            os.set_inheritable(self.fd, False)
            self._check_open_file()
            if os.name == "nt":
                import ctypes
                from ctypes import wintypes
                import msvcrt

                class Overlapped(ctypes.Structure):
                    _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                                ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                                ("hEvent", wintypes.HANDLE)]

                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                lock = kernel.LockFileEx
                lock.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                 wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]
                lock.restype = wintypes.BOOL
                self._overlapped = Overlapped()
                if not lock(msvcrt.get_osfhandle(self.fd), (0 if blocking else 1) | (0 if shared else 2), 0, 1, 0,
                            ctypes.byref(self._overlapped)):
                    error = ctypes.get_last_error()
                    if error in (32, 33, 158):
                        raise FileLockBusy(f"File lock occupied: {self.path.name}")
                    raise ctypes.WinError(error)
            else:
                import fcntl
                try:
                    fcntl.flock(self.fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | (0 if blocking else fcntl.LOCK_NB))
                except OSError as error:
                    if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise FileLockBusy(f"File lock occupied: {self.path.name}") from error
                    raise
            # Файл мог быть подменён во время ожидания блокировки другого процесса.
            self._check_open_file()
            # Заполнять нулевой байт можно только после получения исключительной блокировки.
            # Windows допускает блокировку байта за концом пустого файла.
            if create and not shared and os.fstat(self.fd).st_size == 0:
                os.write(self.fd, b"\0")
        except BaseException:
            self.close()
            raise

    def _check_open_file(self):
        """Проверить обычный файл без псевдонимов и соответствие дескриптору."""
        opened = os.fstat(self.fd)
        current = self.path.lstat()
        if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                or not stat.S_ISREG(current.st_mode) or current.st_nlink != 1
                or not os.path.samestat(current, opened)
                or self.path.resolve() != self.path):
            raise ValueError("File lock must be one unaliased regular file")

    def close(self):
        """Закрыть дескриптор и освободить блокировку, если он ещё открыт."""

        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        """Вернуть экземпляр контекстного менеджера.

        Returns
        -------
        FileLock
            Тот же экземпляр; повторно блокировка не запрашивается.
        """

        return self

    def __exit__(self, *_):
        """Освободить блокировку при выходе из контекста."""

        self.close()
