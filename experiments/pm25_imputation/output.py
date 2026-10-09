"""Проверка отдельных каталогов результатов и метрик заполнения PM₂.₅."""
from pathlib import Path


def outputs(root, names, *, protected=()):
    """Проверить пути вывода и вернуть абсолютный каталог результатов.

    Parameters
    ----------
    root : str or Path
        Явно заданный корневой каталог вывода.
    names : iterable of str or Path
        Имена создаваемых файлов относительно `root`.
    protected : iterable of str or Path, optional
        Файлы и каталоги входов, с которыми вывод не должен пересекаться.

    Returns
    -------
    Path
        Разрешённый абсолютный корневой каталог. Проверка не создаёт каталоги
        или файлы.

    Raises
    ------
    ValueError
        Путь выходит за ``root``, пересекает защищённый вход, проходит через
        символическую ссылку или точку соединения либо указывает на каталог
        или файл с несколькими жёсткими ссылками.
    """
    if root is None:
        raise ValueError("An explicit output root is required")
    root = Path(root).expanduser().absolute()
    protected = [Path(path).resolve() for path in protected]
    for name in names:
        candidate = root / name
        if any(part.is_symlink() or getattr(part,"is_junction",lambda:False)()
               for part in (candidate, *candidate.parents)):
            raise ValueError("Workflow output must not traverse symlinks or junctions")
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError("Workflow output escapes the explicit output root")
        if any(resolved == path or resolved.is_relative_to(path) or path.is_relative_to(resolved)
               for path in protected):
            raise ValueError("Workflow output overlaps an input")
        if candidate.exists() and (not candidate.is_file() or candidate.stat().st_nlink > 1):
            raise ValueError("Workflow output is not an independent regular file")
    return root.resolve()
