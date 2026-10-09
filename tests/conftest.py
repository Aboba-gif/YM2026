"""Общие фикстуры тестов приложения YM2026."""
from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def project_root(request):
    return Path(request.config.rootpath).resolve()
