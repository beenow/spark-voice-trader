# Root conftest: puts the repo root on sys.path so `mockmarket`, `tools`, `agent` import in tests.

# Tests never touch the real conversation store ($VOICE_HOME/sessions): each test gets its own.
import pytest as _pytest


@_pytest.fixture(autouse=True)
def _memory_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICE_MEMORY_DIR", str(tmp_path / "sessions"))
