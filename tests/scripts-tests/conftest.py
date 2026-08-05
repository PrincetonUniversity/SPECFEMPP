import subprocess
import sys
from pathlib import Path

import pytest

specfem_root = Path(__file__).resolve().parent.parent.parent
if not (specfem_root / "pyproject.toml").is_file():
    e = RuntimeError(
        'scripts-tests/conftest.py could not resolve "specfem_root" correctly.'
        f'Got "{str(specfem_root)}".'
    )
    raise e
scripts_root = specfem_root / "scripts"


@pytest.fixture
def path_specfem_root():
    return specfem_root


@pytest.fixture
def path_scripts_root():
    return scripts_root


@pytest.fixture
def scripts_in_path(monkeypatch):
    monkeypatch.syspath_prepend(scripts_root)


@pytest.fixture
def execute_script():
    """Returns a function to execute a python script.
    Properly captures the subprocess module and correct python executable."""

    def execute(cmd: list[str], execute_as_module: bool = False):
        if execute_as_module:
            full_cmd = [sys.executable, "-m"] + cmd
        else:
            full_cmd = [sys.executable] + cmd
        out = subprocess.run(full_cmd, capture_output=True)
        if out.returncode != 0:
            pytest.fail(
                f"nonzero return code ({out.returncode}) for subprocess.run:\n$ "
                f"{' '.join(full_cmd)}\n\n{out.stderr.decode()}"
            )

        return out.stdout

    return execute
