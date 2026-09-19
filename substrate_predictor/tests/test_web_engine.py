"""The browser engine must reproduce the Python package (full windows to ~1e-9, gapped percentiles within 0.03)."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_js_engine_matches_python():
    subprocess.run([sys.executable, str(WEB / "make_fixture.py")], check=True, capture_output=True)
    r = subprocess.run(["node", str(WEB / "test_engine.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout
