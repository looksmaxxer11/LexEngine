"""
Basic smoke tests for the new pipeline skeleton.
Run: python -m pytest -q (if pytest available), or python test_pipeline_skeleton.py
"""

import sys
from pathlib import Path


def test_import():
    from src.pipeline import Pipeline  # noqa: F401


def test_cli_help():
    import subprocess
    # Just invoke without args to print help and exit 0
    proc = subprocess.run([sys.executable, 'pipeline.py'], capture_output=True, text=True)
    assert proc.returncode == 0 or proc.returncode == 2  # argparse may return 2 for help
    assert 'Local OCR pipeline' in (proc.stdout + proc.stderr)


if __name__ == "__main__":
    test_import()
    test_cli_help()
    print("OK")
