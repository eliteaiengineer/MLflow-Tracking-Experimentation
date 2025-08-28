import os
import subprocess
import sys
from pathlib import Path


def test_train_smoke(tmp_path):
    ROOT = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    # one run
    cmd = [
        sys.executable,
        "src/train.py",
        "--model",
        "rf",
        "--n_estimators",
        "50",
        "--max_depth",
        "3",
        "--test_size",
        "0.2",
    ]
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)

    # mlruns directory should exist
    assert (ROOT / "mlruns").exists()
