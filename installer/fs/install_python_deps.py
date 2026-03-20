import subprocess
import sys
from pathlib import Path

INSTALLER_ROOT = Path(__file__).resolve().parents[1]
REQ_FILE = INSTALLER_ROOT / "runtime" / "configs" / "requirements.txt"

def install_python_deps():
    BASE = Path.home() / ".federated"
    VENV_DIR = BASE / "venv"

    python_path = VENV_DIR / "Scripts" / "python.exe"

    if not python_path.exists():
        raise RuntimeError("Venv creation failed: python.exe not found")

    print("[STEP] Installing dependencies into venv")

    result = subprocess.run(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "-r",
            str(REQ_FILE)
        ],
        capture_output=True,
        text=True
    )

    print("[PIP STDOUT]")
    print(result.stdout)

    print("[PIP STDERR]")
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError("pip install failed")
