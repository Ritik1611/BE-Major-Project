import subprocess
import sys
from pathlib import Path

INSTALLER_ROOT = Path(__file__).resolve().parents[1]
REQ_FILE = INSTALLER_ROOT / "runtime" / "configs" / "requirements.txt"

def install_python_deps():
    BASE = Path.home() / ".federated"
    VENV_DIR = BASE / "venv"
    python_path = VENV_DIR / "Scripts" / "python.exe"

    subprocess.run([
        str(python_path),
        "-m",
        "pip",
        "install",
        "-r",
        str(REQ_FILE)
    ], check=True)

    print("[OK] Dependencies installed in venv")
