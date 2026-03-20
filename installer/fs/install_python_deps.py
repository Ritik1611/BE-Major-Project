import subprocess
import sys
from pathlib import Path

INSTALLER_ROOT = Path(__file__).resolve().parents[1]
REQ_FILE = INSTALLER_ROOT / "runtime" / "configs" / "requirements.txt"

def install_python_deps():
    BASE = Path.home() / ".federated"
    VENV_DIR = BASE / "venv"
    python_path = VENV_DIR / "Scripts" / "python.exe"

    print("[STEP] Installing dependencies (safe mode)")

    # Upgrade pip first
    subprocess.run([
        str(python_path),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip"
    ], check=True)

    if not REQ_FILE.exists():
        raise RuntimeError("requirements.txt not found")

    with open(REQ_FILE, "r") as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    for pkg in packages:
        print(f"[INSTALL] {pkg}")

        result = subprocess.run(
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                pkg
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"[SKIPPED] {pkg}")
            print(result.stderr.splitlines()[-1] if result.stderr else "")
        else:
            print(f"[OK] {pkg}")

    print("[OK] Dependency installation complete (with skips)")
