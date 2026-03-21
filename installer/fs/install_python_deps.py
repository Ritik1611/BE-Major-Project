import subprocess
import sys
from pathlib import Path

INSTALLER_ROOT = Path(__file__).resolve().parents[1]
REQ_FILE = INSTALLER_ROOT / "runtime" / "configs" / "requirements.txt"

def install_python_deps():
    BASE = Path.home() / ".federated"
    VENV_DIR = BASE / "venv"
    python_path = VENV_DIR / "Scripts" / "python.exe"

    print("[STEP] Installing dependencies (robust mode)")

    # Upgrade pip
    subprocess.run([
        str(python_path),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip"
    ])

    with open(REQ_FILE, "r") as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    for pkg in packages:
        print(f"\n[INSTALL] {pkg}")

        try:
            process = subprocess.Popen(
                [
                    str(python_path),
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    pkg
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # 🔥 STREAM OUTPUT LIVE
            for line in process.stdout:
                print(line.strip())

            process.wait(timeout=120)  # ⏱️ max 2 min per package

            if process.returncode != 0:
                print(f"[SKIPPED] {pkg}")

            else:
                print(f"[OK] {pkg}")

        except subprocess.TimeoutExpired:
            process.kill()
            print(f"[TIMEOUT - SKIPPED] {pkg}")

        except Exception as e:
            print(f"[ERROR - SKIPPED] {pkg}: {e}")

    print("\n[OK] Dependency installation complete")