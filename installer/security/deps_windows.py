import subprocess
import sys

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 10

def _run(cmd):
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)

def verify_python_and_pip():
    print("🔥 ENTERED verify_python_and_pip")

    try:
        print("[DEBUG] Checking python version")
        subprocess.run([sys.executable, "--version"], check=True)

        print("[DEBUG] Checking pip version")
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True)

        print("🔥 EXITING verify_python_and_pip")

    except Exception as e:
        print("[ERROR] verify_python_and_pip failed:", e)
        raise

from pathlib import Path
import sys

BASE = Path.home() / ".federated" / "deps" / "windows"

REQUIRED = [
    BASE / "OpenFace" / "FeatureExtraction.exe",
    BASE / "OpenFace" / "model",
    BASE / "opensmile" / "build" / "progsrc" / "smilextract" / "Release" / "SMILExtract.exe",
    BASE / "opensmile" / "config" / "egemaps" / "v02" / "eGeMAPSv02.conf",
]

def verify_windows_deps():
    for p in REQUIRED:
        if not p.exists():
            sys.exit(f"[INSTALLER] Missing dependency: {p}")

    print("[OK] Windows dependencies verified")
