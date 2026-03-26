#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
import importlib

BASE = Path.home() / ".federated"
DEPS = BASE / "deps"

# ✅ IMPORTANT: use venv python
VENV_PYTHON = BASE / "venv" / "Scripts" / "python.exe"

REQUIRED = {
    "pydantic": "pydantic",
    "PyYAML": "yaml",
    "torch": "torch",
    "transformers": "transformers",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "opencv-python": "cv2",
}

OPTIONAL = {
    "librosa": "librosa",
    "webrtcvad": "webrtcvad",
    "spacy": "spacy",
    "boto3": "boto3",
    "Pyfhel": "Pyfhel",
}

# --------------------------------------------------
# Run commands using VENV PYTHON
# --------------------------------------------------
def run(cmd, name):
    print(f"[TEST] {name}")
    try:
        out = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            timeout=30
        )
        print("[OK]")
        return out
    except subprocess.CalledProcessError as e:
        print(e.output.decode(errors="ignore"))
        sys.exit(f"[FAIL] {name}")
    except Exception as e:
        sys.exit(f"[FAIL] {name}: {e}")

# --------------------------------------------------
# Validate Python imports INSIDE VENV
# --------------------------------------------------
def check():
    print("\n[CHECK] Validating dependencies...\n")

    for pip_name, import_name in REQUIRED.items():
        try:
            subprocess.run(
                [str(VENV_PYTHON), "-c", f"import {import_name}"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"[OK] {pip_name}")
        except Exception:
            print(f"[FAIL] {pip_name}")
            raise RuntimeError(f"Missing REQUIRED dependency: {pip_name}")

    for pip_name, import_name in OPTIONAL.items():
        try:
            subprocess.run(
                [str(VENV_PYTHON), "-c", f"import {import_name}"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"[OK] {pip_name}")
        except Exception:
            print(f"[WARN] Optional missing: {pip_name}")

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    # 1. Python sanity (VENV)
    run([str(VENV_PYTHON), "--version"], "Venv Python available")

    # 2. OpenFace
    openface = DEPS / "windows" / "OpenFace" / "FeatureExtraction.exe"
    if not openface.exists():
        sys.exit("[FAIL] FeatureExtraction.exe missing")

    run([str(openface), "-h"], "OpenFace executable runs")

    # 3. openSMILE
    smile = list(
        (DEPS / "windows" / "opensmile" / "build" / "progsrc" / "smilextract" / "Release")
        .rglob("SMILExtract.exe")
    )
    if not smile:
        sys.exit("[FAIL] SMILExtract.exe missing")

    run([str(smile[0]), "-h"], "openSMILE executable runs")

    # 4. Python deps check
    check()

    print("\n[ALL CHECKS PASSED] Runtime dependencies validated")

if __name__ == "__main__":
    main()