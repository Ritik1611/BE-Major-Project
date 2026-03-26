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
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        if process.stdout:
            for line in process.stdout:
                print(line.strip())

        process.wait(timeout=30)

        if process.returncode != 0:
            sys.exit(f"[FAIL] {name}")

        print("[OK]")

    except subprocess.TimeoutExpired:
        process.kill()
        sys.exit(f"[FAIL] {name}: timeout")

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
    if not VENV_PYTHON.exists():
        sys.exit(f"[FAIL] Venv Python not found: {VENV_PYTHON}")
    # 1. Python sanity (VENV)
    print("[DEBUG] Using venv python:", VENV_PYTHON)
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