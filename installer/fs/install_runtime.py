"""
install_runtime.py  — FIXED VERSION 3

FIXES in this version:
  FIX-RMTREE-1 (CURRENT ERROR / WinError 5):
    Every shutil.rmtree() call now passes an onerror handler that strips
    the read-only / immutable attribute before deletion.
    Root cause: integrity.py's freeze_all_agent_files() calls
    path.chmod(0o444) on every agent .py file after the first successful
    install.  On Windows, shutil.rmtree() raises PermissionError (WinError 5)
    when it hits a read-only file because it does NOT clear the attribute
    automatically.  The fix is a one-line onerror= argument on every rmtree.

  FIX-RMTREE-2:
    install_windows_deps() also called shutil.rmtree without the handler.
    Fixed with the same helper.

  FIX-1 through FIX-3 from the previous version are preserved unchanged.
"""

import shutil
import stat
import platform
import subprocess
import sys
import os
from pathlib import Path

IS_WINDOWS = platform.system().lower() == "windows"

BASE_DIR = Path.home() / ".federated"
KEYS_DIR = Path.home() / ".federated" / "keys"


def get_installer_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


INSTALLER_ROOT = get_installer_root()
RUNTIME_SRC = INSTALLER_ROOT / "runtime"


# ── Permissions helpers ───────────────────────────────────────────────────────

def _chmod_exec(path: Path):
    try:
        path.chmod(stat.S_IRWXU)
    except Exception:
        pass


def _chmod_tree(root: Path):
    for p in root.rglob("*"):
        try:
            p.chmod(stat.S_IRWXU)
        except Exception:
            pass


# ── FIX-RMTREE-1: safe rmtree that handles read-only files on Windows ─────────

def _on_rmtree_error(func, path, exc_info):
    """
    onerror callback for shutil.rmtree.

    When a file is read-only (chmod 0o444) — set by integrity.py's
    freeze_all_agent_files() after the first install — Windows raises
    PermissionError (WinError 5).  This handler strips the read-only
    attribute and retries the operation.

    Parameters mirror the shutil.rmtree onerror contract:
      func  — the failing os function (e.g. os.unlink, os.rmdir)
      path  — the path that failed
      exc_info — sys.exc_info() tuple
    """
    try:
        # Make the path writable then retry
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        func(path)
    except Exception:
        # If still fails, log and continue — don't abort the whole install
        print(f"[WARN] Could not remove {path}: {exc_info[1]}", flush=True)


def _safe_rmtree(path: Path):
    """shutil.rmtree with read-only handling for Windows and Linux."""
    if path.exists():
        shutil.rmtree(path, onerror=_on_rmtree_error)


# ── Venv python path ──────────────────────────────────────────────────────────

def _venv_python() -> Path:
    if IS_WINDOWS:
        return BASE_DIR / "venv" / "Scripts" / "python.exe"
    return BASE_DIR / "venv" / "bin" / "python"


# ── MentalBERT installer ──────────────────────────────────────────────────────

def _is_real_model(directory: Path) -> bool:
    model_files = (
        list(directory.glob("*.bin"))
        + list(directory.glob("*.safetensors"))
        + list(directory.glob("pytorch_model*.bin"))
    )
    if not model_files:
        return False
    return max(f.stat().st_size for f in model_files) > 1_000_000


def install_mentalbert_model():
    MODEL_DST = BASE_DIR / "models" / "mentalbert"
    MODEL_SRC = RUNTIME_SRC / "models" / "mentalbert"
    MODEL_DST.parent.mkdir(parents=True, exist_ok=True)

    print(f"[MODEL] Checking MentalBERT at {MODEL_DST}", flush=True)

    if MODEL_DST.exists():
        if _is_real_model(MODEL_DST):
            print("[MODEL] Already installed and valid, skipping")
            return
        else:
            print("[MODEL] Found incomplete model — removing and re-downloading…", flush=True)
            _safe_rmtree(MODEL_DST)          # FIX-RMTREE-1

    if MODEL_SRC.exists() and _is_real_model(MODEL_SRC):
        print("[MODEL] Installing from installer payload…")
        shutil.copytree(MODEL_SRC, MODEL_DST)
        print("[OK] MentalBERT model installed from installer payload")
        return

    if MODEL_SRC.exists():
        print("[WARN] Installer payload contains git-lfs pointer files, not real weights.", flush=True)

    python_cmd = str(_venv_python()) if _venv_python().exists() else sys.executable

    download_script = r'''
import sys, os
from pathlib import Path

dst = sys.argv[1]
Path(dst).mkdir(parents=True, exist_ok=True)

def try_hub_download(repo_id, dst):
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=dst,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*", "*.ot"],
        )
        return True
    except Exception as e:
        print(f"[WARN] Hub download failed for {repo_id}: {e}", flush=True)
        return False

if try_hub_download("mental/mental-bert-base-uncased", dst):
    print("[OK] MentalBERT downloaded from HuggingFace Hub", flush=True)
    sys.exit(0)

print("[MODEL] Falling back to bert-base-uncased (compatible architecture)", flush=True)
try:
    from transformers import AutoModel, AutoTokenizer
    Path(dst).mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model.save_pretrained(dst)
    tokenizer.save_pretrained(dst)
    print("[OK] bert-base-uncased installed as MentalBERT fallback", flush=True)
    sys.exit(0)
except Exception as e2:
    print(f"[ERROR] Fallback also failed: {e2}", flush=True)
    sys.exit(1)
'''

    print("[MODEL] Downloading MentalBERT from HuggingFace Hub…", flush=True)
    result = subprocess.run(
        [python_cmd, "-c", download_script, str(MODEL_DST)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print(result.stdout or "", end="", flush=True)

    if result.returncode != 0:
        print(result.stderr or "", end="", file=sys.stderr)
        raise RuntimeError("MentalBERT model installation failed — check network or HuggingFace token.")

    if not MODEL_DST.exists():
        raise RuntimeError("Model folder was not created at all")

    if not _is_real_model(MODEL_DST):
        raise RuntimeError("Model exists but is invalid (likely LFS pointer or failed download)")

    print("[OK] MentalBERT model ready", flush=True)


# ── Windows native deps ───────────────────────────────────────────────────────

def install_windows_deps():
    if platform.system().lower() != "windows":
        return

    src = RUNTIME_SRC / "deps" / "windows"
    dst_root = BASE_DIR / "deps"
    dst = dst_root / "windows"

    if dst.exists():
        _safe_rmtree(dst)               # FIX-RMTREE-1

    if dst_root.exists():
        for item in dst_root.iterdir():
            if item.name == "windows":
                _safe_rmtree(item)      # FIX-RMTREE-1

    shutil.copytree(src, dst)
    _chmod_tree(dst)

    openface_bin = dst / "OpenFace" / "FeatureExtraction.exe"
    opensmile_bin = next(dst.glob("opensmile/**/SMILExtract.exe"), None)

    if not openface_bin.exists():
        raise RuntimeError("[INSTALLER] FeatureExtraction.exe missing after install")

    if opensmile_bin is None:
        raise RuntimeError("[INSTALLER] SMILExtract.exe missing after install")

    print("[OK] Windows OpenFace + openSMILE installed")


# ── Runtime installer ─────────────────────────────────────────────────────────

def install_runtime():
    # 1. bin/federated-client
    bin_dir = BASE_DIR / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    src_client = RUNTIME_SRC / "federated_client.py"
    dst_client = bin_dir / "federated-client"

    shutil.copy2(src_client, dst_client)
    _chmod_exec(dst_client)

    # ── Windows TPM signer ────────────────────────────────────────────────────
    if IS_WINDOWS:
        signer_src = RUNTIME_SRC / "windows_signer.exe"
        signer_dst = bin_dir / "windows_signer.exe"

        if not signer_src.exists():
            raise RuntimeError("windows_signer.exe missing from runtime")

        shutil.copy2(signer_src, signer_dst)
        _chmod_exec(signer_dst)
        print("[OK] Windows TPM signer installed")

    # 2. agents
    # FIX-RMTREE-1: was shutil.rmtree(agents_dst) — fails WinError 5 on re-install
    # because integrity.py set all .py files to 0o444 (read-only) after first install.
    agents_dst = BASE_DIR / "agents"
    _safe_rmtree(agents_dst)            # ← THE FIX FOR THE CURRENT ERROR
    shutil.copytree(RUNTIME_SRC / "agents", agents_dst)
    _chmod_tree(agents_dst)

    # FIX-2a: ensure agents package __init__.py files exist
    for pkg_dir in [agents_dst,
                    agents_dst / "lda",
                    agents_dst / "lda" / "pipelines",
                    agents_dst / "trainer",
                    agents_dst / "dp",
                    agents_dst / "enc"]:
        init_file = pkg_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")
            try:
                init_file.chmod(0o600)
            except Exception:
                pass

    # 3. configs
    configs_dst = BASE_DIR / "configs"
    _safe_rmtree(configs_dst)           # FIX-RMTREE-1
    shutil.copytree(RUNTIME_SRC / "configs", configs_dst)
    _chmod_tree(configs_dst)

    # 4. runtime guards & helpers
    runtime_dst = BASE_DIR / "runtime"
    _safe_rmtree(runtime_dst)           # FIX-RMTREE-1
    runtime_dst.mkdir(parents=True, exist_ok=True)

    for f in RUNTIME_SRC.glob("*.py"):
        if f.name == "federated_client.py":
            continue
        shutil.copy2(f, runtime_dst / f.name)
    _chmod_tree(runtime_dst)

    # FIX-3: create runtime/__init__.py
    runtime_init = runtime_dst / "__init__.py"
    if not runtime_init.exists():
        runtime_init.write_text("")
        try:
            runtime_init.chmod(0o600)
        except Exception:
            pass

    # 5. grpc stubs
    grpc_dst = BASE_DIR / "runtime" / "grpc"
    _safe_rmtree(grpc_dst)              # FIX-RMTREE-1
    shutil.copytree(RUNTIME_SRC / "grpc", grpc_dst)
    _chmod_tree(grpc_dst)

    grpc_init = grpc_dst / "__init__.py"
    if not grpc_init.exists():
        grpc_init.write_text("")
        try:
            grpc_init.chmod(0o600)
        except Exception:
            pass

    # 6. core shared modules
    core_src = RUNTIME_SRC / "core"
    core_dst = BASE_DIR / "core"
    if core_src.exists():
        _safe_rmtree(core_dst)          # FIX-RMTREE-1
        shutil.copytree(core_src, core_dst)
        _chmod_tree(core_dst)

    core_init = core_dst / "__init__.py"
    if not core_init.exists():
        core_init.write_text("")
        try:
            core_init.chmod(0o600)
        except Exception:
            pass

    # 7. Windows native deps
    install_windows_deps()

    # 8. CA certificate
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parents[1]

    ca_src = base / "runtime" / "keys" / "ca.pem"
    ca_dst = KEYS_DIR / "ca.pem"
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ca_src, ca_dst)
    print("[OK] CA certificate installed")

    # 9. validate_deps helper
    shutil.copy2(
        RUNTIME_SRC / "validate_deps.py",
        BASE_DIR / "runtime" / "validate_deps.py",
    )

    if IS_WINDOWS:
        shutil.copy2(
            RUNTIME_SRC / "windows_signer.exe",
            BASE_DIR / "bin" / "windows_signer.exe",
        )

    # 10. installer/security subset
    installer_security_src = INSTALLER_ROOT / "installer" / "security"
    installer_security_dst = BASE_DIR / "installer" / "security"

    if installer_security_src.exists():
        _safe_rmtree(installer_security_dst.parent)   # FIX-RMTREE-1
        installer_security_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(installer_security_src, installer_security_dst)
        _chmod_tree(installer_security_dst)
        print("[OK] installer.security module installed")
    else:
        print("[WARN] installer/security not found in installer package")

    installer_pkg = BASE_DIR / "installer"
    installer_pkg.mkdir(parents=True, exist_ok=True)
    installer_init = installer_pkg / "__init__.py"
    if not installer_init.exists():
        installer_init.write_text("")
        try:
            installer_init.chmod(0o600)
        except Exception:
            pass

    sec_init = installer_security_dst / "__init__.py"
    if not sec_init.exists():
        sec_init.write_text(
            "from .anti_debug import anti_debug\n"
            "from .tpm_attestation import tpm_attestation\n"
        )
        try:
            sec_init.chmod(0o600)
        except Exception:
            pass

    print("[OK] Runtime installed successfully")