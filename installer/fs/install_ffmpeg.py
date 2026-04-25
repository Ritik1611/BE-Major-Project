"""
install_ffmpeg.py — FIXED VERSION WITH AUTOMATED HASH VERIFICATION
FIXES:
  FIX-FFMPEG-404: Use stable redirect URL that always resolves to current release.
  FIX-FFMPEG-HASH-AUTO: Compute and store SHA-256 on first install, verify on subsequent runs.
"""
import os
import platform
import subprocess
import shutil
import hashlib
import urllib.request
import zipfile
from pathlib import Path

# Stable redirect URL — always points to current FFmpeg release
FFMPEG_WIN_URL = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-release-essentials.zip"

# Path to store the computed hash after first successful download
FFMPEG_HASH_STORE = Path.home() / ".federated" / "state" / "ffmpeg_sha256.txt"

def _compute_sha256(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().lower()

def _get_expected_hash() -> str:
    """Get expected hash from env var or stored file, or None if not set."""
    # Priority 1: Environment variable override
    env_hash = os.environ.get("FFMPEG_SHA256")
    if env_hash:
        return env_hash.lower().strip()
    
    # Priority 2: Stored hash from first install
    if FFMPEG_HASH_STORE.exists():
        try:
            return FFMPEG_HASH_STORE.read_text().strip().lower()
        except Exception:
            pass
    
    # Priority 3: No hash configured — skip verification (dev mode)
    return None

def _verify_sha256(path: str, expected: str) -> None:
    """Verify file hash matches expected. Raises RuntimeError on mismatch."""
    actual = _compute_sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"FFmpeg ZIP integrity check FAILED.\n"
            f"  expected: {expected}\n"
            f"  actual  : {actual}\n"
            "The download may have been tampered with. Do not proceed."
        )

def _store_hash(hash_value: str) -> None:
    """Store computed hash for future verification."""
    FFMPEG_HASH_STORE.parent.mkdir(parents=True, exist_ok=True)
    FFMPEG_HASH_STORE.write_text(hash_value)
    try:
        os.chmod(FFMPEG_HASH_STORE, 0o600)
    except Exception:
        pass

def install_ffmpeg():
    print("[DEBUG] Checking FFmpeg...", flush=True)
    if shutil.which("ffmpeg"):
        print("[DEBUG] FFmpeg already installed", flush=True)
        return
    
    system = platform.system()
    try:
        if system == "Windows":
            zip_path = "ffmpeg.zip"
            extract_dir = "ffmpeg"
            
            print(f"[DEBUG] Downloading FFmpeg from {FFMPEG_WIN_URL}...", flush=True)
            req = urllib.request.Request(
                FFMPEG_WIN_URL,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out:
                out.write(response.read())
            
            # Compute and verify hash
            computed_hash = _compute_sha256(zip_path)
            expected_hash = _get_expected_hash()
            
            if expected_hash:
                print("[DEBUG] Verifying FFmpeg ZIP integrity...", flush=True)
                _verify_sha256(zip_path, expected_hash)
                print("[DEBUG] FFmpeg ZIP integrity OK", flush=True)
            else:
                print("[WARN] No FFmpeg hash configured — skipping integrity check (first install)", flush=True)
                print(f"[INFO] Computed hash: {computed_hash}", flush=True)
                print(f"[INFO] To enable verification, set env var: FFMPEG_SHA256={computed_hash}", flush=True)
                # Store for future runs
                _store_hash(computed_hash)
            
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            
            ffmpeg_bin = None
            for root, dirs, files in os.walk(extract_dir):
                if "ffmpeg.exe" in files:
                    ffmpeg_bin = root
                    break
            
            if not ffmpeg_bin:
                raise RuntimeError("ffmpeg.exe not found in extracted archive")
            
            os.environ["PATH"] += os.pathsep + ffmpeg_bin
            print(f"[DEBUG] FFmpeg installed at {ffmpeg_bin}", flush=True)
            
            # Persist the path for daemon processes
            state_dir = Path.home() / ".federated" / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "ffmpeg_path.txt").write_text(ffmpeg_bin)
            
            # Clean up
            try:
                os.remove(zip_path)
            except OSError:
                pass
        
        elif system == "Linux":
            print("[DEBUG] Installing FFmpeg on Linux...", flush=True)
            subprocess.run(["sudo", "apt", "update"], check=True)
            subprocess.run(["sudo", "apt", "install", "-y", "ffmpeg"], check=True)
        
        elif system == "Darwin":
            print("[DEBUG] Installing FFmpeg on macOS...", flush=True)
            if shutil.which("brew"):
                subprocess.run(["brew", "install", "ffmpeg"], check=True)
            else:
                raise RuntimeError("Homebrew not found. Install from https://brew.sh/")
        else:
            raise RuntimeError(f"Unsupported OS: {system}")
        
        print("[DEBUG] FFmpeg installed successfully", flush=True)
    
    except Exception as e:
        print(f"[ERROR] FFmpeg installation failed: {e}", flush=True)
        print("Please install FFmpeg manually from https://ffmpeg.org/download.html", flush=True)
        raise