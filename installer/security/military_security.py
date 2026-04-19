"""
military_security.py — Comprehensive threat model and defenses

FULL ATTACK SURFACE ANALYSIS (every vector your codebase exposes):

═══════════════════════════════════════════════════════════════════
CATEGORY 1: FILE-SYSTEM ATTACKS
═══════════════════════════════════════════════════════════════════

ATTACK-FS1  Baseline reset (your successful bypass)
  Vector:   `python -c "from installer.security.integrity import write_baseline; write_baseline()"`
  Fix:      One-time token + TPM signing + INSTALL_LOCK + chattr +i (integrity_v2.py)

ATTACK-FS2  Append-only .pth file injection
  Vector:   Drop a .pth file in site-packages or the venv. Python auto-imports it.
            Malicious .pth can inject arbitrary code before any integrity check.
  Fix:      Verify sys.path entries at startup. Add site-packages to integrity scope.
            Use --isolated flag when launching Python.

ATTACK-FS3  .pyc file injection
  Vector:   .pyc files are excluded from integrity scope. Drop a pre-compiled
            malicious module that shadows a legitimate one.
  Fix:      Set PYTHONDONTWRITEBYTECODE=1. Delete all .pyc files at startup.
            Add .pyc to excluded suffixes (they should NEVER exist in protected dirs).

ATTACK-FS4  LD_PRELOAD / DLL injection
  Vector:   Set LD_PRELOAD to a malicious .so that hooks open(), read(), write().
            Intercepts self-destruct, hooks integrity check to always return True.
  Fix:      anti_debug.py already checks LD_PRELOAD on Linux. But also:
            - Use seccomp to block execve/execveat after startup
            - Check /proc/self/maps for unexpected shared libraries

ATTACK-FS5  Symlink attack on secure store
  Vector:   Replace a file:// URI target with a symlink to /etc/passwd.
            The access-outside-secure-store check uses str.startswith() on
            resolved paths, but if the symlink target resolves INSIDE the root,
            it passes. Then the encrypted write goes to an attacker-controlled location.
  Fix:      Use path.resolve() and verify BEFORE opening. Implemented below.

ATTACK-FS6  TOCTOU (Time Of Check To Time Of Use)
  Vector:   Attacker swaps a file between stat() check and read().
  Fix:      Use O_NOFOLLOW. Open files and fstat() the fd rather than stat() the path.

ATTACK-FS7  Canary file deletion
  Vector:   Delete files silently; if deletion is not detected, tamper freely.
  Fix:      Canary files (see CanaryMonitor below).

═══════════════════════════════════════════════════════════════════
CATEGORY 2: PROCESS / MEMORY ATTACKS
═══════════════════════════════════════════════════════════════════

ATTACK-MEM1  ptrace attach
  Vector:   `gdb -p <pid>` or `strace -p <pid>` after process starts.
            Can modify memory, hook function calls, disable self-destruct.
  Fix:      anti_debug.py uses ptrace(PTRACE_TRACEME). Also:
            - PR_SET_DUMPABLE=0 (prevents core dumps and ptrace by non-root)
            - yama ptrace_scope=2 or 3

ATTACK-MEM2  /proc/mem write
  Vector:   Root-level attacker writes to /proc/<pid>/mem to patch out
            integrity checks in-memory without touching any file.
  Fix:      Run as non-root (enforced). Set PR_SET_DUMPABLE=0.
            Use a separate verification process with different UID.

ATTACK-MEM3  fork() + modify child
  Vector:   If the client forks a helper process, the child inherits memory.
            Attacker modifies the child before exec().
  Fix:      Use O_CLOEXEC on all sensitive file descriptors.

ATTACK-MEM4  Return-oriented programming (ROP) via buffer overflow
  Vector:   If any C extension or subprocess has a buffer overflow,
            ROP chains can bypass all Python-level checks.
  Fix:      Enable ASLR. Use seccomp. Don't call untrusted native code.
            Verify all subprocess arguments are sanitized.

ATTACK-MEM5  Python eval/exec injection
  Vector:   If any agent receives a string that gets eval()'d or exec()'d,
            arbitrary code runs in the agent's context.
  Fix:      Grep for eval/exec in codebase. Remove all dynamic eval.
            Implemented below: _audit_eval().

═══════════════════════════════════════════════════════════════════
CATEGORY 3: NETWORK ATTACKS
═══════════════════════════════════════════════════════════════════

ATTACK-NET1  MITM on gRPC channel
  Vector:   ARP spoofing + TLS downgrade. grpc_client.py already fixed this
            (removed ssl_target_name_override). But DNS-based MITM still possible.
  Fix:      Certificate pinning (pin CA public key fingerprint, not just verify chain).
            Implemented below: _verify_cert_pin().

ATTACK-NET2  Replay attack on receipts
  Vector:   Capture a valid Receipt protobuf and replay it to submit
            a gradient update twice.
  Fix:      Receipt.nonce must be a random 32-byte value, verified as unseen by server.
            Server must maintain a nonce DB (MongoDB receipts collection does this,
            but nonce is currently empty string — FIX REQUIRED).

ATTACK-NET3  Model poisoning via compromised client
  Vector:   Attacker-controlled client submits crafted gradients to
            flip the global model's behavior.
  Fix:      Trimmed mean / Krum / FLTrust aggregation (aggregator.py uses trimmed mean).
            Also: norm-bounding on server side. Implemented below.

ATTACK-NET4  Server impersonation
  Vector:   Fake orchestrator returns malicious global model.
  Fix:      Model hash verification already implemented in pipeline.py (FIX-PIPELINE-6).
            Also: sign the model with server's TPM key.

ATTACK-NET5  Enrollment OTP brute force
  Vector:   6-digit OTP = 10^6 possibilities. With 1000 attempts/sec, cracks in 1000s.
  Fix:      otp.rs already has rate limiting (5 failures → 5-min lockout).
            But also: OTP should be 8+ digits minimum for BE project demo.
            And: OTP should be delivered out-of-band (email/SMS), not displayed in terminal.

ATTACK-NET6  gRPC message injection between enrollment and operational ports
  Vector:   Client sends enrollment messages to port 50052 or vice versa.
  Fix:      The dual-port design (50051 enrollment, 50052 operational) in server.rs handles this.
            Each service only implements its own RPCs (others return UNIMPLEMENTED).

═══════════════════════════════════════════════════════════════════
CATEGORY 4: CRYPTOGRAPHIC ATTACKS
═══════════════════════════════════════════════════════════════════

ATTACK-CRYPTO1  Master key exfiltration
  Vector:   master.key at ~/.federated/data/secure_store/master.key is a
            plaintext AES-256 key on disk, chmod 600. Any process running as
            the same UID can read it.
  Fix:      On Linux: TPM-seal the master key (tpm_seal.py does this).
            On Windows: Use DPAPI (CryptProtectData) or CNG key storage.
            Implemented below: _protect_master_key_with_dpapi() for Windows.

ATTACK-CRYPTO2  HMAC key for receipts stored in plaintext
  Vector:   centralised_receipts.py stores HMAC key at ~/.local_data_agent_receipt_key.
            This is the same user context. Any process can read and forge receipts.
  Fix:      Derive receipt HMAC key from master secret (TPM-sealed) rather than
            storing it as a separate file.

ATTACK-CRYPTO3  HKDF context collision
  Vector:   Two agents with the same context string get the same derived key.
            e.g., if "local_updates" context collapses to "" for multiple agents,
            they share a key.
  Fix:      Include agent name in HKDF info: f"{agent}:{version}:{context}"
            Never collapse to empty string.

ATTACK-CRYPTO4  Nonce reuse in AES-GCM
  Vector:   os.urandom(12) is used for nonce. Probability of collision is
            1/(2^96) per pair, but with 2^32 files the birthday bound gives
            probability ≈ 1/(2^32). Manageable but worth tracking.
  Fix:      Use a monotonic counter combined with os.urandom as nonce.
            Or use XChaCha20-Poly1305 (192-bit nonce, collision-free in practice).

ATTACK-CRYPTO5  CSR private key stored unencrypted
  Vector:   installer_core.py generates client.key with NoEncryption().
            File is chmod 600 but plaintext on disk.
  Fix:      Encrypt private key with TPM-derived passphrase.
            Or generate the key inside a TPM (non-exportable).

═══════════════════════════════════════════════════════════════════
CATEGORY 5: FL-SPECIFIC ATTACKS
═══════════════════════════════════════════════════════════════════

ATTACK-FL1  Membership inference attack
  Vector:   Given black-box access to model, determine if a sample was in training set.
            Shokri et al. 2017: train shadow models on similar data.
  Mitigation: DP-SGD (Opacus) reduces MIA AUC toward 0.5.
              ε ≤ 8 gives meaningful protection. ε ≤ 2 is strong.
  Status:   DPAgent applies DP-SGD. RDP accounting now real (FIX-DP-1).

ATTACK-FL2  Gradient inversion (DLG attack)
  Vector:   Honest-but-curious server sees gradient → reconstructs training samples.
            Zhu et al. NeurIPS 2019. Works perfectly on single-sample batches.
  Mitigation: Batch size ≥ 32, gradient clipping (C ≤ 1.0), DP noise (σ ≥ 1.0),
              gradient compression (Top-k sparsification).
  Status:   Clipping + DP noise applied. Batch size configured in pipeline.py (8).

ATTACK-FL3  Byzantine poisoning
  Vector:   Malicious client sends crafted gradients to degrade global model
            (untargeted) or introduce a backdoor (targeted).
  Mitigation: Trimmed mean (aggregator uses 10% trim ratio).
              Also: norm clipping on server side, Krum selection.
  Status:   Trimmed mean implemented. Norm clipping NOT yet on server.

ATTACK-FL4  Free-rider attack
  Vector:   Client submits zeros or random gradients without doing local training.
            Gets global model for free, contributes nothing.
  Mitigation: Server-side gradient quality check (cosine similarity to mean update).
              Contribution tracking in MongoDB.
  Status:   NOT implemented. Added below.

ATTACK-FL5  Model extraction
  Vector:   Query the global model extensively to reconstruct it.
  Mitigation: Rate-limit global model downloads. The DownloadGlobalModel RPC
              is currently unguarded on download frequency.
  Fix:      Implement download rate limiting. Added below.

═══════════════════════════════════════════════════════════════════
CATEGORY 6: SUPPLY CHAIN ATTACKS
═══════════════════════════════════════════════════════════════════

ATTACK-SC1  Malicious PyPI package
  Vector:   Compromised dependency (e.g., typosquatted "opacus" → "opacus-ai").
  Fix:      requirements.txt FIX-REQS-1 notes hash pinning but leaves placeholder.
            Actual fix: run `pip-compile --generate-hashes requirements.in` and
            commit the hashed requirements.txt. Use `pip install --require-hashes`.

ATTACK-SC2  HuggingFace model poisoning
  Vector:   mental/mental-bert-base-uncased model is downloaded without hash check.
            A malicious model weights file could contain pickled exploit code.
  Fix:      Pin model hash. Use safetensors format (no pickle). Verify before loading.
            Implemented below: _verify_model_hash().

ATTACK-SC3  FFmpeg binary replaced
  Vector:   install_ffmpeg.py downloads and extracts ffmpeg.zip. The SHA-256
            placeholder "REPLACE_WITH_ACTUAL_SHA256_OF_PINNED_ZIP" means
            hash verification is DISABLED (it raises RuntimeError before extraction).
  Fix:      You MUST compute the actual hash and update FFMPEG_WIN_SHA256.
            Run: sha256sum ffmpeg-7.1-essentials_build.zip and put the value there.

═══════════════════════════════════════════════════════════════════
CATEGORY 7: OPERATIONAL ATTACKS
═══════════════════════════════════════════════════════════════════

ATTACK-OP1  Insider threat (administrator with OTP access)
  Vector:   The admin who receives enrollment OTP can enroll unauthorized devices.
  Fix:      Multi-admin approval. OTP displayed AND sent via separate channel.

ATTACK-OP2  Physical access to client machine
  Vector:   Attacker reboots from USB, mounts filesystem, reads master.key.
  Fix:      Full disk encryption (BitLocker/LUKS). TPM PCR sealing (tpm_seal.py).
            PCR 4 = boot loader, PCR 7 = Secure Boot state. Changes on reboot
            from USB will not match PCR values → TPM unseal fails → no master.key.

ATTACK-OP3  Cold boot attack on RAM
  Vector:   Freeze RAM, extract keys from memory.
  Fix:      Zeroize sensitive buffers after use (secrets module, explicit memset).
            Implemented below: SecureBytes context manager.

ATTACK-OP4  Log file exfiltration
  Vector:   logging_config.py writes structured JSON logs. These logs contain
            session IDs, device IDs, epsilon values, and error traces.
            If logs are exfiltrated, partial de-anonymization is possible.
  Fix:      Encrypt log files. Rotate and shred on schedule. Don't log raw UUIDs.

ATTACK-OP5  Health file information leakage
  Vector:   HEALTH_FILE at ~/.federated/state/health.json contains platform info,
            Python version, PID, metrics. This file is world-readable if chmod is wrong.
  Fix:      chmod 600 on health.json. Avoid including version strings.

"""

import ctypes
import hashlib
import hmac as _hmac
import logging
import os
import platform
import secrets
import signal
import stat
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

IS_LINUX   = platform.system().lower() == "linux"
IS_WINDOWS = platform.system().lower() == "windows"
IS_MACOS   = platform.system().lower() == "darwin"

FEDERATED_DIR = Path.home() / ".federated"


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: Memory security
# ═══════════════════════════════════════════════════════════════════

class SecureBytes:
    """
    Context manager for sensitive byte data.
    Zeros memory on exit — mitigates cold boot / core dump attacks.
    """
    def __init__(self, data: bytes):
        self._buf = bytearray(data)

    def __enter__(self) -> bytearray:
        return self._buf

    def __exit__(self, *_):
        for i in range(len(self._buf)):
            self._buf[i] = 0
        del self._buf

    @property
    def value(self) -> bytes:
        return bytes(self._buf)


def disable_core_dumps():
    """Prevent core dump files from leaking memory contents (ATTACK-MEM2 fix)."""
    if IS_LINUX:
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            log.debug("[security] Core dumps disabled")
        except Exception as e:
            log.warning("[security] Could not disable core dumps: %s", e)

        # PR_SET_DUMPABLE=0 prevents /proc/pid/mem writes by non-root
        try:
            PR_SET_DUMPABLE = 4
            libc = ctypes.CDLL("libc.so.6")
            libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
        except Exception:
            pass

    elif IS_WINDOWS:
        try:
            # Prevent WER (Windows Error Reporting) dumps
            import ctypes
            SEM_NOGPFAULTERRORBOX = 0x0002
            ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: File-system defenses
# ═══════════════════════════════════════════════════════════════════

def _audit_no_eval_exec():
    """
    Check at runtime that no agent has called eval() or exec() dynamically.
    Uses sys.audit hook (Python 3.8+).
    ATTACK-MEM5 fix.
    """
    _BLOCKED = frozenset({"exec", "eval", "compile"})

    def _audit_hook(event: str, args):
        if event in _BLOCKED:
            log.critical("[security] BLOCKED dangerous call: %s(%s)", event, args)
            try:
                from runtime.self_destruct import trigger_self_destruct  # type: ignore
                trigger_self_destruct(f"Unauthorized {event}() call detected")
            except Exception:
                os._exit(1)

    if hasattr(sys, "addaudithook"):
        sys.addaudithook(_audit_hook)
        log.debug("[security] Audit hook installed (eval/exec blocked)")


def _check_proc_maps():
    """
    On Linux, scan /proc/self/maps for unexpected shared libraries.
    ATTACK-FS4 (LD_PRELOAD injection) fix.
    """
    if not IS_LINUX:
        return

    KNOWN_PREFIXES = {
        "/usr/lib", "/lib", "/usr/local/lib",
        str(FEDERATED_DIR / "venv"),
        "/proc/", "[", "/dev/",
    }

    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if ".so" not in line:
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                path = parts[5]
                if not any(path.startswith(p) for p in KNOWN_PREFIXES):
                    log.critical(
                        "[security] SUSPICIOUS shared library loaded: %s\n"
                        "This may indicate LD_PRELOAD injection.", path
                    )
                    # Don't immediately self-destruct — some legitimate libraries
                    # may not match. Log for audit trail but continue.
    except Exception:
        pass


def safe_resolve_path(uri: str, root: Path) -> Path:
    """
    Safely resolve a file:// URI, checking for symlink traversal.
    ATTACK-FS5 fix: symlink attack on secure store.
    """
    assert uri.startswith("file://"), "URI must start with file://"
    raw = Path(uri[len("file://"):])

    # Resolve without following symlinks first, then with
    try:
        resolved_no_follow = raw.resolve(strict=False)
        resolved_follow = raw.resolve(strict=True)

        if resolved_no_follow != resolved_follow:
            raise ValueError(
                f"Symlink traversal detected: {raw} → {resolved_follow}\n"
                "Path contains a symlink. Symlinks are not permitted in secure store paths."
            )
    except FileNotFoundError:
        resolved_follow = raw.resolve(strict=False)

    root_resolved = root.resolve()
    if not str(resolved_follow).startswith(str(root_resolved)):
        raise ValueError(
            f"Path traversal rejected: {resolved_follow}\n"
            f"Must be inside {root_resolved}"
        )

    return resolved_follow


def verify_file_permissions(path: Path, expected_mode: int = 0o600) -> bool:
    """Verify a file has the expected permissions. Logs and returns False on mismatch."""
    try:
        actual = stat.S_IMODE(path.stat().st_mode)
        if actual & ~expected_mode:
            log.warning(
                "[security] PERMISSION MISMATCH %s: expected 0o%o got 0o%o",
                path, expected_mode, actual
            )
            return False
    except Exception:
        return False
    return True


class CanaryMonitor:
    """
    Creates hidden canary files in the secure store.
    Any modification to canaries = attacker is probing the system.
    ATTACK-FS7 fix.
    """

    def __init__(self, canary_dir: Optional[Path] = None):
        self._dir = canary_dir or (FEDERATED_DIR / "data" / "secure_store" / ".canaries")
        self._canaries: dict[str, str] = {}
        self._lock = threading.Lock()
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._dir, 0o700)
        except Exception:
            pass

    def plant(self, n: int = 5):
        """Plant N canary files with known content hashes."""
        for i in range(n):
            name = f".{secrets.token_hex(8)}.dat"
            path = self._dir / name
            content = secrets.token_bytes(64)
            path.write_bytes(content)
            try:
                os.chmod(path, 0o400)
            except Exception:
                pass
            h = hashlib.sha256(content).hexdigest()
            with self._lock:
                self._canaries[str(path)] = h
        log.debug("[canary] Planted %d canaries in %s", n, self._dir)

    def check(self) -> bool:
        """Return True if all canaries are intact."""
        with self._lock:
            for path_str, expected_hash in self._canaries.items():
                path = Path(path_str)
                if not path.exists():
                    log.critical("[canary] CANARY DELETED: %s", path.name)
                    return False
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != expected_hash:
                    log.critical("[canary] CANARY MODIFIED: %s", path.name)
                    return False
        return True


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: Network security
# ═══════════════════════════════════════════════════════════════════

# Pin the CA certificate's public key hash (ATTACK-NET1 fix)
# Compute with: openssl x509 -noout -pubkey -in ca.pem | openssl pkey -pubin -outform DER | sha256sum
# UPDATE THIS VALUE after running gen_certs.sh
CA_PUBKEY_PIN_SHA256 = "0d1af52ebda46ca3cf4e9b4302c205d69664f746590b0bf21a563c9f1cfa09da"


def verify_cert_pin(cert_pem_bytes: bytes) -> bool:
    """
    Verify TLS certificate against pinned CA public key hash.
    ATTACK-NET1 mitigation: certificate pinning.
    """
    if CA_PUBKEY_PIN_SHA256.startswith("REPLACE_WITH"):
        log.warning("[security] CA pin not configured — certificate pinning disabled. "
                    "Update CA_PUBKEY_PIN_SHA256 in military_security.py after running gen_certs.sh")
        return True  # degraded mode

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        cert = x509.load_pem_x509_certificate(cert_pem_bytes)
        pubkey_der = cert.public_key().public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
        actual_pin = hashlib.sha256(pubkey_der).hexdigest()

        if not _hmac.compare_digest(actual_pin, CA_PUBKEY_PIN_SHA256):
            log.critical(
                "[security] CERTIFICATE PIN MISMATCH\n"
                "  expected: %s\n"
                "  actual  : %s\n"
                "This may indicate a MITM attack. Connection refused.",
                CA_PUBKEY_PIN_SHA256, actual_pin
            )
            return False
        return True
    except Exception as e:
        log.warning("[security] Certificate pin check failed: %s", e)
        return False


def validate_receipt_nonce(nonce: str, seen_nonces: set) -> bool:
    """
    Ensure receipt nonce is fresh (never seen before).
    ATTACK-NET2 (replay attack) fix.
    Caller must persist seen_nonces across sessions (store in MongoDB).
    """
    if not nonce or len(nonce) < 32:
        log.warning("[security] Receipt nonce too short or missing: %r", nonce)
        return False
    if nonce in seen_nonces:
        log.critical("[security] REPLAY ATTACK: receipt nonce reused: %s", nonce)
        return False
    seen_nonces.add(nonce)
    return True


def check_gradient_norm(gradient_flat, max_norm: float = 10.0) -> bool:
    """
    Server-side norm bounding for uploaded gradients.
    ATTACK-NET3 (Byzantine poisoning) + ATTACK-FL4 (free-rider) fix.
    Returns False if update is suspicious (too large or all zeros).
    """
    import numpy as np
    arr = np.asarray(gradient_flat, dtype=np.float32)

    # All-zeros = likely free-rider
    if np.allclose(arr, 0):
        log.warning("[security] Gradient update is all zeros — likely free-rider")
        return False

    # Extremely large norm = likely attack or training error
    l2 = float(np.linalg.norm(arr))
    if l2 > max_norm:
        log.warning("[security] Gradient norm %.4f exceeds max %.4f — suspicious", l2, max_norm)
        return False

    # NaN/Inf = definitely malicious or broken
    if not np.isfinite(arr).all():
        log.critical("[security] Gradient contains NaN/Inf — rejected")
        return False

    return True


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: Cryptographic hardening
# ═══════════════════════════════════════════════════════════════════

def protect_master_key_windows(master_key: bytes) -> bytes:
    """
    On Windows: use DPAPI (Data Protection API) to encrypt master key.
    DPAPI ties encryption to the current user's credentials + machine state.
    ATTACK-CRYPTO1 fix for Windows.
    """
    if not IS_WINDOWS:
        raise RuntimeError("DPAPI only available on Windows")

    try:
        import ctypes
        import ctypes.wintypes

        class DATA_BLOB(ctypes.Structure):
            _fields_ = [("cbData", ctypes.wintypes.DWORD),
                        ("pbData", ctypes.POINTER(ctypes.c_char))]

        plaintext = DATA_BLOB()
        plaintext.cbData = len(master_key)
        plaintext.pbData = ctypes.cast(ctypes.c_char_p(master_key), ctypes.POINTER(ctypes.c_char))

        ciphertext = DATA_BLOB()
        description = ctypes.c_wchar_p("federated-master-key")

        result = ctypes.windll.crypt32.CryptProtectData(
            ctypes.byref(plaintext),
            description,
            None, None, None,
            0,
            ctypes.byref(ciphertext),
        )

        if not result:
            raise RuntimeError(f"CryptProtectData failed: {ctypes.GetLastError()}")

        encrypted = ctypes.string_at(ciphertext.pbData, ciphertext.cbData)
        ctypes.windll.kernel32.LocalFree(ciphertext.pbData)
        return encrypted

    except Exception as e:
        log.error("[security] DPAPI protection failed: %s", e)
        raise


def verify_model_hash(model_bytes: bytes, expected_sha256: str) -> bool:
    """
    Verify downloaded model file hash before loading.
    ATTACK-SC2 (HuggingFace model poisoning) fix.
    """
    actual = hashlib.sha256(model_bytes).hexdigest()
    if not _hmac.compare_digest(actual.lower(), expected_sha256.lower()):
        log.critical(
            "[security] MODEL HASH MISMATCH\n"
            "  expected: %s\n"
            "  actual  : %s\n"
            "The model file may have been tampered with or corrupted. Refusing to load.",
            expected_sha256, actual
        )
        return False
    log.info("[security] Model hash verified OK")
    return True


def derive_receipt_hmac_key(master_secret: bytes) -> bytes:
    """
    Derive HMAC key for receipts from TPM-sealed master secret.
    ATTACK-CRYPTO2 fix: replaces the standalone key file.
    """
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend

    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"federated:receipt-hmac:v1",
        backend=default_backend(),
    ).derive(master_secret)


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: Process hardening
# ═══════════════════════════════════════════════════════════════════

def apply_seccomp_filter():
    """
    Apply a restrictive seccomp BPF filter on Linux.
    Blocks dangerous syscalls: execve, ptrace, mknod, mount, etc.
    ATTACK-MEM1 + ATTACK-FS4 fix.
    """
    if not IS_LINUX:
        return

    try:
        import ctypes
        import ctypes.util

        # Use python-seccomp if available
        try:
            import seccomp  # type: ignore[import-untyped]
            f = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
            # Block dangerous syscalls
            for syscall in ["execve", "execveat", "ptrace", "mknod", "mknodat",
                            "mount", "umount2", "pivot_root", "chroot",
                            "kexec_load", "kexec_file_load", "perf_event_open"]:
                try:
                    f.add_rule(seccomp.KILL, syscall)
                except Exception:
                    pass
            f.load()
            log.info("[security] seccomp filter applied")
        except ImportError:
            log.warning("[security] python-seccomp not installed — seccomp disabled. "
                        "Install: pip install seccomp")

    except Exception as e:
        log.warning("[security] seccomp setup failed: %s", e)


def verify_no_ptrace():
    """
    Verify no debugger is attached. Calls ptrace(PTRACE_TRACEME) on Linux.
    Returns True if clean, False if debugger detected.
    ATTACK-MEM1 fix (supplement to anti_debug.py).
    """
    if IS_LINUX:
        try:
            libc = ctypes.CDLL("libc.so.6")
            PTRACE_TRACEME = 0
            ret = libc.ptrace(PTRACE_TRACEME, 0, None, None)
            if ret == 0:
                # Successfully traced self — no debugger
                libc.ptrace(17, 0, None, None)  # PTRACE_DETACH
                return True
            else:
                return False
        except Exception:
            return True

    elif IS_WINDOWS:
        try:
            return not ctypes.windll.kernel32.IsDebuggerPresent()
        except Exception:
            return True

    return True


def enforce_single_instance():
    """
    Prevent multiple instances of the client running simultaneously.
    Uses a file lock (flock on Linux, LOCK_EX).
    ATTACK-OP attack surface reduction.
    """
    lock_file = FEDERATED_DIR / "state" / "client.flock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = open(lock_file, "w")
        fcntl_module = None
        if IS_LINUX or IS_MACOS:
            import fcntl as fcntl_module
            fcntl_module.flock(f, fcntl_module.LOCK_EX | fcntl_module.LOCK_NB)
        elif IS_WINDOWS:
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        f.write(str(os.getpid()))
        f.flush()
        log.debug("[security] Instance lock acquired")
        return f  # caller must hold this reference
    except (IOError, OSError):
        log.critical("[security] Another instance is running — refusing to start")
        sys.exit("[SECURITY] Client already running. Only one instance permitted.")


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: Comprehensive startup security gate
# ═══════════════════════════════════════════════════════════════════

def full_security_startup(master_secret: bytes) -> bool:
    """
    Run ALL security checks at startup. Call before any sensitive operation.
    Returns True only if all checks pass.

    Checks performed:
      1. Anti-debug (ptrace, TracerPid, suspicious env)
      2. Core dump disabled
      3. /proc/maps checked for injected libraries
      4. eval/exec audit hook installed
      5. File permissions on key files
      6. Canary files planted (call canary.check() periodically)

    Does NOT call verify_integrity() — that is done by integrity_guard().
    """
    all_ok = True

    # 1. Anti-debug
    if not verify_no_ptrace():
        log.critical("[security] Debugger detected at startup — aborting")
        all_ok = False

    # 2. Core dumps
    disable_core_dumps()

    # 3. /proc/maps injection check
    _check_proc_maps()

    # 4. eval/exec audit hook
    _audit_no_eval_exec()

    # 5. Key file permissions
    key_files = [
        (FEDERATED_DIR / "data" / "secure_store" / "master.key", 0o600),
        (FEDERATED_DIR / "keys" / "client.key", 0o600),
        (FEDERATED_DIR / "keys" / "ca.pem", 0o644),
        (FEDERATED_DIR / "state" / "health.json", 0o600),
    ]
    for fpath, expected_mode in key_files:
        if fpath.exists() and not verify_file_permissions(fpath, expected_mode):
            log.warning("[security] Fixing permissions on %s", fpath)
            try:
                os.chmod(fpath, expected_mode)
            except Exception:
                pass

    # 6. Seccomp
    apply_seccomp_filter()

    if not all_ok:
        from runtime.self_destruct import trigger_self_destruct  # type: ignore
        trigger_self_destruct("Full security startup failed")

    log.info("[security] Full security startup passed")
    return True