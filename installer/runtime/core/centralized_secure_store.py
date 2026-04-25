# centralized_secure_store.py  — FIXED VERSION
#
# FIXES:
#   FIX-SS-1  Path traversal via startswith().
#             The old check:
#               if not str(p).startswith(str(self.root)):
#             fails for adjacent directories.  Example:
#               root  = /home/user/.federated/data/secure_store
#               path  = /home/user/.federated/data/secure_store_evil/x.bin
#             str(path).startswith(str(root)) → TRUE  ← WRONG, should be FALSE
#             Fix: append os.sep so the check becomes a proper prefix test,
#             OR use Path.is_relative_to() (Python 3.9+).  We use both for
#             maximum compatibility.
#
#   FIX-SS-2  Agent / context header byte overflow.
#             bytes([len(agent_b)]) raises ValueError if agent or context
#             name exceeds 255 bytes.  Added explicit length checks.
#
#   FIX-SS-3  decrypt_read did not validate the stored agent_len /
#             context_len against actual buffer length — an attacker who
#             can write a crafted .enc file could trigger an IndexError or
#             cause the wrong context to be used for key derivation.
#             Added bounds checks on every offset step.

import os
import json
import base64
import secrets
from pathlib import Path
from typing import Union
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL PATHS — single source of truth for the whole system
# ─────────────────────────────────────────────────────────────────────────────
_FEDERATED_BASE  = Path.home() / ".federated"
_CANONICAL_ROOT  = _FEDERATED_BASE / "data" / "secure_store"
_GLOBAL_KEY_PATH = _CANONICAL_ROOT / "master.key"


def _path_is_within(child: Path, parent: Path) -> bool:
    """
    FIX-SS-1: Safe containment check that avoids the startswith prefix attack.

    Uses Path.is_relative_to() on Python 3.9+; falls back to comparing
    resolved string paths with an explicit os.sep suffix so that
    /data/secure_store_evil is NOT considered inside /data/secure_store.
    """
    try:
        # Python 3.9+
        return child.is_relative_to(parent)
    except AttributeError:
        # Python 3.8 fallback
        parent_str = str(parent).rstrip(os.sep) + os.sep
        child_str  = str(child)
        return child_str.startswith(parent_str) or child_str == str(parent)


class SecureStore:
    """
    Centralized AES-GCM encrypted store.

    master.key is ALWAYS stored at _GLOBAL_KEY_PATH so every agent instance
    encrypts/decrypts with the same underlying secret.  Per-agent key
    isolation is achieved via HKDF with an (agent, context) info tag.
    """

    _MAX_NAME_BYTES = 200   # hard cap on agent / context label length

    def __init__(
        self,
        agent: str = "generic",
        root: Union[str, Path] = None,
        key_path: Union[str, Path] = None,
    ):
        self.agent = agent

        if root is None:
            root = _CANONICAL_ROOT
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

        if key_path is None:
            key_path = _GLOBAL_KEY_PATH
        self.key_path = Path(key_path).expanduser().resolve()
        self.key_path.parent.mkdir(parents=True, exist_ok=True)

        self.master_key = self._load_or_create_master_key()

    # ─────────────────────────────────────────────────────────────────────────
    # Key management
    # ─────────────────────────────────────────────────────────────────────────

    def _load_or_create_master_key(self) -> bytes:
        if self.key_path.exists():
            txt = self.key_path.read_text().strip()
            try:
                return base64.b64decode(txt)
            except Exception:
                return self.key_path.read_bytes()
        else:
            k = secrets.token_bytes(32)
            self.key_path.write_text(base64.b64encode(k).decode())
            try:
                os.chmod(self.key_path, 0o600)
            except Exception:
                pass
            return k

    def _derive_key(self, context: str) -> bytes:
        info = f"{self.agent}:{context}".encode()
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=info,
            backend=default_backend(),
        )
        return hkdf.derive(self.master_key)

    # ─────────────────────────────────────────────────────────────────────────
    # Context helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _uri_to_context(uri: str) -> str:
        parent_name = Path(uri[len("file://"):]).parent.name
        if "local_updates" in parent_name:
            return ""
        return parent_name

    # ─────────────────────────────────────────────────────────────────────────
    # Internal path resolver
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_and_validate(self, uri: str) -> Path:
        """
        Resolve a file:// URI to a canonical Path that is guaranteed to be
        inside self.root.  Raises ValueError on path-traversal attempts.
        """
        assert uri.startswith("file://"), "URI must start with file://"
        p = Path(uri[len("file://"):]).resolve()
        if not _path_is_within(p, self.root):       # FIX-SS-1
            raise ValueError(
                f"Access outside secure store is not allowed.\n"
                f"  requested : {p}\n"
                f"  store root: {self.root}"
            )
        return p

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def encrypt_write(self, uri: str, data: bytes) -> str:
        assert data, "Refusing to encrypt empty payload"

        p = self._resolve_and_validate(uri)         # FIX-SS-1
        p.parent.mkdir(parents=True, exist_ok=True)

        context = self._uri_to_context(uri)
        key     = self._derive_key(context)
        aesgcm  = AESGCM(key)
        nonce   = os.urandom(12)
        ct      = aesgcm.encrypt(nonce, data, None)

        # FIX-SS-2: guard against names longer than 255 bytes
        agent_b   = self.agent.encode()
        context_b = context.encode()
        if len(agent_b) > self._MAX_NAME_BYTES:
            raise ValueError(
                f"SecureStore: agent name too long ({len(agent_b)} bytes > {self._MAX_NAME_BYTES})"
            )
        if len(context_b) > self._MAX_NAME_BYTES:
            raise ValueError(
                f"SecureStore: context label too long ({len(context_b)} bytes > {self._MAX_NAME_BYTES})"
            )

        # BINARY FORMAT:
        # [1 byte  version=0x01]
        # [1 byte  agent_len]
        # [N bytes agent]
        # [1 byte  context_len]
        # [M bytes context]
        # [12 bytes nonce]
        # [remaining: ciphertext + 16-byte GCM tag]
        header = (
            b"\x01"
            + bytes([len(agent_b)])   + agent_b
            + bytes([len(context_b)]) + context_b
        )
        p.write_bytes(header + nonce + ct)
        return uri

    def decrypt_read(self, uri: str) -> bytes:
        p = self._resolve_and_validate(uri)         # FIX-SS-1

        raw = p.read_bytes()
        if not raw:
            raise ValueError(f"SecureStore: file is empty: {p}")

        if raw[0] == 0x01:
            # FIX-SS-3: validate every offset before use
            offset = 1

            if offset >= len(raw):
                raise ValueError(f"SecureStore: truncated header in {p}")
            agent_len = raw[offset]; offset += 1

            if offset + agent_len > len(raw):
                raise ValueError(f"SecureStore: agent_len={agent_len} overruns buffer in {p}")
            offset += agent_len          # skip agent bytes (not used for decryption)

            if offset >= len(raw):
                raise ValueError(f"SecureStore: truncated context_len in {p}")
            context_len = raw[offset]; offset += 1

            if offset + context_len > len(raw):
                raise ValueError(f"SecureStore: context_len={context_len} overruns buffer in {p}")
            context_b      = raw[offset:offset + context_len]; offset += context_len
            stored_context = context_b.decode()

            if offset + 12 > len(raw):
                raise ValueError(f"SecureStore: nonce missing in {p}")
            nonce  = raw[offset:offset + 12]; offset += 12
            ct     = raw[offset:]

            if not ct:
                raise ValueError(f"SecureStore: ciphertext is empty in {p}")

        else:
            # Legacy JSON format (backward compat — read-only)
            try:
                parsed = json.loads(raw.decode())
                nonce  = base64.b64decode(parsed["nonce"])
                ct     = base64.b64decode(parsed["ct"])
                stored_context = parsed.get("context", self._uri_to_context(uri))
            except Exception as e:
                raise ValueError(
                    f"SecureStore: unrecognised file format in {p}: {e}"
                )

        key    = self._derive_key(stored_context)
        aesgcm = AESGCM(key)

        try:
            return aesgcm.decrypt(nonce, ct, None)
        except Exception:
            raise ValueError(
                f"SecureStore: decryption failed for {p}\n"
                f"  agent={self.agent!r}  context={stored_context!r}\n"
                f"  master_key_path={self.key_path}\n"
                "  Likely cause: encrypted by a different master.key."
            )