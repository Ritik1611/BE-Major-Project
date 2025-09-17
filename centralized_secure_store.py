# centralized_secure_store.py
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


class SecureStore:
    """
    Centralized AES-GCM encrypted store.
    Keys are derived deterministically from the parent directory path
    using HKDF, so files in the same directory can always be decrypted.
    """

    def __init__(self, root: Union[str, Path] = "./secure_store", key_path: Union[str, Path] = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if key_path is None:
            key_path = self.root / "master.key"
        self.key_path = Path(key_path)
        self.master_key = self._load_or_create_master_key()

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
            return k

    def _derive_key(self, context: str) -> bytes:
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=context.encode(),
            backend=default_backend(),
        )
        return hkdf.derive(self.master_key)

    # ------------------ write ------------------
    def encrypt_write(self, uri: str, data: bytes):
        assert uri.startswith("file://"), "URI must start with file://"
        p = Path(uri[len("file://"):])
        p.parent.mkdir(parents=True, exist_ok=True)

        key = self._derive_key(str(p.parent))
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ct = aesgcm.encrypt(nonce, data, None)

        payload = {"nonce": base64.b64encode(nonce).decode(),
                   "ct": base64.b64encode(ct).decode()}
        p.write_text(json.dumps(payload))

    # ------------------ read ------------------
    def decrypt_read(self, uri: str) -> bytes:
        assert uri.startswith("file://"), "URI must start with file://"
        p = Path(uri[len("file://"):])
        payload = json.loads(p.read_text())
        nonce = base64.b64decode(payload["nonce"])
        ct = base64.b64decode(payload["ct"])

        key = self._derive_key(str(p.parent))
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ct, None)
