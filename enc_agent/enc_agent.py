# enc_agent/enc_agent.py
import os
import json
import time
import base64
import hmac
import hashlib
from typing import Optional, Dict, Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.fernet import Fernet

# Optional libs for HE/SMPC. Install as required.
try:
    from Pyfhel import Pyfhel, PyCtxt, PyPtxt
    HAS_PYFHEL = True
except Exception:
    HAS_PYFHEL = False

# placeholder for KMS (AWS example)
try:
    import boto3
    HAS_BOTO3 = True
except Exception:
    HAS_BOTO3 = False


class EncryptionAgent:
    def __init__(self,
                 final_store_dir: str = "secure_store/final_updates",
                 receipts_dir: str = "receipts",
                 mode: str = "aes",
                 symmetric_key: Optional[bytes] = None,
                 kms_key_id: Optional[str] = None,
                 hmac_key: Optional[bytes] = None):

        # 🔹 set mode first
        self.mode = mode.lower()

        self.final_store_dir = final_store_dir
        self.receipts_dir = receipts_dir
        os.makedirs(self.final_store_dir, exist_ok=True)
        os.makedirs(self.receipts_dir, exist_ok=True)

        # 🔹 Fernet mode
        if self.mode == "fernet":
            if symmetric_key is None:
                if os.path.exists("keys/fernet.key"):
                    with open("keys/fernet.key", "rb") as f:
                        symmetric_key = f.read().strip()
                else:
                    os.makedirs("keys", exist_ok=True)
                    symmetric_key = Fernet.generate_key()
                    with open("keys/fernet.key", "wb") as f:
                        f.write(symmetric_key)
            self.fernet = Fernet(symmetric_key)
        else:
            self.fernet = None

        # 🔹 AES mode
        if symmetric_key is None and self.mode == "aes":
            symmetric_key = AESGCM.generate_key(bit_length=256)
        self.aes_key = symmetric_key

        # 🔹 fallback Fernet if still none
        if symmetric_key is None and self.mode == "fernet":
            symmetric_key = Fernet.generate_key()
        self.fernet = Fernet(symmetric_key) if (self.mode == "fernet" and symmetric_key is not None) else None

        self.kms_key_id = kms_key_id
        self.hmac_key = hmac_key or b"enc_demo_hmac_32bytes_long____"[:32]


        # Setup KMS client if available
        if HAS_BOTO3 and self.mode == "kms_envelope":
            self.kms = boto3.client("kms")
        else:
            self.kms = None

        # Pyfhel for CKKS (if mode selected)
        self.pyfhel = None
        if self.mode == "he_ckks":
            if not HAS_PYFHEL:
                raise RuntimeError("Pyfhel not available. Install with `pip install pyfhel`")
            self.pyfhel = Pyfhel()
            # example params, tune per use-case
            self.pyfhel.contextGen(scheme='CKKS', n=2**14, scale=2**30)  # heavy; tune n/scale
            self.pyfhel.keyGen()

    # ---------------- symmetric AES-GCM ----------------
    def encrypt_aes_gcm(self, plaintext: bytes) -> Dict[str, Any]:
        aes = AESGCM(self.aes_key)
        nonce = os.urandom(12)  # 96-bit recommended for GCM
        ct = aes.encrypt(nonce, plaintext, associated_data=None)
        return {"ciphertext": ct, "nonce": nonce, "scheme": "AES-GCM-256"}

    def decrypt_aes_gcm(self, nonce: bytes, ciphertext: bytes) -> bytes:
        aes = AESGCM(self.aes_key)
        return aes.decrypt(nonce, ciphertext, associated_data=None)

    # ---------------- fernet (simple) ----------------
    def encrypt_fernet(self, plaintext: bytes) -> Dict[str, Any]:
        token = self.fernet.encrypt(plaintext)
        return {"ciphertext": token, "nonce": None, "scheme": "Fernet"}

    def decrypt_fernet(self, token: bytes) -> bytes:
        return self.fernet.decrypt(token)

    # ---------------- KMS envelope encryption (AWS) ----------------
    def encrypt_kms_envelope(self, plaintext: bytes) -> Dict[str, Any]:
        """
        Envelope encryption pattern:
        - Generate a data-key via KMS (data key plaintext + ciphertext)
        - Use data-key to encrypt locally with AES-GCM
        - Return ciphertext and encrypted data-key (so aggregator can decrypt via KMS)
        """
        if not HAS_BOTO3 or self.kms is None:
            raise RuntimeError("boto3/AWS KMS not configured")

        # Generate data key (plaintext + ciphertext)
        resp = self.kms.generate_data_key(KeyId=self.kms_key_id, KeySpec='AES_256')
        data_key_plain = resp['Plaintext']         # bytes
        data_key_ciphertext = resp['CiphertextBlob']  # encrypted under KMS
        # local AES encrypt using the plaintext data key:
        aes = AESGCM(data_key_plain)
        nonce = os.urandom(12)
        ct = aes.encrypt(nonce, plaintext, associated_data=None)
        # return encrypted payload + wrapped key
        return {
            "ciphertext": ct,
            "nonce": nonce,
            "scheme": "KMS-Envelope-AES-GCM",
            "wrapped_key": base64.b64encode(data_key_ciphertext).decode('utf-8'),
            "kms_key_id": self.kms_key_id
        }

    def decrypt_kms_envelope(self, nonce: bytes, ciphertext: bytes, wrapped_key_b64: str) -> bytes:
        if not HAS_BOTO3 or self.kms is None:
            raise RuntimeError("boto3/AWS KMS not configured")
        wrapped = base64.b64decode(wrapped_key_b64)
        resp = self.kms.decrypt(CiphertextBlob=wrapped)
        data_key_plain = resp['Plaintext']
        aes = AESGCM(data_key_plain)
        return aes.decrypt(nonce, ciphertext, associated_data=None)

    # ---------------- Homomorphic encryption (CKKS) ----------------
    def encrypt_he_ckks(self, plaintext: bytes) -> Dict[str, Any]:
        """
        Demo: we will treat plaintext as serialized float32 array (e.g., model weights flattened).
        Real HE requires packing floats into CKKS plaintexts.
        Uses Pyfhel for CKKS encryption. Note: Pyfhel objects are not serializable directly;
        we save ciphertext bytes (Pyfhel.to_bytes) and store public params separately.
        """
        if not self.pyfhel:
            raise RuntimeError("Pyfhel not initialized")
        # For demo: assume user passes a list of floats in JSON bytes
        import numpy as np
        arr = np.frombuffer(plaintext, dtype=np.float32)
        # pack in plaintext(s) — CKKS packs arrays
        ptxt = self.pyfhel.encodeFrac(arr.tolist())
        ctxt = self.pyfhel.encryptPtxt(ptxt)
        ctxt_bytes = ctxt.to_bytes()
        return {"ciphertext": ctxt_bytes, "scheme": "CKKS-Pyfhel", "params": "pyfhel-ctx-placeholder"}

    def decrypt_he_ckks(self, ctxt_bytes: bytes) -> bytes:
        if not self.pyfhel:
            raise RuntimeError("Pyfhel not initialized")
        ctxt = PyCtxt(pyfhel=self.pyfhel, bytestring=ctxt_bytes)
        ptxt = self.pyfhel.decryptFrac(ctxt)
        import numpy as np
        arr = np.array(ptxt, dtype=np.float32)
        return arr.tobytes()

    # ---------------- SMPC (placeholder) ----------------
    def create_smpc_shares(self, plaintext: bytes) -> Dict[str, Any]:
        """
        In SMPC flow, the Enc Agent generates shares or converts the ciphertext into the
        protocol-specific representation expected by Aggregator/SMPC.
        Real SMPC requires a framework (CrypTen / MP-SPDZ). This is a placeholder to show structure.
        """
        # Example: split bytes into N shares (simple XOR secret sharing demo)
        N = 3
        import os
        shares = []
        prev = plaintext
        for i in range(N - 1):
            r = os.urandom(len(plaintext))
            shares.append(base64.b64encode(r).decode('utf-8'))
            prev = bytes(x ^ y for x, y in zip(prev, r))
        shares.append(base64.b64encode(prev).decode('utf-8'))
        return {"shares": shares, "scheme": "XOR-secret-sharing-demo", "num_shares": N}

    # ---------------- main entry ----------------
    def process_dp_update(self, dp_receipt_path: str) -> Dict[str, Any]:
        # Read dp receipt
        with open(dp_receipt_path, "r") as rf:
            dp_receipt = json.load(rf)

        dp_update_uri = dp_receipt.get("local_update_uri")
        if not dp_update_uri or not dp_update_uri.startswith("file://"):
            raise ValueError("dp_receipt must include file:// local_update_uri")

        dp_path = dp_update_uri[len("file://"):]

        with open(dp_path, "rb") as f:
            dp_bytes = f.read()

        # Choose algorithm
        if self.mode == "aes":
            res = self.encrypt_aes_gcm(dp_bytes)
            ciphertext = res["ciphertext"]
            meta = {"nonce": base64.b64encode(res["nonce"]).decode('utf-8'), "scheme": res["scheme"]}
        elif self.mode == "fernet":
            res = self.encrypt_fernet(dp_bytes)
            ciphertext = res["ciphertext"]
            meta = {"scheme": res["scheme"]}
        elif self.mode == "kms_envelope":
            res = self.encrypt_kms_envelope(dp_bytes)
            ciphertext = res["ciphertext"]
            meta = {"nonce": base64.b64encode(res["nonce"]).decode('utf-8'),
                    "scheme": res["scheme"],
                    "wrapped_key": res["wrapped_key"],
                    "kms_key_id": res["kms_key_id"]}
        elif self.mode == "he_ckks":
            res = self.encrypt_he_ckks(dp_bytes)
            ciphertext = res["ciphertext"]
            meta = {"scheme": res["scheme"], "params": res.get("params")}
        elif self.mode == "smpc":
            res = self.create_smpc_shares(dp_bytes)
            ciphertext = None
            meta = res
        else:
            raise ValueError(f"Unknown encryption mode: {self.mode}")

        # write final ciphertext to final store (if available)
        ts = int(time.time() * 1000)
        final_fname = f"encdp_{ts}.pt.enc2" if ciphertext is not None else f"encdp_shares_{ts}.json"
        final_path = os.path.join(self.final_store_dir, final_fname)

        if ciphertext is not None:
            with open(final_path, "wb") as wf:
                wf.write(ciphertext)
            final_uri = "file://" + final_path
        else:
            # SMPC shares stored as JSON
            with open(final_path, "w") as wf:
                json.dump(meta, wf)
            final_uri = "file://" + final_path

        # signature (demo: HMAC over final ciphertext or JSON)
        if ciphertext is not None:
            sig = hmac.new(self.hmac_key, ciphertext, hashlib.sha256).hexdigest()
        else:
            sig = hmac.new(self.hmac_key, json.dumps(meta).encode('utf-8'), hashlib.sha256).hexdigest()

        enc_receipt = {
            "type": "encryption_receipt",
            "dp_receipt": dp_receipt_path,
            "dp_update_uri": dp_update_uri,
            "final_update_uri": final_uri,
            "encryption_scheme": meta.get("scheme"),
            "metadata": meta,
            "timestamp": time.time(),
            "signature": sig
        }

        # merge some useful dp metadata for audit
        for k in ("epochs", "batch_size", "dataset_size", "session_parquet"):
            if k in dp_receipt:
                enc_receipt[k] = dp_receipt[k]

        # store receipt
        outname = os.path.basename(final_fname).replace(".pt.enc2", ".json")
        rpath = os.path.join(self.receipts_dir, outname)
        with open(rpath, "w") as rf:
            json.dump(enc_receipt, rf, indent=2)

        return enc_receipt
