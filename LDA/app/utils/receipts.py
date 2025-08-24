# app/utils/receipts.py
import json
import hashlib
import hmac
import base64
from datetime import datetime
from pathlib import Path
import secrets
from typing import Dict, Any

class ReceiptManager:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        key_path = self.root / "receipt.key"
        if key_path.exists():
            self.hmac_key = base64.b64decode(key_path.read_text())
        else:
            self.hmac_key = secrets.token_bytes(32)
            key_path.write_text(base64.b64encode(self.hmac_key).decode())

    def _build_payload(self, operation: str, input_meta: dict, output_uri: str, extra: Dict[str, Any] = None) -> Dict[str, Any]:
        payload = {
            "operation": operation,
            "input_meta": input_meta,
            "output_uri": output_uri,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        if extra:
            payload["extra"] = extra
        return payload

    def _sign_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        signature = hmac.new(self.hmac_key, payload_bytes, hashlib.sha256).digest()
        payload_signed = dict(payload)  # copy
        payload_signed["signature"] = base64.b64encode(signature).decode()
        return payload_signed

    def create_receipt(self, operation: str, input_meta: dict, output_uri: str, extra: Dict[str, Any] = None) -> str:
        """Generates a signed receipt, stores it on disk, and returns the path."""
        payload = self._build_payload(operation, input_meta, output_uri, extra)
        signed = self._sign_payload(payload)

        # Store receipt
        filename = f"{operation}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
        path = self.root / filename
        path.write_text(json.dumps(signed, indent=2))
        return str(path)

    def sign_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Sign a payload dict and return the signed dict (does not write to disk)."""
        return self._sign_payload(payload)

    def verify_receipt(self, receipt_path: str) -> bool:
        """Verifies that a receipt file has a valid signature."""
        payload = json.loads(Path(receipt_path).read_text())
        sig = base64.b64decode(payload.pop("signature"))
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        expected_sig = hmac.new(self.hmac_key, payload_bytes, hashlib.sha256).digest()
        return hmac.compare_digest(sig, expected_sig)


# Convenience top-level function expected by app.main
_default_receipt_mgr = ReceiptManager("./receipts")

def make_receipt(agent: str, session_id: str, op: str, params: dict, outputs: list) -> Dict[str, Any]:
    """
    Build a standard receipt dict for agent operations. This returns a signed dict
    (so the caller can encrypt/store it as they wish).
    """
    payload = {
        "agent": agent,
        "session_id": session_id,
        "op": op,
        "params": params,
        "outputs": outputs,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    return _default_receipt_mgr.sign_payload(payload)
