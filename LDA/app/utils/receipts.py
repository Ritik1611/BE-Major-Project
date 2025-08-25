# app/utils/receipts.py
import json
import hashlib
import hmac
import base64
from datetime import datetime
from pathlib import Path
import secrets
from typing import Any, Dict, List

class ReceiptManager:
    """
    Simple receipt manager that stores per-operation receipts under a root_dir.
    Each root_dir will get a 'receipt.key' (HMAC key) to sign receipts.
    """
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        key_path = self.root / "receipt.key"
        if key_path.exists():
            self.hmac_key = base64.b64decode(key_path.read_text())
        else:
            self.hmac_key = secrets.token_bytes(32)
            key_path.write_text(base64.b64encode(self.hmac_key).decode())

    def create_receipt(self, operation: str, input_meta: Dict[str, Any], output_uri: str) -> str:
        payload = {
            "operation": operation,
            "input_meta": input_meta,
            "output_uri": output_uri,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        signature = hmac.new(self.hmac_key, payload_bytes, hashlib.sha256).digest()
        payload["signature"] = base64.b64encode(signature).decode()

        filename = f"{operation}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
        path = self.root / filename
        path.write_text(json.dumps(payload, indent=2))
        return str(path)

    def verify_receipt(self, receipt_path: str) -> bool:
        p = Path(receipt_path)
        payload = json.loads(p.read_text())
        sig_b64 = payload.pop("signature", None)
        if sig_b64 is None:
            return False
        sig = base64.b64decode(sig_b64)
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        expected = hmac.new(self.hmac_key, payload_bytes, hashlib.sha256).digest()
        return hmac.compare_digest(sig, expected)


# ----------------------------
# Top-level helper (used by app.main)
# ----------------------------
_global_key_path = Path.home() / ".local_data_agent_receipt_key"

def _load_global_hmac_key() -> bytes:
    if _global_key_path.exists():
        return base64.b64decode(_global_key_path.read_text())
    else:
        k = secrets.token_bytes(32)
        _global_key_path.write_text(base64.b64encode(k).decode())
        return k

def make_receipt(agent: str, session_id: str, op: str, params: Dict[str, Any], outputs: List[str]) -> Dict[str, Any]:
    """
    Create a session-level receipt payload and sign it with a global HMAC key.
    Returned object is JSON-serializable.
    """
    payload = {
        "agent": agent,
        "session_id": session_id,
        "operation": op,
        "params": params,
        "outputs": outputs,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    payload_bytes = json.dumps(payload, sort_keys=True).encode()
    key = _load_global_hmac_key()
    signature = hmac.new(key, payload_bytes, hashlib.sha256).digest()
    payload["signature"] = base64.b64encode(signature).decode()
    return payload
