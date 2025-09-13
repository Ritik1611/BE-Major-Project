import hmac, hashlib, time, json, uuid
from pathlib import Path

MASTER_KEY = b"supersecretkey"  # ⚠️ replace with env var in production

class ReceiptManager:
    @staticmethod
    def sign(data: dict) -> dict:
        payload = json.dumps(data, sort_keys=True).encode()
        sig = hmac.new(MASTER_KEY, payload, hashlib.sha256).hexdigest()
        return {**data, "signature": sig}

    @staticmethod
    def write_receipt(data: dict, out_dir="./receipts") -> str:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        signed = ReceiptManager.sign(data)
        fname = f"receipt_{uuid.uuid4().hex}.json"
        path = Path(out_dir) / fname
        with open(path, "w") as f:
            json.dump(signed, f, indent=2)
        return f"file://{path}"
