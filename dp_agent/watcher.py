# dp_agent/watcher.py
import os, time, json
from dp_agent.dp_agent import DPAgent
from cryptography.fernet import Fernet

def run(receipts_dir="receipts", secure_store="secure_store/local_updates"):
    # DEMO: Use a shared key hardcoded for local testing.
    # In production, DPAgent should fetch decryption key from KMS or call Encryption Agent.
    demo_key = Fernet.generate_key()
    hmac_key = b"dp_demo_hmac_key_32_bytes_long____"[:32]
    dp = DPAgent(clip_norm=1.0, noise_multiplier=1.2,
                 secure_store_dir=secure_store, receipts_dir=receipts_dir,
                 fernet_key=demo_key, hmac_key=hmac_key)

    print("DP watcher started (demo). Replace demo_key with real key in production.")
    processed = set()
    while True:
        for f in os.listdir(receipts_dir):
            if not f.endswith(".json"):
                continue
            path = os.path.join(receipts_dir, f)
            if path in processed:
                continue
            try:
                with open(path, "r") as rf:
                    rec = json.load(rf)
            except:
                continue
            if rec.get("type","").startswith("train_receipt"):
                try:
                    dp_receipt = dp.process_local_update(rec["local_update_uri"], metadata=rec)
                    print("DP produced:", dp_receipt["local_update_uri"])
                except Exception as e:
                    print("DP processing error for", path, e)
            processed.add(path)
        time.sleep(1)

if __name__ == "__main__":
    run()
