# enc_agent/watcher.py
import os, time
from enc_agent.enc_agent import EncryptionAgent

def watch_for_dp_updates(receipts_dir="receipts"):
    agent = EncryptionAgent()
    seen = set()
    print("[EncryptionAgent] Watching for DP receipts...")

    while True:
        for fname in os.listdir(receipts_dir):
            if fname.startswith("dp_") and fname.endswith(".json"):
                fpath = os.path.join(receipts_dir, fname)
                if fpath not in seen:
                    seen.add(fpath)
                    agent.process_dp_update(fpath)
        time.sleep(2)

if __name__ == "__main__":
    watch_for_dp_updates()
