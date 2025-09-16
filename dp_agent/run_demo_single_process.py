# dp_agent/run_demo_single_process.py
import os, io, json, time
from cryptography.fernet import Fernet
import torch
from dp_agent.dp_agent import DPAgent

def make_state_dict():
    return {"w1": torch.randn(20,20), "b1": torch.randn(20)}

def main():
    # 🔹 Load or generate Fernet key (shared across pipeline)
    os.makedirs("keys", exist_ok=True)
    key_path = "keys/fernet.key"
    if os.path.exists(key_path):
        with open(key_path, "rb") as f:
            demo_key = f.read().strip()
    else:
        demo_key = Fernet.generate_key()
        with open(key_path, "wb") as f:
            f.write(demo_key)
        print("Generated new Fernet key -> keys/fernet.key")

    hmac_key = b"dp_demo_hmac_key_32_bytes_long____"[:32]

    # 🔹 Trainer creates encrypted update
    os.makedirs("secure_store/local_updates", exist_ok=True)
    fname = f"trainer_{int(time.time()*1000)}.pt.enc"
    path = os.path.join("secure_store/local_updates", fname)

    sd = make_state_dict()
    buf = io.BytesIO()
    torch.save(sd, buf)
    raw = buf.getvalue()

    f = Fernet(demo_key)
    enc = f.encrypt(raw)

    with open(path, "wb") as wf:
        wf.write(enc)

    # 🔹 Trainer receipt
    receipt = {
        "type": "train_receipt",
        "local_update_uri": "file://" + path,
        "epochs": 1,
        "batch_size": 32,
        "dataset_size": 1000,
        "timestamp": time.time()
    }
    os.makedirs("receipts", exist_ok=True)
    rfname = fname.replace(".pt.enc", ".json")
    with open(os.path.join("receipts", rfname), "w") as rf:
        json.dump(receipt, rf, indent=2)

    # 🔹 Run DP agent with same Fernet key
    dp = DPAgent(
        clip_norm=1.0,
        noise_multiplier=1.2,
        secure_store_dir="secure_store/local_updates",
        receipts_dir="receipts",
        fernet_key=demo_key,   # same Fernet key
        hmac_key=hmac_key
    )
    dp_receipt = dp.process_local_update(receipt['local_update_uri'], metadata=receipt)
    print("DP receipt created:", dp_receipt)

if __name__ == "__main__":
    main()
