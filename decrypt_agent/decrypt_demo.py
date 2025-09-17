import os
import json
from cryptography.fernet import Fernet
import torch

def decrypt_update(receipt_path: str):
    # Load Fernet key
    with open("keys/fernet.key", "rb") as f:
        fernet_key = f.read()
    fernet = Fernet(fernet_key)

    # Load encryption receipt
    with open(receipt_path, "r") as f:
        enc_receipt = json.load(f)

    cipher_path = enc_receipt["final_update_uri"].replace("file://", "")
    if not os.path.exists(cipher_path):
        raise FileNotFoundError(f"Ciphertext not found at {cipher_path}")

    # Read ciphertext
    with open(cipher_path, "rb") as f:
        ciphertext = f.read()

    # Decrypt
    plaintext = fernet.decrypt(ciphertext)
    print("✅ Decryption successful! Bytes length:", len(plaintext))

    # Save decrypted model
    out_dir = "secure_store/decrypted_updates"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(cipher_path).replace(".pt.enc2", ".pt"))

    with open(out_path, "wb") as f:
        f.write(plaintext)

    print("💾 Decrypted model saved to:", out_path)

    # Try to load with torch
    try:
        state_dict = torch.load(out_path, weights_only=False)  # 🔹 force load full pickle
        print("🔑 Loaded state_dict keys:", list(state_dict.keys()))
    except Exception as e:
        print("⚠️ Could not load decrypted model with torch:", e)

    return out_path

if __name__ == "__main__":
    receipts = [f for f in os.listdir("receipts") if f.startswith("encdp_") and f.endswith(".json")]
    if not receipts:
        print("No encrypted receipts found.")
    else:
        latest = max(receipts, key=lambda f: os.path.getmtime(os.path.join("receipts", f)))
        print("Using receipt:", latest)
        decrypt_update(os.path.join("receipts", latest))
