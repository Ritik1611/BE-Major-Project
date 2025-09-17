import os
from dp_agent.run_demo_single_process import diff_privacy as run_dp_demo
from enc_agent.enc_agent import EncryptionAgent


def encrypt_agent():
    # 🔹 Run DP demo first (produces dp_receipt in receipts/)
    run_dp_demo()

    # 🔹 Find the latest DP receipt
    receipts_dir = "receipts"
    files = [f for f in os.listdir(receipts_dir) if f.endswith(".json")]
    if not files:
        print("No DP receipts found!")
        return
    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(receipts_dir, f)))
    dp_receipt_path = os.path.join(receipts_dir, latest)

    print(f"Encrypting DP update from: {dp_receipt_path}")

    # 🔹 Use AES mode by default (can switch to "kms_envelope", "he_ckks", "smpc")
    enc_agent = EncryptionAgent(mode="aes")
    result = enc_agent.process_dp_update(dp_receipt_path)

    print("✅ Encryption receipt created and stored:")
    print("   Receipt URI:", result["receipt_uri"])
    print("   Receipt JSON:", result["receipt"])


if __name__ == "__main__":
    encrypt_agent()
