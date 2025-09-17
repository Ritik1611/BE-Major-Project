import os, io, time, torch
from dp_agent.dp_agent import DPAgent

# 👇 centralized receipts + secure store
from ..centralised_receipts import CentralReceiptManager
from ..centralized_secure_store import SecureStore


def make_state_dict():
    return {"w1": torch.randn(20, 20), "b1": torch.randn(20)}


def diff_privacy():
    # 🔹 Ensure dirs exist
    os.makedirs("secure_store/local_updates", exist_ok=True)
    os.makedirs("receipts", exist_ok=True)

    # 🔹 Trainer creates an update and stores via SecureStore
    fname = f"trainer_{int(time.time() * 1000)}.pt.enc"
    path = os.path.join("secure_store/local_updates", fname)

    sd = make_state_dict()
    buf = io.BytesIO()
    torch.save(sd, buf)
    raw = buf.getvalue()

    store = SecureStore("./secure_store")
    store.encrypt_write("file://" + path, raw)

    # 🔹 Create trainer receipt using CentralReceiptManager
    rm = CentralReceiptManager()
    trainer_receipt = rm.create_receipt(
        agent="trainer-agent",
        session_id=f"sess-{int(time.time())}",
        operation="train_step",
        params={
            "epochs": 1,
            "batch_size": 32,
            "dataset_size": 1000,
        },
        outputs=["file://" + path],
    )
    trainer_receipt_uri = rm.write_receipt(trainer_receipt, out_dir="receipts")

    print("Trainer receipt created:", trainer_receipt_uri)

    # 🔹 Run DP agent
    dp = DPAgent(
        clip_norm=1.0,
        noise_multiplier=1.2,
        secure_store_dir="secure_store/local_updates",
        receipts_dir="receipts",
    )

    dp_result = dp.process_local_update(
        trainer_receipt["outputs"][0], metadata=trainer_receipt
    )
    print("DP receipt created:", dp_result["receipt_uri"])


if __name__ == "__main__":
    diff_privacy()
