# trainer_agent/main.py
import torch, io, uuid, time, json
from pathlib import Path
from typing import Dict, Any

from trainer import train_model
from utils import read_embeddings_from_parquet

# 👇 centralized imports
from ..centralized_secure_store import SecureStore
from ..centralised_receipts import CentralReceiptManager


def train(
    session_parquet: str,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Train on embeddings from parquet and produce:
      - Encrypted local update (via SecureStore)
      - Signed + encrypted receipt (via CentralReceiptManager)
    """
    # 1) Init secure store + receipts
    store = SecureStore("./secure_store")
    rm = CentralReceiptManager()

    # 2) Read embeddings
    embs = read_embeddings_from_parquet(session_parquet)
    if len(embs) == 0:
        raise RuntimeError("No embeddings found in parquet")

    X = torch.tensor(embs, dtype=torch.float32)
    y = torch.zeros(X.shape[0], dtype=torch.long)  # dummy labels for demo

    # 3) Train model -> delta update
    delta, _ = train_model(
        X, y,
        input_dim=X.shape[1],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device
    )

    # 4) Serialize delta update
    buf = io.BytesIO()
    torch.save(delta, buf)
    raw_update = buf.getvalue()

    # 5) Write encrypted update into SecureStore
    session_id = f"sess-{int(time.time())}"
    update_rel = f"{session_id}/local_updates/{uuid.uuid4().hex}.pt.enc"
    update_uri = store.encrypt_write(f"file://{store.root / update_rel}", raw_update)

    # 6) Create + store signed receipt
    receipt = rm.create_receipt(
        agent="trainer-agent",
        session_id=session_id,
        operation="train",
        params={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "device": device,
            "session_parquet": session_parquet,
        },
        outputs=[update_uri],
    )

    receipt_rel = f"{session_id}/receipts/train_{int(time.time()*1000)}.json.enc"
    receipt_uri = store.encrypt_write(
        f"file://{store.root / receipt_rel}", json.dumps(receipt).encode()
    )

    return {
        "local_update_uri": update_uri,
        "receipt": receipt,
        "receipt_uri": receipt_uri,
    }
