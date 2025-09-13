from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch, io, uuid, json
from pathlib import Path
from trainer import train_model
from utils import read_embeddings_from_parquet
from receipts import ReceiptManager
import time
# import SecureStore
import sys
import os


from security.secure_store import SecureStore



app = FastAPI(title="Trainer Agent")

class TrainRequest(BaseModel):
    session_parquet: str
    epochs: int = 1
    batch_size: int = 32
    lr: float = 1e-3
    device: str = "cpu"

store = SecureStore("./secure_store")

@app.post("/train")
def train(req: TrainRequest):
    try:
        embs = read_embeddings_from_parquet(req.session_parquet)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read parquet: {e}")

    if len(embs) == 0:
        raise HTTPException(status_code=400, detail="No embeddings found")

    X = torch.tensor(embs, dtype=torch.float32)
    y = torch.zeros(X.shape[0], dtype=torch.long)  # dummy labels

    delta, _ = train_model(
        X, y, input_dim=X.shape[1],
        epochs=req.epochs,
        batch_size=req.batch_size,
        lr=req.lr,
        device=req.device
    )

    # save delta encrypted
    tmp = io.BytesIO()
    torch.save(delta, tmp)
    tmp.seek(0)
    update_uri = f"file://secure_store/local_updates/{uuid.uuid4().hex}.pt.enc".encode()  # now bytes
    store.encrypt_write(tmp.getvalue(), update_uri)


    # create signed receipt
    receipt = {
        "type": "train_receipt",
        "session_parquet": req.session_parquet,
        "local_update_uri": update_uri,
        "epochs": req.epochs,
        "batch_size": req.batch_size,
        "timestamp": time.time()
    }
    receipt_uri = ReceiptManager.write_receipt(receipt)

    return {"local_update_uri": update_uri, "receipt_uri": receipt_uri}
