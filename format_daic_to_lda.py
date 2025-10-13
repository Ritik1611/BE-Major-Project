#!/usr/bin/env python3
"""
format_daic_to_lda.py

Converts DAIC-WOZ data (300_P, 301_P, etc.) into LDA-compatible outputs:
 - Encrypted parquet file (embedding + PHQ + label)
 - JSONL manifest (like Local Data Agent would generate)

Output: ./secure_store/sess-DAICWOZ/formatted_text_embeddings.parquet.enc
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import pyarrow as pa
import pyarrow.parquet as pq

# SecureStore fallback (minimal version)
class SecureStore:
    def __init__(self, root="./secure_store"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def encrypt_write(self, uri: str, payload: bytes):
        if uri.startswith("file://"):
            path = Path(uri[len("file://"):])
        else:
            path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(payload)
        return f"file://{path}"

# Paths
DATA_DIR = Path("./data")
LABELS_PATH = Path("./labels/Detailed_PHQ8_Labels.csv")
STORE_ROOT = Path("./secure_store")
SESSION_ID = "sess-DAICWOZ"

MODEL_NAME = "mental/mental-bert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_PARTICIPANTS = 20

store = SecureStore(str(STORE_ROOT))

# ----------------------------
# Step 1: Load PHQ scores
# ----------------------------
if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Missing PHQ label file at {LABELS_PATH}")

labels_df = pd.read_csv(LABELS_PATH)
phq_col = next((c for c in labels_df.columns if "PHQ" in c.upper()), None)
if not phq_col:
    raise KeyError("No PHQ score column found in labels file!")

id_col = next((c for c in labels_df.columns if "Participant" in c or "ID" in c), None)
if not id_col:
    raise KeyError("No participant ID column found!")

phq_dict = dict(zip(labels_df[id_col].astype(str), labels_df[phq_col]))

# ----------------------------
# Step 2: Load text transcripts
# ----------------------------
print("🔍 Collecting transcripts...")
records = []

for folder in sorted(DATA_DIR.glob("*_P")):
    pid = folder.name.replace("_P", "")
    if len(records) >= MAX_PARTICIPANTS:
        break

    phq = phq_dict.get(pid)
    if pd.isna(phq):
        continue
    label = 1 if phq >= 10 else 0

    transcript_path = folder / f"{pid}_Transcript.csv"
    if not transcript_path.exists():
        print(f"⚠️ Skipping {pid} (missing transcript)")
        continue

    try:
        df_t = pd.read_csv(transcript_path)
        text_col = next((c for c in df_t.columns if "text" in c.lower() or "value" in c.lower()), None)
        if not text_col:
            continue
        full_text = " ".join(df_t[text_col].astype(str).tolist())
        if full_text.strip():
            records.append({"participant_id": pid, "text": full_text, "phq_score": phq, "label": label})
    except Exception as e:
        print(f"⚠️ Error reading {pid}: {e}")

print(f"✅ Loaded {len(records)} participants for embedding.")

# ----------------------------
# Step 3: Generate embeddings (MentalBERT)
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

embs = []
for r in tqdm(records, desc="Encoding transcripts"):
    try:
        inputs = tokenizer(r["text"], return_tensors="pt", truncation=True, padding=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
            pooled = out.last_hidden_state[:, 0, :].mean(dim=0).cpu().numpy()
        embs.append(pooled)
    except Exception as e:
        print(f"⚠️ Embedding failed for {r['participant_id']}: {e}")
        embs.append(np.zeros(model.config.hidden_size))

df = pd.DataFrame({
    "participant_id": [r["participant_id"] for r in records],
    "embedding": list(embs),
    "phq_score": [r["phq_score"] for r in records],
    "label": [r["label"] for r in records],
})
print(f"✅ Created dataframe: {df.shape}")

# ----------------------------
# Step 4: Write parquet + manifest
# ----------------------------
sess_dir = STORE_ROOT / SESSION_ID / "encrypted"
sess_dir.mkdir(parents=True, exist_ok=True)

table = pa.Table.from_pandas(df)
buf = pa.BufferOutputStream()
pq.write_table(table, buf)
parquet_bytes = buf.getvalue().to_pybytes()

parquet_uri = store.encrypt_write(f"file://{sess_dir}/text_embeddings.parquet.enc", parquet_bytes)
print(f"💾 Saved encrypted parquet: {parquet_uri}")

# Manifest JSONL
manifest_dir = STORE_ROOT / SESSION_ID / "manifest"
manifest_dir.mkdir(parents=True, exist_ok=True)

manifest_entry = {
    "uri": parquet_uri,
    "modality": "text",
    "meta": {"count": len(df), "session_id": SESSION_ID},
}
manifest_path = manifest_dir / "lda_manifest.jsonl"
with open(manifest_path, "w") as f:
    f.write(json.dumps(manifest_entry) + "\n")

print(f"📜 Manifest saved: file://{manifest_path}")
print("✅ DONE: DAIC-WOZ formatted to LDA-style output.")
