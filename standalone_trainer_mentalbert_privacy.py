#!/usr/bin/env python3
# standalone_trainer_mentalbert_privacy.py
"""
Fine-tunes MentalBERT on a 20-session subset of DAIC-WOZ.
Trains a multitask model:
  1. Regression → PHQ-8 score
  2. Classification → depressed (>=10) / not depressed (<10)

Outputs:
  - model weights (.pt)
  - trainer receipt (JSON)
  - artifact manifest JSONL and session summary JSON (LDA-like)
This is a standalone trainer intended to produce a base model for the
Privacy-Preserving AI Framework.
"""

import os
import json
import tarfile
import random
import hashlib
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score

# ===============================
# CONFIG
# ===============================
MENTALBERT_MODEL = "mental/mental-bert-base-uncased"
DATA_DIR = "./data"                      # expected to contain DAIC-WOZ tarballs or extracted folders
LABELS_PATH = "./labels/Detailed_PHQ8_Labels.csv"
OUTPUT_DIR = "./trainer_outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_PARTICIPANTS = 20                    # subset size requested
RANDOM_SEED = 42
BATCH_SIZE = 4
EPOCHS = 2
LR = 2e-5
MAX_LEN = 256
BIN_THRESHOLD = 10.0                    # PHQ threshold for depressed vs not depressed

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ===============================
# DATASET HANDLER
# ===============================
class DAICSubsetDataset(Dataset):
    def __init__(self, texts, labels_regression, labels_classification, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels_regression = labels_regression
        self.labels_classification = labels_classification
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # use safe tokenization: if empty, replace with placeholder
        if not isinstance(text, str) or text.strip() == "":
            text = "[NO_TRANSCRIPT]"
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels_regression": torch.tensor(self.labels_regression[idx], dtype=torch.float),
            "labels_classification": torch.tensor(self.labels_classification[idx], dtype=torch.long),
        }


# ===============================
# DATA EXTRACTION (subset)
# ===============================
def extract_and_load_subset(data_dir=DATA_DIR, labels_path=LABELS_PATH, max_participants=MAX_PARTICIPANTS):
    """
    - Expects labels CSV that contains participant IDs and a PHQ8 score column.
    - Attempts to extract tar.gz files in data_dir if not already extracted.
    - Loads transcripts for participants and returns texts, phq_scores, binaries, and list of participant IDs used.
    """
    print(f"[INFO] Extracting subset of up to {max_participants} participants from DAIC-WOZ...")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Label file not found at: {labels_path}")

    df_labels = pd.read_csv(labels_path)
    print(f"[INFO] Loaded labels CSV with columns: {list(df_labels.columns)}")

    # detect PHQ column automatically (common variants)
    phq_col = None
    for candidate in ["PHQ8_Score", "PHQ_8Total", "PHQ8_Total", "PHQ8_TOTAL", "PHQ_8TOTAL", "PHQ_8", "PHQ8"]:
        if candidate in df_labels.columns:
            phq_col = candidate
            break
    if phq_col is None:
        # fallback: try to find any column with 'phq' in its name
        for c in df_labels.columns:
            if "phq" in c.lower():
                phq_col = c
                break
    if phq_col is None:
        raise KeyError("Could not find PHQ score column in labels file!")

    # choose subset deterministically (head or random sample if many)
    if len(df_labels) <= max_participants:
        subset = df_labels.copy()
    else:
        subset = df_labels.sample(n=max_participants, random_state=RANDOM_SEED).reset_index(drop=True)

    # Step 1: try to extract archives in data_dir
    for f in os.listdir(data_dir):
        if f.endswith(".tar.gz") or f.endswith(".tgz"):
            folder = os.path.join(data_dir, f.replace(".tar.gz", "").replace(".tgz", ""))
            if not os.path.exists(folder):
                print(f"[INFO] Extracting {f} ...")
                with tarfile.open(os.path.join(data_dir, f)) as tar:
                    tar.extractall(path=data_dir)

    texts = []
    phq_scores = []
    binaries = []
    participant_ids = []

    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Loading transcripts"):
        # Participant ID field heuristics
        pid = None
        for candidate in ["Participant_ID", "participant_id", "ID", "id"]:
            if candidate in row.index:
                pid = str(int(row[candidate])) if not pd.isna(row[candidate]) else None
                break
        if pid is None:
            # try first column
            pid = str(row.iloc[0]) if not pd.isna(row.iloc[0]) else None

        if pid is None:
            continue

        try:
            phq_raw = row[phq_col]
            phq = float(phq_raw)
        except Exception:
            # skip rows with bad PHQ
            print(f"[WARN] invalid PHQ for participant {pid}, skipping")
            continue

        binary = 1 if phq >= BIN_THRESHOLD else 0

        # Folder naming heuristics
        folder_variants = [
            os.path.join(data_dir, f"{pid}_P"),
            os.path.join(data_dir, f"{pid}-P"),
            os.path.join(data_dir, pid),
            os.path.join(data_dir, f"{int(pid)}_P") if pid.isdigit() else None,
        ]
        folder_variants = [v for v in folder_variants if v]

        transcript_path = None
        for folder_path in folder_variants:
            if not os.path.exists(folder_path):
                continue
            # candidate transcript file names
            possible_names = [
                f"{pid}_Transcript.csv",
                f"{pid}_TRANSCRIPT.csv",
                f"{pid}_transcript.csv",
                "transcript.csv",
                "Transcript.csv",
                "TRANSCRIPT.csv",
            ]
            for name in possible_names:
                p = os.path.join(folder_path, name)
                if os.path.exists(p):
                    transcript_path = p
                    break
            if transcript_path:
                break

        if not transcript_path:
            # try searching subfolders for any CSV that looks like transcript
            found = False
            for root, _, files in os.walk(data_dir):
                for fn in files:
                    if fn.lower().endswith(".csv") and pid in root:
                        # pick this as fallback
                        p = os.path.join(root, fn)
                        transcript_path = p
                        found = True
                        break
                if found:
                    break

        if not transcript_path:
            print(f"[WARN] Missing transcript for participant {pid}; skipping.")
            continue

        try:
            df_trans = pd.read_csv(transcript_path)
            # find text-like column
            text_col = None
            speaker_col = None
            for c in df_trans.columns:
                lc = c.lower()
                if any(k in lc for k in ("transcript", "sentence", "text", "value", "utterance")) and text_col is None:
                    text_col = c
                if "speaker" in lc or "spkr" in lc:
                    speaker_col = c
            if text_col is None:
                # fallback: pick first non-numeric column
                for c in df_trans.columns:
                    if df_trans[c].dtype == object:
                        text_col = c
                        break

            if text_col is None:
                print(f"[WARN] No text column found in transcript for {pid}; skipping.")
                continue

            # If speaker column exists, try to select participant utterances; else use all
            if speaker_col is not None:
                # Normalize speaker strings
                speakers = df_trans[speaker_col].astype(str).str.lower().fillna("")
                # pick rows with 'participant' or 'interviewee' or 'client'
                mask = speakers.str.contains("participant|interviewee|client|user", na=False)
                if mask.any():
                    user_lines = df_trans.loc[mask, text_col].astype(str).tolist()
                else:
                    user_lines = df_trans[text_col].astype(str).tolist()
            else:
                user_lines = df_trans[text_col].astype(str).tolist()

            text = " ".join([t for t in user_lines if isinstance(t, str)])
            if not text.strip():
                print(f"[WARN] empty transcript for {pid}; skipping.")
                continue

            texts.append(text)
            phq_scores.append(float(phq))
            binaries.append(int(binary))
            participant_ids.append(pid)
            print(f"[INFO] Loaded transcript for {pid}")
        except Exception as e:
            print(f"[WARN] Error reading transcript for {pid}: {e}")
            continue

    print(f"[INFO] Loaded {len(texts)} valid transcripts from subset.")
    return texts, phq_scores, binaries, participant_ids


# ===============================
# MULTITASK MODEL
# ===============================
class MentalBERTMultiTask(torch.nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.reg_head = torch.nn.Linear(hidden_size, 1)  # regression
        self.cls_head = torch.nn.Linear(hidden_size, 2)  # binary classification

    def forward(self, input_ids, attention_mask, labels_regression=None, labels_classification=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # use CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        reg_out = self.reg_head(pooled_output).squeeze(-1)  # (batch,)
        cls_out = self.cls_head(pooled_output)              # (batch,2)

        loss = None
        if labels_regression is not None and labels_classification is not None:
            mse_loss = torch.nn.MSELoss()(reg_out, labels_regression)
            ce_loss = torch.nn.CrossEntropyLoss()(cls_out, labels_classification)
            loss = mse_loss + ce_loss

        return {"loss": loss, "regression": reg_out, "classification": cls_out}


# ===============================
# TRAIN / EVAL UTILITIES
# ===============================
def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels_regression": torch.stack([b["labels_regression"] for b in batch]),
        "labels_classification": torch.stack([b["labels_classification"] for b in batch]),
    }


def evaluate_model(model, dataloader, device):
    model.eval()
    all_reg_preds = []
    all_cls_preds = []
    all_reg_labels = []
    all_cls_labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"],
                        labels_regression=None, labels_classification=None)
            reg_out = out["regression"].detach().cpu().numpy()
            cls_out = out["classification"].detach().cpu().numpy()
            all_reg_preds.extend(reg_out.tolist())
            all_cls_preds.extend(np.argmax(cls_out, axis=1).tolist())
            all_reg_labels.extend(batch["labels_regression"].cpu().numpy().tolist())
            all_cls_labels.extend(batch["labels_classification"].cpu().numpy().tolist())

    mae = float(mean_absolute_error(all_reg_labels, all_reg_preds)) if len(all_reg_preds) > 0 else None
    acc = float(accuracy_score(all_cls_labels, all_cls_preds)) if len(all_cls_preds) > 0 else None
    prec = float(precision_score(all_cls_labels, all_cls_preds, zero_division=0)) if len(all_cls_preds) > 0 else None
    rec = float(recall_score(all_cls_labels, all_cls_preds, zero_division=0)) if len(all_cls_preds) > 0 else None
    f1 = float(f1_score(all_cls_labels, all_cls_preds, zero_division=0)) if len(all_cls_preds) > 0 else None

    return {"mae": mae, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, 
            "reg_preds": all_reg_preds, "cls_preds": all_cls_preds,
            "reg_labels": all_reg_labels, "cls_labels": all_cls_labels}


# ===============================
# TRAIN FUNCTION
# ===============================
def train_privacy_model(output_dir=OUTPUT_DIR):
    print("[INFO] Loading tokenizer and extracting subset...")
    tokenizer = AutoTokenizer.from_pretrained(MENTALBERT_MODEL)

    texts, phq_scores, binaries, participant_ids = extract_and_load_subset(max_participants=MAX_PARTICIPANTS)
    if len(texts) == 0:
        raise RuntimeError("No transcripts loaded; cannot train.")

    # train/val split (deterministic)
    train_texts, val_texts, train_reg, val_reg, train_bin, val_bin, train_pids, val_pids = train_test_split(
        texts, phq_scores, binaries, participant_ids, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
    )

    train_ds = DAICSubsetDataset(train_texts, train_reg, train_bin, tokenizer, max_len=MAX_LEN)
    val_ds = DAICSubsetDataset(val_texts, val_reg, val_bin, tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # build model
    model = MentalBERTMultiTask(MENTALBERT_MODEL).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    # optionally a simple LR scheduler
    total_steps = max(1, len(train_loader) * EPOCHS)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)

    print(f"[INFO] Starting training: {len(train_ds)} train samples, {len(val_ds)} val samples")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        count = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"],
                        labels_regression=batch["labels_regression"], labels_classification=batch["labels_classification"])
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
            count += 1
        avg_loss = float(total_loss / max(1, count))
        print(f"[INFO] Epoch {epoch+1} finished. Avg loss: {avg_loss:.6f}")

        # quick eval each epoch
        eval_metrics = evaluate_model(model, val_loader, DEVICE)
        print(f"[EVAL] val MAE={eval_metrics['mae']:.4f} acc={eval_metrics['accuracy']:.4f} f1={eval_metrics['f1']:.4f}")

    # final evaluation
    final_metrics = evaluate_model(model, val_loader, DEVICE)

    # Save model weights
    model_path = os.path.join(output_dir, "mentalbert_privacy_subset.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved: {model_path}")

    # Create a receipt JSON (trainer receipt)
    session_id = f"sess-{int(torch.tensor(0).random_().item() if False else int(random.random()*1e9))}"
    receipt = {
        "trainer": "mentalbert_privacy_subset",
        "session_id": session_id,
        "outputs": [f"file://{os.path.abspath(model_path)}"],
        "metrics": {
            "train_avg_loss_estimate": avg_loss if 'avg_loss' not in locals() else locals().get('avg_loss', avg_loss),
            "val_mae": final_metrics["mae"],
            "val_accuracy": final_metrics["accuracy"],
            "val_precision": final_metrics["precision"],
            "val_recall": final_metrics["recall"],
            "val_f1": final_metrics["f1"],
            "num_train_samples": len(train_ds),
            "num_val_samples": len(val_ds),
        },
        "participants_used": len(participant_ids),
        "participant_ids": participant_ids,
        "tasks": ["PHQ8_regression", "binary_classification"]
    }
    receipt_path = os.path.join(output_dir, "receipt_privacy_subset.json")
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"[INFO] Receipt saved: {receipt_path}")

    # Create an artifact manifest (JSONL) that mimics what LDA would produce for downstream pipelines.
    # We'll write a small JSONL with one artifact (the model weights) and then create a tiny session summary.
    manifest_lines = []
    artifact_entry = {
        "uri": f"file://{os.path.abspath(model_path)}",
        "type": "model_weights",
        "filename": os.path.basename(model_path),
        "size_bytes": os.path.getsize(model_path),
        "created": pd.Timestamp.utcnow().isoformat() + "Z"
    }
    manifest_lines.append(artifact_entry)
    manifest_path = os.path.join(output_dir, "artifact_manifest.jsonl")
    with open(manifest_path, "w") as f:
        for entry in manifest_lines:
            f.write(json.dumps(entry) + "\n")
    print(f"[INFO] Artifact manifest written: {manifest_path}")

    # Session summary (LDA-like)
    session_summary = {
        "session_id": session_id,
        "artifact_manifest": f"file://{os.path.abspath(manifest_path)}",
        "receipts": [f"file://{os.path.abspath(receipt_path)}"],
        "count": len(participant_ids)
    }
    session_path = os.path.join(output_dir, "session_result.json")
    with open(session_path, "w") as f:
        json.dump(session_summary, f, indent=2)
    print(f"[INFO] Session summary written: {session_path}")

    # Also save detailed predictions for interpretability (per validation item)
    explain_path = os.path.join(output_dir, "val_predictions.jsonl")
    # We have reg_preds/cls_preds/reg_labels/cls_labels from final_metrics only if evaluate_model returned them
    # Re-run to get per-sample predictions with participant ids from val_pids mapping
    model.eval()
    per_item = []
    with torch.no_grad():
        all_batches = []
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"], labels_regression=None, labels_classification=None)
            reg_out = out["regression"].detach().cpu().tolist()
            cls_out = out["classification"].detach().cpu().tolist()
            input_ids = batch["input_ids"].detach().cpu().tolist()
            # convert predictions to probabilities for classification (softmax)
            probs = torch.nn.functional.softmax(torch.tensor(cls_out), dim=1).numpy().tolist()
            for i in range(len(reg_out)):
                per_item.append({
                    "reg_pred": float(reg_out[i]),
                    "cls_pred_label": int(np.argmax(probs[i])),
                    "cls_pred_probs": probs[i],
                    # We cannot map individual sample to participant id robustly here unless we kept val_pids per sample.
                })

    with open(explain_path, "w") as f:
        for item in per_item:
            f.write(json.dumps(item) + "\n")
    print(f"[INFO] Validation predictions written (jsonl): {explain_path}")

    # Return paths and metrics
    return {
        "model_path": os.path.abspath(model_path),
        "receipt_path": os.path.abspath(receipt_path),
        "manifest_path": os.path.abspath(manifest_path),
        "session_path": os.path.abspath(session_path),
        "explain_path": os.path.abspath(explain_path),
        "metrics": final_metrics
    }


if __name__ == "__main__":
    res = train_privacy_model()
    print("\n=== TRAINING FINISHED ===")
    print(json.dumps(res, indent=2))
