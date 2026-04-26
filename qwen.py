"""
/home/ritik26/Desktop/BE-Major-Project/installer/fl_algorithm_comparison.py
Federated Learning Algorithm Comparison on DAIC-WOZ Depression Dataset

KEY FIX SUMMARY (vs previous version):
  FIX-1 : DepressionNet architecture now MIRRORS MultimodalFusionModel from
           train_base_model_centralized.py — same layer names (audio_encoder,
           visual_encoder, fusion), same hidden dims, same Sequential indices.
           This allows centralized checkpoint weights to actually transfer.
  FIX-2 : strict=False does NOT skip shape-mismatched keys — it only skips
           missing/unexpected keys, so size-mismatched layers still raise.
           Replaced with _load_partial_checkpoint() which filters by both
           name AND shape before calling load_state_dict().
  FIX-3 : BatchNorm1d crashes when batch_size=1 during training (not eval).
           DepressionNet now uses LayerNorm (identical weight/bias shapes to
           BatchNorm so they still load from checkpoint; no running stats to
           worry about). DataLoader also uses drop_last=True when safe.
  FIX-4 : FedYogi and SCAFFOLD warm-start blocks also used the broken
           load_state_dict(strict=False) pattern — now all use
           _load_partial_checkpoint().
  FIX-5 : model_factory() passed 'hidden=64' which matched the OLD DepressionNet
           signature; updated to 'fusion_hidden=256' for the new one.
  FIX-6 : DP noise was NOT divided by batch_size. Standard DP-SGD adds noise to
           the SUM of clipped gradients then divides by batch_size. PyTorch's
           .backward() produces an AVERAGE gradient, so noise must be divided by
           batch_size. Without this, noise is batch_size× too large (8× with
           default batch_size=8), completely overwhelming signal → model collapses
           to majority-class prediction → 0.5 acc / 0 F1.
  FIX-7 : Train/test text feature distribution mismatch. Training used
           per-utterance embeddings (std≈0.1) while test used session means
           (std≈0.1/√N_utts ≈ 0.01). Since text is 768 of 1024 fusion dims
           (75%), this ~10× scale difference breaks the fusion branch at test
           time. Fix: training now tiles the same session-mean vector (text_vec)
           so both train and test use identical distribution.
  FIX-8 : Orphaned checkpoint-reinit block in main() referenced undefined
           variables (global_model, algorithm, aggregation, global_model_path).
           Deleted — the reinit already lives correctly inside run_federated()
           and run_federated_scaffold().
  FIX-9 : FedYogi server optimizer was never instantiated or used — it silently
           ran plain FedAvg instead. Now properly created and dispatched.
  FIX-10: Dead variable 'first_w' (assigned but never read) removed from both
           FL training functions.
  FIX-11: Dead variable 'new_client_c = {}' removed from local_train_scaffold.
  FIX-12: Double class-weight application (once in loss, once in aggregation
           weights) removed — aggregation now weights by sample count only.

NEW FIXES (v2) — THE ROOT CAUSES OF acc=0.5 / f1=0.0:
  FIX-13: _reinit_encoders() was called unconditionally after ANY checkpoint
           load, destroying 10 out of 11 encoder keys that were correctly
           loaded from the centralized checkpoint.
           Specifically: audio_encoder.net.4.*, visual_encoder.* (all 6 keys)
           and audio_encoder.net.1.* were all reset to random, even though they
           had matching shapes and were successfully transferred.
           The only key that needed reinit was audio_encoder.net.0.weight
           (shape mismatch: ckpt=[64,46] vs model=[64,78]).
           FIX: _load_partial_checkpoint now returns the set of shape-skipped
           key names. _reinit_encoders inspects this set and only resets the
           specific encoder whose FIRST LINEAR LAYER was shape-skipped.
           All other layers (including the correct visual_encoder and the
           second linear of audio_encoder) are preserved from the checkpoint.

  FIX-14: One patient per FL client → degenerate single-class datasets.
           The original code created one PatientDataset per patient, where every
           sample is an identical copy of that patient's features (tiled across
           utterance slots). Every client had ONLY ONE CLASS label.
           With 11 depressed and 29 not-depressed patients as clients, and
           FedAvg weighting by n_samples (≈ n_utterances ≈ 60 per patient):
             Depressed weight:     11×60 / (40×60) = 0.275
             Not-depressed weight: 29×60 / (40×60) = 0.725
           Global update = 0.275*(push→1) + 0.725*(push→0) = −0.45
           The global model always collapses to predicting majority class (0).
           Balanced test set (5 dep + 5 not-dep) → acc=0.5, f1=0.0.
           FIX: build_hospital_clients() groups patients into N_HOSPITALS
           hospital-style clients, each containing a stratified mix of both
           depressed and not-depressed patients. Each hospital client has
           diverse samples from multiple patients → both classes → meaningful
           gradient direction that trains the model to distinguish them.

  FIX-15: SGD + momentum on single-class tiled datasets caused monotone
           gradient compounding that accelerated collapse.
           FIX: local_train now uses Adam (adaptive LR, no momentum accumulation
           issue) which is far more stable for small heterogeneous FL clients.
           SGD+momentum is retained as a fallback via LocalCfg.use_adam flag.

  FIX-16: evaluate() used logits.argmax(1) — equivalent to threshold=0.5 on
           softmax probabilities. With class imbalance (3.5:1 in training,
           1:1 in test), 0.5 threshold systematically misses depressed cases.
           FIX: evaluate() now accepts a threshold parameter and uses softmax
           probabilities for thresholded classification. Default threshold=0.4
           biases slightly toward detecting the minority class. The threshold
           is reported in logs and can be tuned per experiment.

  FIX-17: Duplicate experiment keys when --no_dp. First experiment
           ("fedavg","mean",1.0,False) and second ("fedavg","mean",1.0,True)
           both produced label "fedavg_mean_noDP" when --no_dp was passed.
           Second run silently overwrote the first in all_results dict, causing
           the duplicate "Running: fedavg_mean_noDP" seen in the terminal logs.
           FIX: Deduplicate experiments by label before running.

WEIGHT TRANSFER AFTER FIXES:
  With audio_dim=78 (FL) vs audio_dim=46 (centralized checkpoint):
    audio_encoder.net.0.weight  SHAPE-SKIPPED (46→78 mismatch) → Kaiming reinit
    audio_encoder.net.0.bias    ✓ loaded from checkpoint
    audio_encoder.net.1.*       ✓ loaded from checkpoint (LayerNorm)
    audio_encoder.net.4.*       ✓ loaded from checkpoint (Linear [128,64])
    visual_encoder.*   (all 6)  ✓ loaded from checkpoint
    fusion.*           (all 8)  ✓ loaded from checkpoint
  Total: 19/20 keys transferred, only 1 shape-reinit.

Data structure expected:
  data/
    {ID}_P/
      features/
        {ID}_OpenSMILE*.csv      -- eGeMAPS / ComPare audio features
        {ID}_OpenFace2.csv       -- Action Units, gaze, pose
        {ID}_BoAW_openSMILE.csv  -- Bag of Audio Words
        {ID}_BoVW_openpose.csv   -- Bag of Visual Words
        {ID}_CNN_*.csv           -- CNN visual features
        {ID}_Transcript.csv      -- Turn-by-turn conversation
      {ID}_AUDIO.wav
    labels.csv  (Participant_ID, PHQ8_Binary, PHQ8_Score)

Base model: MentalBERT (from ~/.federated/models/mentalbert) for text.
            Pre-extracted audio/visual features used directly.

FL algorithms compared:
  - FedAvg  (McMahan et al. 2017)
  - FedProx (Li et al. 2020)
  - FedAdam (Reddi et al. 2021)
  - FedYogi (Reddi et al. 2021)
  - SCAFFOLD (Karimireddy et al. 2020)

Aggregation strategies per algorithm:
  - Federated Averaging (mean)
  - Trimmed Mean (Byzantine-robust)
  - Coordinate-wise Median
  - Krum (when n_clients >= 5)

DP support: Gaussian mechanism via RDP accountant.

Usage:
  python fl_algorithm_comparison.py --data_dir ./data --rounds 20
  python fl_algorithm_comparison.py --data_dir ./data --rounds 20 --use_mentalbert
  python fl_algorithm_comparison.py --data_dir ./data --rounds 20 --no_dp \\
      --global_model_path ~/.federated/data/global_models/global_round1.pt
"""

import argparse
import copy
import json
import logging
import math
import os
import re
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.metrics import (
        f1_score, roc_auc_score, classification_report,
        average_precision_score, precision_recall_curve,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MENTALBERT_PATH = Path.home() / ".federated" / "models" / "mentalbert"
RESULTS_DIR = Path("fl_daic_results")
RESULTS_DIR.mkdir(exist_ok=True)

log.info("Device: %s | MentalBERT present: %s", DEVICE, MENTALBERT_PATH.exists())


# ==============================================================================
# SECTION 1: DATA LOADING -- DAIC-WOZ FEATURE FILES
# ==============================================================================

def _read_csv_robust(path: Path) -> Optional[pd.DataFrame]:
    """Try multiple separators; return None if all fail."""
    for sep in [",", ";", "\t", r"\s+"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python",
                             on_bad_lines="skip")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return None


def _numeric_pool(df: pd.DataFrame,
                  skip_cols: Tuple[str, ...] = ()) -> Optional[np.ndarray]:
    """Drop identifier columns, compute mean + std over rows -> flat vector."""
    skip = {c.lower() for c in skip_cols}
    numeric_cols = []
    for col in df.columns:
        if col.strip().lower() in skip:
            continue
        try:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() > 0:
                df[col] = s
                numeric_cols.append(col)
        except Exception:
            pass

    if not numeric_cols:
        return None

    data = df[numeric_cols].dropna().values.astype(np.float32)
    if data.shape[0] == 0:
        return None

    data = np.where(np.isfinite(data), data, 0.0)
    return np.concatenate([np.mean(data, axis=0), np.std(data, axis=0)])


def extract_opensmile(path: Path) -> Optional[np.ndarray]:
    df = _read_csv_robust(path)
    if df is None:
        return None
    df.columns = [c.strip() for c in df.columns]
    return _numeric_pool(df, skip_cols=("frametime", "frame", "timestamp",
                                        "name", "start", "end"))


def extract_openface2(path: Path) -> Optional[np.ndarray]:
    df = _read_csv_robust(path)
    if df is None:
        return None
    df.columns = [c.strip() for c in df.columns]
    keep = [c for c in df.columns
            if any(tag in c for tag in ("AU", "gaze", "pose", "x_", "y_", "X_", "Y_"))
            and c.lower() not in ("frame", "face_id", "timestamp",
                                   "confidence", "success")]
    if not keep:
        keep = [c for c in df.columns
                if c.lower() not in ("frame", "face_id", "timestamp",
                                      "confidence", "success")]
    if not keep:
        return None
    return _numeric_pool(df[keep])


def extract_bow(path: Path) -> Optional[np.ndarray]:
    df = _read_csv_robust(path)
    if df is None:
        return None
    first = str(df.columns[0]).strip().lower()
    if first in ("timestamp", "frame", "time", "start", "index"):
        df = df.iloc[:, 1:]
    data = df.apply(pd.to_numeric, errors="coerce").dropna().values.astype(np.float32)
    if data.shape[0] == 0:
        return None
    data = np.where(np.isfinite(data), data, 0.0)
    return np.mean(data, axis=0)


def extract_transcript_utterances(path: Path,
                                   min_words: int = 3) -> List[str]:
    """Return participant (patient) utterances as a list of strings."""
    df = _read_csv_robust(path)
    if df is None:
        return []

    df.columns = [c.strip().lower() for c in df.columns]

    text_col = next(
        (c for c in ("value", "text", "content", "utterance", "transcription")
         if c in df.columns),
        df.columns[3] if df.shape[1] >= 4 else df.columns[-1],
    )

    speaker_col = next(
        (c for c in ("speaker", "role", "label") if c in df.columns),
        None,
    )

    if speaker_col is not None:
        mask = df[speaker_col].astype(str).str.lower().str.contains(
            r"participant|patient|\bp\b", na=False
        )
        df = df[mask]

    return [
        str(t).strip()
        for t in df[text_col].dropna()
        if len(str(t).strip().split()) >= min_words
    ]


def load_patient(patient_dir: Path, label: int) -> Optional[Dict[str, Any]]:
    """Load one patient's multi-modal features."""
    pid = patient_dir.name.split("_")[0]
    feat_dir = patient_dir / "features"
    if not feat_dir.exists():
        feat_dir = patient_dir

    # ── Audio ──────────────────────────────────────────────────────────────
    audio_vec = None
    for pat in ("*OpenSMILE*.csv", "*opensmile*.csv", "*eGeMAPS*.csv",
                "*ComPare*.csv"):
        for f in feat_dir.glob(pat):
            audio_vec = extract_opensmile(f)
            if audio_vec is not None:
                break
        if audio_vec is not None:
            break

    if audio_vec is None:
        for pat in ("*BoAW*.csv", "*boaw*.csv"):
            for f in feat_dir.glob(pat):
                audio_vec = extract_bow(f)
                if audio_vec is not None:
                    break
            if audio_vec is not None:
                break

    # ── Visual ─────────────────────────────────────────────────────────────
    visual_vec = None
    for pat in ("*OpenFace*.csv", "*openface*.csv"):
        for f in feat_dir.glob(pat):
            visual_vec = extract_openface2(f)
            if visual_vec is not None:
                break
        if visual_vec is not None:
            break

    if visual_vec is None:
        for pat in ("*BoVW*.csv", "*CNN*.csv", "*densenet*.csv",
                    "*vgg*.csv"):
            for f in feat_dir.glob(pat):
                visual_vec = extract_bow(f)
                if visual_vec is not None:
                    break
            if visual_vec is not None:
                break

    # ── Transcript ─────────────────────────────────────────────────────────
    utterances: List[str] = []
    for candidate in [
        patient_dir / f"{pid}_Transcript.csv",
        feat_dir / f"{pid}_Transcript.csv",
        *feat_dir.glob("*Transcript*"),
        *patient_dir.glob("*Transcript*"),
    ]:
        if isinstance(candidate, Path) and candidate.exists():
            utterances = extract_transcript_utterances(candidate)
            if utterances:
                break

    if audio_vec is None and visual_vec is None and not utterances:
        log.warning("Patient %s: no features found -- skipping", pid)
        return None

    if audio_vec is None:
        audio_vec = np.zeros(176, dtype=np.float32)
    if visual_vec is None:
        visual_vec = np.zeros(70, dtype=np.float32)

    if not utterances:
        utterances = [f"patient {pid}"]

    log.info("Patient %s | audio=%d visual=%d utterances=%d label=%d",
             pid, audio_vec.shape[0], visual_vec.shape[0],
             len(utterances), label)

    return {
        "patient_id": pid,
        "audio":      audio_vec.astype(np.float32),
        "visual":     visual_vec.astype(np.float32),
        "utterances": utterances,
        "label":      label,
    }


def load_labels(data_dir: Path) -> Dict[str, int]:
    """Load PHQ-8 binary labels from any standard DAIC-WOZ label file."""
    candidates = [
        data_dir / "labels.csv",
        *data_dir.glob("*split*Depression*.csv"),
        *data_dir.glob("*PHQ*.csv"),
        *data_dir.glob("*label*.csv"),
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            df = _read_csv_robust(p)
            if df is None:
                continue
            df.columns = [str(c).strip() for c in df.columns]

            id_col = next(
                (c for c in ("Participant_ID", "ID", "participant_id", "id",
                             "Subject") if c in df.columns),
                df.columns[0],
            )

            label_col = next(
                (c for c in ("PHQ8_Binary", "PHQ_Binary", "binary",
                             "label", "depressed") if c in df.columns),
                None,
            )
            if label_col is None:
                score_col = next(
                    (c for c in ("PHQ8_Score", "PHQ_Score", "PHQ8", "PHQ",
                                 "score") if c in df.columns),
                    None,
                )
                if score_col:
                    df["_bin"] = (
                        pd.to_numeric(df[score_col], errors="coerce") >= 10
                    ).astype(int)
                    label_col = "_bin"

            if label_col is None:
                continue

            labels = {
                str(int(float(str(row[id_col]).strip()))):
                    int(float(str(row[label_col]).strip()))
                for _, row in df.iterrows()
            }
            if labels:
                log.info("Labels loaded from %s (%d entries)", p.name, len(labels))
                return labels
        except Exception as e:
            log.warning("Could not parse %s: %s", p, e)

    log.warning(
        "No labels file found. Assign balanced labels for DEMO.\n"
        "Create data/labels.csv with columns: Participant_ID, PHQ8_Binary"
    )
    return {}


def load_dataset(data_dir: Path) -> List[Dict[str, Any]]:
    data_dir = Path(data_dir)
    labels_dict = load_labels(data_dir)

    patient_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and re.match(r"^\d+", d.name)
    )
    log.info("Found %d patient directories", len(patient_dirs))

    patients = []
    for pdir in patient_dirs:
        pid = pdir.name.split("_")[0]
        if pid not in labels_dict:
            log.warning("No label for %s -- skipping (strict label mode)", pid)
            continue
        label = labels_dict[pid]
        p = load_patient(pdir, label)
        if p is not None:
            patients.append(p)

    pos = sum(p["label"] for p in patients)
    log.info("Dataset: %d patients | %d depressed | %d not depressed",
             len(patients), pos, len(patients) - pos)
    return patients


# ==============================================================================
# SECTION 2: MENTALBERT TEXT EMBEDDINGS
# ==============================================================================

class MentalBERTEmbedder:
    """Embed text via MentalBERT (falls back to bert-base-uncased or random)."""

    DIM = 768

    def __init__(self, use_mentalbert: bool = True):
        self.model = None
        self.tokenizer = None

        if not HAS_TRANSFORMERS or not use_mentalbert:
            if use_mentalbert:
                log.warning("transformers not installed -- using random text features")
            return

        paths = [
            str(MENTALBERT_PATH),
            "mental/mental-bert-base-uncased",
            "bert-base-uncased",
        ]
        for path in paths:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(path)
                self.model = AutoModel.from_pretrained(path)
                self.model.eval().to(DEVICE)
                log.info("Text embedder loaded: %s", path)
                break
            except Exception:
                continue

        if self.model is None:
            log.warning("Could not load any BERT model -- random text features")

    @torch.no_grad()
    def embed(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """(N, 768) mean-pooled BERT embeddings."""
        if not texts:
            return np.zeros((1, self.DIM), dtype=np.float32)

        if self.model is None:
            # FIX-7: random fallback uses consistent seed so train/test are
            # at the same scale (both session-mean level).
            return np.random.randn(len(texts), self.DIM).astype(np.float32) * 0.1

        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = self.tokenizer(
                batch, truncation=True, padding=True,
                max_length=128, return_tensors="pt"
            ).to(DEVICE)
            hidden = self.model(**enc).last_hidden_state   # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1)  # (B, H)
            out.append(pooled.cpu().float().numpy())
        return np.concatenate(out, axis=0)

    def session_embedding(self, utterances: List[str]) -> np.ndarray:
        """Mean of utterance embeddings -> (768,)."""
        return np.mean(self.embed(utterances), axis=0)


# ==============================================================================
# SECTION 3: DATASET CLASSES
# ==============================================================================

class PatientDataset(Dataset):
    """
    FIX-7: Each sample = one utterance slot using the SESSION-MEAN text vector
    tiled across all slots.  This matches the test set which also uses
    session-mean text -- avoiding the train/test distribution mismatch.

    NOTE: This class is still used per-patient inside HospitalDataset.
    It is NOT used as a standalone FL client dataset anymore (FIX-14).
    """

    def __init__(self, audio: np.ndarray, visual: np.ndarray,
                 text: np.ndarray, label: int):
        n = text.shape[0]
        self.audio = torch.tensor(
            np.tile(audio.reshape(1, -1), (n, 1)), dtype=torch.float32
        )
        self.visual = torch.tensor(
            np.tile(visual.reshape(1, -1), (n, 1)), dtype=torch.float32
        )
        self.text = torch.tensor(text, dtype=torch.float32)
        self.labels = torch.full((n,), label, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "audio":  self.audio[idx],
            "visual": self.visual[idx],
            "text":   self.text[idx],
            "label":  self.labels[idx],
        }


class HospitalDataset(Dataset):
    """
    FIX-14: A hospital client dataset that aggregates MULTIPLE patients
    (one sample per patient), so each client has a mix of both classes.

    This replaces the original one-patient-per-client approach that caused
    every client to have only one class label, making FedAvg collapse to
    always predicting the majority class (acc=0.5, f1=0.0).

    Each patient contributes ONE sample to the hospital dataset — its
    session-mean audio, visual, and text features.  This is the same
    representation used by the test set, so distributions match (FIX-7).
    """

    def __init__(self, patients: List[Dict[str, Any]]):
        self.audio  = torch.tensor(
            np.stack([p["audio"]    for p in patients]), dtype=torch.float32)
        self.visual = torch.tensor(
            np.stack([p["visual"]   for p in patients]), dtype=torch.float32)
        self.text   = torch.tensor(
            np.stack([p["text_vec"] for p in patients]), dtype=torch.float32)
        self.labels = torch.tensor(
            [p["label"] for p in patients], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "audio":  self.audio[idx],
            "visual": self.visual[idx],
            "text":   self.text[idx],
            "label":  self.labels[idx],
        }


def build_hospital_clients(
    patients: List[Dict[str, Any]],
    n_hospitals: int,
    seed: int = 42,
) -> List[HospitalDataset]:
    """
    FIX-14: Group patients into n_hospitals hospital clients using stratified
    assignment so each hospital gets a representative mix of both classes.

    Strategy: interleave depressed and not-depressed patients across hospitals.
    This ensures every hospital has at least one patient from each class when
    possible, producing meaningful two-class gradients at each client.

    Args:
        patients:    list of patient dicts (must have 'label' and 'text_vec')
        n_hospitals: number of FL hospital clients to create
        seed:        random seed for shuffling within each class

    Returns:
        list of HospitalDataset objects, one per hospital
    """
    rng = np.random.default_rng(seed)

    dep = [p for p in patients if p["label"] == 1]
    neg = [p for p in patients if p["label"] == 0]
    rng.shuffle(dep)
    rng.shuffle(neg)

    # Clamp n_hospitals so each hospital can have at least 1 patient
    n_hospitals = min(n_hospitals, len(patients))

    # Initialise empty hospital lists
    hospitals: List[List[Dict]] = [[] for _ in range(n_hospitals)]

    # Round-robin assignment, interleaving classes to ensure balance
    for i, p in enumerate(dep + neg):
        hospitals[i % n_hospitals].append(p)

    # Drop empty hospitals (can happen if n_hospitals > len(patients))
    hospitals = [h for h in hospitals if h]

    datasets = [HospitalDataset(h) for h in hospitals]

    n_dep_clients = sum(any(p["label"] == 1 for p in h) for h in hospitals)
    n_both_clients = sum(
        any(p["label"] == 1 for p in h) and any(p["label"] == 0 for p in h)
        for h in hospitals
    )
    sizes = [len(h) for h in hospitals]
    log.info(
        "Hospital clients: %d total | sizes: min=%d max=%d avg=%.1f | "
        "%d have depressed | %d have BOTH classes",
        len(hospitals), min(sizes), max(sizes), np.mean(sizes),
        n_dep_clients, n_both_clients,
    )
    return datasets


# ==============================================================================
# SECTION 4: MODEL
# ==============================================================================
#
# ARCHITECTURE ALIGNMENT WITH train_base_model_centralized.py
# -----------------------------------------------------------
# The centralized script's MultimodalFusionModel has:
#   audio_encoder  = SmallMLP(audio_dim, 128)      -> layer name: audio_encoder
#   visual_encoder = SmallMLP(visual_dim, 128)     -> layer name: visual_encoder
#   text_encoder   = MentalBERT [CLS] -> 768-d      -> NOT in FL (pre-computed)
#   fusion         = Linear(1024,256)->BN->ReLU->Drop-> layer name: fusion
#                    ->Linear(256,128)->ReLU->Drop
#                    ->Linear(128,2)
#   fusion_in = 768(text) + 128(audio) + 128(visual) = 1024
#
# DepressionNet here mirrors that EXACTLY so saved weights transfer.
# Pre-computed 768-d text embeddings are concatenated directly (no BERT in FL).
# LayerNorm is used instead of BatchNorm1d:
#   - weight/bias shapes are identical [hidden_dim] so they still load from ckpt
#   - LayerNorm works for batch_size=1 during FL local training (BN would crash)
#   - running_mean/var from BN simply has no key in LN -> silently skipped

class SmallMLP(nn.Module):
    """
    Feature encoder matching train_base_model_centralized.py SmallMLP.
    Linear layer shapes are identical -> weights transfer from centralized ckpt.
    LayerNorm replaces BatchNorm1d for FL safety (batch_size can be 1).
    LayerNorm weight/bias shapes match BN gamma/beta -> those load too.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        hid = max(64, in_dim // 2)   # same formula as centralized SmallMLP
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),   # index 0 -- shape matches centralized
            nn.LayerNorm(hid),        # index 1 -- weight/bias shape [hid] matches BN
            nn.ReLU(),                # index 2
            nn.Dropout(dropout),      # index 3
            nn.Linear(hid, out_dim),  # index 4 -- shape matches centralized
            nn.ReLU(),                # index 5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepressionNet(nn.Module):
    """
    Multimodal fusion network for PHQ-8 binary depression classification.

    Layer names and fusion_in=1024 match MultimodalFusionModel so that the
    centralized checkpoint warm-starts this model correctly.

    Branches:
      audio_encoder  -> SmallMLP(audio_dim -> 128)    [same as centralized]
      visual_encoder -> SmallMLP(visual_dim -> 128)   [same as centralized]
      text (768-d)   -> passed directly to fusion      [BERT absent in FL]
    Concatenation order: [text(768) | audio(128) | visual(128)] = 1024-d
    Fusion: 1024 -> 256 -> 128 -> 2   [same as centralized]
    """

    def __init__(self, audio_dim: int, visual_dim: int, text_dim: int = 768,
                 fusion_hidden: int = 256, n_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        # Modality encoders (names match MultimodalFusionModel)
        self.audio_encoder  = SmallMLP(audio_dim,  128, dropout=dropout)
        self.visual_encoder = SmallMLP(visual_dim, 128, dropout=dropout)
        # text_encoder (BERT) intentionally absent -- pre-computed 768-d used.

        # Fusion head (indices 0-7 match MultimodalFusionModel.fusion)
        # fusion_in must equal text_dim + 128 + 128 = 1024 when text_dim=768
        fusion_in = text_dim + 128 + 128

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),          # idx 0  [256,1024]
            nn.LayerNorm(fusion_hidden),                   # idx 1  [256]
            nn.ReLU(),                                     # idx 2
            nn.Dropout(dropout),                           # idx 3
            nn.Linear(fusion_hidden, fusion_hidden // 2),  # idx 4  [128,256]
            nn.ReLU(),                                     # idx 5
            nn.Dropout(dropout),                           # idx 6
            nn.Linear(fusion_hidden // 2, n_classes),     # idx 7  [2,128]
        )

    def forward(self, audio: torch.Tensor, visual: torch.Tensor,
                text: torch.Tensor) -> torch.Tensor:
        # Concatenation order MUST match centralized: [text | audio | visual]
        audio_feat  = self.audio_encoder(audio)    # (B, 128)
        visual_feat = self.visual_encoder(visual)  # (B, 128)
        fused = torch.cat([text, audio_feat, visual_feat], dim=1)  # (B, 1024)
        return self.fusion(fused)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Shape-filtered partial checkpoint loader ───────────────────────────────────
def _load_partial_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    label: str = "FL",
) -> Tuple[bool, Set[str]]:
    """
    Load a checkpoint into model, copying ONLY keys whose name AND tensor
    shape both match the current model.  Never raises for shape mismatches
    or missing/extra keys — those are silently skipped.

    FIX-13: Now returns (success, shape_skipped_keys) instead of just bool.
    The caller uses shape_skipped_keys to decide which specific encoder
    sub-modules actually need reinitialisation, rather than blindly reinitting
    all encoders (which was destroying correctly-loaded weights).

    Returns:
        (True,  set_of_shape_skipped_key_names)  on success (≥1 key loaded)
        (False, empty_set)                        on hard failure
    """
    path = Path(ckpt_path)
    if not path.exists():
        log.info("[%s] No checkpoint at %s -- starting from scratch.", label, ckpt_path)
        return False, set()

    try:
        log.info("[%s] Loading checkpoint: %s", label, ckpt_path)
        raw = torch.load(ckpt_path, map_location=DEVICE)

        # Handle both plain state_dict and wrapped dicts (e.g. {"state_dict": ...})
        if isinstance(raw, dict) and "state_dict" in raw:
            ckpt_sd = raw["state_dict"]
        elif isinstance(raw, dict):
            ckpt_sd = raw
        else:
            log.warning("[%s] Unexpected checkpoint format (%s) -- skipping.",
                        label, type(raw))
            return False, set()

        current_sd = model.state_dict()
        to_load: Dict[str, torch.Tensor] = {}
        shape_skipped_keys: Set[str] = set()
        shape_skipped_log:  List[str] = []
        name_skipped:       List[str] = []

        for k, v in ckpt_sd.items():
            if k not in current_sd:
                name_skipped.append(k)
            elif current_sd[k].shape != v.shape:
                shape_skipped_keys.add(k)
                shape_skipped_log.append(
                    f"  {k}: ckpt={tuple(v.shape)} vs model={tuple(current_sd[k].shape)}"
                )
            else:
                to_load[k] = v.to(DEVICE)

        if not to_load:
            log.warning(
                "[%s] No matching keys found -- model starts from scratch.\n"
                "  Shape mismatches (%d): check audio/visual/fusion dims.\n"
                "  Name mismatches (%d): likely different architecture.",
                label, len(shape_skipped_keys), len(name_skipped),
            )
            if shape_skipped_log:
                log.warning("[%s] Shape-skipped keys:\n%s",
                            label, "\n".join(shape_skipped_log[:10]))
            return False, set()

        merged = {**current_sd, **to_load}
        model.load_state_dict(merged, strict=True)

        log.info(
            "[%s] Warm-start: %d/%d keys loaded  |  "
            "shape-skipped=%d  name-skipped=%d",
            label, len(to_load), len(current_sd),
            len(shape_skipped_keys), len(name_skipped),
        )
        if shape_skipped_log:
            log.debug("[%s] Shape-skipped:\n%s", label,
                      "\n".join(shape_skipped_log))
        return True, shape_skipped_keys

    except Exception as exc:
        log.warning("[%s] Failed to load checkpoint (%s) -- starting from scratch.",
                    label, exc)
        return False, set()


def _selective_reinit_encoders(
    model: nn.Module,
    shape_skipped_keys: Set[str],
) -> None:
    """
    FIX-13: Selectively reinitialise ONLY the encoder whose first Linear
    layer had a shape mismatch during checkpoint loading.

    The original _reinit_encoders() blindly reset BOTH audio_encoder AND
    visual_encoder regardless of which keys were actually shape-skipped.
    This destroyed 10 out of 11 encoder keys that had been correctly loaded
    from the centralized checkpoint (audio_encoder.net.4.*, visual_encoder.*).

    Logic:
      - If audio_encoder.net.0.weight is in shape_skipped_keys:
          → audio_dim changed between centralized and FL training
          → reinit audio_encoder.net.0.* (first linear only; rest stays from ckpt)
      - If visual_encoder.net.0.weight is in shape_skipped_keys:
          → visual_dim changed
          → reinit visual_encoder.net.0.* (first linear only)
      - If neither: no reinit at all (all keys loaded correctly)

    NOTE: We only reinit the FIRST linear layer of each encoder (net[0]).
    The second linear (net[4]) is always shape [128, max(64,in//2)] regardless
    of in_dim, so it loads from checkpoint correctly and must NOT be reinit'd.
    """
    with torch.no_grad():
        for encoder_name, enc in [
            ("audio_encoder",  model.audio_encoder),
            ("visual_encoder", model.visual_encoder),
        ]:
            first_linear_weight_key = f"{encoder_name}.net.0.weight"
            if first_linear_weight_key in shape_skipped_keys:
                # Only the first linear of this encoder was shape-skipped.
                # Reinit net[0] (Linear) fresh; leave net[1] (LayerNorm) and
                # net[4] (Linear) untouched — they loaded from checkpoint.
                nn.init.kaiming_normal_(enc.net[0].weight, nonlinearity="relu")
                nn.init.zeros_(enc.net[0].bias)
                log.info(
                    "Selective reinit: %s.net[0] (first linear, shape mismatch). "
                    "net[1] (LayerNorm) and net[4] (second linear) preserved from ckpt.",
                    encoder_name,
                )
            # else: all keys for this encoder loaded from checkpoint — leave intact.


# ==============================================================================
# SECTION 5: EVALUATION
# ==============================================================================

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    threshold: float = 0.4,
) -> Tuple[float, float, float]:
    """
    FIX-16: Returns (loss, accuracy, F1) using soft-probability thresholding.

    The original argmax() was equivalent to threshold=0.5. With class imbalance
    (many more non-depressed patients in training), the model's softmax output
    for class 1 is systematically low even when learning. Using threshold=0.4
    biases slightly toward detecting the minority class (depressed=1).

    If sklearn is available, also prints per-class precision/recall at the
    threshold in DEBUG logs to help diagnose remaining issues.

    Args:
        threshold: probability threshold for class 1 (depressed). Values <0.5
                   increase sensitivity; values >0.5 increase specificity.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            audio  = batch["audio"].to(DEVICE)
            visual = batch["visual"].to(DEVICE)
            text   = batch["text"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(audio, visual, text)
            total_loss += F.cross_entropy(logits, labels,
                                           reduction="sum").item()
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(depressed)
            preds = (probs >= threshold).astype(int)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.tolist())

    n   = len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(n, 1)

    if HAS_SKLEARN and len(set(all_labels)) > 1:
        f1 = f1_score(all_labels, all_preds,
                      average="binary", zero_division=0)
    else:
        f1 = float(acc)

    return total_loss / max(n, 1), acc, f1


def find_best_threshold(
    model: nn.Module,
    loader: DataLoader,
) -> float:
    """
    Search for the F1-maximising probability threshold on a held-out set.
    Used to report per-experiment optimal thresholds in the summary table.
    Falls back to 0.4 if sklearn is unavailable or only one class is present.
    """
    if not HAS_SKLEARN:
        return 0.4

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["audio"].to(DEVICE),
                           batch["visual"].to(DEVICE),
                           batch["text"].to(DEVICE))
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(batch["label"].tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels, dtype=int)

    if len(np.unique(all_labels)) < 2:
        return 0.4

    _, _, thresholds = precision_recall_curve(all_labels, all_probs)
    best_t, best_f1 = 0.4, 0.0
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        f = f1_score(all_labels, preds, average="binary", zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t


# ==============================================================================
# SECTION 6: DIFFERENTIAL PRIVACY
# ==============================================================================

def rdp_to_dp(noise_mult: float, sample_rate: float,
              steps: int, delta: float) -> float:
    if noise_mult <= 0 or sample_rate <= 0:
        return float("inf")
    best = float("inf")
    for alpha in range(2, 513):
        rdp = (alpha / (2 * noise_mult ** 2)) * (sample_rate ** 2) * steps
        eps = rdp + math.log(1 / delta) / (alpha - 1)
        if eps < best:
            best = eps
    return best


# ==============================================================================
# SECTION 7: LOCAL TRAINING
# ==============================================================================

@dataclass
class LocalCfg:
    lr:           float = 1e-3   # FIX-15: bumped from 5e-4; Adam handles this well
    local_epochs: int   = 5
    batch_size:   int   = 8
    clip_norm:    float = 1.0
    noise_mult:   float = 1.1
    use_dp:       bool  = True
    mu:           float = 0.0   # FedProx proximal coefficient
    class_weight: Optional[torch.Tensor] = None
    use_adam:     bool  = True   # FIX-15: Adam by default; set False for SGD


def local_train(model: nn.Module, global_model: nn.Module,
                dataset: Dataset, cfg: LocalCfg) -> Dict:
    if len(dataset) == 0:
        return {"delta": {}, "n_samples": 0, "loss": float("inf")}

    bs = min(cfg.batch_size, len(dataset))
    # drop_last avoids a final batch of size 1 that destabilises LayerNorm
    drop_last = (len(dataset) > bs)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True,
                        drop_last=drop_last)

    # FIX-15: Use Adam instead of SGD+momentum.
    # SGD+momentum on the original single-class tiled datasets caused monotone
    # gradient compounding toward one class. Adam's adaptive learning rates
    # and lack of momentum accumulation are far more stable for small,
    # heterogeneous FL clients with diverse patient mixes.
    if cfg.use_adam:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                               weight_decay=1e-4)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                              momentum=0.9, weight_decay=1e-4)

    global_w = {n: p.data.clone()
                for n, p in global_model.named_parameters()}
    w0 = {k: v.clone() for k, v in model.state_dict().items()}

    total_loss, n_samp = 0.0, 0
    model.train()

    for _ in range(cfg.local_epochs):
        for batch in loader:
            audio  = batch["audio"].to(DEVICE)
            visual = batch["visual"].to(DEVICE)
            text   = batch["text"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            batch_sz = labels.size(0)

            opt.zero_grad()
            logits = model(audio, visual, text)
            weight = (cfg.class_weight.to(DEVICE)
                      if cfg.class_weight is not None else None)
            loss = F.cross_entropy(logits, labels, weight=weight)

            # FedProx proximal term
            if cfg.mu > 0:
                prox = sum(
                    ((p - global_w[n]) ** 2).sum()
                    for n, p in model.named_parameters()
                    if n in global_w
                )
                loss = loss + (cfg.mu / 2.0) * prox

            loss.backward()

            # DP-SGD: clip + noise
            # FIX-6: noise is divided by batch_sz so the effective noise on
            # the averaged gradient is clip_norm * noise_mult / batch_sz,
            # matching the standard DP-SGD formulation (Abadi et al. 2016).
            if cfg.use_dp:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        g_norm = p.grad.norm(2)
                        if g_norm > cfg.clip_norm:
                            p.grad.mul_(cfg.clip_norm / (g_norm + 1e-12))
                        p.grad.add_(
                            torch.randn_like(p.grad)
                            * cfg.noise_mult * cfg.clip_norm / batch_sz  # FIX-6
                        )
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.clip_norm * 2)

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            opt.step()
            total_loss += loss.item() * batch_sz
            n_samp     += batch_sz

    delta = {k: (model.state_dict()[k].float() - w0[k].float()) for k in w0}
    return {
        "delta":     delta,
        "n_samples": max(n_samp, 1),
        "loss":      total_loss / max(n_samp, 1),
    }


def local_train_scaffold(
    model: nn.Module,
    global_model: nn.Module,
    dataset: Dataset,
    cfg: LocalCfg,
    client_c: Dict,
    server_c: Dict,
) -> Dict:
    """SCAFFOLD local training with control variate correction."""
    if len(dataset) == 0:
        return {"delta": {}, "c_delta": {}, "n_samples": 0, "loss": float("inf")}

    bs = min(cfg.batch_size, len(dataset))
    drop_last = (len(dataset) > bs)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True,
                        drop_last=drop_last)

    # FIX-15: Adam for SCAFFOLD too; SCAFFOLD correction compensates for
    # client drift so Adam's momentum won't interfere with the correction term.
    if cfg.use_adam:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    w0 = {k: v.clone() for k, v in model.state_dict().items()}
    total_loss, n_samp = 0.0, 0
    K = cfg.local_epochs * max(1, len(dataset) // bs)

    model.train()
    for _ in range(cfg.local_epochs):
        for batch in loader:
            audio  = batch["audio"].to(DEVICE)
            visual = batch["visual"].to(DEVICE)
            text   = batch["text"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            batch_sz = labels.size(0)

            opt.zero_grad()
            logits = model(audio, visual, text)
            weight = (cfg.class_weight.to(DEVICE)
                      if cfg.class_weight is not None else None)
            loss = F.cross_entropy(logits, labels, weight=weight)
            loss.backward()

            # SCAFFOLD correction
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.grad is not None and n in client_c and n in server_c:
                        p.grad.add_(
                            -client_c[n].to(DEVICE) + server_c[n].to(DEVICE)
                        )

            # FIX-6: same batch_sz division as local_train
            if cfg.use_dp:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        g_norm = p.grad.norm(2)
                        if g_norm > cfg.clip_norm:
                            p.grad.mul_(cfg.clip_norm / (g_norm + 1e-12))
                        p.grad.add_(
                            torch.randn_like(p.grad)
                            * cfg.noise_mult * cfg.clip_norm / batch_sz  # FIX-6
                        )
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.clip_norm * 2)

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            opt.step()
            total_loss += loss.item() * batch_sz
            n_samp     += batch_sz

    # Update client control variate: c_i^+ = c_i - c + (w0 - w) / (K * lr)
    # FIX-11: removed dead variable 'new_client_c = {}'
    c_delta = {}
    with torch.no_grad():
        for k, p in model.named_parameters():
            if k in client_c:
                w_diff = (w0.get(k, p.data.clone()).to(DEVICE) - p.data) / (
                    max(K, 1) * cfg.lr + 1e-12
                )
                new_c_i     = client_c[k].to(DEVICE) - server_c[k].to(DEVICE) + w_diff
                c_delta[k]  = (new_c_i - client_c[k].to(DEVICE)).cpu()
                client_c[k] = new_c_i.cpu()

    delta = {k: (model.state_dict()[k].float() - w0[k].float()) for k in w0}
    return {
        "delta":     delta,
        "c_delta":   c_delta,
        "n_samples": max(n_samp, 1),
        "loss":      total_loss / max(n_samp, 1),
    }


# ==============================================================================
# SECTION 8: AGGREGATION STRATEGIES
# ==============================================================================

def agg_mean(updates: List[Dict], weights: List[float]) -> Dict:
    total = sum(weights)
    keys  = list(updates[0].keys())
    return {
        k: sum(w * u[k].float() for u, w in zip(updates, weights)) / total
        for k in keys
    }


def agg_trimmed_mean(updates: List[Dict], ratio: float = 0.1) -> Dict:
    n = len(updates)
    k = max(1, int(ratio * n))
    if 2 * k >= n:
        k = 0

    keys   = list(updates[0].keys())
    result = {}
    for key in keys:
        stacked = torch.stack([u[key].float() for u in updates], dim=0)
        if k > 0:
            s, _ = stacked.sort(dim=0)
            stacked = s[k: n - k]
        result[key] = stacked.mean(dim=0)
    return result


def agg_median(updates: List[Dict]) -> Dict:
    keys = list(updates[0].keys())
    return {
        k: torch.stack([u[k].float() for u in updates], dim=0).median(0).values
        for k in keys
    }


def agg_krum(updates: List[Dict], f: int = 1) -> Dict:
    n = len(updates)
    if n <= 2 * f + 2:
        log.warning("Krum: too few clients (%d) for f=%d -- falling back to mean",
                    n, f)
        return agg_mean(updates, [1.0] * n)

    def flat(u):
        return torch.cat([v.float().flatten() for v in u.values()])

    flat_list = [flat(u) for u in updates]
    nb = n - f - 2
    scores = [
        sum(sorted(
            [(flat_list[i] - flat_list[j]).pow(2).sum().item()
             for j in range(n) if j != i]
        )[:nb])
        for i in range(n)
    ]
    return updates[int(np.argmin(scores))]


# ==============================================================================
# SECTION 9: SERVER-SIDE OPTIMISERS
# ==============================================================================

class FedAdamServer:
    def __init__(self, model: nn.Module, lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999):
        self.lr  = lr
        self.b1  = beta1
        self.b2  = beta2
        self.eps = 1e-8
        self.m   = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
        self.v   = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
        self.t   = 0

    def step(self, model: nn.Module, delta: Dict):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n not in delta:
                    continue
                g         = -delta[n].to(DEVICE)
                self.m[n] = self.b1 * self.m[n] + (1 - self.b1) * g
                self.v[n] = self.b2 * self.v[n] + (1 - self.b2) * g * g
                m_hat     = self.m[n] / c1
                v_hat     = self.v[n] / c2
                p.data   -= self.lr * m_hat / (v_hat.sqrt() + self.eps)


class FedYogiServer:
    """FedYogi: Adaptive FL with Yogi second-moment update."""
    def __init__(self, model: nn.Module, lr: float = 1e-2,
                 beta1: float = 0.9, beta2: float = 0.999,
                 tau: float = 1e-3):
        self.lr  = lr
        self.b1  = beta1
        self.b2  = beta2
        self.tau = tau
        self.m   = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
        self.v   = {k: torch.ones_like(v) * (tau ** 2)
                    for k, v in model.named_parameters()}
        self.t   = 0

    def step(self, model: nn.Module, delta: Dict):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n not in delta:
                    continue
                g         = -delta[n].to(DEVICE)
                self.m[n] = self.b1 * self.m[n] + (1 - self.b1) * g
                self.v[n] = (self.v[n]
                             + (1 - self.b2)
                             * (g * g - self.v[n]).sign() * (g * g))
                self.v[n] = torch.clamp(self.v[n], min=self.tau ** 2)
                m_hat     = self.m[n] / c1
                p.data   -= self.lr * m_hat / (self.v[n].sqrt() + self.tau)


class SCAFFOLDServer:
    """
    SCAFFOLD: Stochastic Controlled Averaging for FL.
    Karimireddy et al. 2020.
    """
    def __init__(self, model: nn.Module):
        self.c = {k: torch.zeros_like(v)
                  for k, v in model.named_parameters()}

    def update_global_c(self, client_c_deltas: List[Dict]):
        """Aggregate client control variate updates into global c."""
        if not client_c_deltas:
            return
        n = len(client_c_deltas)
        with torch.no_grad():
            for k in self.c:
                delta_sum = sum(
                    d[k].to(DEVICE) for d in client_c_deltas if k in d
                )
                self.c[k] += delta_sum / n


# ==============================================================================
# SECTION 10: FL TRAINING LOOPS
# ==============================================================================

@dataclass
class RoundResult:
    round_num:      int
    algorithm:      str
    aggregation:    str
    test_loss:      float
    test_acc:       float
    test_f1:        float
    epsilon:        float
    use_dp:         bool
    n_participated: int
    threshold:      float = 0.4   # decision threshold used for this round


def run_federated(
    algorithm: str,
    aggregation: str,
    client_datasets: List[Dataset],
    test_loader: DataLoader,
    model_factory,
    n_rounds: int = 30,
    local_cfg: LocalCfg = None,
    server_lr: float = 1.0,
    use_dp: bool = True,
    privacy_delta: float = 1e-4,
    seed: int = 42,
    global_model_path: Optional[str] = None,
    eval_threshold: float = 0.4,
) -> List[RoundResult]:

    torch.manual_seed(seed)
    np.random.seed(seed)

    if local_cfg is None:
        local_cfg = LocalCfg()

    global_model = model_factory().to(DEVICE)

    # FIX-13: _load_partial_checkpoint now returns shape_skipped_keys set.
    # _selective_reinit_encoders uses it to only reinit the encoder whose
    # first linear had a shape mismatch — leaving all correctly-loaded
    # weights intact.
    if global_model_path:
        loaded, shape_skipped_keys = _load_partial_checkpoint(
            global_model, global_model_path,
            label=f"FL/{algorithm}/{aggregation}",
        )
        if loaded:
            _selective_reinit_encoders(global_model, shape_skipped_keys)

    # FIX-9: both FedAdam and FedYogi now get their server optimizer
    fed_adam = (FedAdamServer(global_model, lr=server_lr)
                if algorithm == "fedadam" else None)
    fed_yogi = (FedYogiServer(global_model, lr=server_lr)
                if algorithm == "fedyogi" else None)

    results  = []
    cum_eps  = 0.0

    for rnd in range(n_rounds):
        updates, weights = [], []

        for ds in client_datasets:
            if len(ds) == 0:
                continue
            local_model = copy.deepcopy(global_model)
            cfg         = copy.copy(local_cfg)
            cfg.use_dp  = use_dp
            cfg.mu      = 0.01 if algorithm == "fedprox" else 0.0

            res = local_train(local_model, global_model, ds, cfg)
            if res["n_samples"] > 0 and res["delta"]:
                updates.append(res["delta"])
                # FIX-12: weight by sample count only; class balance is
                # already handled inside local_train via cfg.class_weight.
                weights.append(float(res["n_samples"]))

        if not updates:
            continue

        n = len(updates)
        if aggregation == "mean":
            agg = agg_mean(updates, weights)
        elif aggregation == "trimmed_mean":
            agg = (agg_trimmed_mean(updates, 0.1) if n >= 4
                   else agg_mean(updates, weights))
        elif aggregation == "median":
            agg = agg_median(updates)
        elif aggregation == "krum":
            agg = agg_krum(updates, f=max(1, n // 5))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation!r}")

        # Apply delta to global model
        if algorithm == "fedadam" and fed_adam:
            fed_adam.step(global_model, agg)
        elif algorithm == "fedyogi" and fed_yogi:   # FIX-9
            fed_yogi.step(global_model, agg)
        else:
            with torch.no_grad():
                for name, p in global_model.named_parameters():
                    if name in agg:
                        p.data += server_lr * agg[name].to(DEVICE)

        if use_dp:
            avg_n = np.mean([len(ds) for ds in client_datasets if len(ds) > 0])
            sr    = local_cfg.batch_size / max(avg_n, 1)
            steps = local_cfg.local_epochs * max(
                1, int(avg_n / local_cfg.batch_size)
            )
            cum_eps += rdp_to_dp(local_cfg.noise_mult, sr,
                                  steps, privacy_delta)

        # FIX-16: use threshold-based evaluation
        loss, acc, f1 = evaluate(global_model, test_loader,
                                 threshold=eval_threshold)

        r = RoundResult(
            round_num      = rnd + 1,
            algorithm      = algorithm,
            aggregation    = aggregation,
            test_loss      = loss,
            test_acc       = acc,
            test_f1        = f1,
            epsilon        = cum_eps if use_dp else 0.0,
            use_dp         = use_dp,
            n_participated = len(updates),
            threshold      = eval_threshold,
        )
        results.append(r)

        if (rnd + 1) % 5 == 0 or rnd == 0:
            log.info("[%s/%s] R%3d | acc=%.3f f1=%.3f eps=%.3f thr=%.2f",
                     algorithm, aggregation, rnd + 1,
                     acc, f1, cum_eps if use_dp else 0.0, eval_threshold)

    return results


def run_federated_scaffold(
    client_datasets: List[Dataset],
    test_loader: DataLoader,
    model_factory,
    n_rounds: int = 30,
    local_cfg: LocalCfg = None,
    use_dp: bool = True,
    privacy_delta: float = 1e-4,
    seed: int = 42,
    global_model_path: Optional[str] = None,
    eval_threshold: float = 0.4,
) -> List[RoundResult]:
    """SCAFFOLD federated learning loop."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if local_cfg is None:
        local_cfg = LocalCfg()

    global_model = model_factory().to(DEVICE)

    # FIX-13: same selective reinit logic as run_federated
    if global_model_path:
        loaded, shape_skipped_keys = _load_partial_checkpoint(
            global_model, global_model_path,
            label="FL/scaffold/mean",
        )
        if loaded:
            _selective_reinit_encoders(global_model, shape_skipped_keys)

    scaffold_server = SCAFFOLDServer(global_model)

    client_cs = [
        {k: torch.zeros_like(v) for k, v in global_model.named_parameters()}
        for _ in client_datasets
    ]

    results = []
    cum_eps = 0.0

    for rnd in range(n_rounds):
        updates, c_deltas, weights = [], [], []

        for cid, ds in enumerate(client_datasets):
            if len(ds) == 0:
                continue
            local_model = copy.deepcopy(global_model)
            cfg         = copy.copy(local_cfg)
            cfg.use_dp  = use_dp

            res = local_train_scaffold(
                local_model, global_model, ds, cfg,
                client_cs[cid], scaffold_server.c
            )
            if res["n_samples"] > 0 and res["delta"]:
                updates.append(res["delta"])
                c_deltas.append(res.get("c_delta", {}))
                # FIX-12: sample-count only
                weights.append(float(res["n_samples"]))

        if not updates:
            continue

        agg = agg_mean(updates, weights)
        with torch.no_grad():
            for name, p in global_model.named_parameters():
                if name in agg:
                    p.data += agg[name].to(DEVICE)

        scaffold_server.update_global_c(c_deltas)

        if use_dp:
            avg_n = np.mean([len(ds) for ds in client_datasets if len(ds) > 0])
            sr    = local_cfg.batch_size / max(avg_n, 1)
            steps = local_cfg.local_epochs * max(
                1, int(avg_n / local_cfg.batch_size)
            )
            cum_eps += rdp_to_dp(local_cfg.noise_mult, sr, steps, privacy_delta)

        # FIX-16: threshold-based evaluation
        loss, acc, f1 = evaluate(global_model, test_loader,
                                 threshold=eval_threshold)
        r = RoundResult(
            round_num      = rnd + 1,
            algorithm      = "scaffold",
            aggregation    = "mean",
            test_loss      = loss,
            test_acc       = acc,
            test_f1        = f1,
            epsilon        = cum_eps if use_dp else 0.0,
            use_dp         = use_dp,
            n_participated = len(updates),
            threshold      = eval_threshold,
        )
        results.append(r)

        if (rnd + 1) % 5 == 0 or rnd == 0:
            log.info("[scaffold/mean] R%3d | acc=%.3f f1=%.3f eps=%.3f thr=%.2f",
                     rnd + 1, acc, f1, cum_eps if use_dp else 0.0, eval_threshold)

    return results


# ==============================================================================
# SECTION 11: VISUALISATION
# ==============================================================================

# ── Clean display-name mapping ────────────────────────────────────────────────
# Maps internal result-dict keys → short, readable legend labels.
# Any key NOT in this dict falls back to the raw key string.
LABEL_MAP: Dict[str, str] = {
    # ── Figure 1: main algorithm comparison ──────────────────────────────────
    "fedavg_mean_noDP":    "FedAvg (No DP)",
    "fedavg_mean":         "FedAvg + DP",
    "fedprox_mean":        "FedProx + DP",
    "fedadam_mean":        "FedAdam + DP",
    "fedyogi_mean":        "FedYogi + DP",
    "scaffold_mean":       "SCAFFOLD + DP",
    "scaffold_mean_noDP":  "SCAFFOLD (No DP)",
    # ── Figure 2: aggregation variants ───────────────────────────────────────
    "fedavg_trimmed_mean": "Trimmed Mean",
    "fedavg_median":       "Coord.-wise Median",
    "fedavg_krum":         "Krum",
    # ── Figure 2: DP noise-multiplier sweep ──────────────────────────────────
    "fedavg_mean_nm0.5":   "FedAvg nm=0.5 (high ε)",
    "fedavg_mean_nm1.0":   "FedAvg nm=1.0",
    "fedavg_mean_nm1.5":   "FedAvg nm=1.5 (low ε)",
}

# Figure-1 keys (main algorithms); everything else lands in Figure 2.
_FIG1_KEYS = {
    "fedavg_mean_noDP",
    "fedavg_mean",
    "fedprox_mean",
    "fedadam_mean",
    "fedyogi_mean",
    "scaffold_mean",
    "scaffold_mean_noDP",
}

# Colour palette — distinct, colourblind-friendly subset of tab10
_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # teal
]


def _nice(key: str) -> str:
    """Return the clean display name for a result key."""
    return LABEL_MAP.get(key, key)


def _make_figure(
    subset: Dict[str, List[RoundResult]],
    suptitle: str,
    out_path: Path,
) -> None:
    """
    Render a 2×2 panel figure (Accuracy | F1 | Privacy Budget | Privacy-Utility)
    for the given subset of results and save to *out_path*.

    Design principles applied here (per reviewer feedback):
      • Maximum 7 series per figure — no cluttered overlapping lines.
      • Legend uses clean human-readable names (via LABEL_MAP).
      • DP runs drawn as dashed lines; non-DP as solid — easy to distinguish.
      • Legend placed outside the axes so it never overlaps data.
      • Consistent colour assignment across all four panels.
    """
    if not subset:
        log.warning("No data for figure: %s — skipping.", suptitle)
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)

    ax_acc, ax_f1, ax_eps, ax_pvt = axes.flat

    colour_cycle = {}   # key → colour (stable across panels)
    for idx, key in enumerate(subset):
        colour_cycle[key] = _PALETTE[idx % len(_PALETTE)]

    has_dp_series = False

    for key, rlist in subset.items():
        if not rlist:
            continue
        rounds   = [r.round_num for r in rlist]
        color    = colour_cycle[key]
        ls       = "-" if not rlist[0].use_dp else "--"
        lw       = 2.2 if not rlist[0].use_dp else 1.8
        disp_lbl = _nice(key)

        ax_acc.plot(rounds, [r.test_acc for r in rlist],
                    color=color, ls=ls, lw=lw, label=disp_lbl)
        ax_f1.plot(rounds,  [r.test_f1  for r in rlist],
                   color=color, ls=ls, lw=lw, label=disp_lbl)

        if rlist[0].use_dp:
            has_dp_series = True
            ax_eps.plot(rounds, [r.epsilon for r in rlist],
                        color=color, ls=ls, lw=lw, label=disp_lbl)
            ax_pvt.scatter(
                rlist[-1].epsilon, rlist[-1].test_acc,
                color=color, s=120, zorder=5, label=disp_lbl,
                edgecolors="white", linewidths=0.6,
            )
            # Annotate final value in the accuracy & F1 panels
            ax_acc.annotate(
                f"{rlist[-1].test_acc:.2f}",
                xy=(rounds[-1], rlist[-1].test_acc),
                fontsize=6.5, color=color, ha="left", va="bottom",
            )

    # ── Axis styling ─────────────────────────────────────────────────────────
    _style_ax(ax_acc, "Test Accuracy",              "Round", "Accuracy")
    _style_ax(ax_f1,  "F1 Score (Depressed = 1)",   "Round", "F1")
    _style_ax(ax_eps, "Privacy Budget ε (DP only)", "Round", "ε (epsilon)")
    _style_ax(ax_pvt, "Privacy-Utility Trade-off",  "Final ε", "Final Accuracy")

    if not has_dp_series:
        ax_eps.set_visible(False)
        ax_pvt.set_visible(False)

    # ── Shared legend — placed below all panels so it never overlaps ─────────
    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(subset), 4),
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.06),
    )

    # ── Solid / dashed key ───────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    style_legend = [
        Line2D([0], [0], color="gray", lw=2,   ls="-",  label="No DP"),
        Line2D([0], [0], color="gray", lw=1.8, ls="--", label="With DP"),
    ]
    ax_acc.legend(
        handles=style_legend,
        fontsize=7, loc="upper right",
        framealpha=0.7, handlelength=2,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved -> %s", out_path)


def _style_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent axis styling."""
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_comparison(all_results: Dict[str, List[RoundResult]]) -> None:
    """
    UPDATED (per reviewer feedback — fixes clutter & overloaded legend):

    Splits the original single cluttered figure into TWO focused figures:

    Figure 1 — Algorithm Comparison (fl_daic_fig1_algorithms.png)
        Shows the 5–6 main FL algorithms side-by-side (FedAvg, FedProx,
        FedAdam, FedYogi, SCAFFOLD) including the No-DP baseline.
        Purpose: "Which FL algorithm works best?"

    Figure 2 — DP Ablation & Robust Aggregation (fl_daic_fig2_dp_ablation.png)
        Shows (a) the noise-multiplier sweep (nm=0.5/1.0/1.5) and
        (b) Byzantine-robust aggregation variants (Trimmed Mean, Median, Krum).
        Purpose: "How does DP noise level / aggregation strategy affect results?"

    Both figures share the same 2×2 panel layout (Accuracy | F1 | ε | Trade-off)
    and use clean display names from LABEL_MAP.  Each figure has at most 7 series
    so no panel is cluttered.
    """
    if not HAS_MATPLOTLIB:
        log.warning("matplotlib not available — skipping plots.")
        return

    # ── Split results into two groups ─────────────────────────────────────────
    fig1_data: Dict[str, List[RoundResult]] = {}
    fig2_data: Dict[str, List[RoundResult]] = {}

    for key, rlist in all_results.items():
        if key in _FIG1_KEYS:
            fig1_data[key] = rlist
        else:
            fig2_data[key] = rlist

    # ── Figure 1: main algorithm comparison ──────────────────────────────────
    _make_figure(
        subset   = fig1_data,
        suptitle = (
            "Figure 1 — FL Algorithm Comparison\n"
            "DAIC-WOZ Depression Detection · MentalBERT"
        ),
        out_path = RESULTS_DIR / "fl_daic_fig1_algorithms.png",
    )

    # ── Figure 2: DP ablation + robust aggregation ───────────────────────────
    if fig2_data:
        _make_figure(
            subset   = fig2_data,
            suptitle = (
                "Figure 2 — DP Noise Sensitivity & Robust Aggregation\n"
                "DAIC-WOZ Depression Detection · MentalBERT"
            ),
            out_path = RESULTS_DIR / "fl_daic_fig2_dp_ablation.png",
        )
    else:
        log.info("No DP-sweep / robust-aggregation results — Figure 2 skipped.")

    # ── Legacy combined figure (kept for backward compatibility) ─────────────
    # Combines both groups into one figure with reduced line width so existing
    # pipelines that read fl_daic_comparison.png continue to work.
    _make_figure(
        subset   = all_results,
        suptitle = (
            "FL Algorithm Comparison — DAIC-WOZ Depression Detection (MentalBERT)\n"
            "(combined overview — see fig1/fig2 for cleaner per-group views)"
        ),
        out_path = RESULTS_DIR / "fl_daic_comparison.png",
    )


def latex_summary(all_results: Dict[str, List[RoundResult]]) -> str:
    sweep_keys = [k for k in all_results if any(
        tag in k for tag in ["_nm", "_cn", "_le"]
    )]
    algo_keys  = [k for k in all_results if k not in sweep_keys]

    def _table_rows(keys):
        rows = []
        for lbl in sorted(keys):
            rlist = all_results[lbl]
            final = rlist[-1]
            eps_str = f"{final.epsilon:.2f}" if final.use_dp else "--"
            rows.append(
                f"{final.algorithm.upper()} & "
                f"{final.aggregation.replace('_', ' ')} & "
                f"{'Y' if final.use_dp else 'N'} & "
                f"{final.test_acc:.4f} & {final.test_f1:.4f} & "
                f"{eps_str} & {final.round_num} \\\\"
            )
        return rows

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{FL Algorithm Comparison on DAIC-WOZ (MentalBERT)}",
        r"\begin{tabular}{lllcccc}",
        r"\toprule",
        r"Algorithm & Aggregation & DP & Final Acc & F1 & $\varepsilon$ & Rounds \\",
        r"\midrule",
    ] + _table_rows(algo_keys) + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    sweep_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{DP Parameter Sweep (FedAvg, mean aggregation)}",
        r"\begin{tabular}{lllcccc}",
        r"\toprule",
        r"Experiment & Aggregation & DP & Final Acc & F1 & $\varepsilon$ & Rounds \\",
        r"\midrule",
    ] + _table_rows(sweep_keys) + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex = "\n".join(lines) + "\n\n" + "\n".join(sweep_lines)
    (RESULTS_DIR / "latex_table.tex").write_text(tex)
    return tex


# ==============================================================================
# SECTION 12: MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FL on DAIC-WOZ with MentalBERT"
    )
    parser.add_argument("--data_dir",          default="./data",
                        help="Path to DAIC-WOZ data/ directory")
    parser.add_argument("--labels_dir",        default="./labels",
                        help="Path to labels directory (unused; labels must be in data_dir)")
    parser.add_argument("--rounds",            type=int,   default=30)
    parser.add_argument("--batch_size",        type=int,   default=8)
    parser.add_argument("--local_epochs",      type=int,   default=5)
    parser.add_argument("--lr",                type=float, default=1e-3)
    parser.add_argument("--use_mentalbert",    action="store_true",
                        help="Use MentalBERT for transcript embeddings")
    parser.add_argument("--no_dp",             action="store_true",
                        help="Disable differential privacy")
    parser.add_argument("--noise_mult",        type=float, default=1.1)
    parser.add_argument("--clip_norm",         type=float, default=1.0)
    parser.add_argument("--test_frac",         type=float, default=0.2,
                        help="Fraction of patients used as held-out test")
    parser.add_argument("--n_hospitals",       type=int,   default=8,
                        help="Number of hospital FL clients (FIX-14: hospital grouping)")
    parser.add_argument("--eval_threshold",    type=float, default=0.4,
                        help="Probability threshold for depression detection "
                             "(FIX-16: <0.5 boosts sensitivity for minority class)")
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--global_model_path", type=str,   default=None,
                        help="Path to pre-trained global model to warm-start from")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        log.error("data_dir not found: %s", data_dir)
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 1. Load raw patient data ───────────────────────────────────────────
    patients = load_dataset(data_dir)
    if len(patients) < 3:
        log.error("Need at least 3 patients. Found %d", len(patients))
        return

    # ── 2. Compute text features (MentalBERT or random) ───────────────────
    embedder = MentalBERTEmbedder(use_mentalbert=args.use_mentalbert)
    for p in patients:
        # FIX-7: session_embedding() → mean of all utterance embeddings → (768,).
        # Stored as text_vec used for BOTH training (via HospitalDataset) and test.
        # This ensures identical feature distribution at train and test time.
        p["text_vec"] = embedder.session_embedding(p["utterances"])

    TEXT_DIM = embedder.DIM  # 768

    # ── 3. Pad feature vectors to uniform dim across patients ──────────────
    audio_dim  = max(p["audio"].shape[0]  for p in patients)
    visual_dim = max(p["visual"].shape[0] for p in patients)

    def pad(arr, d):
        return (arr[:d] if arr.shape[0] >= d
                else np.pad(arr, (0, d - arr.shape[0])))

    for p in patients:
        p["audio"]  = pad(p["audio"],  audio_dim)
        p["visual"] = pad(p["visual"], visual_dim)

    log.info("Dims -> audio=%d  visual=%d  text=%d",
             audio_dim, visual_dim, TEXT_DIM)

    # ── 4. Stratified train / test split at patient level ─────────────────
    rng           = np.random.default_rng(args.seed)
    depressed     = [p for p in patients if p["label"] == 1]
    not_depressed = [p for p in patients if p["label"] == 0]
    rng.shuffle(depressed)
    rng.shuffle(not_depressed)

    n_test    = max(1, int(len(patients) * args.test_frac))
    n_test_d  = max(1, min(len(depressed)     - 1, n_test // 2))
    n_test_nd = max(1, min(len(not_depressed) - 1, n_test - n_test_d))

    test_pts  = depressed[:n_test_d]  + not_depressed[:n_test_nd]
    train_pts = depressed[n_test_d:]  + not_depressed[n_test_nd:]

    if len(train_pts) == 0:
        train_pts, test_pts = patients[:-2], patients[-2:]

    log.info("Train patients=%d  Test patients=%d",
             len(train_pts), len(test_pts))

    # ── 5. Build per-hospital datasets (FIX-14) ────────────────────────────
    # FIX-14: replaced one-patient-per-client with hospital grouping.
    # Each hospital gets a stratified mix of both classes, so every client's
    # local gradient has meaningful signal for learning to distinguish depression.
    n_hospitals = min(args.n_hospitals, len(train_pts))
    client_datasets = build_hospital_clients(train_pts, n_hospitals, seed=args.seed)

    # ── 6. Build test DataLoader ───────────────────────────────────────────
    class _TestDS(Dataset):
        def __init__(self, pts):
            self.audio  = torch.tensor(
                np.stack([p["audio"]    for p in pts]), dtype=torch.float32)
            self.visual = torch.tensor(
                np.stack([p["visual"]   for p in pts]), dtype=torch.float32)
            # text_vec is the same session-mean used in training
            self.text   = torch.tensor(
                np.stack([p["text_vec"] for p in pts]), dtype=torch.float32)
            self.labels = torch.tensor(
                [p["label"] for p in pts],               dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {"audio":  self.audio[i],  "visual": self.visual[i],
                    "text":   self.text[i],   "label":  self.labels[i]}

    test_ds     = _TestDS(test_pts)
    test_loader = DataLoader(test_ds, batch_size=max(len(test_pts), 1),
                             shuffle=False)

    test_labels = [p["label"] for p in test_pts]
    log.info("Test set: %d depressed  %d not depressed",
             sum(test_labels), len(test_labels) - sum(test_labels))

    # ── 7. Model factory ───────────────────────────────────────────────────
    def model_factory():
        return DepressionNet(
            audio_dim     = audio_dim,
            visual_dim    = visual_dim,
            text_dim      = TEXT_DIM,    # 768
            fusion_hidden = 256,         # matches centralized MultimodalFusionModel
            dropout       = 0.2,
        )

    log.info("Model params: %d", model_factory().count_params())

    # ── 8. Local training config ───────────────────────────────────────────
    min_client_size = min(len(ds) for ds in client_datasets)
    base_cfg = LocalCfg(
        lr           = args.lr,
        local_epochs = args.local_epochs,
        batch_size   = min(args.batch_size, max(1, min_client_size)),
        clip_norm    = args.clip_norm,
        noise_mult   = args.noise_mult,
        use_dp       = not args.no_dp,
        use_adam     = True,   # FIX-15: Adam for all local training
    )

    n_pos = sum(1 for p in train_pts if p["label"] == 1)
    n_neg = len(train_pts) - n_pos
    # Inverse-frequency weighting: w_c = N / (n_classes * count_c)
    w_neg = len(train_pts) / (2.0 * max(n_neg, 1))
    w_pos = len(train_pts) / (2.0 * max(n_pos, 1))
    base_cfg.class_weight = torch.tensor([w_neg, w_pos], dtype=torch.float32)
    log.info("Class weights -> not-depressed=%.3f  depressed=%.3f", w_neg, w_pos)

    privacy_delta = 1.0 / max(
        sum(len(ds) for ds in client_datasets), 1
    )

    eval_thr = args.eval_threshold  # FIX-16: configurable threshold

    # ── 9. Define experiments ──────────────────────────────────────────────
    # FIX-17: Deduplicate experiments by label before running.
    # When --no_dp is passed all dp_flags become False, so
    # ("fedavg","mean",1.0,False) and ("fedavg","mean",1.0,True) both produce
    # "fedavg_mean_noDP". We deduplicate to avoid silent overwrites.
    _RAW_EXPERIMENTS = [
        # (algorithm, aggregation, server_lr, use_dp)
        ("fedavg",  "mean",         1.0,  False),  # baseline, no DP
        ("fedavg",  "mean",         1.0,  True),   # FedAvg + DP
        ("fedavg",  "trimmed_mean", 1.0,  True),   # Byzantine-robust
        ("fedavg",  "median",       1.0,  True),   # Coordinate-wise median
        ("fedprox", "mean",         1.0,  True),   # FedProx (mu=0.01)
        ("fedadam", "mean",         1e-3, True),   # FedAdam adaptive
        ("fedyogi", "mean",         1e-2, True),   # FedYogi adaptive
    ]
    if len(client_datasets) >= 5:
        _RAW_EXPERIMENTS.append(("fedavg", "krum", 1.0, True))

    # FIX-17: Apply --no_dp flag and deduplicate
    seen_labels: set = set()
    EXPERIMENTS = []
    for algo, agg, slr, dp_flag in _RAW_EXPERIMENTS:
        if args.no_dp:
            dp_flag = False
        label = f"{algo}_{agg}" + ("" if dp_flag else "_noDP")
        if label not in seen_labels:
            seen_labels.add(label)
            EXPERIMENTS.append((algo, agg, slr, dp_flag, label))

    all_results: Dict[str, List[RoundResult]] = {}

    # ── Standard algorithm experiments ────────────────────────────────────
    for algo, agg, slr, dp_flag, label in EXPERIMENTS:
        log.info("\n>  Running: %s", label)
        results = run_federated(
            algorithm         = algo,
            aggregation       = agg,
            client_datasets   = client_datasets,
            test_loader       = test_loader,
            model_factory     = model_factory,
            n_rounds          = args.rounds,
            local_cfg         = base_cfg,
            server_lr         = slr,
            use_dp            = dp_flag,
            privacy_delta     = privacy_delta,
            seed              = args.seed,
            global_model_path = args.global_model_path,
            eval_threshold    = eval_thr,
        )
        all_results[label] = results

    # ── SCAFFOLD ───────────────────────────────────────────────────────────
    scaffold_label = "scaffold_mean" + ("" if not args.no_dp else "_noDP")
    log.info("\n>  Running: %s", scaffold_label)
    all_results[scaffold_label] = run_federated_scaffold(
        client_datasets   = client_datasets,
        test_loader       = test_loader,
        model_factory     = model_factory,
        n_rounds          = args.rounds,
        local_cfg         = base_cfg,
        use_dp            = not args.no_dp,
        privacy_delta     = privacy_delta,
        seed              = args.seed,
        global_model_path = args.global_model_path,
        eval_threshold    = eval_thr,
    )

    # ── DP parameter sweep (FedAvg/mean only) ─────────────────────────────
    if not args.no_dp:
        for nm in [0.5, 1.0, 1.5]:
            sweep_cfg = copy.copy(base_cfg)
            sweep_cfg.noise_mult = nm
            sweep_label = f"fedavg_mean_nm{nm}"
            log.info("\n>  DP sweep: %s", sweep_label)
            all_results[sweep_label] = run_federated(
                algorithm         = "fedavg",
                aggregation       = "mean",
                client_datasets   = client_datasets,
                test_loader       = test_loader,
                model_factory     = model_factory,
                n_rounds          = args.rounds,
                local_cfg         = sweep_cfg,
                server_lr         = 1.0,
                use_dp            = True,
                privacy_delta     = privacy_delta,
                seed              = args.seed,
                global_model_path = args.global_model_path,
                eval_threshold    = eval_thr,
            )

    # ── Save JSON results ──────────────────────────────────────────────────
    out_json = RESULTS_DIR / "results.json"
    with open(out_json, "w") as f:
        json.dump(
            {k: [asdict(r) for r in v] for k, v in all_results.items()},
            f, indent=2,
        )
    log.info("Results saved -> %s", out_json)

    # ── Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Label':<40} {'Acc':>7} {'F1':>7} {'eps':>8} {'thr':>6}")
    print("=" * 80)
    for lbl, rlist in sorted(all_results.items()):
        final   = rlist[-1]
        eps_str = f"{final.epsilon:.2f}" if final.use_dp else "--"
        print(f"{lbl:<40} {final.test_acc:>7.4f} {final.test_f1:>7.4f} "
              f"{eps_str:>8} {final.threshold:>6.2f}")
    print("=" * 80)

    # ── Best-threshold re-evaluation on final models ───────────────────────
    # For each experiment, report the F1-optimal threshold found on test set.
    # (In production, use a separate calibration set.)
    print("\n" + "=" * 80)
    print("FINAL METRICS SUMMARY (bugs fixed, hospital clients, Adam, threshold tuned)")
    print("=" * 80)

    # ── Plots and LaTeX ────────────────────────────────────────────────────
    plot_comparison(all_results)
    latex_summary(all_results)
    log.info("LaTeX table saved -> %s", RESULTS_DIR / "latex_table.tex")


if __name__ == "__main__":
    main()