#!/usr/bin/env python3
"""
/home/ritik26/Desktop/BE-Major-Project/train_base_model_centralized.py — FIXED VERSION
================================================
Rigorous centralized training of MentalBERT multimodal depression classifier
on DAIC-WOZ dataset. NO differential privacy, NO federated learning.

BUGS FIXED (18 total):
  BUG-1:  Duplicate 'early_stop_metric' field in TrainingConfig dataclass
  BUG-2:  Optional not imported (used in type hints)
  BUG-3:  Tuple not imported (used in type hints)
  BUG-4:  average_precision_score, matthews_corrcoef not imported at top
  BUG-5:  get_linear_schedule_with_warmup imported only inside HAS_TRANSFORMERS
          block but used unconditionally — crashes when transformers missing
  BUG-6:  pos_ratio used in two_stage_training block but never defined
  BUG-7:  train_labels used before assignment in multiple code paths
  BUG-8:  FocalLoss defined twice (once as top-level class, once inside fn)
  BUG-9:  FocalLoss.__init__ accepted 'weight' param but forward() ignored it
  BUG-10: Stage-1 loop called optimizer.step() without loss.backward() first
  BUG-11: find_optimal_threshold defined as a mid-class snippet, not a function
  BUG-12: evaluate_with_threshold defined as a snippet inside class body
  BUG-13: stratify_clients was a stub (just 'pass') — now actually implemented
  BUG-14: criterion created twice in train_model — second creation overwrote first
  BUG-15: two_stage_training block ran before pos_ratio was ever computed
  BUG-16: detected_audio/visual dims computed after two_stage block that needs them
  BUG-17: WeightedRandomSampler set up before train_labels was defined
  BUG-18: get_linear_schedule_with_warmup missing when HAS_TRANSFORMERS=False;
          now falls back to CosineAnnealingLR from torch

CLASS IMBALANCE HANDLING (DAIC-WOZ is ~30% depressed / 70% not depressed):
  - FocalLoss (alpha=0.75, gamma=2.0): down-weights easy negatives
  - WeightedRandomSampler: oversamples minority class so every batch is balanced
  - Class-weighted evaluation: reports per-class F1, MCC, AUC-PR
  - Two-stage training: Stage-1 on balanced subset, Stage-2 on full data
  - Optimal threshold search on dev set (maximises F1 for minority class)
  - AUC-PR as primary early-stopping metric (better than F1/ROC for imbalance)
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR       # always available
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    precision_recall_curve, classification_report,
)

# ── Optional: transformers for MentalBERT ─────────────────────────────────────
try:
    from transformers import (
        AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    # FIX-BUG-5: provide a no-op so rest of code compiles either way
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        return CosineAnnealingLR(optimizer, T_max=max(num_training_steps, 1))

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    # ── Paths ──────────────────────────────────────────────────────────────────
    data_dir:        str = "./data"
    labels_dir:      str = "./labels"
    output_path:     str = "~/.federated/data/global_models/global_round1.pt"
    mentalbert_path: str = str(Path.home() / ".federated" / "models" / "mentalbert")

    # ── Model dims ─────────────────────────────────────────────────────────────
    text_dim:      int = 768    # MentalBERT hidden size
    audio_dim:     int = 176    # detected at runtime; default = eGeMAPS mean+std
    visual_dim:    int = 256    # detected at runtime; default = CNN pool
    fusion_hidden: int = 256
    dropout:       float = 0.3
    n_classes:     int = 2      # binary: depressed / not-depressed

    # ── Training ───────────────────────────────────────────────────────────────
    epochs:         int   = 50
    batch_size:     int   = 16
    lr:             float = 2e-5
    weight_decay:   float = 1e-4
    max_grad_norm:  float = 1.0
    warmup_ratio:   float = 0.1

    # ── Class-imbalance handling ───────────────────────────────────────────────
    use_weighted_sampler: bool  = True    # oversample minority class in each batch
    use_focal_loss:       bool  = True    # Focal Loss (down-weights easy negatives)
    focal_alpha:          float = 0.75   # weight for minority class (0.5 = none)
    focal_gamma:          float = 2.0    # focusing parameter (0 = CE, 2 = standard)

    # ── Two-stage training ─────────────────────────────────────────────────────
    two_stage_training:    bool  = True
    stage1_epochs:         int   = 10
    stage1_lr_multiplier:  float = 2.0

    # ── Early stopping ─────────────────────────────────────────────────────────
    patience:           int   = 10
    min_delta:          float = 0.001
    # FIX-BUG-1: removed duplicate 'early_stop_metric' field.
    # AUC-PR is the best metric for imbalanced binary classification.
    early_stop_metric:  str   = "auc_pr"   # options: auc_pr | f1 | roc_auc | loss

    # ── Optimal-threshold search ───────────────────────────────────────────────
    find_best_threshold: bool = True    # search F1-optimal threshold on dev set

    # ── Data ───────────────────────────────────────────────────────────────────
    max_text_len: int   = 512
    test_frac:    float = 0.0 
    seed:         int   = 42

    # ── Hardware ───────────────────────────────────────────────────────────────
    device:      str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4

    # ── Checkpointing ──────────────────────────────────────────────────────────
    log_every:       int  = 10
    save_best_only:  bool = True
    metrics_file:    str  = "training_metrics.json"

    def __post_init__(self):
        self.output_path = os.path.expanduser(self.output_path)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_labels(labels_dir: str, split: str = "train") -> pd.DataFrame:
    """Load labels from metadata_mapped.csv and filter by split CSV."""
    labels_path = Path(labels_dir) / "metadata_mapped.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    df = pd.read_csv(labels_path)
    df.columns = df.columns.str.strip()
    df["Participant_ID"] = df["Participant_ID"].astype(str).str.strip()
    df["PHQ_Binary"]     = df["PHQ_Binary"].astype(int)
    df["PHQ_Score"]      = pd.to_numeric(df["PHQ_Score"], errors="coerce").fillna(0).astype(int)

    split_file = Path(labels_dir) / f"{split}_split.csv"
    if split_file.exists():
        split_df  = pd.read_csv(split_file)
        valid_ids = set(split_df["Participant_ID"].astype(str))
        df = df[df["Participant_ID"].isin(valid_ids)].copy()
        log.info("Loaded %d labels for '%s' split", len(df), split)
    else:
        log.warning("Split file %s not found — using all labels", split_file)

    return df


def _read_csv_robust(path: Path) -> Optional[pd.DataFrame]:
    """Try multiple separators; return None if all fail."""
    for sep in [",", ";", "\t", r"\s+"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return None


def _pool_numeric_features(
    df: pd.DataFrame,
    exclude_cols: Tuple[str, ...] = ("name", "timeStamp", "frame", "frametime"),
) -> np.ndarray:
    """
    Pool frame-level features to session-level: mean + std per feature.
    Returns flat float32 vector of shape (2 * n_features,).
    """
    exclude_lower = {c.lower() for c in exclude_cols}
    cols = [c for c in df.columns if c.strip().lower() not in exclude_lower]
    if not cols:
        return np.array([], dtype=np.float32)

    num_df = df[cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    num_df = num_df.dropna(how="all").fillna(0.0)
    if num_df.empty:
        return np.array([], dtype=np.float32)

    mean_v = num_df.mean(axis=0).values.astype(np.float32)
    std_v  = num_df.std(axis=0).fillna(0.0).values.astype(np.float32)
    return np.concatenate([mean_v, std_v])


def load_patient_features(patient_dir: Path, label: int) -> Optional[Dict[str, Any]]:
    """Load all available modality features for one patient."""
    pid      = patient_dir.name.split("_")[0]
    feat_dir = patient_dir / "features"
    if not feat_dir.exists():
        feat_dir = patient_dir

    result = {"patient_id": pid, "label": label}

    # ── Audio ─────────────────────────────────────────────────────────────────
    audio_vec: Optional[np.ndarray] = None
    for patterns in [
        ["*OpenSMILE*eGeMAPS*.csv", "*egemaps*.csv", "*eGeMAPS*.csv"],
        ["*MFCC*.csv", "*mfcc*.csv"],
        ["*BoAW*.csv", "*boaw*.csv"],
    ]:
        for pat in patterns:
            for f in feat_dir.glob(pat):
                df = _read_csv_robust(f)
                if df is not None:
                    v = _pool_numeric_features(df)
                    if len(v) > 0:
                        audio_vec = v
                        break
            if audio_vec is not None:
                break
        if audio_vec is not None:
            break

    result["audio_vec"] = (audio_vec if audio_vec is not None
                           else np.zeros(176, dtype=np.float32)).astype(np.float32)

    # ── Visual ────────────────────────────────────────────────────────────────
    visual_vec: Optional[np.ndarray] = None
    for pat in ["*OpenFace*.csv", "*openface*.csv", "*Pose_gaze_AUs*.csv"]:
        for f in feat_dir.glob(pat):
            df = _read_csv_robust(f)
            if df is not None:
                keep = [c for c in df.columns
                        if any(t in c.lower() for t in ["au", "gaze", "pose", "blink"])
                        and c.lower() not in ("frame", "face_id", "timestamp", "confidence")]
                v = _pool_numeric_features(df[keep] if keep else df)
                if len(v) > 0:
                    visual_vec = v
                    break
        if visual_vec is not None:
            break

    # CNN fallback
    if visual_vec is None:
        for pat in ["*vgg*.csv", "*densenet*.csv", "*resnet*.csv", "*CNN_*.csv"]:
            for f in feat_dir.glob(pat):
                df = _read_csv_robust(f)
                if df is not None:
                    neuron_cols = [c for c in df.columns if "neuron" in c.lower()]
                    v = (df[neuron_cols].iloc[0].values.astype(np.float32)
                         if neuron_cols else _pool_numeric_features(df))
                    if len(v) > 0:
                        visual_vec = v
                        break
            if visual_vec is not None:
                break

    result["visual_vec"] = (visual_vec if visual_vec is not None
                            else np.zeros(256, dtype=np.float32)).astype(np.float32)

    # ── Transcript ────────────────────────────────────────────────────────────
    utterances: List[str] = []
    candidates = [
        patient_dir / f"{pid}_Transcript.csv",
        feat_dir    / f"{pid}_Transcript.csv",
        *feat_dir.glob("*Transcript*"),
        *patient_dir.glob("*Transcript*"),
    ]
    for cand in candidates:
        if not cand.exists():
            continue
        df = _read_csv_robust(cand)
        if df is None:
            continue
        text_col = next(
            (c for c in ["Text", "text", "Value", "value", "Utterance", "utterance"]
             if c in df.columns),
            df.columns[-1] if df.shape[1] >= 3 else None,
        )
        if text_col is None:
            continue
        speaker_col = next(
            (c for c in ["Speaker", "speaker", "Role", "role"] if c in df.columns), None
        )
        if speaker_col:
            mask = df[speaker_col].astype(str).str.lower().str.contains(
                r"participant|patient|\bp\b", na=False
            )
            df = df[mask]
        utterances = [t for t in df[text_col].dropna().astype(str).str.strip()
                      if len(t.split()) >= 3]
        break

    if not utterances:
        utterances = [f"patient {pid} session"]

    result["utterances"] = utterances
    result["full_text"]  = " ".join(utterances)

    log.info("Patient %s | audio:%d visual:%d utts:%d label:%d",
             pid, len(result["audio_vec"]), len(result["visual_vec"]),
             len(utterances), label)
    return result


def load_split(data_dir: str, labels_dir: str, split: str) -> List[Dict[str, Any]]:
    """Load all patients for one named split (train / dev / test)."""
    labels_df = load_labels(labels_dir, split)
    data_path = Path(data_dir)
    patients  = []
    for _, row in labels_df.iterrows():
        pid  = str(row["Participant_ID"])
        dirs = list(data_path.glob(f"{pid}_P"))
        if not dirs:
            log.warning("Patient %s directory not found", pid)
            continue
        p = load_patient_features(dirs[0], int(row["PHQ_Binary"]))
        if p is not None:
            patients.append(p)
    log.info("Loaded %d patients for '%s' split", len(patients), split)
    return patients


# FIX-BUG-13: stratify_clients now actually implemented
def stratify_clients(
    patients: List[Dict[str, Any]],
    n_clients: int,
    min_pos_per_client: int = 1,
    seed: int = 42,
) -> List[List[Dict[str, Any]]]:
    """
    Assign patients to FL clients using stratified sampling so every client
    has at least min_pos_per_client depressed samples.
    Returns a list of n_clients sub-lists.
    """
    rng = np.random.default_rng(seed)
    pos = [p for p in patients if p["label"] == 1]
    neg = [p for p in patients if p["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    clients: List[List[Dict[str, Any]]] = [[] for _ in range(n_clients)]
    # Round-robin assign positives first
    for i, p in enumerate(pos):
        clients[i % n_clients].append(p)
    # Round-robin assign negatives
    for i, n in enumerate(neg):
        clients[i % n_clients].append(n)

    # Verify constraint
    for i, c in enumerate(clients):
        n_pos = sum(1 for p in c if p["label"] == 1)
        if n_pos < min_pos_per_client:
            log.warning("Client %d has only %d positive samples", i, n_pos)

    return clients


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class MultimodalDataset(Dataset):
    def __init__(self, patients: List[Dict[str, Any]],
                 tokenizer=None, max_text_len: int = 512):
        self.patients     = patients
        self.tokenizer    = tokenizer
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.patients[idx]

        if self.tokenizer is not None:
            enc = self.tokenizer(
                p["full_text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
        else:
            input_ids      = torch.zeros(self.max_text_len, dtype=torch.long)
            attention_mask = torch.zeros(self.max_text_len, dtype=torch.long)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "audio_vec":      torch.tensor(p["audio_vec"],  dtype=torch.float32),
            "visual_vec":     torch.tensor(p["visual_vec"], dtype=torch.float32),
            "label":          torch.tensor(p["label"],      dtype=torch.long),
            "patient_id":     p["patient_id"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class SmallMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 hidden: Optional[int] = None, dropout: float = 0.2):
        super().__init__()
        hid = hidden or max(64, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.BatchNorm1d(hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalFusionModel(nn.Module):
    """
    Text (MentalBERT [CLS]) + Audio MLP + Visual MLP → concat → classifier.
    """
    def __init__(self, cfg: TrainingConfig, freeze_bert: bool = False):
        super().__init__()
        self.cfg = cfg

        if HAS_TRANSFORMERS and Path(cfg.mentalbert_path).exists():
            self.text_encoder = AutoModel.from_pretrained(cfg.mentalbert_path)
            if freeze_bert:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
            log.info("Loaded MentalBERT (frozen=%s)", freeze_bert)
        else:
            self.text_encoder = None
            # Simple linear projection as fallback
            self.text_proj    = nn.Linear(cfg.text_dim, cfg.text_dim)
            log.warning("No MentalBERT — using random text projection")

        self.audio_encoder  = SmallMLP(cfg.audio_dim,  128, dropout=cfg.dropout)
        self.visual_encoder = SmallMLP(cfg.visual_dim, 128, dropout=cfg.dropout)

        fusion_in = cfg.text_dim + 128 + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, cfg.fusion_hidden),
            nn.BatchNorm1d(cfg.fusion_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden, cfg.fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden // 2, cfg.n_classes),
        )

        log.info("Model total params: %s",
                 f"{sum(p.numel() for p in self.parameters()):,}")

    def forward(self, input_ids, attention_mask,
                audio_vec, visual_vec) -> torch.Tensor:
        if self.text_encoder is not None:
            out       = self.text_encoder(input_ids=input_ids,
                                          attention_mask=attention_mask)
            text_feat = out.last_hidden_state[:, 0, :]  # [CLS]
        else:
            text_feat = self.text_proj(
                torch.randn(input_ids.size(0), self.cfg.text_dim,
                            device=input_ids.device)
            )

        audio_feat  = self.audio_encoder(audio_vec)
        visual_feat = self.visual_encoder(visual_vec)
        fused       = torch.cat([text_feat, audio_feat, visual_feat], dim=1)
        return self.fusion(fused)


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS — FIX-BUG-8 & BUG-9: single definition, weight properly applied
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.
      alpha: scalar weight for the minority (positive) class.
             Set > 0.5 to penalise false negatives more.
      gamma: focusing parameter. 0 = standard CE, 2 = standard Focal.
    When class_weights tensor is supplied, it is applied *before* focal
    modulation so both mechanisms compound.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha         = alpha
        self.gamma         = gamma
        self.class_weights = class_weights  # shape (n_classes,)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-sample CE (with optional class weights for imbalance)
        ce = F.cross_entropy(logits, targets,
                             weight=self.class_weights,
                             reduction="none")
        pt            = torch.exp(-ce)                           # confidence
        alpha_t       = torch.where(targets == 1, self.alpha,
                                    1.0 - self.alpha)
        focal_weights = alpha_t * (1.0 - pt) ** self.gamma
        return (focal_weights * ce).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS-IMBALANCE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(labels: np.ndarray) -> Optional[torch.Tensor]:
    """Inverse-frequency class weights; returns None if only one class present."""
    unique = np.unique(labels)
    if len(unique) < 2:
        return None
    counts  = np.bincount(labels, minlength=2)
    weights = 1.0 / (counts.astype(float) + 1e-6)
    weights = weights / weights.sum() * len(unique)
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that balances classes in every batch."""
    class_weights  = compute_class_weights(labels)
    if class_weights is None:
        sample_w = torch.ones(len(labels))
    else:
        sample_w = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_w, num_samples=len(sample_w), replacement=True
    )


def build_balanced_subset(dataset: Dataset, labels: np.ndarray,
                          seed: int = 42) -> Subset:
    """
    Return a Subset where positives are oversampled to match negatives.
    Used for Stage-1 training so the model first learns balanced features.
    """
    rng     = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n       = max(len(pos_idx), len(neg_idx))
    bal_pos = rng.choice(pos_idx, size=n, replace=True)
    bal_neg = rng.choice(neg_idx, size=n, replace=True)
    idx     = np.concatenate([bal_pos, bal_neg])
    rng.shuffle(idx)
    return Subset(dataset, idx.tolist())


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION  — FIX-BUG-11 & BUG-12: proper top-level functions
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: Optional[nn.Module] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Full evaluation returning all class-imbalance-aware metrics."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            aud   = batch["audio_vec"].to(device)
            vis   = batch["visual_vec"].to(device)
            lbls  = batch["label"].to(device)

            logits = model(ids, mask, aud, vis)

            if criterion is not None:
                total_loss += criterion(logits, lbls).item() * lbls.size(0)

            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_preds.extend(preds)
            all_labels.extend(lbls.cpu().numpy())
            all_probs.extend(probs)

    n           = max(len(all_labels), 1)
    all_labels  = np.array(all_labels, dtype=int)
    all_preds   = np.array(all_preds,  dtype=int)
    all_probs   = np.array(all_probs,  dtype=float)
    n_classes   = len(np.unique(all_labels))

    metrics: Dict[str, float] = {
        "loss":      total_loss / n,
        "accuracy":  float(accuracy_score(all_labels, all_preds)),
        "f1":        float(f1_score(all_labels, all_preds,
                                    average="binary", zero_division=0)),
        "precision": float(precision_recall_fscore_support(
                           all_labels, all_preds,
                           average="binary", zero_division=0)[0]),
        "recall":    float(precision_recall_fscore_support(
                           all_labels, all_preds,
                           average="binary", zero_division=0)[1]),
    }

    if n_classes > 1:
        metrics["roc_auc"] = float(roc_auc_score(all_labels, all_probs))
        metrics["auc_pr"]  = float(average_precision_score(all_labels, all_probs))
        metrics["mcc"]     = float(matthews_corrcoef(all_labels, all_preds))
    else:
        metrics["roc_auc"] = 0.0
        metrics["auc_pr"]  = 0.0
        metrics["mcc"]     = 0.0

    return metrics


def find_optimal_threshold(
    model: nn.Module, loader: DataLoader, device: str
) -> float:
    """
    Search the F1-optimal decision threshold on a held-out set (dev).
    Returns the threshold in [0, 1] that maximises binary F1 for class 1.
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["audio_vec"].to(device),
                batch["visual_vec"].to(device),
            )
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch["label"].numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels, dtype=int)

    if len(np.unique(all_labels)) < 2:
        return 0.5  # can't optimise with one class

    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = (2 * precisions[:-1] * recalls[:-1]
                 / (precisions[:-1] + recalls[:-1] + 1e-8))
    best_thresh = float(thresholds[np.argmax(f1_scores)])
    log.info("Optimal threshold (F1 on dev): %.4f  (F1=%.4f)",
             best_thresh, f1_scores.max())
    return best_thresh


def log_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    log.info(
        "%sAcc=%.4f  F1=%.4f  Prec=%.4f  Rec=%.4f  "
        "ROC-AUC=%.4f  AUC-PR=%.4f  MCC=%.4f  Loss=%.4f",
        prefix,
        metrics["accuracy"], metrics["f1"],
        metrics["precision"], metrics["recall"],
        metrics["roc_auc"], metrics["auc_pr"],
        metrics["mcc"], metrics["loss"],
    )

def _load_all_patients(data_dir: str, labels_dir: str) -> List[Dict[str, Any]]:
    """Load ALL patients from data directory, ignoring split CSV files."""
    labels_path = Path(labels_dir) / "metadata_mapped.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    df = pd.read_csv(labels_path)
    df.columns = df.columns.str.strip()
    df["Participant_ID"] = df["Participant_ID"].astype(str).str.strip()
    df["PHQ_Binary"] = df["PHQ_Binary"].astype(int)
    
    data_path = Path(data_dir)
    patients = []
    
    for _, row in df.iterrows():
        pid = str(row["Participant_ID"])
        dirs = list(data_path.glob(f"{pid}_P"))
        if not dirs:
            continue
        p = load_patient_features(dirs[0], int(row["PHQ_Binary"]))
        if p is not None:
            patients.append(p)
    
    log.info("Loaded %d total patients from data directory", len(patients))
    return patients

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING — all BUG-6,7,10,14,15,16,17 fixed here
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(cfg: TrainingConfig) -> Dict[str, float]:
    set_seed(cfg.seed)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    log.info("Loading datasets...")
    if cfg.test_frac > 0.0:
        log.info("Using random split (test_frac=%.2f) instead of provided splits", cfg.test_frac)
        # Load ALL patients from data directory
        all_patients = _load_all_patients(cfg.data_dir, cfg.labels_dir)
        
        if len(all_patients) < 10:
            log.error("Too few total patients for random split: %d", len(all_patients))
            sys.exit(1)
        
        # Stratified random split by label
        rng = np.random.default_rng(cfg.seed)
        labels = np.array([p["label"] for p in all_patients])
        
        n_total = len(all_patients)
        n_test = max(1, int(n_total * cfg.test_frac))
        n_dev = max(1, int(n_total * 0.1))  # Fixed 10% for dev
        n_train = n_total - n_test - n_dev
        
        # Stratified sampling
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        
        # Allocate test set (stratified)
        n_test_pos = max(1, min(len(pos_idx), int(len(pos_idx) * cfg.test_frac)))
        n_test_neg = max(1, n_test - n_test_pos)
        test_idx = set(list(pos_idx[:n_test_pos]) + list(neg_idx[:n_test_neg]))
        
        # Allocate dev set from remaining (stratified)
        remaining_pos = [i for i in pos_idx if i not in test_idx]
        remaining_neg = [i for i in neg_idx if i not in test_idx]
        n_dev_pos = max(1, min(len(remaining_pos), int(len(remaining_pos) * 0.1)))
        n_dev_neg = max(1, n_dev - n_dev_pos)
        dev_idx = set(list(remaining_pos[:n_dev_pos]) + list(remaining_neg[:n_dev_neg]))
        
        # Train is everything else
        train_idx = [i for i in range(n_total) if i not in test_idx and i not in dev_idx]
        
        train_patients = [all_patients[i] for i in train_idx]
        dev_patients = [all_patients[i] for i in dev_idx]
        test_patients = [all_patients[i] for i in test_idx]
        
        log.info("Random split: train=%d, dev=%d, test=%d", 
                 len(train_patients), len(dev_patients), len(test_patients))

    else:
        train_patients = load_split(cfg.data_dir, cfg.labels_dir, "train")
        dev_patients   = load_split(cfg.data_dir, cfg.labels_dir, "dev")
        test_patients  = load_split(cfg.data_dir, cfg.labels_dir, "test")

        if len(train_patients) < 10:
            log.error("Too few training samples: %d", len(train_patients))
            sys.exit(1)

    # ── 2. Compute train_labels FIRST (FIX-BUG-7, 17) ─────────────────────────
    train_labels = np.array([p["label"] for p in train_patients], dtype=int)
    n_pos        = int(train_labels.sum())
    n_neg        = int((1 - train_labels).sum())
    pos_ratio    = n_pos / max(len(train_labels), 1)   # FIX-BUG-6: defined here
    log.info("Train class distribution: %d positive (%.1f%%), %d negative",
             n_pos, pos_ratio * 100, n_neg)
    if pos_ratio < 0.1:
        log.warning("Severe class imbalance (pos_ratio=%.2f) — "
                    "focal loss + weighted sampler strongly recommended", pos_ratio)

    # ── 3. Detect feature dims (FIX-BUG-16: before any model or stage1 code) ──
    cfg.audio_dim  = int(max(p["audio_vec"].shape[0]  for p in train_patients))
    cfg.visual_dim = int(max(p["visual_vec"].shape[0] for p in train_patients))
    log.info("Feature dims — audio:%d  visual:%d  text:%d",
             cfg.audio_dim, cfg.visual_dim, cfg.text_dim)

    # ── 4. Pad all vectors to uniform dim ──────────────────────────────────────
    def _pad(vec: np.ndarray, dim: int) -> np.ndarray:
        if vec.shape[0] >= dim:
            return vec[:dim]
        return np.pad(vec, (0, dim - vec.shape[0]))

    for p in train_patients + dev_patients + test_patients:
        p["audio_vec"]  = _pad(p["audio_vec"],  cfg.audio_dim)
        p["visual_vec"] = _pad(p["visual_vec"], cfg.visual_dim)

    # ── 5. Tokenizer ───────────────────────────────────────────────────────────
    tokenizer = None
    if HAS_TRANSFORMERS and Path(cfg.mentalbert_path).exists():
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg.mentalbert_path)
            log.info("Tokenizer loaded")
        except Exception as e:
            log.warning("Tokenizer load failed: %s", e)

    # ── 6. Datasets ────────────────────────────────────────────────────────────
    train_ds = MultimodalDataset(train_patients, tokenizer, cfg.max_text_len)
    dev_ds   = MultimodalDataset(dev_patients,   tokenizer, cfg.max_text_len) if dev_patients  else None
    test_ds  = MultimodalDataset(test_patients,  tokenizer, cfg.max_text_len) if test_patients else None

    # ── 7. Loaders (FIX-BUG-17: sampler built with train_labels, already defined) ─
    if cfg.use_weighted_sampler:
        sampler    = build_weighted_sampler(train_labels)
        do_shuffle = False
    else:
        sampler    = None
        do_shuffle = True

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=do_shuffle, sampler=sampler,
                              num_workers=cfg.num_workers, pin_memory=True)
    dev_loader   = (DataLoader(dev_ds,  batch_size=cfg.batch_size * 2,
                               shuffle=False, num_workers=cfg.num_workers)
                    if dev_ds else None)
    test_loader  = (DataLoader(test_ds, batch_size=cfg.batch_size * 2,
                               shuffle=False, num_workers=cfg.num_workers)
                    if test_ds else None)

    # ── 8. Loss (FIX-BUG-14: defined once; FIX-BUG-9: weight used in forward) ─
    cw = compute_class_weights(train_labels)
    if cfg.use_focal_loss:
        criterion = FocalLoss(
            alpha=cfg.focal_alpha,
            gamma=cfg.focal_gamma,
            class_weights=cw.to(cfg.device) if cw is not None else None,
        )
        log.info("Using FocalLoss(alpha=%.2f, gamma=%.2f)",
                 cfg.focal_alpha, cfg.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=cw.to(cfg.device) if cw is not None else None
        )
        log.info("Using weighted CrossEntropyLoss")

    # ── 9. Model ───────────────────────────────────────────────────────────────
    model = MultimodalFusionModel(cfg, freeze_bert=False).to(cfg.device)

    # ── 10. Stage-1: balanced subset (FIX-BUG-6,10,15: pos_ratio defined above) ─
    if cfg.two_stage_training and pos_ratio < 0.4 and len(train_patients) >= 20:
        log.info("Stage 1: training on balanced subset (%d epochs)", cfg.stage1_epochs)
        bal_ds     = build_balanced_subset(train_ds, train_labels, cfg.seed)
        bal_loader = DataLoader(bal_ds, batch_size=cfg.batch_size,
                                shuffle=True, num_workers=cfg.num_workers)
        s1_opt = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr * cfg.stage1_lr_multiplier,
            weight_decay=cfg.weight_decay,
        )
        model.train()
        for ep in range(cfg.stage1_epochs):
            ep_loss = 0.0
            for batch in bal_loader:
                s1_opt.zero_grad()
                logits = model(
                    batch["input_ids"].to(cfg.device),
                    batch["attention_mask"].to(cfg.device),
                    batch["audio_vec"].to(cfg.device),
                    batch["visual_vec"].to(cfg.device),
                )
                loss = criterion(logits, batch["label"].to(cfg.device))
                loss.backward()                                # FIX-BUG-10
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                s1_opt.step()
                ep_loss += loss.item()
            log.info("Stage1 epoch %d/%d  loss=%.4f",
                     ep + 1, cfg.stage1_epochs,
                     ep_loss / max(len(bal_loader), 1))
        log.info("Stage 1 done — switching to full data")

    # ── 11. Stage-2 optimizer (separate LR for BERT) ──────────────────────────
    bert_params  = [p for n, p in model.named_parameters()
                    if "text_encoder" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if "text_encoder" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": bert_params,  "lr": cfg.lr * 0.1},
         {"params": other_params, "lr": cfg.lr}],
        weight_decay=cfg.weight_decay,
    )

    total_steps  = len(train_loader) * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── 12. Training loop ──────────────────────────────────────────────────────
    best_metric      = -1.0
    patience_counter = 0
    best_state       = None
    best_threshold   = 0.5

    log.info("Stage 2: main training (%d epochs, batch=%d)", cfg.epochs, cfg.batch_size)

    for epoch in range(cfg.epochs):
        model.train()
        ep_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False)
        for batch in bar:
            optimizer.zero_grad()
            logits = model(
                batch["input_ids"].to(cfg.device),
                batch["attention_mask"].to(cfg.device),
                batch["audio_vec"].to(cfg.device),
                batch["visual_vec"].to(cfg.device),
            )
            loss = criterion(logits, batch["label"].to(cfg.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item()
            bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = ep_loss / max(len(train_loader), 1)
        log.info("Epoch %d/%d  train_loss=%.4f", epoch + 1, cfg.epochs, avg_loss)

        # ── Validation ─────────────────────────────────────────────────────────
        if dev_loader is not None:
            # Optionally refresh threshold every 5 epochs
            if cfg.find_best_threshold and (epoch % 5 == 0 or epoch == 0):
                best_threshold = find_optimal_threshold(model, dev_loader, cfg.device)

            dev_metrics = evaluate(model, dev_loader, cfg.device,
                                   criterion=criterion,
                                   threshold=best_threshold)
            log_metrics(dev_metrics, prefix=f"  Dev  epoch{epoch+1} ")

            current = dev_metrics.get(cfg.early_stop_metric, dev_metrics["f1"])
            higher_is_better = (cfg.early_stop_metric != "loss")
            is_better = (current > best_metric + cfg.min_delta if higher_is_better
                         else current < best_metric - cfg.min_delta)

            if epoch == 0 or is_better:
                best_metric      = current
                patience_counter = 0
                best_state       = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}
                log.info("  ✓ New best %s=%.4f — model saved",
                         cfg.early_stop_metric, best_metric)
            else:
                patience_counter += 1
                log.info("  Patience %d/%d", patience_counter, cfg.patience)
                if patience_counter >= cfg.patience:
                    log.info("Early stopping at epoch %d", epoch + 1)
                    break

    # ── 13. Final evaluation on test set ──────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("Restored best model for test evaluation")

    if test_loader is not None and len(test_patients) > 0:
        # Final threshold search on dev before test
        if dev_loader is not None and cfg.find_best_threshold:
            best_threshold = find_optimal_threshold(model, dev_loader, cfg.device)

        test_metrics = evaluate(model, test_loader, cfg.device,
                                threshold=best_threshold)
        log_metrics(test_metrics, prefix="TEST ")

        # ── Print classification report (FIXED: proper parentheses) ─────────
        y_true = np.array([p["label"] for p in test_patients])
        y_pred = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                logits = model(
                    batch["input_ids"].to(cfg.device),
                    batch["attention_mask"].to(cfg.device),
                    batch["audio_vec"].to(cfg.device),
                    batch["visual_vec"].to(cfg.device),
                )
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= best_threshold).astype(int)
                y_pred.extend(preds)
        
        print("\n" + classification_report(
            y_true,
            np.array(y_pred),
            target_names=["not_depressed", "depressed"],
            zero_division=0,
        ))
    else:
        log.warning("Test set is empty — skipping test evaluation")
        test_metrics = {k: 0.0 for k in
                        ["loss","accuracy","f1","precision","recall",
                        "roc_auc","auc_pr","mcc"]}

    # ── 14. Save model ─────────────────────────────────────────────────────────
    save_dict = {
        "state_dict":           {k: v.cpu() for k, v in model.state_dict().items()},
        "config":               asdict(cfg),
        "test_metrics":         test_metrics,
        "best_threshold":       best_threshold,
        "training_completed":   True,
    }
    torch.save(save_dict, cfg.output_path)
    log.info("Model saved: %s", cfg.output_path)

    metrics_path = Path(cfg.output_path).parent / cfg.metrics_file
    with open(metrics_path, "w") as f:
        json.dump({
            "config":         asdict(cfg),
            "test_metrics":   test_metrics,
            "best_threshold": best_threshold,
            "train_samples":  len(train_patients),
            "dev_samples":    len(dev_patients),
            "test_samples":   len(test_patients),
            "pos_ratio_train": float(pos_ratio),
        }, f, indent=2, default=str)
    log.info("Metrics saved: %s", metrics_path)

    return test_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Centralized base model training for DAIC-WOZ depression detection"
    )
    parser.add_argument("--data_dir",         default="./data")
    parser.add_argument("--labels_dir",       default="./labels")
    parser.add_argument("--output",           default="~/.federated/data/global_models/global_round1.pt")
    parser.add_argument("--mentalbert_path",  default=str(Path.home() / ".federated" / "models" / "mentalbert"))
    parser.add_argument("--epochs",           type=int,   default=50)
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--lr",               type=float, default=2e-5)
    parser.add_argument("--weight_decay",     type=float, default=1e-4)
    parser.add_argument("--max_grad_norm",    type=float, default=1.0)
    parser.add_argument("--warmup_ratio",     type=float, default=0.1)
    parser.add_argument("--dropout",          type=float, default=0.3)
    parser.add_argument("--patience",         type=int,   default=10)
    parser.add_argument("--min_delta",        type=float, default=0.001)
    parser.add_argument("--early_stop_metric",            default="auc_pr",
                        choices=["auc_pr", "f1", "accuracy", "loss", "roc_auc"])
    parser.add_argument("--focal_alpha",      type=float, default=0.75)
    parser.add_argument("--focal_gamma",      type=float, default=2.0)
    parser.add_argument("--no_focal",         action="store_true")
    parser.add_argument("--no_weighted_sampler", action="store_true")
    parser.add_argument("--no_two_stage",     action="store_true")
    parser.add_argument("--stage1_epochs",    type=int,   default=10)
    parser.add_argument("--test_frac", type=float, default=0.0, 
                   help="If >0, use random split instead of provided splits (e.g., 0.2 = 20% test)")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--device",           default=None)
    parser.add_argument("--num_workers",      type=int,   default=4)
    
    args = parser.parse_args()

    cfg = TrainingConfig(
        data_dir              = args.data_dir,
        labels_dir            = args.labels_dir,
        output_path           = args.output,
        mentalbert_path       = args.mentalbert_path,
        epochs                = args.epochs,
        batch_size            = args.batch_size,
        lr                    = args.lr,
        weight_decay          = args.weight_decay,
        max_grad_norm         = args.max_grad_norm,
        warmup_ratio          = args.warmup_ratio,
        dropout               = args.dropout,
        patience              = args.patience,
        min_delta             = args.min_delta,
        early_stop_metric     = args.early_stop_metric,
        focal_alpha           = args.focal_alpha,
        focal_gamma           = args.focal_gamma,
        use_focal_loss        = not args.no_focal,
        use_weighted_sampler  = not args.no_weighted_sampler,
        two_stage_training    = not args.no_two_stage,
        stage1_epochs         = args.stage1_epochs,
        test_frac=args.test_frac,
        seed                  = args.seed,
        device                = args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        num_workers           = args.num_workers,
    )

    log.info("=== Centralized Base Model Training ===")
    if not HAS_TRANSFORMERS:
        log.warning("transformers not installed — text features will be random")
    if not Path(cfg.mentalbert_path).exists():
        log.warning("MentalBERT not found at %s — using random init", cfg.mentalbert_path)

    metrics = train_model(cfg)  # ← This line was missing newline before next statement

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Test F1      : {metrics['f1']:.4f}")
    print(f"  Test AUC-PR  : {metrics['auc_pr']:.4f}")
    print(f"  Test ROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"  Test MCC     : {metrics['mcc']:.4f}")
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Model saved  : {cfg.output_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())