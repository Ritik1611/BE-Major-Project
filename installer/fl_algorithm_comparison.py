"""
fl_daic_comparison.py
Federated Learning Algorithm Comparison on DAIC-WOZ Depression Dataset

Data structure expected:
  data/
    {ID}_P/
      features/
        {ID}_OpenSMILE*.csv      — eGeMAPS / ComPare audio features
        {ID}_OpenFace2.csv       — Action Units, gaze, pose
        {ID}_BoAW_openSMILE.csv  — Bag of Audio Words
        {ID}_BoVW_openpose.csv   — Bag of Visual Words
        {ID}_CNN_*.csv           — CNN visual features
        {ID}_Transcript.csv      — Turn-by-turn conversation
      {ID}_AUDIO.wav
    labels.csv  (Participant_ID, PHQ8_Binary, PHQ8_Score)  ← optional but recommended

Base model: MentalBERT (from ~/.federated/models/mentalbert) for text.
            Pre-extracted audio/visual features used directly.

FL algorithms compared:
  - FedAvg  (McMahan et al. 2017)
  - FedProx (Li et al. 2020)
  - FedAdam (Reddi et al. 2021)

Aggregation strategies per algorithm:
  - Federated Averaging (mean)
  - Trimmed Mean (Byzantine-robust)
  - Coordinate-wise Median
  - Krum (when n_clients >= 5)

DP support: Gaussian mechanism via RDP accountant.

Usage:
  python fl_daic_comparison.py --data_dir ./data --rounds 30
  python fl_daic_comparison.py --data_dir ./data --rounds 30 --use_mentalbert
  python fl_daic_comparison.py --data_dir ./data --rounds 30 --no_dp
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
from typing import Dict, List, Optional, Tuple, Any

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
    from sklearn.metrics import f1_score, roc_auc_score, classification_report
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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING — DAIC-WOZ FEATURE FILES
# ══════════════════════════════════════════════════════════════════════════════

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
    """Drop identifier columns, compute mean + std over rows → flat vector."""
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
    # Prefer AU / gaze / pose columns
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
    # Drop first col if it looks like a time index
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

    # Locate text column
    text_col = next(
        (c for c in ("value", "text", "content", "utterance", "transcription")
         if c in df.columns),
        df.columns[3] if df.shape[1] >= 4 else df.columns[-1],
    )

    # Locate speaker column
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
        feat_dir = patient_dir  # flat layout

    # ── Audio ─────────────────────────────────────────────────────────────
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

    # ── Visual ────────────────────────────────────────────────────────────
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

    # ── Transcript ────────────────────────────────────────────────────────
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
        log.warning("Patient %s: no features found — skipping", pid)
        return None

    # Defaults when a modality is missing
    if audio_vec is None:
        audio_vec = np.zeros(176, dtype=np.float32)   # 88 eGeMAPS × 2 (mean+std)
    if visual_vec is None:
        visual_vec = np.zeros(70, dtype=np.float32)   # 35 AU × 2

    if not utterances:
        utterances = [f"patient {pid}"]

    log.info("Patient %s | audio=%d visual=%d utterances=%d label=%d",
             pid, audio_vec.shape[0], visual_vec.shape[0],
             len(utterances), label)

    return {
        "patient_id": pid,
        "audio": audio_vec.astype(np.float32),
        "visual": visual_vec.astype(np.float32),
        "utterances": utterances,
        "label": label,
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
                # Derive from score
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
    for i, pdir in enumerate(patient_dirs):
        pid = pdir.name.split("_")[0]
        label = labels_dict.get(pid, i % 2)
        if pid not in labels_dict:
            log.warning("No label for %s — using demo label %d (balanced)", pid, label)
        p = load_patient(pdir, label)
        if p is not None:
            patients.append(p)

    pos = sum(p["label"] for p in patients)
    log.info("Dataset: %d patients | %d depressed | %d not depressed",
             len(patients), pos, len(patients) - pos)
    return patients


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MENTALBERT TEXT EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

class MentalBERTEmbedder:
    """Embed text via MentalBERT (falls back to bert-base-uncased or random)."""

    DIM = 768

    def __init__(self, use_mentalbert: bool = True):
        self.model = None
        self.tokenizer = None

        if not HAS_TRANSFORMERS or not use_mentalbert:
            if use_mentalbert:
                log.warning("transformers not installed — using random text features")
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
            log.warning("Could not load any BERT model — random text features")

    @torch.no_grad()
    def embed(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """(N, 768) mean-pooled BERT embeddings."""
        if not texts:
            return np.zeros((1, self.DIM), dtype=np.float32)

        if self.model is None:
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
        """Mean of utterance embeddings → (768,)."""
        return np.mean(self.embed(utterances), axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DATASET CLASS
# ══════════════════════════════════════════════════════════════════════════════

class PatientDataset(Dataset):
    """
    Each sample = one utterance from the patient's session.
    Audio and visual features are session-level (tiled across utterances).
    All utterances from one patient share the same PHQ label.
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
            "audio": self.audio[idx],
            "visual": self.visual[idx],
            "text": self.text[idx],
            "label": self.labels[idx],
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: MODEL
# ══════════════════════════════════════════════════════════════════════════════

class DepressionNet(nn.Module):
    """
    Multimodal fusion network for PHQ-8 binary depression classification.

    Branches:
      audio  → small MLP → 64-d
      visual → small MLP → 64-d
      text   → small MLP → 64-d
    Fusion → 192-d → 128-d → 2 (logits)
    """

    def __init__(self, audio_dim: int, visual_dim: int, text_dim: int,
                 hidden: int = 64, n_classes: int = 2, dropout: float = 0.25):
        super().__init__()

        def branch(in_d):
            return nn.Sequential(
                nn.Linear(in_d, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )

        self.audio_branch = branch(audio_dim) if audio_dim > 0 else None
        self.visual_branch = branch(visual_dim) if visual_dim > 0 else None
        self.text_branch = branch(text_dim) if text_dim > 0 else None

        n_branches = sum(1 for b in [self.audio_branch,
                                      self.visual_branch,
                                      self.text_branch] if b)
        fusion_in = n_branches * hidden

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, n_classes),
        )

    def forward(self, audio, visual, text):
        parts = []
        if self.audio_branch is not None:
            parts.append(self.audio_branch(audio))
        if self.visual_branch is not None:
            parts.append(self.visual_branch(visual))
        if self.text_branch is not None:
            parts.append(self.text_branch(text))

        fused = torch.cat(parts, dim=1) if parts else audio  # fallback
        return self.fusion(fused)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float, float]:
    """Returns (loss, accuracy, F1)."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(DEVICE)
            visual = batch["visual"].to(DEVICE)
            text = batch["text"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(audio, visual, text)
            total_loss += F.cross_entropy(logits, labels,
                                           reduction="sum").item()
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(n, 1)

    if HAS_SKLEARN and len(set(all_labels)) > 1:
        f1 = f1_score(all_labels, all_preds,
                      average="binary", zero_division=0)
    else:
        f1 = float(acc)  # single class fallback

    return total_loss / max(n, 1), acc, f1


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DIFFERENTIAL PRIVACY
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: LOCAL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LocalCfg:
    lr: float = 5e-4
    local_epochs: int = 5
    batch_size: int = 8
    clip_norm: float = 1.0
    noise_mult: float = 1.1
    use_dp: bool = True
    mu: float = 0.0           # FedProx proximal coefficient


def local_train(model: nn.Module, global_model: nn.Module,
                dataset: Dataset, cfg: LocalCfg) -> Dict:
    if len(dataset) == 0:
        return {"delta": {}, "n_samples": 0, "loss": float("inf")}

    bs = min(cfg.batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    global_w = {n: p.data.clone()
                for n, p in global_model.named_parameters()}
    w0 = {k: v.clone() for k, v in model.state_dict().items()}

    total_loss, n_samp = 0.0, 0
    model.train()

    for _ in range(cfg.local_epochs):
        for batch in loader:
            audio = batch["audio"].to(DEVICE)
            visual = batch["visual"].to(DEVICE)
            text = batch["text"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            opt.zero_grad()
            logits = model(audio, visual, text)
            loss = F.cross_entropy(logits, labels)

            # FedProx proximal term
            if cfg.mu > 0:
                prox = sum(
                    ((p - global_w[n]) ** 2).sum()
                    for n, p in model.named_parameters()
                    if n in global_w
                )
                loss = loss + (cfg.mu / 2.0) * prox

            loss.backward()

            # DP-SGD: clip + noise per gradient
            if cfg.use_dp:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        g_norm = p.grad.norm(2)
                        if g_norm > cfg.clip_norm:
                            p.grad.mul_(cfg.clip_norm / (g_norm + 1e-12))
                        p.grad.add_(torch.randn_like(p.grad) * cfg.noise_mult * cfg.clip_norm)
                # Global clip once after DP noise is added
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm * 2)
            opt.step()
            total_loss += loss.item() * labels.size(0)
            n_samp += labels.size(0)

    delta = {k: (model.state_dict()[k].float() - w0[k].float()) for k in w0}
    return {
        "delta": delta,
        "n_samples": max(n_samp, 1),
        "loss": total_loss / max(n_samp, 1),
    }

def local_train_scaffold(
    model: nn.Module,
    global_model: nn.Module,
    dataset: Dataset,
    cfg: LocalCfg,
    client_c: Dict,          # per-client control variate (modified in place)
    server_c: Dict,          # global control variate from SCAFFOLDServer
) -> Dict:
    """SCAFFOLD local training with control variate correction."""
    if len(dataset) == 0:
        return {"delta": {}, "c_delta": {}, "n_samples": 0, "loss": float("inf")}

    bs = min(cfg.batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)  # SCAFFOLD uses SGD

    w0 = {k: v.clone() for k, v in model.state_dict().items()}
    total_loss, n_samp = 0.0, 0
    K = cfg.local_epochs * max(1, len(dataset) // bs)  # total local steps

    model.train()
    for _ in range(cfg.local_epochs):
        for batch in loader:
            audio = batch["audio"].to(DEVICE)
            visual = batch["visual"].to(DEVICE)
            text = batch["text"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            opt.zero_grad()
            logits = model(audio, visual, text)
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            # SCAFFOLD correction: subtract c_i, add c
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.grad is not None and n in client_c and n in server_c:
                        p.grad.add_(
                            -client_c[n].to(DEVICE) + server_c[n].to(DEVICE)
                        )

            # DP-SGD if enabled
            if cfg.use_dp:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        g_norm = p.grad.norm(2)
                        if g_norm > cfg.clip_norm:
                            p.grad.mul_(cfg.clip_norm / (g_norm + 1e-12))
                        p.grad.add_(torch.randn_like(p.grad) * cfg.noise_mult * cfg.clip_norm)
                # Global clip once after DP noise is added
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm * 2)
            opt.step()
            total_loss += loss.item() * labels.size(0)
            n_samp += labels.size(0)

    # Update client control variate
    # c_i^+ = c_i - c + (w0 - w) / (K * lr)
    new_client_c = {}
    c_delta = {}
    with torch.no_grad():
        for k, p in model.named_parameters():
            if k in client_c:
                w_diff = (w0.get(k, p.data.clone()).to(DEVICE) - p.data) / (
                    max(K, 1) * cfg.lr + 1e-12
                )
                new_c_i = client_c[k].to(DEVICE) - server_c[k].to(DEVICE) + w_diff
                c_delta[k] = (new_c_i - client_c[k].to(DEVICE)).cpu()
                client_c[k] = new_c_i.cpu()

    delta = {k: (model.state_dict()[k].float() - w0[k].float()) for k in w0}
    return {
        "delta": delta,
        "c_delta": c_delta,
        "n_samples": max(n_samp, 1),
        "loss": total_loss / max(n_samp, 1),
    }

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: AGGREGATION STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def agg_mean(updates: List[Dict], weights: List[float]) -> Dict:
    total = sum(weights)
    keys = list(updates[0].keys())
    return {
        k: sum(w * u[k].float() for u, w in zip(updates, weights)) / total
        for k in keys
    }


def agg_trimmed_mean(updates: List[Dict], ratio: float = 0.1) -> Dict:
    n = len(updates)
    k = max(1, int(ratio * n))
    if 2 * k >= n:
        k = 0

    keys = list(updates[0].keys())
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
        log.warning("Krum: too few clients (%d) for f=%d — falling back to mean",
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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: SERVER-SIDE OPTIMISER (FedAdam)
# ══════════════════════════════════════════════════════════════════════════════

class FedAdamServer:
    def __init__(self, model: nn.Module, lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999):
        self.lr = lr
        self.b1, self.b2 = beta1, beta2
        self.eps = 1e-8
        self.m = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
        self.v = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
        self.t = 0

    def step(self, model: nn.Module, delta: Dict):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n not in delta:
                    continue
                g = -delta[n].to(DEVICE)
                self.m[n] = self.b1 * self.m[n] + (1 - self.b1) * g
                self.v[n] = self.b2 * self.v[n] + (1 - self.b2) * g * g
                m_hat = self.m[n] / c1
                v_hat = self.v[n] / c2
                p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

class FedYogiServer:
    """FedYogi: Adaptive FL with Yogi second-moment update."""
    def __init__(self, model: nn.Module, lr: float = 1e-2,
                 beta1: float = 0.9, beta2: float = 0.999,
                 tau: float = 1e-3):
        self.lr = lr
        self.b1, self.b2 = beta1, beta2
        self.tau = tau
        self.m = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
        self.v = {k: torch.ones_like(v) * (tau ** 2) for k, v in model.named_parameters()}
        self.t = 0

    def step(self, model: nn.Module, delta: Dict):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n not in delta:
                    continue
                g = -delta[n].to(DEVICE)
                self.m[n] = self.b1 * self.m[n] + (1 - self.b1) * g
                # Yogi: sign-based second moment to prevent overshooting
                self.v[n] = self.v[n] + (1 - self.b2) * (g * g - self.v[n]).sign() * (g * g)
                self.v[n] = torch.clamp(self.v[n], min=self.tau ** 2)
                m_hat = self.m[n] / c1
                p.data -= self.lr * m_hat / (self.v[n].sqrt() + self.tau)


class SCAFFOLDServer:
    """
    SCAFFOLD: Stochastic Controlled Averaging for FL.
    Karimireddy et al. 2020.
    Maintains a global control variate c to correct client drift.
    """
    def __init__(self, model: nn.Module):
        # Global control variate c (server-side)
        self.c = {k: torch.zeros_like(v)
                  for k, v in model.named_parameters()}

    def update_global_c(self, client_c_deltas: List[Dict]):
        """Aggregate client control variate updates into global c."""
        if not client_c_deltas:
            return
        n = len(client_c_deltas)
        with torch.no_grad():
            for k in self.c:
                if k in client_c_deltas[0]:
                    self.c[k] += sum(d[k].to(DEVICE) for d in client_c_deltas) / n

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: FL TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RoundResult:
    round_num: int
    algorithm: str
    aggregation: str
    test_loss: float
    test_acc: float
    test_f1: float
    epsilon: float
    use_dp: bool
    n_participated: int


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
) -> List[RoundResult]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if local_cfg is None:
        local_cfg = LocalCfg()

    global_model = model_factory().to(DEVICE)
    fed_adam = FedAdamServer(global_model, lr=server_lr) \
        if algorithm == "fedadam" else None

    results = []
    cum_eps = 0.0

    for rnd in range(n_rounds):
        updates, weights = [], []

        for cid, ds in enumerate(client_datasets):
            if len(ds) == 0:
                continue
            local_model = copy.deepcopy(global_model)
            cfg = copy.copy(local_cfg)
            cfg.use_dp = use_dp
            cfg.mu = 0.01 if algorithm == "fedprox" else 0.0

            res = local_train(local_model, global_model, ds, cfg)
            if res["n_samples"] > 0 and res["delta"]:
                updates.append(res["delta"])
                weights.append(float(res["n_samples"]))

        if not updates:
            continue

        n = len(updates)
        if aggregation == "mean":
            agg = agg_mean(updates, weights)
        elif aggregation == "trimmed_mean":
            agg = agg_trimmed_mean(updates, 0.1) if n >= 4 \
                else agg_mean(updates, weights)
        elif aggregation == "median":
            agg = agg_median(updates)
        elif aggregation == "krum":
            agg = agg_krum(updates, f=max(1, n // 5))
        else:
            raise ValueError(aggregation)

        # Apply delta to global model
        if algorithm == "fedadam" and fed_adam:
            fed_adam.step(global_model, agg)
        else:
            with torch.no_grad():
                for name, p in global_model.named_parameters():
                    if name in agg:
                        p.data += server_lr * agg[name].to(DEVICE)

        # Privacy accounting (per round)
        if use_dp:
            avg_n = np.mean([len(ds) for ds in client_datasets if len(ds) > 0])
            sr = local_cfg.batch_size / max(avg_n, 1)
            steps = local_cfg.local_epochs * max(
                1, int(avg_n / local_cfg.batch_size)
            )
            cum_eps += rdp_to_dp(local_cfg.noise_mult, sr,
                                  steps, privacy_delta)

        loss, acc, f1 = evaluate(global_model, test_loader)

        r = RoundResult(
            round_num=rnd + 1,
            algorithm=algorithm,
            aggregation=aggregation,
            test_loss=loss,
            test_acc=acc,
            test_f1=f1,
            epsilon=cum_eps if use_dp else 0.0,
            use_dp=use_dp,
            n_participated=len(updates),
        )
        results.append(r)

        if (rnd + 1) % 5 == 0 or rnd == 0:
            log.info("[%s/%s] R%3d | acc=%.3f f1=%.3f ε=%.3f",
                     algorithm, aggregation, rnd + 1,
                     acc, f1, cum_eps if use_dp else 0.0)

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
) -> List[RoundResult]:
    """SCAFFOLD federated learning loop."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if local_cfg is None:
        local_cfg = LocalCfg()

    global_model = model_factory().to(DEVICE)
    scaffold_server = SCAFFOLDServer(global_model)

    # Per-client control variates (initialised to zero)
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
            cfg = copy.copy(local_cfg)
            cfg.use_dp = use_dp

            res = local_train_scaffold(
                local_model, global_model, ds, cfg,
                client_cs[cid], scaffold_server.c
            )
            if res["n_samples"] > 0 and res["delta"]:
                updates.append(res["delta"])
                c_deltas.append(res.get("c_delta", {}))
                weights.append(float(res["n_samples"]))

        if not updates:
            continue

        # FedAvg aggregation of model deltas
        agg = agg_mean(updates, weights)
        with torch.no_grad():
            for name, p in global_model.named_parameters():
                if name in agg:
                    p.data += agg[name].to(DEVICE)

        # Update global control variate
        scaffold_server.update_global_c(c_deltas)

        if use_dp:
            avg_n = np.mean([len(ds) for ds in client_datasets if len(ds) > 0])
            sr = local_cfg.batch_size / max(avg_n, 1)
            steps = local_cfg.local_epochs * max(1, int(avg_n / local_cfg.batch_size))
            cum_eps += rdp_to_dp(local_cfg.noise_mult, sr, steps, privacy_delta)

        loss, acc, f1 = evaluate(global_model, test_loader)
        r = RoundResult(
            round_num=rnd + 1, algorithm="scaffold", aggregation="mean",
            test_loss=loss, test_acc=acc, test_f1=f1,
            epsilon=cum_eps if use_dp else 0.0,
            use_dp=use_dp, n_participated=len(updates),
        )
        results.append(r)

        if (rnd + 1) % 5 == 0 or rnd == 0:
            log.info("[scaffold/mean] R%3d | acc=%.3f f1=%.3f ε=%.3f",
                     rnd + 1, acc, f1, cum_eps if use_dp else 0.0)

    return results

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11: VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

COLORS = plt.cm.tab10.colors if HAS_MATPLOTLIB else []
DP_MARKERS = {"dp": "--", "nodp": "-"}


def plot_comparison(all_results: Dict[str, List[RoundResult]]):
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "FL Algorithm Comparison — DAIC-WOZ Depression Detection (MentalBERT)",
        fontsize=13, fontweight="bold",
    )

    ax_acc, ax_f1, ax_eps, ax_pvt = axes.flat

    for idx, (lbl, rlist) in enumerate(all_results.items()):
        rounds = [r.round_num for r in rlist]
        color = COLORS[idx % len(COLORS)]
        ls = "--" if rlist[0].use_dp else "-"

        ax_acc.plot(rounds, [r.test_acc for r in rlist],
                    color=color, ls=ls, lw=2, label=lbl)
        ax_f1.plot(rounds, [r.test_f1 for r in rlist],
                   color=color, ls=ls, lw=2, label=lbl)
        if rlist[0].use_dp:
            ax_eps.plot(rounds, [r.epsilon for r in rlist],
                        color=color, ls=ls, lw=2, label=lbl)
            # Privacy-utility: ε vs final acc
            ax_pvt.scatter(rlist[-1].epsilon, rlist[-1].test_acc,
                           color=color, s=100, zorder=5, label=lbl)

    for ax, title, ylabel in [
        (ax_acc, "Test Accuracy", "Accuracy"),
        (ax_f1,  "F1 Score (Depression=1)", "F1"),
        (ax_eps, "Privacy Budget ε (DP runs only)", "ε"),
        (ax_pvt, "Privacy–Utility Trade-off (DP runs)", "Final Accuracy"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Round" if ax is not ax_pvt else "Final ε")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "fl_daic_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Plot saved → %s", out)


def latex_summary(all_results: Dict[str, List[RoundResult]]) -> str:
    # Split into algorithm comparison and parameter sweep
    sweep_keys = [k for k in all_results if any(
        tag in k for tag in ["_nm", "_cn", "_le"]
    )]
    algo_keys  = [k for k in all_results if k not in sweep_keys]

    def _table_rows(keys):
        rows = []
        for lbl in sorted(keys):
            rlist = all_results[lbl]
            final = rlist[-1]
            eps_str = f"{final.epsilon:.2f}" if final.use_dp else "—"
            rows.append(
                f"{final.algorithm.upper()} & {final.aggregation.replace('_',' ')} & "
                f"{'✓' if final.use_dp else '✗'} & "
                f"{final.test_acc:.4f} & {final.test_f1:.4f} & {eps_str} & "
                f"{final.round_num} \\\\"
            )
        return rows

    # Algorithm comparison table
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{FL Algorithm Comparison on DAIC-WOZ (MentalBERT)}",
        r"\begin{tabular}{lllcccc}",
        r"\toprule",
        r"Algorithm & Aggregation & DP & Final Acc & F1 & $\varepsilon$ & Rounds \\",
        r"\midrule",
    ] + _table_rows(algo_keys) + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    # Parameter sweep table
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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12: MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FL on DAIC-WOZ with MentalBERT"
    )
    parser.add_argument("--data_dir", default="./data",
                        help="Path to DAIC-WOZ data/ directory")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--use_mentalbert", action="store_true",
                        help="Use MentalBERT for transcript embeddings")
    parser.add_argument("--no_dp", action="store_true",
                        help="Disable differential privacy")
    parser.add_argument("--noise_mult", type=float, default=1.1)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--test_frac", type=float, default=0.2,
                        help="Fraction of patients used as held-out test")
    parser.add_argument("--seed", type=int, default=42)
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
        p["text_vec"] = embedder.session_embedding(p["utterances"])
        # Also pre-compute per-utterance embeddings for dataset construction
        p["text_utts"] = embedder.embed(p["utterances"])

    TEXT_DIM = embedder.DIM

    # ── 3. Pad feature vectors to uniform dim across patients ─────────────
    audio_dim  = max(p["audio"].shape[0] for p in patients)
    visual_dim = max(p["visual"].shape[0] for p in patients)

    def pad(arr, d):
        return (arr[:d] if arr.shape[0] >= d
                else np.pad(arr, (0, d - arr.shape[0])))

    for p in patients:
        p["audio"]  = pad(p["audio"],  audio_dim)
        p["visual"] = pad(p["visual"], visual_dim)

    log.info("Dims → audio=%d  visual=%d  text=%d",
             audio_dim, visual_dim, TEXT_DIM)

    # ── 4. Stratified train / test split at patient level ─────────────────
    rng = np.random.default_rng(args.seed)
    depressed     = [p for p in patients if p["label"] == 1]
    not_depressed = [p for p in patients if p["label"] == 0]
    rng.shuffle(depressed)
    rng.shuffle(not_depressed)

    n_test = max(1, int(len(patients) * args.test_frac))
    n_test_d  = max(1, min(len(depressed) - 1,     n_test // 2))
    n_test_nd = max(1, min(len(not_depressed) - 1, n_test - n_test_d))

    test_pts  = depressed[:n_test_d] + not_depressed[:n_test_nd]
    train_pts = depressed[n_test_d:] + not_depressed[n_test_nd:]

    if len(train_pts) == 0:
        train_pts, test_pts = patients[:-2], patients[-2:]

    log.info("Train clients=%d  Test patients=%d",
             len(train_pts), len(test_pts))

    # ── 5. Build per-client datasets (train) ──────────────────────────────
    client_datasets: List[Dataset] = []
    for p in train_pts:
        ds = PatientDataset(
            audio=p["audio"],
            visual=p["visual"],
            text=p["text_utts"],
            label=p["label"],
        )
        client_datasets.append(ds)

    # ── 6. Build test DataLoader ───────────────────────────────────────────
    class _TestDS(Dataset):
        def __init__(self, pts):
            self.audio  = torch.tensor(
                np.stack([p["audio"] for p in pts]),  dtype=torch.float32)
            self.visual = torch.tensor(
                np.stack([p["visual"] for p in pts]), dtype=torch.float32)
            self.text   = torch.tensor(
                np.stack([p["text_vec"] for p in pts]), dtype=torch.float32)
            self.labels = torch.tensor(
                [p["label"] for p in pts], dtype=torch.long)
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            return {"audio": self.audio[i], "visual": self.visual[i],
                    "text": self.text[i],   "label": self.labels[i]}

    test_ds = _TestDS(test_pts)
    test_loader = DataLoader(test_ds, batch_size=len(test_pts), shuffle=False)

    test_labels = [p["label"] for p in test_pts]
    log.info("Test set: %d depressed  %d not depressed",
             sum(test_labels), len(test_labels) - sum(test_labels))

    # ── 7. Model factory ───────────────────────────────────────────────────
    def model_factory():
        return DepressionNet(
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            text_dim=TEXT_DIM,
            hidden=64,
            dropout=0.2,
        )

    log.info("Model params: %d", model_factory().count_params())

    # ── 8. Local training config ───────────────────────────────────────────
    min_client_size = min(len(ds) for ds in client_datasets)
    base_cfg = LocalCfg(
        lr=5e-4,
        local_epochs=5,
        batch_size=min(8, max(1, min_client_size)),
        clip_norm=args.clip_norm,
        noise_mult=args.noise_mult,
        use_dp=not args.no_dp,
    )
    privacy_delta = 1.0 / max(
        sum(len(ds) for ds in client_datasets), 1
    )

    # ── 9. Define experiments ──────────────────────────────────────────────
    EXPERIMENTS = [
        # (algorithm, aggregation, server_lr, use_dp)
        ("fedavg",  "mean",         1.0,   False),   # baseline, no DP
        ("fedavg",  "mean",         1.0,   True),    # FedAvg + DP
        ("fedavg",  "trimmed_mean", 1.0,   True),    # Byzantine-robust trimmed mean
        ("fedavg",  "median",       1.0,   True),    # Coordinate-wise median
        ("fedprox", "mean",         1.0,   True),    # FedProx (proximal term μ=0.01)
        ("fedadam", "mean",         1e-3,  True),    # FedAdam adaptive
        ("fedyogi", "mean",         1e-2,  True),    # FedYogi adaptive
    ]
    if len(train_pts) >= 5:
        EXPERIMENTS.append(("fedavg", "krum", 1.0, True))

    all_results: Dict[str, List[RoundResult]] = {}

    # ── Standard algorithm experiments ────────────────────────────────────────
    for algo, agg, slr, dp_flag in EXPERIMENTS:
        if args.no_dp:
            dp_flag = False
        label = f"{algo}_{agg}" + ("" if dp_flag else "_noDP")
        log.info("\n▶  Running: %s", label)

        cfg = copy.copy(base_cfg)
        cfg.use_dp = dp_flag

        if algo == "fedyogi":
            # FedYogi uses its own server optimizer, reuse fedadam path
            global_model_yogi = model_factory().to(DEVICE)
            fed_yogi_server = FedYogiServer(global_model_yogi, lr=slr)

            # Inline run using existing local_train + FedYogi step
            results_yogi: List[RoundResult] = []
            cum_eps_yogi = 0.0
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            for rnd in range(args.rounds):
                updates_y, weights_y = [], []
                for ds in client_datasets:
                    if len(ds) == 0:
                        continue
                    lm = copy.deepcopy(global_model_yogi)
                    c2 = copy.copy(cfg)
                    res = local_train(lm, global_model_yogi, ds, c2)
                    if res["n_samples"] > 0 and res["delta"]:
                        updates_y.append(res["delta"])
                        weights_y.append(float(res["n_samples"]))
                if updates_y:
                    agg_delta = agg_mean(updates_y, weights_y)
                    fed_yogi_server.step(global_model_yogi, agg_delta)
                    if dp_flag:
                        avg_n = np.mean([len(ds) for ds in client_datasets if len(ds) > 0])
                        sr = cfg.batch_size / max(avg_n, 1)
                        steps = cfg.local_epochs * max(1, int(avg_n / cfg.batch_size))
                        cum_eps_yogi += rdp_to_dp(cfg.noise_mult, sr, steps, privacy_delta)
                loss_y, acc_y, f1_y = evaluate(global_model_yogi, test_loader)
                results_yogi.append(RoundResult(
                    round_num=rnd + 1, algorithm="fedyogi", aggregation="mean",
                    test_loss=loss_y, test_acc=acc_y, test_f1=f1_y,
                    epsilon=cum_eps_yogi if dp_flag else 0.0,
                    use_dp=dp_flag, n_participated=len(updates_y),
                ))
                if (rnd + 1) % 5 == 0 or rnd == 0:
                    log.info("[fedyogi/mean] R%3d | acc=%.3f f1=%.3f ε=%.3f",
                             rnd + 1, acc_y, f1_y, cum_eps_yogi if dp_flag else 0.0)
            all_results[label] = results_yogi
            continue

        rlist = run_federated(
            algorithm=algo,
            aggregation=agg,
            client_datasets=client_datasets,
            test_loader=test_loader,
            model_factory=model_factory,
            n_rounds=args.rounds,
            local_cfg=cfg,
            server_lr=slr,
            use_dp=dp_flag,
            privacy_delta=privacy_delta,
            seed=args.seed,
        )
        all_results[label] = rlist

    # ── SCAFFOLD experiment ───────────────────────────────────────────────────
    log.info("\n▶  Running: scaffold_mean")
    scaffold_cfg = copy.copy(base_cfg)
    scaffold_cfg.use_dp = not args.no_dp
    # SCAFFOLD uses SGD internally, so set a slightly higher lr
    scaffold_cfg.lr = 1e-3
    all_results["scaffold_mean"] = run_federated_scaffold(
        client_datasets=client_datasets,
        test_loader=test_loader,
        model_factory=model_factory,
        n_rounds=args.rounds,
        local_cfg=scaffold_cfg,
        use_dp=not args.no_dp,
        privacy_delta=privacy_delta,
        seed=args.seed,
    )

    # ── Parameter sweep: noise multiplier (DP sensitivity) ───────────────────
    if not args.no_dp:
        log.info("\n▶  Parameter sweep: noise_multiplier")
        for nm in [0.5, 1.0, 1.5, 2.0]:
            sweep_cfg = copy.copy(base_cfg)
            sweep_cfg.noise_mult = nm
            sweep_cfg.use_dp = True
            label_nm = f"fedavg_mean_nm{nm}"
            log.info("   noise_mult=%.1f", nm)
            all_results[label_nm] = run_federated(
                algorithm="fedavg", aggregation="mean",
                client_datasets=client_datasets, test_loader=test_loader,
                model_factory=model_factory, n_rounds=args.rounds,
                local_cfg=sweep_cfg, server_lr=1.0,
                use_dp=True, privacy_delta=privacy_delta, seed=args.seed,
            )

        # ── Parameter sweep: clip norm ────────────────────────────────────────
        log.info("\n▶  Parameter sweep: clip_norm")
        for cn in [0.5, 1.0, 2.0]:
            sweep_cfg = copy.copy(base_cfg)
            sweep_cfg.clip_norm = cn
            sweep_cfg.use_dp = True
            label_cn = f"fedavg_mean_cn{cn}"
            log.info("   clip_norm=%.1f", cn)
            all_results[label_cn] = run_federated(
                algorithm="fedavg", aggregation="mean",
                client_datasets=client_datasets, test_loader=test_loader,
                model_factory=model_factory, n_rounds=args.rounds,
                local_cfg=sweep_cfg, server_lr=1.0,
                use_dp=True, privacy_delta=privacy_delta, seed=args.seed,
            )

        # ── Parameter sweep: local epochs ─────────────────────────────────────
        log.info("\n▶  Parameter sweep: local_epochs")
        for le in [1, 3, 5, 10]:
            sweep_cfg = copy.copy(base_cfg)
            sweep_cfg.local_epochs = le
            sweep_cfg.use_dp = True
            label_le = f"fedavg_mean_le{le}"
            log.info("   local_epochs=%d", le)
            all_results[label_le] = run_federated(
                algorithm="fedavg", aggregation="mean",
                client_datasets=client_datasets, test_loader=test_loader,
                model_factory=model_factory, n_rounds=args.rounds,
                local_cfg=sweep_cfg, server_lr=1.0,
                use_dp=True, privacy_delta=privacy_delta, seed=args.seed,
            )

    # ── 10. Persist results ────────────────────────────────────────────────
    out = {
        lbl: [asdict(r) for r in rlist]
        for lbl, rlist in all_results.items()
    }
    out["_config"] = {
        "data_dir":       str(data_dir),
        "rounds":         args.rounds,
        "train_clients":  len(train_pts),
        "test_patients":  len(test_pts),
        "audio_dim":      audio_dim,
        "visual_dim":     visual_dim,
        "text_dim":       TEXT_DIM,
        "use_mentalbert": args.use_mentalbert,
        "noise_mult":     args.noise_mult,
        "clip_norm":      args.clip_norm,
        "seed":           args.seed,
    }
    json_path = RESULTS_DIR / "fl_daic_results.json"
    json_path.write_text(json.dumps(out, indent=2, default=str))
    log.info("Results → %s", json_path)

    # ── 11. Print summary table ────────────────────────────────────────────
    log.info("\n%s", "=" * 70)
    log.info("%-42s %7s %7s %8s", "Experiment", "Acc", "F1", "ε")
    log.info("-" * 70)
    for lbl in sorted(all_results):
        final = all_results[lbl][-1]
        log.info("%-42s %7.4f %7.4f %8.3f",
                 lbl, final.test_acc, final.test_f1,
                 final.epsilon if final.use_dp else 0.0)
    log.info("=" * 70)

    # ── 12. Visualise ──────────────────────────────────────────────────────
    plot_comparison(all_results)
    tex = latex_summary(all_results)
    log.info("LaTeX table → %s", RESULTS_DIR / "latex_table.tex")

    return all_results


if __name__ == "__main__":
    main()