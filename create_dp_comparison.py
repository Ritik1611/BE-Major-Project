#!/usr/bin/env python3
# create_dp_comparison.py
"""
Run DP comparison experiments across training modes (base, rag, vector_rag)
and multiple DP mechanisms/noise multipliers. Produces integrated CSVs and optional plots.

Robust/fallback behavior:
 - Tries to import trainer functions from trainer_agent; if missing, uses local lightweight trainer.
 - Tries different secure store class names.
 - Accepts DPAgent outputs in several common forms.
"""

import os
import io
import time
import json
import math
import argparse
import copy
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
import pyarrow as pa

# sklearn used for clustering / silhouette and metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error

# plotting
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False

# -------------------------
# Local imports (defensive)
# -------------------------
# LDA preprocess entrypoint (assumed)
try:
    from LDA.app.main import preprocess, PreprocessRequest
except Exception as e:
    raise ImportError("Could not import LDA.app.main.preprocess - ensure LDA is on PYTHONPATH") from e

# DP agent (user provided)
try:
    from dp_agent.dp_agent import DPAgent
except Exception:
    DPAgent = None

# Encryption / Enc agent
try:
    from enc_agent.enc_agent import EncryptionAgent
except Exception:
    EncryptionAgent = None

# Centralized receipt manager
try:
    from centralised_receipts import CentralReceiptManager
except Exception:
    try:
        from centralised_receipts import CentralReceiptManager as CentralReceiptManagerFallback
        CentralReceiptManager = CentralReceiptManagerFallback
    except Exception:
        # fallback stub
        class CentralReceiptManager:
            def __init__(self, *a, **k):
                pass
            def create_receipt(self, *a, **k):
                return {"agent": "fallback", "params": {}, "outputs": []}
            def write_receipt(self, receipt, out_dir="receipts"):
                # write a JSON file into out_dir
                os.makedirs(out_dir, exist_ok=True)
                fname = f"receipt_fallback_{int(time.time()*1000)}.json"
                path = Path(out_dir) / fname
                with open(path, "w") as f:
                    json.dump(receipt, f)
                return f"file://{path}"

# Secure store: try several names: SecureStore, CentralizedSecureStore
_store_cls = None
for _name in ("centralized_secure_store", "CentralizedSecureStore", "centralizedSecureStore", "secure_store", "securestore"):
    try:
        mod = __import__(_name)
        # try direct attribute names
        for cand in ("SecureStore", "CentralizedSecureStore", "secure_store", "Secure_Store"):
            if hasattr(mod, cand):
                _store_cls = getattr(mod, cand)
                break
        if _store_cls:
            break
    except Exception:
        pass

# fallback attempt: import from module path expected earlier
if _store_cls is None:
    try:
        from centralized_secure_store import SecureStore as _SS  # common in your files
        _store_cls = _SS
    except Exception:
        # minimal fallback
        class _FallbackStore:
            def __init__(self, root="./secure_store", agent=None):
                self.root = Path(root)
                self.root.mkdir(parents=True, exist_ok=True)
                self.agent = agent
            def encrypt_write(self, uri: str, payload: bytes):
                # accept uri starting with file:// or path
                if uri.startswith("file://"):
                    p = Path(uri[len("file://"):])
                else:
                    p = Path(uri)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "wb") as f:
                    f.write(payload)
                return f"file://{p}"
            def decrypt_read(self, uri: str) -> bytes:
                if uri.startswith("file://"):
                    p = Path(uri[len("file://"):])
                else:
                    p = Path(uri)
                with open(p, "rb") as f:
                    return f.read()
        _store_cls = _FallbackStore

SecureStore = _store_cls

# Trainer imports: try to import train_model and MentalBERTMultiTask from trainer_agent.trainer_mentalbert_privacy
_train_model_fn = None
_MentalBERTModelClass = None
try:
    import trainer_agent.trainer_mentalbert_privacy as _trainer_mod
    if hasattr(_trainer_mod, "train_model"):
        _train_model_fn = getattr(_trainer_mod, "train_model")
    if hasattr(_trainer_mod, "MentalBERTMultiTask"):
        _MentalBERTModelClass = getattr(_trainer_mod, "MentalBERTMultiTask")
    # also accept 'orchestrate' (higher-level)
    _trainer_orchestrate = getattr(_trainer_mod, "orchestrate", None)
except Exception:
    _trainer_orchestrate = None

# If train_model missing, we'll implement a local simple trainer (MLP probe)
if _train_model_fn is None:
    def _local_train_model(
        X: torch.Tensor,
        y: torch.Tensor,
        input_dim: int,
        epochs: int = 1,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "cpu",
        pretrained_model=None
    ):
        """
        Lightweight trainer to produce a delta state-dict and a trained model object.
        This is a simple MLP used as fallback when trainer_agent doesn't expose train_model.
        Returns: (delta_state_dict, model_instance)
        """
        device = torch.device(device)
        X = X.to(device)
        # interpret y: if all zeros or single class -> unsupervised-like; we'll train an autoencoder-like MLP
        num_samples, dim = X.shape[0], input_dim
        # Simple 2-layer MLP
        class ProbeModel(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc1 = torch.nn.Linear(dim, max(16, dim // 2))
                self.act = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(max(16, dim // 2), dim)
            def forward(self, x):
                h = self.act(self.fc1(x))
                return self.fc2(h)
        model = ProbeModel(input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(X, X)  # try to reconstruct
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        before_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.train()
        for epoch in range(epochs):
            total = 0.0
            cnt = 0
            for bx, by in loader:
                bx = bx.to(device)
                pred = model(bx)
                loss = loss_fn(pred, bx)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.item())
                cnt += 1
            avg = total / max(1, cnt)
            print(f"[local_probe] epoch {epoch+1}/{epochs} loss={avg:.6f}")
        after_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # delta
        delta = {}
        for k in after_state:
            if k in before_state and before_state[k].shape == after_state[k].shape:
                delta[k] = after_state[k] - before_state[k]
            else:
                delta[k] = after_state[k].clone()
        return delta, model
    _train_model_fn = _local_train_model

# Accept MentalBERT model class fallback: no-op simple wrapper
if _MentalBERTModelClass is None:
    _MentalBERTModelClass = None  # may not be used


# ------------
# Utilities
# ------------
def read_parquet_from_bytes(b: bytes) -> pa.Table:
    buf = io.BytesIO(b)
    return pq.read_table(buf)


def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    parts = []
    for v in state_dict.values():
        try:
            if isinstance(v, torch.Tensor):
                parts.append(v.detach().cpu().reshape(-1))
            else:
                t = torch.tensor(v)
                parts.append(t.reshape(-1))
        except Exception:
            continue
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    cat = torch.cat(parts).cpu().numpy()
    return cat


def evaluate_unsupervised_X(X_np: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Return cluster labels (best k) and silhouette score. If insufficient samples, return (None, 0.0)"""
    X_np = np.asarray(X_np)
    n = X_np.shape[0]
    if n < 2:
        return None, 0.0
    best_score = -1.0
    best_labels = None
    k_min = 2
    k_max = min(10, max(2, n - 1))
    for k in range(k_min, k_max + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labs = km.fit_predict(X_np)
            score = silhouette_score(X_np, labs) if k > 1 else 0.0
            if score > best_score:
                best_score = float(score)
                best_labels = labs
        except Exception:
            continue
    if best_labels is None:
        return None, 0.0
    return best_labels, float(best_score)


def build_rag_features(X_np: np.ndarray, k: int = 3) -> np.ndarray:
    """Concatenate original feature with mean of k nearest neighbors (RAG-style)."""
    if X_np.shape[0] <= 1 or k <= 0:
        return X_np
    knn = NearestNeighbors(n_neighbors=min(k + 1, X_np.shape[0]), metric='cosine')
    knn.fit(X_np)
    _, idxs = knn.kneighbors(X_np)
    neighbor_mean = []
    for i in range(X_np.shape[0]):
        neigh = idxs[i, 1: min(1 + k, idxs.shape[1])]
        neighbor_mean.append(X_np[neigh].mean(axis=0))
    neighbor_mean = np.stack(neighbor_mean, axis=0)
    return np.concatenate([X_np, neighbor_mean], axis=1)


def apply_dp_to_embeddings(X: torch.Tensor, noisy_delta, dp_params=None, seed: int = 0) -> torch.Tensor:
    """Try to apply noisy delta to embeddings; fallback to isotropic noise injection."""
    torch.manual_seed(seed)
    X = X.detach().cpu().clone()
    n = X.shape[0]
    noisy_vec = None

    # try flattening noisy_delta
    if isinstance(noisy_delta, dict):
        flat = flatten_state_dict(noisy_delta)
        if flat.size > 0:
            noisy_vec = torch.from_numpy(flat).float()
    elif torch.is_tensor(noisy_delta):
        noisy_vec = noisy_delta.detach().float().reshape(-1)
    else:
        try:
            noisy_vec = torch.tensor(noisy_delta).float().reshape(-1)
        except Exception:
            noisy_vec = None

    # direct element-wise match
    total_elems = X.numel()
    if noisy_vec is not None and noisy_vec.numel() == total_elems:
        return X + noisy_vec.view_as(X)

    # per-feature match
    if noisy_vec is not None and noisy_vec.numel() == X.shape[1]:
        return X + noisy_vec.view(1, -1).expand_as(X)

    # fallback: use l2 guidance if present, else small fraction of X norm
    l2_after = None
    if dp_params and isinstance(dp_params, dict):
        l2_after = dp_params.get("l2_norm_after", None)

    if noisy_vec is not None:
        total_noise_l2 = float(torch.norm(noisy_vec).item())
    elif l2_after is not None:
        total_noise_l2 = float(l2_after)
    else:
        total_noise_l2 = float(torch.norm(X).item()) * 0.01

    per_sample_l2 = max(1e-12, total_noise_l2 / max(1, n))
    noise = torch.randn_like(X)
    flat_noise = noise.view(noise.shape[0], -1)
    norms = torch.norm(flat_noise, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    scaled_flat = (flat_noise / norms) * math.sqrt(per_sample_l2)
    scaled = scaled_flat.view_as(X)
    return X + scaled


def try_apply_delta_to_model_and_forward(original_model, delta, X: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Try to apply delta dict to a copy of original_model and forward X.
    Returns resulting embeddings tensor or None.
    """
    if original_model is None:
        return None

    delta_state = None
    if isinstance(delta, dict):
        delta_state = delta
    else:
        try:
            delta_state = dict(delta)
        except Exception:
            delta_state = None

    if delta_state is None:
        return None

    try:
        model_copy = copy.deepcopy(original_model)
    except Exception:
        return None

    model_state = model_copy.state_dict()
    applied = False
    for k, dv in delta_state.items():
        if k in model_state:
            try:
                dv_t = dv if isinstance(dv, torch.Tensor) else torch.tensor(dv, dtype=model_state[k].dtype)
                if dv_t.shape == model_state[k].shape:
                    # move to CPU combine
                    model_state[k] = (model_state[k].detach().cpu() + dv_t.detach().cpu()).to(model_state[k].device)
                    applied = True
            except Exception:
                continue

    if not applied:
        return None

    try:
        model_copy.load_state_dict(model_state)
    except Exception:
        # best-effort continue
        pass

    model_copy.eval()
    # find device for model
    try:
        model_device = next(model_copy.parameters()).device
    except StopIteration:
        model_device = torch.device('cpu')

    with torch.no_grad():
        try:
            inp = X.to(model_device)
            out = model_copy(inp)
        except Exception:
            try:
                out = model_copy.forward(X.to(model_device))
            except Exception:
                try:
                    out = model_copy(X.to(model_device))
                except Exception:
                    return None

    # If model returns tuple (class_logits, reg_out) try to use reg_out or pooled features
    if isinstance(out, (tuple, list)):
        t0, t1 = None, None
        for cand in out:
            if isinstance(cand, torch.Tensor) and cand.shape[0] == X.shape[0]:
                if cand.ndim == 2:
                    t0 = cand.detach().cpu()
                elif cand.ndim == 1 or (cand.ndim == 2 and cand.shape[1] == 1):
                    t1 = cand.detach().cpu()
        if t1 is not None:
            return t1.squeeze(-1)
        if t0 is not None:
            return t0.detach().cpu()
        for cand in out:
            if isinstance(cand, torch.Tensor):
                return cand.detach().cpu()
        return None
    elif isinstance(out, torch.Tensor):
        return out.detach().cpu()
    else:
        return None


def supervised_eval_from_model_output(model_out: torch.Tensor, y_true: np.ndarray):
    """
    model_out: torch.Tensor predictions (logits or continuous)
    y_true: numpy array ground truth (either discrete or continuous)
    returns dict of metrics with keys: accuracy, precision, recall, f1, mae
    """
    res = {"accuracy": None, "precision": None, "recall": None, "f1": None, "mae": None}
    if model_out is None:
        return res
    out_np = model_out.detach().cpu().numpy() if isinstance(model_out, torch.Tensor) else np.array(model_out)
    y_np = np.array(y_true)
    # If discrete labels (integers)
    try:
        if np.issubdtype(y_np.dtype, np.integer) or np.all(np.equal(np.mod(y_np, 1), 0)):
            # classification
            if out_np.ndim == 2 and out_np.shape[1] > 1:
                preds = np.argmax(out_np, axis=1)
            else:
                preds = (out_np.squeeze() >= 0.5).astype(int)
            res["accuracy"] = float(accuracy_score(y_np, preds))
            res["precision"] = float(precision_score(y_np, preds, zero_division=0))
            res["recall"] = float(recall_score(y_np, preds, zero_division=0))
            res["f1"] = float(f1_score(y_np, preds, zero_division=0))
        else:
            preds = out_np.squeeze()
            res["mae"] = float(mean_absolute_error(y_np.astype(float), preds))
    except Exception:
        pass
    return res


# ----------------------
# DP-run orchestration
# ----------------------
def run_dp_experiments_for_update(
    store: SecureStore,
    rm: CentralReceiptManager,
    trainer_receipt: Dict[str, Any],
    trainer_update_uri: str,
    X: torch.Tensor,
    y_np: Optional[np.ndarray],
    noise_mechs: List[str],
    noise_mults: List[float],
    clip_norm: float,
    original_model=None
) -> List[Dict[str, Any]]:
    """
    For a single trainer update, run DPAgent for each mechanism and noise multiplier,
    evaluate and return list of result dicts.
    """
    results: List[Dict[str, Any]] = []
    if DPAgent is None:
        raise ImportError("DPAgent not found - ensure dp_agent.dp_agent is importable")

    for mech in noise_mechs:
        for nm in noise_mults:
            dp = DPAgent(
                clip_norm=clip_norm,
                noise_multiplier=nm,
                mechanism=mech,
                secure_store_dir=str(Path(store.root) / "local_updates"),
                receipts_dir="receipts",
                store=store
            )
            try:
                dp_result = dp.process_local_update(trainer_update_uri, metadata=trainer_receipt)
            except Exception as e:
                print(f"[DP] process_local_update FAILED mech={mech} nm={nm}: {e}")
                print(traceback.format_exc())
                continue

            # Extract dp_update_uri in robust ways
            dp_update_uri = None
            dp_receipt = None
            dp_receipt_uri = None
            try:
                if isinstance(dp_result, dict):
                    dp_receipt = dp_result.get("receipt") or dp_result.get("receipt_obj") or dp_result.get("receipt_data")
                    dp_receipt_uri = dp_result.get("receipt_uri") or dp_result.get("receipt_uri_out") or dp_result.get("receipt_path")
                    # outputs in receipt
                    if dp_receipt and isinstance(dp_receipt, dict):
                        outs = dp_receipt.get("outputs", [])
                        if outs:
                            dp_update_uri = outs[0]
                    # fallback dp_result.outputs
                    if dp_update_uri is None:
                        outs2 = dp_result.get("outputs") or dp_result.get("outputs_paths")
                        if isinstance(outs2, (list, tuple)) and outs2:
                            dp_update_uri = outs2[0]
                else:
                    # unknown form - skip
                    dp_update_uri = None
            except Exception:
                dp_update_uri = None

            # read noisy update (if present)
            noisy_delta = None
            if dp_update_uri:
                try:
                    raw = store.decrypt_read(dp_update_uri)
                    buf = io.BytesIO(raw)
                    noisy_delta = torch.load(buf, map_location='cpu')
                except Exception as e:
                    print(f"[DP] Could not read noisy update {dp_update_uri}: {e}")
                    noisy_delta = None

            # 1) Try model-level application
            model_out = None
            if original_model is not None and noisy_delta is not None:
                try:
                    model_out = try_apply_delta_to_model_and_forward(original_model, noisy_delta, X)
                except Exception:
                    model_out = None

            # 2) Fallback to embeddings-level application
            X_pert = None
            if model_out is None:
                try:
                    X_pert = apply_dp_to_embeddings(X, noisy_delta, dp_params=(dp_result.get("receipt", {}).get("params") if isinstance(dp_result, dict) else None), seed=42)
                except Exception:
                    X_pert = X.clone()

                model_out = None
                if y_np is not None:
                    # quick probe: linear classifier or regressor trained on original X -> y, applied to X_pert
                    try:
                        from sklearn.linear_model import LogisticRegression, LinearRegression
                        X_orig = X.detach().cpu().numpy()
                        X_new = X_pert.detach().cpu().numpy()
                        if np.issubdtype(y_np.dtype, np.integer) or np.all(np.equal(np.mod(y_np, 1), 0)):
                            clf = LogisticRegression(max_iter=200)
                            clf.fit(X_orig, y_np.astype(int))
                            probs = clf.predict_proba(X_new)
                            model_out = torch.tensor(probs)
                        else:
                            reg = LinearRegression()
                            reg.fit(X_orig, y_np.astype(float))
                            preds = reg.predict(X_new)
                            model_out = torch.tensor(preds).unsqueeze(-1)
                    except Exception as e:
                        model_out = None

            # choose representation to compute silhouette
            eval_array = None
            if model_out is not None:
                eval_array = model_out.detach().cpu().numpy()
            elif X_pert is not None:
                eval_array = X_pert.detach().cpu().numpy()
            else:
                eval_array = X.detach().cpu().numpy()

            _, sil = evaluate_unsupervised_X(np.array(eval_array))

            # supervised metrics if available
            sup_metrics = {"accuracy": None, "precision": None, "recall": None, "f1": None, "mae": None}
            if y_np is not None and model_out is not None:
                try:
                    sup_metrics = supervised_eval_from_model_output(model_out, y_np)
                except Exception:
                    pass

            # collect l2 params if present in receipt
            params = None
            if isinstance(dp_result, dict):
                params = (dp_result.get("receipt", {}) or {}).get("params", {}) if isinstance(dp_result.get("receipt", {}), dict) else {}
            l2_before = params.get("l2_norm_before") if isinstance(params, dict) else None
            l2_after = params.get("l2_norm_after") if isinstance(params, dict) else None

            result_entry = {
                "session_id": trainer_receipt.get("session_id") if isinstance(trainer_receipt, dict) else None,
                "mechanism": mech,
                "noise_multiplier": nm,
                "l2_before": l2_before,
                "l2_after": l2_after,
                "distortion_ratio": None if (l2_before is None or l2_after is None) else (l2_after / (l2_before + 1e-12)),
                "silhouette_score": sil,
                "accuracy": sup_metrics.get("accuracy"),
                "precision": sup_metrics.get("precision"),
                "recall": sup_metrics.get("recall"),
                "f1": sup_metrics.get("f1"),
                "mae": sup_metrics.get("mae"),
                "dp_update_uri": dp_update_uri,
                "dp_receipt": dp_result.get("receipt") if isinstance(dp_result, dict) else None,
                "dp_receipt_uri": dp_receipt_uri
            }
            print(f"[DP RUN] mech={mech} nm={nm} silhouette={sil:.4f} acc={result_entry['accuracy']}")
            results.append(result_entry)

    return results


# ----------------------
# Plot helpers
# ----------------------
def plot_metric_vs_noise(df_mode: pd.DataFrame, mode_name: str, out_dir: str):
    if not _HAS_MATPLOTLIB:
        print("[WARN] matplotlib not available, skipping plots")
        return
    os.makedirs(out_dir, exist_ok=True)
    df_mode["noise_multiplier"] = df_mode["noise_multiplier"].astype(float)
    metrics = ["silhouette_score", "accuracy", "mae"]
    plt.figure(figsize=(10, 6))
    for m in metrics:
        if m in df_mode.columns and df_mode[m].notnull().any():
            means = df_mode.groupby("noise_multiplier")[m].mean()
            stds = df_mode.groupby("noise_multiplier")[m].std().fillna(0.0)
            xs = means.index.values
            ys = means.values
            plt.plot(xs, ys, marker='o', label=m)
            plt.fill_between(xs, ys - stds, ys + stds, alpha=0.12)
    plt.title(f"{mode_name} - Metrics vs Noise Multiplier")
    plt.xlabel("Noise Multiplier")
    plt.ylabel("Metric value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = Path(out_dir) / f"dp_comparison_{mode_name}_combined.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    # separate per-metric plots
    if "silhouette_score" in df_mode.columns and df_mode["silhouette_score"].notnull().any():
        fig, ax = plt.subplots(figsize=(8,4))
        for mech, g in df_mode.groupby("mechanism"):
            ax.plot(g["noise_multiplier"], g["silhouette_score"], marker='o', label=mech)
        ax.set_xlabel("Noise Multiplier")
        ax.set_ylabel("Silhouette")
        ax.set_title(f"{mode_name} Silhouette vs Noise")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.savefig(Path(out_dir)/f"silhouette_vs_noise_{mode_name}.png", bbox_inches="tight")
        plt.close(fig)

    if "accuracy" in df_mode.columns and df_mode["accuracy"].notnull().any():
        fig, ax = plt.subplots(figsize=(8,4))
        for mech, g in df_mode.groupby("mechanism"):
            ax.plot(g["noise_multiplier"], g["accuracy"], marker='o', label=mech)
        ax.set_xlabel("Noise Multiplier")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{mode_name} Accuracy vs Noise")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.savefig(Path(out_dir)/f"accuracy_vs_noise_{mode_name}.png", bbox_inches="tight")
        plt.close(fig)

    if "mae" in df_mode.columns and df_mode["mae"].notnull().any():
        fig, ax = plt.subplots(figsize=(8,4))
        for mech, g in df_mode.groupby("mechanism"):
            ax.plot(g["noise_multiplier"], g["mae"], marker='o', label=mech)
        ax.set_xlabel("Noise Multiplier")
        ax.set_ylabel("MAE")
        ax.set_title(f"{mode_name} MAE vs Noise")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.savefig(Path(out_dir)/f"mae_vs_noise_{mode_name}.png", bbox_inches="tight")
        plt.close(fig)


# ----------------------
# Main pipeline
# ----------------------
def run_pipeline(args):
    store = SecureStore(root=args.store_root) if callable(SecureStore) else SecureStore(args.store_root)
    rm = CentralReceiptManager()

    # Ensure single SecureStore instance reused across agents
    from dp_agent import dp_agent as dp_mod
    if hasattr(dp_mod, "DPAgent"):
        dp_mod.GLOBAL_SECURE_STORE = store

    noise_mechs = args.noise_mechanisms or ["gaussian", "laplace", "uniform", "exponential", "student_t", "none"]
    noise_mults = args.noise_multipliers or [0.0, 0.5, 1.0, 1.5, 2.0]

    # 1) Run LDA preprocessing
    print("=== STEP 1: LDA Preprocessing ===")
    lda_req = PreprocessRequest(mode=args.lda_mode, inputs={args.input_type: args.input_path}, config_uri=args.config_uri)
    lda_result = preprocess(lda_req)
    print("LDA result keys:", list(lda_result.keys()))

    # read parquet URIs from manifest saved in SecureStore
    manifest_uri = lda_result.get("artifact_manifest")
    manifest_bytes = store.decrypt_read(manifest_uri)
    manifest_lines = manifest_bytes.decode().splitlines()
    manifest = [json.loads(l) for l in manifest_lines]
    parquet_uris = sorted({m["uri"] for m in manifest if str(m["uri"]).endswith(".parquet.enc")})

    # ------------------------------------------------------------------
    # NEW: fallback if no parquet.enc files (text-only mode)
    # ------------------------------------------------------------------
    if not parquet_uris:
        print("[WARN] No parquet.enc files found in manifest, attempting text-only fallback...")

        # Look for .txt or .csv files directly inside the input directory
        input_dir = Path(args.input_path)
        text_files = list(input_dir.glob("*.txt")) + list(input_dir.glob("*.csv"))

        if not text_files:
            raise RuntimeError("No parquet or text files found for input_path.")

        # Load texts and optional labels (if CSV)
        records = []
        for tf in text_files:
            if tf.suffix == ".txt":
                with open(tf, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                records.append({"text": text})
            elif tf.suffix == ".csv":
                df_txt = pd.read_csv(tf)
                for _, row in df_txt.iterrows():
                    records.append({
                        "text": str(row.get("text", "")),
                        "phq_score": row.get("phq_score") or row.get("label") or row.get("target_phq"),
                    })

        if not records:
            raise RuntimeError("No valid records loaded from text_dir fallback.")

        print(f"[INFO] Loaded {len(records)} text samples from {input_dir}")

        # Tokenize using MentalBERT
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")
        model = AutoModel.from_pretrained("mental/mental-bert-base-uncased").to(args.device)
        embs = []
        model.eval()
        with torch.no_grad():
            for r in records:
                inputs = tokenizer(r["text"], return_tensors="pt", truncation=True, padding=True, max_length=128).to(args.device)
                out = model(**inputs)
                pooled = out.last_hidden_state[:, 0, :].mean(dim=0).cpu().numpy()
                embs.append(pooled)

        X_np = np.stack(embs)
        label_cols = [k for k in records[0].keys() if "phq" in k or "label" in k]
        y_np = np.array([r.get(label_cols[0], 0) for r in records]) if label_cols else None

        df = pd.DataFrame({"embedding": list(X_np), "phq_score": y_np})
        print(f"[INFO] Created fallback dataframe shape={df.shape}")

    else:
        # combine parquet tables (normal path)
        tables = []
        for uri in parquet_uris:
            try:
                raw = store.decrypt_read(uri)
                tables.append(read_parquet_from_bytes(raw))
            except Exception as e:
                print(f"[WARN] unable to read parquet {uri}: {e}")
        combined = pa.concat_tables(tables)
        df = combined.to_pandas()
        print(f"[INFO] Combined parquet dataframe shape={df.shape}")

        # Extract embeddings
        if "embedding" in df.columns:
            X_np = np.stack(df["embedding"].to_numpy())
        else:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] == 0:
                raise RuntimeError("No numeric columns or 'embedding' column found in parquet")
            X_np = numeric_df.to_numpy()

        # Labels
        label_cols = [c for c in df.columns if c.lower() in ("label","phq","phq_score","score")]
        y_np = df[label_cols[0]].to_numpy() if label_cols else None


    X = torch.tensor(X_np, dtype=torch.float32)

    # optionally load pretrained model for model-level delta application tests
    pretrained_model = None
    if args.use_bert and _MentalBERTModelClass is not None:
        try:
            pretrained_model = _MentalBERTModelClass(model_name=args.bert_model_name) if _MentalBERTModelClass is not None else None
            if args.bert_finetuned_path and Path(args.bert_finetuned_path).exists():
                pretrained_model.load_state_dict(torch.load(args.bert_finetuned_path, map_location='cpu'), strict=False)
            if pretrained_model is not None:
                pretrained_model = pretrained_model.to(args.device)
                print("[INFO] MentalBERT loaded for model-level application tests")
        except Exception as e:
            print(f"[WARN] Could not load MentalBERT model: {e}")
            pretrained_model = None
    elif args.use_bert:
        print("[WARN] MentalBERT class not available in trainer module; proceeding without it")

    # training modes
    modes = args.modes or ["base", "rag", "vector_rag"]
    combined_results = []

    for mode in modes:
        print(f"\n=== MODE: {mode} ===")
        if mode == "base":
            X_train_np = X_np.copy()
        elif mode == "rag":
            X_train_np = build_rag_features(X_np, k=args.rag_k)
        elif mode == "vector_rag":
            X_train_np = build_rag_features(X_np, k=args.rag_k)
        else:
            print(f"[WARN] unknown mode {mode}, skipping.")
            continue

        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        # prepare y for trainer
        if y_np is not None:
            if np.issubdtype(y_np.dtype, np.floating):
                y_for_trainer = torch.tensor(y_np, dtype=torch.float32)
            else:
                y_for_trainer = torch.tensor(y_np, dtype=torch.long)
        else:
            y_for_trainer = torch.zeros(X_train.shape[0], dtype=torch.long)

        # Try using trainer.train_model if available
        print("[TRAINER] Starting training for mode:", mode)
        delta_state = None
        trained_model = None
        try:
            delta_state, trained_model = _train_model_fn(
                X_train,
                y_for_trainer,
                input_dim=X_train.shape[1],
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                pretrained_model=pretrained_model if args.use_bert else None
            )
        except TypeError:
            # older signature - no pretrained_model
            delta_state, trained_model = _train_model_fn(
                X_train,
                y_for_trainer,
                input_dim=X_train.shape[1],
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device
            )
        except Exception as e:
            print(f"[ERROR] trainer failed: {e}")
            print(traceback.format_exc())
            # fallback to local trainer
            delta_state, trained_model = _local_train_model(X_train, y_for_trainer, input_dim=X_train.shape[1], epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)

        # Save delta into secure store
        sess = lda_result.get("session_id", f"sess-{int(time.time())}")
        update_rel = f"{sess}/local_updates/trainer_{mode}_{int(time.time()*1000)}.pt.enc"
        update_path = Path(args.store_root) / update_rel
        update_path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        torch.save(delta_state, buf)
        store.encrypt_write(f"file://{update_path}", buf.getvalue())
        print(f"[TRAINER] Delta saved to secure store at file://{update_path}")

        # create receipt
        trainer_receipt = rm.create_receipt(
            agent="trainer-agent",
            session_id=sess,
            operation=f"train_{mode}",
            params={"mode": mode, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr},
            outputs=[f"file://{update_path}"]
        )
        try:
            trainer_receipt_uri = rm.write_receipt(trainer_receipt, out_dir="receipts")
            trainer_receipt["receipt_uri"] = trainer_receipt_uri
        except Exception:
            trainer_receipt["receipt_uri"] = None

        # run DP experiments
        print("[DP] Running DP experiments for this trainer update ...")
        try:
            dp_results = run_dp_experiments_for_update(
                store=store,
                rm=rm,
                trainer_receipt=trainer_receipt,
                trainer_update_uri=f"file://{update_path}",
                X=X,
                y_np=y_np,
                noise_mechs=noise_mechs,
                noise_mults=noise_mults,
                clip_norm=args.clip_norm,
                original_model=trained_model if args.try_model_apply else pretrained_model
            )
        except Exception as e:
            print(f"[ERROR] DP run failed: {e}")
            print(traceback.format_exc())
            dp_results = []

        # write per-mode CSV
        df_mode = pd.DataFrame(dp_results)
        out_csv_mode = f"dp_noise_mechanism_comparison_{mode}.csv"
        df_mode.to_csv(out_csv_mode, index=False)
        print(f"[RESULTS] Saved per-mode CSV: {out_csv_mode}")

        # plotting
        try:
            plot_metric_vs_noise(df_mode, mode, out_dir="plots")
        except Exception as e:
            print(f"[WARN] plotting failed for mode {mode}: {e}")

        # append
        for r in dp_results:
            r["mode"] = mode
            combined_results.append(r)

    # integrated CSV
    df_all = pd.DataFrame(combined_results)
    all_csv = "dp_comparison_all_modes.csv"
    df_all.to_csv(all_csv, index=False)
    print(f"[RESULTS] Saved integrated CSV: {all_csv}")

    return {
        "lda_result": lda_result,
        "per_mode_csvs": [f"dp_noise_mechanism_comparison_{m}.csv" for m in modes],
        "integrated_csv": all_csv,
        "plots_dir": "plots" if _HAS_MATPLOTLIB else None
    }


# ----------------------
# CLI
# ----------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Create DP mechanism comparison across modes")
    p.add_argument("--lda-mode", default="session", choices=["session", "batch", "text", "continuous"])
    p.add_argument("--input-type", default="text_dir", choices=["text_dir", "video_dir", "audio_dir"])
    p.add_argument("--input-path", default="./sample_texts")
    p.add_argument("--config-uri", default="file://configs/local_config.yaml")
    p.add_argument("--store-root", default="./secure_store")
    p.add_argument("--modes", nargs="+", default=["base", "rag", "vector_rag"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--rag-k", type=int, default=3)
    p.add_argument("--binarize-phq", action="store_true")
    p.add_argument("--binarize-threshold", type=float, default=10.0)
    p.add_argument("--enc-mode", default="aes", choices=["aes", "fernet", "kms_envelope", "he_ckks", "smpc"])
    p.add_argument("--noise-mechanisms", nargs="+", default=None)
    p.add_argument("--noise-multipliers", nargs="+", type=float, default=None)
    p.add_argument("--use-bert", action="store_true", help="Load MentalBERT for model-level delta application tests")
    p.add_argument("--bert-finetuned-path", default="./trainer_outputs/mentalbert_privacy_subset.pt")
    p.add_argument("--bert-model-name", default="mental/mental-bert-base-uncased")
    p.add_argument("--try-model-apply", action="store_true", help="Try to apply noisy delta to the model to get updated embeddings")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    res = run_pipeline(args)
    print("=== DONE ===")
    print(json.dumps(res, indent=2, default=str))
