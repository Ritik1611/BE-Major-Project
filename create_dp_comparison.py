#!/usr/bin/env python3
"""
create_dp_comparison.py

Pipeline:
  1) LDA preprocess -> read embeddings
  2) Train (train_model) -> expects (delta, model) ideally
  3) DPAgent.process_local_update -> produces noisy delta receipts
  4) Apply DP delta at model-level (preferred) or embedding-level (fallback)
  5) Evaluate (KMeans + silhouette)
  6) Encrypt the chosen DP receipt via EncryptionAgent.process_dp_update

Key fix in this version:
 - Pass the dp receipt's URI string to enc.process_dp_update (enc expects a string path).
"""
import os
import io
import time
import json
import math
import argparse
import random
import copy
import torch
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- LDA ---
from LDA.app.main import preprocess, PreprocessRequest

# --- Trainer ---
from trainer_agent.trainer import train_model

# --- Secure store / receipts ---
from centralized_secure_store import SecureStore
from centralised_receipts import CentralReceiptManager

# --- DP Agent ---
from dp_agent.dp_agent import DPAgent

# --- Encryption Agent ---
from enc_agent.enc_agent import EncryptionAgent


# -------------------
# Helpers
# -------------------
def read_embeddings_from_parquet(store: SecureStore, uri: str):
    data = store.decrypt_read(uri)
    buf = io.BytesIO(data)
    return pq.read_table(buf)


def read_manifest(store: SecureStore, manifest_uri: str):
    data = store.decrypt_read(manifest_uri)
    lines = data.decode().splitlines()
    manifest = [json.loads(line) for line in lines]
    parquet_uris = sorted(set(m["uri"] for m in manifest if m["uri"].endswith(".parquet.enc")))
    return parquet_uris


def flatten_state_dict(state_dict):
    arr_list = []
    for v in state_dict.values():
        if torch.is_tensor(v):
            arr_list.append(v.detach().cpu().flatten())
        else:
            try:
                arr_list.append(torch.tensor(v).flatten())
            except Exception:
                continue
    if not arr_list:
        return np.zeros((1, 0))
    flat = torch.cat(arr_list).unsqueeze(0).numpy()
    return flat


def evaluate_unsupervised(X):
    """Try k in [2..min(10, n-1)] and pick best silhouette."""
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.array(X)

    n_samples = X_np.shape[0]
    if n_samples < 2:
        return None, 0.0

    best_score = -1.0
    best_labels = None
    k_min = 2
    k_max = min(10, max(2, n_samples - 1))

    for k in range(k_min, k_max + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_np)
            score = silhouette_score(X_np, labels) if k > 1 else 0.0
            if score > best_score:
                best_score = float(score)
                best_labels = labels
        except Exception as e:
            # skip failing k
            print(f"[evaluate_unsupervised] k={k} failed: {e}")
            continue

    if best_labels is None:
        return None, 0.0
    return best_labels, float(best_score)


def apply_dp_update_to_embeddings(X: torch.Tensor, noisy_delta, dp_params=None, seed:int=0):
    """
    Heuristic: try elementwise or per-feature addition, otherwise fallback to random noise
    scaled to l2_norm_after.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    X = X.detach().cpu().clone()
    n_elems_X = X.numel()

    noisy = None
    if isinstance(noisy_delta, dict):
        flat = flatten_state_dict(noisy_delta)  # numpy (1, N)
        if flat.size and flat.shape[1] > 0:
            noisy = torch.from_numpy(flat.reshape(-1))
    elif torch.is_tensor(noisy_delta):
        noisy = noisy_delta.detach().cpu().reshape(-1)
    else:
        try:
            noisy = torch.tensor(noisy_delta).reshape(-1)
        except Exception:
            noisy = None

    # Exact match
    if noisy is not None and noisy.numel() == n_elems_X:
        noisy_view = noisy.view_as(X)
        return X + noisy_view

    # Per-feature
    if noisy is not None and noisy.numel() == X.shape[1]:
        noisy_view = noisy.view(1, -1).expand_as(X)
        return X + noisy_view

    # Guidance from dp_params or noisy norm
    l2_after = None
    if dp_params and isinstance(dp_params, dict):
        l2_after = dp_params.get("l2_norm_after", None)

    if noisy is not None:
        total_noise_l2 = float(torch.norm(noisy).item())
    elif l2_after is not None:
        total_noise_l2 = float(l2_after)
    else:
        total_noise_l2 = float(torch.norm(X).item()) * 0.01

    n = X.shape[0]
    per_sample_l2 = max(1e-12, total_noise_l2 / max(1, n))

    noise = torch.randn_like(X)
    noise_flat = noise.view(noise.shape[0], -1)
    norms = torch.norm(noise_flat, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    scaled_noise = (noise_flat / norms) * math.sqrt(per_sample_l2)
    scaled_noise = scaled_noise.view_as(X)

    X_pert = X + scaled_noise
    return X_pert


def apply_delta_to_model_and_get_updated_embeddings(original_model, delta, X):
    """
    Preferred approach:
     - deepcopy original_model (so constructor args/shape mismatches don't matter)
     - add delta tensors to matching keys in the model state_dict (only when shapes match)
     - run the updated model on X and return embeddings (torch.Tensor)
    Returns None if not possible.
    """
    if original_model is None:
        return None

    # Build delta_state
    delta_state = None
    if isinstance(delta, dict):
        delta_state = delta
    elif torch.is_tensor(delta):
        # try to map flat tensor onto base_state keys
        flat = delta.reshape(-1)
        base_state = original_model.state_dict()
        idx = 0
        tmp = {}
        ok = True
        for k, v in base_state.items():
            num = v.numel()
            if idx + num <= flat.numel():
                chunk = flat[idx: idx + num].reshape(v.shape)
                tmp[k] = chunk
                idx += num
            else:
                ok = False
                break
        if ok:
            delta_state = tmp
        else:
            delta_state = None
    else:
        try:
            delta_state = dict(delta)
        except Exception:
            delta_state = None

    if delta_state is None:
        return None

    # deepcopy model
    try:
        model_copy = copy.deepcopy(original_model)
    except Exception as e:
        print(f"[apply_delta_to_model] deepcopy failed: {e}")
        return None

    # Get model state and add delta to matching keys
    model_state = model_copy.state_dict()
    applied_any = False
    for k, dv in delta_state.items():
        if k in model_state:
            try:
                # coerce to tensor
                if not torch.is_tensor(dv):
                    dv_t = torch.tensor(dv, dtype=model_state[k].dtype)
                else:
                    dv_t = dv.to(dtype=model_state[k].dtype)
                if dv_t.shape == model_state[k].shape:
                    model_state[k] = model_state[k] + dv_t.to(model_state[k].device)
                    applied_any = True
                else:
                    # shape mismatch -> skip key
                    continue
            except Exception as e:
                # skip if incompatible
                print(f"[apply_delta_to_model] can't add delta for key {k}: {e}")
                continue
        else:
            # key not present -> skip
            continue

    if not applied_any:
        # nothing applied -> fallback
        print("[apply_delta_to_model] no delta keys matched model state keys - falling back")
        return None

    # load updated state
    try:
        model_copy.load_state_dict(model_state)
    except Exception as e:
        print(f"[apply_delta_to_model] load_state_dict failed: {e}")
        # try to continue using model_copy as-is

    # Try to forward X through model_copy
    try:
        model_copy.eval()
        with torch.no_grad():
            # try common call signatures
            try:
                out = model_copy(X)
            except Exception:
                try:
                    out = model_copy.forward(X)
                except Exception:
                    try:
                        out = model_copy.embed(X)
                    except Exception as e:
                        print(f"[apply_delta_to_model] model forward failed: {e}")
                        return None

        # pick tensor from out
        if isinstance(out, (tuple, list)):
            for cand in out:
                if isinstance(cand, torch.Tensor) and cand.shape[0] == X.shape[0]:
                    return cand.detach().cpu()
            # otherwise return first tensor found
            for cand in out:
                if isinstance(cand, torch.Tensor):
                    return cand.detach().cpu()
            return None
        elif isinstance(out, torch.Tensor):
            return out.detach().cpu()
        else:
            return None
    except Exception as e:
        print(f"[apply_delta_to_model] error computing embeddings: {e}")
        return None


# -------------------
# Run DP experiments per session
# -------------------
def run_dp_on_session(store, rm, trainer_receipt, X, noise_mechanisms, noise_multipliers, clip_norm, original_model=None):
    results = []
    for mech in noise_mechanisms:
        for nm in noise_multipliers:
            dp = DPAgent(
                clip_norm=clip_norm,
                noise_multiplier=nm,
                mechanism=mech,
                secure_store_dir="secure_store/local_updates",
                receipts_dir="receipts",
            )

            dp_result = dp.process_local_update(
                trainer_receipt["outputs"][0], metadata=trainer_receipt
            )

            noisy_buf = io.BytesIO(store.decrypt_read(dp_result["receipt"]["outputs"][0]))
            noisy_delta = torch.load(noisy_buf, map_location="cpu")

            # Try model-level update
            X_pert = None
            if original_model is not None:
                X_from_model = apply_delta_to_model_and_get_updated_embeddings(original_model, noisy_delta, X)
                if X_from_model is not None:
                    if isinstance(X_from_model, torch.Tensor):
                        X_pert = X_from_model
                    else:
                        try:
                            X_pert = torch.tensor(X_from_model)
                        except Exception:
                            X_pert = None

            # fallback: apply delta to embeddings
            if X_pert is None:
                X_pert = apply_dp_update_to_embeddings(X, noisy_delta, dp_result["receipt"].get("params", {}), seed=42)

            _, silhouette = evaluate_unsupervised(X_pert)

            # store full dp receipt (so encryption agent can accept it later)
            dp_receipt_obj = dp_result["receipt"]

            results.append({
                "session_id": trainer_receipt["session_id"],
                "mechanism": mech,
                "noise_multiplier": nm,
                "l2_before": dp_receipt_obj.get("params", {}).get("l2_norm_before"),
                "l2_after": dp_receipt_obj.get("params", {}).get("l2_norm_after"),
                "distortion_ratio": (dp_receipt_obj.get("params", {}).get("l2_norm_after", 0.0) /
                                     (dp_receipt_obj.get("params", {}).get("l2_norm_before", 1e-8) + 1e-8)),
                "silhouette_score": silhouette,
                "dp_update_uri": dp_receipt_obj.get("outputs", [None])[0],
                "dp_receipt": dp_receipt_obj,
                "receipt_uri": dp_result.get("receipt_uri"),
            })
            print(f"[DP Evaluation] session={trainer_receipt['session_id']}, mech={mech}, nm={nm}, silhouette={silhouette:.4f}")
    return results


# -------------------
# Main pipeline
# -------------------
def run_pipeline(args, noise_mechanisms=None, noise_multipliers=None):
    noise_mechanisms = noise_mechanisms or ["gaussian", "laplace", "uniform", "exponential", "none"]
    noise_multipliers = noise_multipliers or [0.0, 0.5, 1.0, 1.5, 2.0]

    store = SecureStore("./secure_store")
    rm = CentralReceiptManager()

    print("\n=== STEP 1: LDA Preprocessing ===")
    lda_req = PreprocessRequest(
        mode=args.lda_mode,
        inputs={args.input_type: args.input_path},
        config_uri="file://configs/local_config.yaml",
    )
    lda_result = preprocess(lda_req)
    print("LDA outputs:", lda_result)

    parquet_uris = read_manifest(store, lda_result["artifact_manifest"])
    if not parquet_uris:
        raise RuntimeError("No parquet embeddings found in manifest")

    tables = [read_embeddings_from_parquet(store, uri) for uri in parquet_uris]
    combined = pa.concat_tables(tables)
    df = combined.to_pandas()

    if "embedding" in df.columns:
        embs = np.stack(df["embedding"].to_numpy())
    else:
        embs = df.select_dtypes(include=[np.number]).to_numpy()

    X = torch.tensor(embs, dtype=torch.float32)

    print("\n=== STEP 2: Training Agent ===")
    dummy_y = torch.zeros(X.shape[0], dtype=torch.long)

    try:
        train_out = train_model(
            X, dummy_y, input_dim=X.shape[1],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device
        )
        if isinstance(train_out, (tuple, list)):
            delta = train_out[0]
            maybe_model = train_out[1] if len(train_out) > 1 else None
        else:
            delta = train_out
            maybe_model = None
    except Exception as e:
        raise RuntimeError(f"train_model failed: {e}")

    original_model = None
    if maybe_model is not None:
        original_model = maybe_model
        print("[Trainer] Received model instance from train_model() - will use model-level DP evaluation.")
    else:
        print("[Trainer] train_model did not return model instance. Model-level DP evaluation will be skipped (fallback applied).")

    # Save trainer update (delta)
    update_fname = f"trainer_{int(time.time() * 1000)}.pt.enc"
    update_path = os.path.join("secure_store/local_updates", update_fname)
    os.makedirs(os.path.dirname(update_path), exist_ok=True)
    buf = io.BytesIO()
    try:
        torch.save(delta, buf)
    except Exception:
        import pickle
        buf = io.BytesIO()
        pickle.dump(delta, buf)
    store.encrypt_write("file://" + update_path, buf.getvalue())

    trainer_receipt = rm.create_receipt(
        agent="trainer-agent",
        session_id=lda_result["session_id"],
        operation="train_step",
        params={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "device": args.device},
        outputs=["file://" + update_path],
    )
    trainer_receipt_uri = rm.write_receipt(trainer_receipt, out_dir="receipts")
    trainer_receipt["receipt_uri"] = trainer_receipt_uri
    print("Trainer receipt:", trainer_receipt_uri)

    print("\n=== STEP 3: DP Agent with Unsupervised Evaluation ===")
    all_results = run_dp_on_session(store, rm, trainer_receipt, X, noise_mechanisms, noise_multipliers, args.clip_norm, original_model=original_model)

    df_cmp = pd.DataFrame(all_results)
    cmp_csv = "dp_noise_mechanism_comparison_full.csv"
    df_cmp.to_csv(cmp_csv, index=False)
    print(f"\nSaved aggregated comparison table -> {cmp_csv}")

    print("\n=== STEP 4: Encryption Agent ===")
    enc = EncryptionAgent(mode=args.enc_mode)

    # Choose best silhouette result for encryption (fallback first)
    dp_receipt_path = None
    if all_results:
        try:
            best_idx = int(df_cmp["silhouette_score"].astype(float).idxmax())
            # use the dp receipt URI string (receipt_uri) that we stored per DP run
            dp_receipt_path = df_cmp.loc[best_idx, "receipt_uri"]
        except Exception:
            # fallback to first run's receipt_uri
            dp_receipt_path = all_results[0].get("receipt_uri")

    # Ensure dp_receipt_path is a string and has file:// prefix
    if dp_receipt_path:
        if not isinstance(dp_receipt_path, str):
            dp_receipt_path = str(dp_receipt_path)
        if not dp_receipt_path.startswith("file://"):
            dp_receipt_path = "file://" + dp_receipt_path

        # Pass the receipt URI string to the encryption agent (enc expects a path string)
        enc_result = enc.process_dp_update(dp_receipt_path)
        print("Encryption receipt:", enc_result.get("receipt_uri"))
    else:
        print("No DP receipt found for encryption - skipping encryption step.")
        enc_result = None

    print("\n=== PIPELINE COMPLETE ===")
    return {
        "lda": lda_result,
        "trainer": trainer_receipt,
        "dp_comparison_csv": cmp_csv,
        "enc_receipt": enc_result.get("receipt") if enc_result else None
    }


# -------------------
# CLI
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DP comparison pipeline over multiple sessions")

    parser.add_argument("--lda-mode", default="text",
                        choices=["text", "session", "batch", "continuous"])
    parser.add_argument("--input-type", default="text_dir",
                        choices=["text_dir", "video_dir", "audio_dir"])
    parser.add_argument("--input-path", default="./sample_texts")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--enc-mode", default="aes",
                        choices=["aes", "fernet", "kms_envelope", "he_ckks", "smpc"])

    args = parser.parse_args()
    result = run_pipeline(args)
    print("\nFinal pipeline outputs:")
    print(json.dumps(result, indent=2))
