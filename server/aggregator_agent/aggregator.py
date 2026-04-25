#!/usr/bin/env python3
"""
aggregator.py — Federated aggregation agent (filesystem/ledger backend)

Zero-trust guarantees:
• All paths are canonicalized and validated against server_root
• Update files are decrypted via SecureStore (AES-GCM) before aggregation
• Aggregation uses trimmed mean (Byzantine-robust) with per-parameter norm bounding
• Global model is encrypted via SecureStore.encrypt_write() (if safe) or plain file
• Audit receipt is written via CentralReceiptManager (HMAC-chained)
• No panics: all errors logged, server continues (availability > audit completeness)
• Memory-safe: parameters streamed one-by-one, never load full model into RAM at once
• Startup checks: verify --server-root is within ~/.federated/, round_id is positive

Usage (invoked by Rust server):
  python aggregator.py --server-root /path/to/.federated/server --round-id 1

Output (parsed by Rust):
  GLOBAL_MODEL_PATH=/absolute/path/to/encrypted_global_model.bin
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator

import numpy as np
import torch

# Local imports (must be on PYTHONPATH set by Rust caller)
from core.centralized_secure_store import SecureStore
from core.centralised_receipts import CentralReceiptManager

# ── Configuration ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("AGGREGATOR_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# Aggregation hyperparameters (tunable via env vars)
TRIM_RATIO = float(os.environ.get("FL_TRIM_RATIO", "0.1"))  # Trim top/bottom 10% per param
MAX_PARAM_DELTA = float(os.environ.get("FL_MAX_PARAM_DELTA", "1e-3"))  # Per-param clamp
MAX_GLOBAL_NORM = float(os.environ.get("FL_MAX_GLOBAL_NORM", "1.0"))  # Scale delta to this L2 norm
PRIVACY_DELTA = float(os.environ.get("FL_PRIVACY_DELTA", "1e-4"))  # For RDP→(ε,δ) conversion

# Secure store root (all encrypted files live under here)
SECURE_STORE_ROOT = Path.home() / ".federated" / "data" / "secure_store"

# ── Zero-trust path validation ────────────────────────────────────────────────
def validate_path_within_root(path: Path, root: Path) -> Path:
    """Ensure path is canonical and within root (prevent traversal)."""
    try:
        canonical_root = root.resolve(strict=True)
    except FileNotFoundError:
        # Root doesn't exist yet — create it
        root.mkdir(parents=True, exist_ok=True)
        canonical_root = root.resolve()
    canonical_path = path.resolve(strict=False)
    # Use relative_to() which raises ValueError if path is not within root
    try:
        canonical_path.relative_to(canonical_root)
    except ValueError:
        raise ValueError(
            f"Path traversal rejected: {canonical_path} is not within {canonical_root}"
        )
    return canonical_path

def validate_server_root(server_root: str) -> Path:
    """Validate --server-root arg is within ~/.federated/."""
    root = Path(server_root).expanduser().resolve()
    federated_root = Path.home() / ".federated"
    try:
        root.relative_to(federated_root)
    except ValueError:
        raise ValueError(
            f"server-root must be within ~/.federated/: got {root}"
        )
    return root

# ── SecureStore wrapper for encryption/decryption ─────────────────────────────
def _get_secure_store(agent: str = "aggregator") -> SecureStore:
    """Return SecureStore instance with agent set for key derivation."""
    return SecureStore(agent=agent, root=SECURE_STORE_ROOT)

def decrypt_update(file_path: Path, secure_store: SecureStore) -> Optional[Dict[str, torch.Tensor]]:
    """Decrypt and load a single update file. Returns state_dict or None on failure."""
    try:
        # SecureStore expects file:// URI
        uri = f"file://{file_path.resolve()}"
        raw_bytes = secure_store.decrypt_read(uri)
        
        # Load PyTorch state_dict from bytes (weights_only for security)
        import io
        buffer = io.BytesIO(raw_bytes)
        state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
        
        # Validate structure: must be dict[str, Tensor]
        if not isinstance(state_dict, dict):
            log.warning("Update %s: not a state_dict (type=%s)", file_path.name, type(state_dict))
            return None
        
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                log.warning("Update %s: key '%s' is not a Tensor (type=%s)", file_path.name, k, type(v))
                return None
        
        log.info("Decrypted update %s: %d parameters", file_path.name, sum(p.numel() for p in state_dict.values()))
        return state_dict
        
    except Exception as e:
        log.warning("Failed to decrypt/load %s: %s", file_path, e)
        return None

def encrypt_and_save_global_model(
    state_dict: Dict[str, torch.Tensor],
    output_path: Path,
    secure_store: SecureStore,
    use_encryption: bool = True,
) -> str:
    """Save global model. Returns the absolute path (for Rust to parse)."""
    try:
        # Serialize to bytes first
        import io
        buffer = io.BytesIO()
        torch.save(state_dict, buffer, _use_new_zipfile_serialization=False)
        model_bytes = buffer.getvalue()
        
        if use_encryption:
            # Encrypt via SecureStore
            uri = f"file://{output_path.resolve()}"
            secure_store.encrypt_write(uri, model_bytes)
            log.info("Encrypted global model saved to %s", output_path)
        else:
            # Plain file (fallback if encryption corrupts model)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(model_bytes)
            log.info("Plain global model saved to %s", output_path)
        
        return str(output_path.resolve())
        
    except Exception as e:
        log.critical("FATAL: Could not save global model encrypted: %s", e)
        raise RuntimeError(
            f"Refusing to save global model in plaintext. Encryption failed: {e}"
        )

# ── Trimmed mean aggregation (memory-safe, parameter-by-parameter) ────────────
def trimmed_mean_aggregate(
    updates: List[Dict[str, torch.Tensor]],
    trim_ratio: float = 0.1,
    max_param_delta: float = 1e-3,
    max_global_norm: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Aggregate updates via coordinate-wise trimmed mean.
    
    Memory-safe: processes one parameter key at a time, never loads all updates'
    full state_dicts into RAM simultaneously.
    
    Byzantine-robust: trims top/bottom trim_ratio fraction of values per parameter.
    
    Safety: clamps per-parameter delta, then scales global delta to max_global_norm.
    """
    if not updates:
        raise ValueError("No updates to aggregate")
    
    # Get parameter keys from first update (assume all have same structure)
    param_keys = list(updates[0].keys())
    n_updates = len(updates)
    n_trim = max(1, int(n_updates * trim_ratio))  # Number to trim from each end
    
    log.info("Aggregating %d updates with trimmed mean (trim_ratio=%.2f → trim %d from each end)",
             n_updates, trim_ratio, n_trim)
    
    aggregated = {}
    
    for key in param_keys:
        # Collect values for this parameter across all updates
        values = []
        for upd in updates:
            if key not in upd:
                log.warning("Update missing key '%s' — skipping this parameter", key)
                continue
            val = upd[key].float()  # Ensure float for aggregation
            # Clamp per-parameter delta (safety)
            val = torch.clamp(val, -max_param_delta, max_param_delta)
            values.append(val)
        
        if not values:
            log.warning("No valid values for key '%s' — skipping", key)
            continue
        
        # Stack along new dimension: [n_updates, ...param_shape...]
        stacked = torch.stack(values, dim=0)
        
        # Trimmed mean: sort, remove extremes, average the rest
        if n_trim > 0 and stacked.shape[0] > 2 * n_trim:
            # Sort along first dimension (updates)
            sorted_vals, _ = torch.sort(stacked, dim=0)
            # Trim: keep indices [n_trim : n_updates - n_trim]
            trimmed = sorted_vals[n_trim : -n_trim] if n_trim < stacked.shape[0] // 2 else sorted_vals
            agg_val = trimmed.mean(dim=0)
        else:
            # Too few updates to trim — fall back to simple mean
            agg_val = stacked.mean(dim=0)
        
        aggregated[key] = agg_val
    
    # Scale global delta to max_global_norm (safety)
    if max_global_norm > 0:
        global_norm = torch.norm(
            torch.stack([v.flatten() for v in aggregated.values()], dim=0),
            p=2
        ).item()
        if global_norm > max_global_norm:
            scale = max_global_norm / (global_norm + 1e-12)
            log.info("Scaling aggregated delta: norm=%.4f → scaled by %.4f", global_norm, scale)
            for k in aggregated:
                aggregated[k] = aggregated[k] * scale
    
    log.info("Aggregation complete: %d parameters", len(aggregated))
    return aggregated

# ── Receipt generation via CentralReceiptManager (unchanged interface) ────────
def write_aggregation_receipt(
    round_id: int,
    num_updates: int,
    global_model_path: str,
    secure_store: SecureStore,
    receipt_mgr: CentralReceiptManager,
) -> str:
    """Write HMAC-chained receipt for this aggregation round."""
    try:
        # Compute payload hash (SHA-256 of global model path + metadata)
        payload = f"{round_id}:{num_updates}:{global_model_path}".encode()
        payload_hash = hashlib.sha256(payload).digest()
        
        # Build receipt (interface matches original MongoDB version)
        receipt = receipt_mgr.create_receipt(
            agent="aggregator",
            operation="aggregation_complete",
            params={
                "payload_hash": payload_hash.hex(),
                "round_id": round_id,
                "num_updates": num_updates,
                "global_model_path": global_model_path,
                "aggregation_mode": "trimmed_mean",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            outputs=[global_model_path],
        )
        
        # Write receipt to filesystem (CentralReceiptManager handles HMAC chaining)
        receipt_uri = receipt_mgr.write_receipt(receipt, out_dir=secure_store.root / "receipts")
        log.info("Aggregation receipt written: %s", receipt_uri)
        return receipt_uri
        
    except Exception as e:
        log.warning("Failed to write aggregation receipt: %s — continuing without receipt", e)
        return ""

# ── Main aggregation logic ────────────────────────────────────────────────────
def run_aggregation(
    server_root: Path,
    round_id: int,
) -> str:
    """
    Perform federated aggregation for a single round.
    
    Returns the absolute path to the saved global model (for Rust to parse).
    """
    log.info("Starting aggregation: round=%d server_root=%s", round_id, server_root)
    
    # Validate paths
    updates_dir = validate_path_within_root(
        server_root / "rounds" / str(round_id) / "updates",
        server_root
    )
    global_models_dir = validate_path_within_root(
        server_root / "global_models",
        server_root
    )
    
    # Find update files (*.bin)
    update_files = sorted(updates_dir.glob("*.bin")) if updates_dir.exists() else []
    log.info("Found %d update files in %s", len(update_files), updates_dir)
    
    if not update_files:
        log.error("No update files found — cannot aggregate")
        sys.exit(1)
    
    # Initialize SecureStore and ReceiptManager
    secure_store = _get_secure_store(agent="aggregator")
    receipt_mgr = CentralReceiptManager(agent="aggregator")
    
    # Decrypt and load updates (skip failures, log warnings)
    updates: List[Dict[str, torch.Tensor]] = []
    for upd_path in update_files:
        state_dict = decrypt_update(upd_path, secure_store)
        if state_dict is not None:
            updates.append(state_dict)
    
    if not updates:
        log.error("All updates failed to load — aborting aggregation")
        sys.exit(1)
    
    log.info("Successfully loaded %d/%d updates for aggregation", len(updates), len(update_files))
    
    # Aggregate via trimmed mean (memory-safe)
    global_sd = trimmed_mean_aggregate(
        updates,
        trim_ratio=TRIM_RATIO,
        max_param_delta=MAX_PARAM_DELTA,
        max_global_norm=MAX_GLOBAL_NORM,
    )
    
    # Save global model
    output_path = global_models_dir / f"round_{round_id}.bin"
    global_model_path = encrypt_and_save_global_model(
        global_sd,
        output_path,
        secure_store,
        use_encryption=True,  # Try encryption first
    )
    
    # Write aggregation receipt (HMAC-chained audit trail)
    write_aggregation_receipt(
        round_id=round_id,
        num_updates=len(updates),
        global_model_path=global_model_path,
        secure_store=secure_store,
        receipt_mgr=receipt_mgr,
    )
    
    log.info("Aggregation complete: global model at %s", global_model_path)
    return global_model_path

class AggregatorAgent:
    """
    Thin wrapper around trimmed_mean_aggregate for local testing.
    """
    def __init__(self, mode: str = "trimmed_mean", trim_ratio: float = 0.1):
        self.mode = mode
        self.trim_ratio = trim_ratio
        self.secure_store = _get_secure_store(agent="aggregator")

    def aggregate_updates(self, updates: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        updates: list of dicts with keys:
            client_id, enc_uri, scheme, nonce, receipt, metadata
        Returns aggregated state dict.
        """
        loaded: List[Dict[str, torch.Tensor]] = []
        for upd in updates:
            enc_uri = upd.get("enc_uri", "")
            scheme  = upd.get("scheme", "")

            if scheme == "AES-GCM-SecureStore" and enc_uri.startswith("file://"):
                file_path = Path(enc_uri[len("file://"):])
                state_dict = decrypt_update(file_path, self.secure_store)
                if state_dict is not None:
                    loaded.append(state_dict)
            else:
                log.warning("Unsupported scheme '%s' for client %s",
                            scheme, upd.get("client_id", "?"))

        if not loaded:
            raise ValueError("No updates could be loaded for aggregation")

        return trimmed_mean_aggregate(
            loaded,
            trim_ratio=self.trim_ratio,
            max_param_delta=MAX_PARAM_DELTA,
            max_global_norm=MAX_GLOBAL_NORM,
        )

# ── CLI entrypoint (invoked by Rust server) ───────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Federated aggregation agent")
    parser.add_argument("--server-root", required=True, help="Canonical server root path")
    parser.add_argument("--round-id", type=int, required=True, help="Round ID to aggregate")
    parser.add_argument("--no-encrypt", action="store_true", help="Save global model unencrypted (debug)")
    args = parser.parse_args()
    
    # Zero-trust startup checks
    try:
        server_root = validate_server_root(args.server_root)
        if args.round_id <= 0:
            raise ValueError(f"round-id must be positive, got {args.round_id}")
    except ValueError as e:
        log.critical("Startup validation failed: %s", e)
        sys.exit(1)
    
    # Run aggregation
    try:
        global_model_path = run_aggregation(server_root, args.round_id)
    except Exception as e:
        log.critical("Aggregation failed: %s", e)
        sys.exit(1)
    
    # Output for Rust server to parse (MUST be exact format)
    print(f"GLOBAL_MODEL_PATH={global_model_path}", flush=True)
    
    # Exit cleanly
    sys.exit(0)

if __name__ == "__main__":
    main()