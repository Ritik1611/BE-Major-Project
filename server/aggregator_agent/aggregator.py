"""
aggregator.py

SECURITY FIXES:
  FIX-AGG-1: SecureStore root is canonical (~/.federated/data/secure_store)
  FIX-AGG-2: enc_uri validated; arbitrary client paths rejected
  FIX-AGG-3: GridFS mode fetches bytes from MongoDB by ObjectId

ARCHITECTURE FIXES (this revision):
  FIX-AGG-4: aggregate_updates() now returns BOTH a flat numpy array (for
             backward compat) AND a reconstructed state_dict keyed by
             parameter name. Previously the aggregator flattened every update
             to a 1-D array and threw away parameter names, making it
             impossible for clients to warm-start MentalBERT from the
             aggregated model.

  FIX-AGG-5: run_job() writes the aggregated state_dict to MongoDB GridFS
             and upserts a document into the `global_models` collection so
             that GetRound.global_model_available returns true and clients
             can call DownloadGlobalModel.  Previously the `global_models`
             collection was NEVER written to, so every client always
             cold-started MentalBERT from scratch.

  FIX-AGG-6: The aggregated .pt file (proper torch state dict) is now saved
             alongside the legacy .npy for backward compatibility.
"""

import os
import io
import json
import hashlib
import logging
import numpy as np
import torch
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

# Canonical paths — must match _CANONICAL_ROOT in centralized_secure_store.py
_FEDERATED_BASE  = Path.home() / ".federated"
_CANONICAL_ROOT  = _FEDERATED_BASE / "data" / "secure_store"

# MongoDB connection string — set via environment variable, never hardcoded
_MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")

log = logging.getLogger(__name__)


class AggregatorAgent:
    """
    AggregatorAgent
    ---------------
    Receives encrypted DP-updates (via GridFS ObjectId or canonical path),
    decrypts them using the shared SecureStore, performs robust aggregation,
    and stores the resulting global model in MongoDB GridFS so clients can
    download it via DownloadGlobalModel.

    Aggregation modes: mean | trimmed_mean | coordinate_median
    """

    def __init__(
        self,
        mode: str = "trimmed_mean",
        trim_ratio: float = 0.1,
        decrypt_callback=None,
    ):
        self.mode        = mode
        self.trim_ratio  = trim_ratio
        self._decrypt_cb = decrypt_callback or self._default_decrypt

    # ─────────────────────────────────────────────────────────────────────────
    # Decryption helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _default_decrypt(
        self,
        gridfs_id: Optional[str],
        enc_path:  Optional[str],
        scheme:    str,
        nonce:     Optional[str],
    ) -> tuple:
        """
        Decrypt an update.

        Returns (flat_array: np.ndarray, state_dict: Optional[dict])
        The state_dict preserves parameter names and shapes so the aggregated
        result can be used as a proper PyTorch model warm-start.
        """
        if gridfs_id:
            return self._decrypt_from_gridfs(gridfs_id)
        if enc_path:
            return self._decrypt_from_store(enc_path, scheme)
        raise ValueError("Neither gridfs_id nor enc_path provided")

    def _bytes_to_flat_and_dict(self, raw: bytes) -> tuple:
        """
        Load raw bytes (AES-GCM decrypted) into (flat_array, state_dict).

        The client serialises the local delta as a torch.save() state dict.
        We return both the flat concatenation (for aggregation) and the
        original structure (so we can reconstruct parameter names after
        aggregation — see FIX-AGG-4).
        """
        buf = io.BytesIO(raw)
        try:
            obj = torch.load(buf, map_location="cpu", weights_only=False)
        except Exception:
            flat = np.frombuffer(raw, dtype=np.float32)
            return flat, None

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().float().numpy().flatten(), None

        if isinstance(obj, dict):
            # State dict — return structured AND flat representations
            parts = []
            state_dict = {}
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    t = v.detach().cpu().float()
                    parts.append(t.flatten().numpy())
                    state_dict[k] = t
                else:
                    print("Values dropped...")
            flat = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
            return flat, state_dict if state_dict else None

        raise TypeError(f"Unexpected object type: {type(obj)}")

    def _decrypt_from_gridfs(self, gridfs_id: str) -> tuple:
        """Fetch bytes from MongoDB GridFS, return (flat, state_dict)."""
        from pymongo import MongoClient
        from bson.objectid import ObjectId
        import gridfs

        client = MongoClient(_MONGO_URI)
        db     = client["federated"]
        fs     = gridfs.GridFS(db)

        try:
            oid = ObjectId(gridfs_id)
        except Exception:
            raise ValueError(
                f"Invalid GridFS ObjectId: {gridfs_id!r}\n"
                "enc_handle must be the server_handle from UploadAck."
            )

        grid_out = fs.get(oid)
        raw      = grid_out.read()
        client.close()

        return self._bytes_to_flat_and_dict(raw)

    def _decrypt_from_store(self, enc_path: str, scheme: str) -> tuple:
        """Decrypt from canonical SecureStore, return (flat, state_dict)."""
        from server.aggregator_agent.core.centralized_secure_store import SecureStore

        if enc_path.startswith("file://"):
            enc_path = enc_path[len("file://"):]

        resolved = Path(enc_path).resolve()
        canonical = _CANONICAL_ROOT.resolve()
        if not str(resolved).startswith(str(canonical)):
            raise ValueError(
                f"Path traversal rejected: {resolved}\n"
                f"Paths must be inside {canonical}"
            )

        store = SecureStore(agent="aggregator", root=_CANONICAL_ROOT)

        if scheme.lower().startswith(("aes", "kms")):
            raw = store.decrypt_read("file://" + str(resolved))
        else:
            raise ValueError(f"Unsupported scheme for path-based decrypt: {scheme}")

        return self._bytes_to_flat_and_dict(raw)

    # ─────────────────────────────────────────────────────────────────────────
    # Aggregation
    # ─────────────────────────────────────────────────────────────────────────

    def aggregate_updates(self, updates: List[Dict]) -> tuple:
        """
        Decrypt and aggregate all updates.

        Returns:
            (aggregated_flat: np.ndarray,
             aggregated_state_dict: Optional[dict])

        aggregated_state_dict is the reconstructed named-parameter dict;
        it is None only if every update lacked structural information (i.e.
        raw tensors rather than state dicts were uploaded).
        """
        flat_list  = []
        dict_list  = []   # list of {param_name: tensor}

        for u in updates:
            gridfs_id = u.get("gridfs_id")
            enc_path  = u.get("enc_uri")
            scheme    = u.get("scheme", "AES-GCM-SecureStore")
            nonce     = u.get("nonce")

            flat, sd = self._decrypt_cb(gridfs_id, enc_path, scheme, nonce)

            if isinstance(flat, torch.Tensor):
                flat = flat.detach().cpu().numpy()

            flat_list.append(flat.astype(np.float32))
            if sd is not None:
                dict_list.append(sd)

        # Ensure all flat vectors have the same length
        shapes = [d.shape for d in flat_list]
        if len(set(s[0] for s in shapes)) > 1:
            min_len = min(d.size for d in flat_list)
            flat_list = [d.flatten()[:min_len] for d in flat_list]

        stacked = np.stack([d.flatten() for d in flat_list], axis=0)
        agg_flat = self._apply_aggregation(stacked)

        # FIX-AGG-4: Reconstruct state dict from aggregated flat array
        agg_dict = None
        if dict_list:
            agg_dict = self._reconstruct_state_dict(agg_flat, dict_list[0])

        return agg_flat, agg_dict

    def _reconstruct_state_dict(self,
                                 flat: np.ndarray,
                                 template: dict) -> dict:
        """
        Rebuild a {name: tensor} state dict from the aggregated flat array
        using one of the original updates as a structural template.

        This is the key fix (FIX-AGG-4): it allows the aggregated model to
        be loaded back into MentalBERT via model.load_state_dict().
        """
        result = {}
        offset = 0
        for name, ref_tensor in template.items():
            n = ref_tensor.numel()
            if offset + n > flat.size:
                log.warning("Flat array shorter than template at '%s' — padding",
                            name)
                chunk = np.zeros(n, dtype=np.float32)
                available = flat.size - offset
                if available > 0:
                    chunk[:available] = flat[offset:offset + available]
            else:
                chunk = flat[offset: offset + n]
            result[name] = torch.tensor(
                chunk, dtype=ref_tensor.dtype
            ).view(ref_tensor.shape)
            offset += n
        return result

    def _apply_aggregation(self, arr: np.ndarray) -> np.ndarray:
        if self.mode == "mean":
            return np.mean(arr, axis=0)

        elif self.mode == "trimmed_mean":
            n     = arr.shape[0]
            lower = max(1, int(self.trim_ratio * n))
            upper = n - lower
            if lower >= upper:
                raise ValueError(
                    f"trim_ratio {self.trim_ratio} too large for {n} updates"
                )
            sorted_arr = np.sort(arr, axis=0)
            return np.mean(sorted_arr[lower:upper], axis=0)

        elif self.mode in ("median", "coordinate_median"):
            return np.median(arr, axis=0)

        else:
            raise NotImplementedError(f"Unknown mode: {self.mode}")

    # ─────────────────────────────────────────────────────────────────────────
    # MongoDB global model storage (FIX-AGG-5)
    # ─────────────────────────────────────────────────────────────────────────

    def _write_global_model_to_mongodb(
        self,
        round_id: int,
        state_dict: dict,
        aggregation_mode: str,
    ) -> Optional[str]:
        """
        FIX-AGG-5: Store the aggregated model in MongoDB GridFS and
        upsert a document into the `global_models` collection.

        Previously this was never done, so GetRound always returned
        global_model_available=false and clients could never warm-start
        from the aggregated model.

        Returns the GridFS ObjectId hex string, or None on failure.
        """
        try:
            from pymongo import MongoClient
            from bson.objectid import ObjectId
            import gridfs

            client = MongoClient(_MONGO_URI)
            db     = client["federated"]
            fs     = gridfs.GridFS(db)

            # Serialise state dict as a torch .pt file
            buf = io.BytesIO()
            torch.save(state_dict, buf)
            model_bytes = buf.getvalue()

            model_hash = hashlib.sha256(model_bytes).hexdigest()

            filename = f"global_model_round_{round_id}.pt"
            file_id  = fs.put(model_bytes, filename=filename)

            db["global_models"].update_one(
                {"round_id": round_id},
                {
                    "$set": {
                        "round_id":         round_id,
                        "file_id":          file_id,
                        "model_hash":       model_hash,
                        "aggregation_mode": aggregation_mode,
                        "created_at":       datetime.now(timezone.utc),
                        "size_bytes":       len(model_bytes),
                    }
                },
                upsert=True,
            )
            client.close()

            log.info(
                "Global model for round %d stored in MongoDB "
                "(file_id=%s, %d bytes, hash=%s…)",
                round_id, file_id, len(model_bytes), model_hash[:16]
            )
            return file_id.binary.hex()

        except Exception as e:
            log.error("Failed to write global model to MongoDB: %s", e)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run_job(self, job: Dict) -> Dict:
        self.mode       = job.get("mode",       self.mode)
        self.trim_ratio = job.get("trim_ratio", self.trim_ratio)

        # FIX-AGG-4/5: aggregate returns both flat array and state dict
        agg_flat, agg_dict = self.aggregate_updates(job["updates"])

        # ── Legacy .npy (backward compat) ────────────────────────────────
        npy_path = f"./aggregated_round_{job['round_id']}.npy"
        np.save(npy_path, agg_flat)

        # ── FIX-AGG-6: also save proper .pt state dict ────────────────────
        pt_path = f"./aggregated_round_{job['round_id']}.pt"
        if agg_dict is not None:
            torch.save(agg_dict, pt_path)
        else:
            # Fall back: save flat tensor with a generic key
            torch.save({"aggregated": torch.tensor(agg_flat)}, pt_path)

        # ── FIX-AGG-5: write to MongoDB global_models collection ──────────
        gridfs_model_id = None
        if agg_dict is not None:
            gridfs_model_id = self._write_global_model_to_mongodb(
                round_id=job["round_id"],
                state_dict=agg_dict,
                aggregation_mode=self.mode,
            )
        else:
            log.warning(
                "No structured state dict available for round %d — "
                "global_models collection NOT updated. "
                "Clients will continue to cold-start.",
                job["round_id"],
            )

        return {
            "round_id":          job["round_id"],
            "aggregated_uri":    "file://" + os.path.abspath(npy_path),
            "aggregated_pt_uri": "file://" + os.path.abspath(pt_path),
            "gridfs_model_id":   gridfs_model_id,   # new field for server.rs
            "num_updates":       len(job["updates"]),
            "mode":              self.mode,
        }


if __name__ == "__main__":
    import sys
    job    = json.load(sys.stdin)
    agent  = AggregatorAgent(
        mode=job.get("mode", "trimmed_mean"),
        trim_ratio=job.get("trim_ratio", 0.1),
    )
    result = agent.run_job(job)
    print(json.dumps(result))