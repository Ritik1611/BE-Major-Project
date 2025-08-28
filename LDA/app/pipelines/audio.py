# app/pipelines/audio.py
"""
Audio pipeline that computes a single wav2vec2 embedding per audio file.

Behavior:
 - Loads file (librosa preferred, wave fallback).
 - Optionally splits long audio into chunks to avoid OOM.
 - Uses HuggingFace Wav2Vec2Processor + Wav2Vec2Model to compute last_hidden_state,
   then aggregates (mean pool) into a fixed-size vector.
 - Returns a single row (list with one dict) matching the project's expected schema.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import wave
import contextlib
import logging
import math

from app.utils.receipts import ReceiptManager

log = logging.getLogger(__name__)

# optional heavy libs
try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    librosa = None
    _HAS_LIBROSA = False

try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    import torch
    _HAS_TRANSFORMERS = True
except Exception:
    Wav2Vec2Processor = None
    Wav2Vec2Model = None
    torch = None
    _HAS_TRANSFORMERS = False

# Global cached model/processor to avoid reloading for each file
_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def _wav_duration(path: str) -> float:
    """Return duration in seconds; prefer wave module (fast)."""
    try:
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        # fallback to librosa if available
        if _HAS_LIBROSA:
            try:
                y, sr = librosa.load(path, sr=None, mono=True)
                return len(y) / float(sr)
            except Exception:
                pass
    return 0.0


def _safe_load_audio(path: str, sr: int = 16000):
    """
    Return tuple (y: numpy.ndarray, sr: int)
    Prefer librosa.load; otherwise use wave + manual decoding (short files).
    """
    if _HAS_LIBROSA:
        try:
            y, s = librosa.load(path, sr=sr, mono=True)
            return y, s
        except Exception as e:
            log.debug("librosa.load failed, falling back to wave: %s", e)

    # fallback -> read raw frames and normalize to float32 in [-1,1]
    try:
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            s = wf.getframerate()
            nchan = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        if sampwidth == 2:
            import numpy as _np_local
            fmt = "<{}h".format(int(len(frames) / 2))
            import struct
            samples = struct.unpack(fmt, frames)
            arr = _np_local.array(samples, dtype=_np_local.float32)
            if nchan > 1:
                arr = arr.reshape(-1, nchan).mean(axis=1)
            maxv = float(max(1.0, _np_local.max(_np_local.abs(arr))))
            arr = arr / maxv
            # if requested sr differs, we do not resample here; prefer librosa for resampling
            return arr, s
    except Exception as e:
        log.debug("wave fallback load failed: %s", e)

    return None, None


def _ensure_model(model_name: str, device: Optional[str] = None):
    """
    Load processor + model once and cache in _MODEL_CACHE.
    device: "cpu" or "cuda" (if available). If None, chooses cuda if available else cpu.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]["processor"], _MODEL_CACHE[model_name]["model"], _MODEL_CACHE[model_name]["device"]

    if not _HAS_TRANSFORMERS:
        raise RuntimeError("transformers/torch not available. Install 'transformers' and 'torch' to extract wav2vec2 embeddings.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)
        model.to(device)
        model.eval()
    except Exception as e:
        # surface loader error
        raise RuntimeError(f"Failed to load wav2vec2 model '{model_name}': {e}") from e

    _MODEL_CACHE[model_name] = {"processor": processor, "model": model, "device": device}
    return processor, model, device


def _chunk_audio(y, sr: int, chunk_s: float):
    """
    Yield (start_frame, end_frame, chunk_array) for non-overlapping chunks of chunk_s seconds.
    """
    if chunk_s <= 0:
        yield 0, len(y), y
        return
    chunk_frames = int(chunk_s * sr)
    total = len(y)
    start = 0
    while start < total:
        end = min(total, start + chunk_frames)
        yield start, end, y[start:end]
        start = end


def _compute_wav2vec2_embedding_for_waveform(y, sr: int, model_name: str = "facebook/wav2vec2-base-960h",
                                             pool: str = "mean", device: Optional[str] = None,
                                             chunk_seconds: float = 20.0) -> List[float]:
    """
    Compute a single fixed-length embedding (list[float]) for waveform y at sample rate sr.
    Strategy:
      - Split into chunks of chunk_seconds to avoid OOM.
      - For each chunk: tokenize with processor -> model -> take last_hidden_state and mean-pool over time -> vector
      - Aggregate chunk vectors by mean to produce final single vector.
    pool: "mean" (default) or "first" (take first token)
    """
    if not _HAS_TRANSFORMERS:
        raise RuntimeError("transformers/torch not available.")

    processor, model, model_device = _ensure_model(model_name, device=device or None)

    all_chunk_vecs = []
    # iterate chunks
    for start_frame, end_frame, chunk in _chunk_audio(y, sr, chunk_seconds):
        if len(chunk) == 0:
            continue
        try:
            # processor accepts numpy array and sampling_rate
            inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
            # move to model device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                last_hidden = outputs.last_hidden_state  # shape (batch, time, dim)
                # take mean over time dimension
                if pool == "mean":
                    vec = last_hidden.mean(dim=1).squeeze(0).cpu().numpy()
                else:
                    # fallback to take first timestep
                    vec = last_hidden[:, 0, :].squeeze(0).cpu().numpy()
            all_chunk_vecs.append(vec)
        except RuntimeError as e:
            # common cause: CUDA OOM or CPU memory; re-raise with hint
            log.exception("Runtime error computing chunk embedding (may be OOM): %s", e)
            raise

    if not all_chunk_vecs:
        raise RuntimeError("No chunk vectors computed for audio (empty input?)")

    # aggregate chunk vectors by mean
    import numpy as _np_local
    stacked = _np_local.stack(all_chunk_vecs, axis=0)  # (n_chunks, dim)
    final = _np_local.mean(stacked, axis=0)
    # convert to python floats list
    return [float(x) for x in final.tolist()]


def process_audio_file(p: Path, cfg: dict, session_id: str) -> List[Dict[str, Any]]:
    """
    Compute wav2vec2 embedding for a single file.

    cfg keys used:
      cfg["ingest"]["audio"]["features"]["wav2vec2"]["enabled"] : bool
      cfg["ingest"]["audio"]["features"]["wav2vec2"]["model"] : model name (HF id), default "facebook/wav2vec2-base-960h"
      cfg["ingest"]["audio"]["features"]["wav2vec2"]["pool"] : "mean" or "first"
      cfg["ingest"]["audio"]["features"]["wav2vec2"]["chunk_seconds"] : seconds per chunk (float), default 20.0
      cfg["ingest"]["audio"]["features"]["wav2vec2"]["device"] : optional "cpu" or "cuda"
    """
    p = Path(p)
    out_rows: List[Dict[str, Any]] = []
    receipt_dir = cfg.get("ingest", {}).get("audio", {}).get("receipt_dir", "./receipts")
    receipts = ReceiptManager(receipt_dir)

    # Basic duration (fast)
    dur = _wav_duration(str(p))
    features: Dict[str, Any] = {"duration": dur}

    # If model disabled, return minimal row
    enabled = cfg.get("ingest", {}).get("audio", {}).get("features", {}).get("wav2vec2", {}).get("enabled", True)
    if not enabled:
        processed_uri = str(p)
        receipt_path = receipts.create_receipt(
            operation="audio_preprocessing",
            input_meta={"source": str(p), "duration": dur},
            output_uri=processed_uri
        )
        out_rows.append({
            "session_id": session_id,
            "modality": "audio",
            "source": str(p),
            "filename": p.name,
            "processed_uri": processed_uri,
            "receipt_path": receipt_path,
            "features": features
        })
        return out_rows

    # Load audio waveform (prefer 16k)
    target_sr = cfg.get("ingest", {}).get("audio", {}).get("sr", 16000)
    y, sr = _safe_load_audio(str(p), sr=target_sr)
    if y is None or sr is None:
        log.warning("Failed to load audio for wav2vec2 embedding: %s", p)
        # fallback to minimal row
        processed_uri = str(p)
        receipt_path = receipts.create_receipt(
            operation="audio_preprocessing",
            input_meta={"source": str(p), "duration": dur, "error": "load_failed"},
            output_uri=processed_uri
        )
        out_rows.append({
            "session_id": session_id,
            "modality": "audio",
            "source": str(p),
            "filename": p.name,
            "processed_uri": processed_uri,
            "receipt_path": receipt_path,
            "features": features
        })
        return out_rows

    # model config
    model_name = cfg.get("ingest", {}).get("audio", {}).get("features", {}).get("wav2vec2", {}).get("model", "facebook/wav2vec2-base-960h")
    pool = cfg.get("ingest", {}).get("audio", {}).get("features", {}).get("wav2vec2", {}).get("pool", "mean")
    chunk_seconds = float(cfg.get("ingest", {}).get("audio", {}).get("features", {}).get("wav2vec2", {}).get("chunk_seconds", 20.0))
    device = cfg.get("ingest", {}).get("audio", {}).get("features", {}).get("wav2vec2", {}).get("device", None)

    # compute embedding
    embedding = None
    try:
        embedding = _compute_wav2vec2_embedding_for_waveform(y, sr,
                                                             model_name=model_name,
                                                             pool=pool,
                                                             device=device,
                                                             chunk_seconds=chunk_seconds)
    except Exception as e:
        log.exception("wav2vec2 embedding computation failed for %s: %s", p, e)
        embedding = None

    if embedding is not None:
        features["w2v2_embedding"] = embedding
        features["w2v2_dim"] = len(embedding)

    # Save processed marker + receipt (unchanged behavior)
    processed_uri = str(p)
    receipt_path = receipts.create_receipt(
        operation="audio_preprocessing",
        input_meta={"source": str(p), "duration": dur, "w2v2_dim": features.get("w2v2_dim")},
        output_uri=processed_uri
    )

    row = {
        "session_id": session_id,
        "modality": "audio",
        "source": str(p),
        "filename": p.name,
        "processed_uri": processed_uri,
        "receipt_path": receipt_path,
        "features": features
    }

    out_rows.append(row)
    return out_rows
