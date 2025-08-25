# app/pipelines/audio.py
from pathlib import Path
from typing import List, Dict, Any
import wave
import contextlib
import hashlib
from app.utils.receipts import ReceiptManager

try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False

def _wav_duration(wav_path: str) -> float:
    try:
        with contextlib.closing(wave.open(wav_path, 'rb')) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        if _HAS_LIBROSA:
            y, sr = librosa.load(wav_path, sr=None)
            return len(y) / float(sr)
        return 0.0

def process_audio_file(p: Path, cfg: dict, session_id: str) -> List[Dict[str, Any]]:
    """
    Simple audio processing adapter. Returns a list of rows (one per file) with metadata and basic features.
    """
    p = Path(p)
    out_rows = []
    receipt_dir = cfg.get("ingest", {}).get("audio", {}).get("receipt_dir", "./receipts")
    receipts = ReceiptManager(receipt_dir)

    dur = _wav_duration(str(p))
    features = {"duration": dur}

    # Try basic librosa features if available
    if _HAS_LIBROSA:
        try:
            y, sr = librosa.load(str(p), sr=cfg.get("ingest", {}).get("audio", {}).get("sr", 16000), mono=True)
            import numpy as np
            features["rms_mean"] = float((y ** 2).mean())
            features["zero_crossing_rate"] = float((librosa.feature.zero_crossing_rate(y).mean()))
        except Exception:
            pass

    # Save a trivial processed marker (we don't rewrite the file)
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
