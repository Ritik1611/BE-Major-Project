# app/pipelines/audio.py
import wave
import contextlib
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
try:
    import librosa
except Exception:
    librosa = None

from app.utils.receipts import ReceiptManager

def _read_basic_wave_info(path: str) -> Dict[str, Any]:
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
    return {"duration": duration, "sample_rate": rate, "channels": channels, "sampwidth": sampwidth}

def _rms_from_wave(path: str) -> float:
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        frames = wf.readframes(wf.getnframes())
        sampwidth = wf.getsampwidth()
        dtype = None
        if sampwidth == 2:
            dtype = np.int16
        elif sampwidth == 4:
            dtype = np.int32
        elif sampwidth == 1:
            dtype = np.uint8
        else:
            return 0.0
        arr = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if arr.size == 0:
            return 0.0
        maxval = float(np.iinfo(dtype).max)
        arr = arr / maxval
        return float(np.sqrt(np.mean(arr ** 2)))

def process_audio_file(audio_path: str, cfg: dict, session_id: str) -> List[Dict[str, Any]]:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    rows = []
    receipt_dir = cfg.get("ingest", {}).get("audio", {}).get("receipt_dir", "./receipts")
    receipt_mgr = ReceiptManager(receipt_dir)

    info = _read_basic_wave_info(str(path))
    duration = info["duration"]
    sample_rate = info["sample_rate"]

    features = {
        "session_id": session_id,
        "source": str(path),
        "filename": path.name,
        "duration": duration,
        "sample_rate": sample_rate,
        "channels": info.get("channels"),
    }

    try:
        if librosa is not None:
            y, sr = librosa.load(str(path), sr=None, mono=True)
            rms = float(np.mean(librosa.feature.rms(y=y).flatten()))
            spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).flatten()))
            features.update({"rms": rms, "spectral_centroid_mean": spec_cent})
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = [float(np.mean(mfcc[i])) for i in range(min(13, mfcc.shape[0]))]
            for i, v in enumerate(mfcc_means):
                features[f"mfcc_{i+1}_mean"] = v
        else:
            features["rms"] = _rms_from_wave(str(path))
    except Exception as e:
        features["feature_error"] = str(e)
        if "rms" not in features:
            try:
                features["rms"] = _rms_from_wave(str(path))
            except Exception:
                features["rms"] = 0.0

    # create a receipt for this processing step
    try:
        receipt_path = receipt_mgr.create_receipt(
            operation="audio_preprocessing",
            input_meta={"source": str(path), "duration": duration, "sample_rate": sample_rate},
            output_uri=f"local://{session_id}/audio/{path.name}"
        )
        features["receipt_path"] = receipt_path
    except Exception:
        features["receipt_path"] = None

    rows.append(features)
    return rows
