import os
import subprocess
import tempfile
import json
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Dict, Any, List

from centralized_secure_store import SecureStore
from centralised_receipts import CentralReceiptManager


def _extract_wav2vec2_features(wav_path: str, model_id: str, pool: str = "mean") -> Dict[str, Any]:
    """
    Extract wav2vec2 embeddings using HuggingFace transformers.
    Returns mean/pooled representation.
    """
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
    except ImportError:
        print("⚠️ transformers not installed; skipping wav2vec2 features.")
        return {}

    print(f"🔍 Extracting wav2vec2 features from {wav_path} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2Model.from_pretrained(model_id).to(device)

    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values).last_hidden_state.cpu().numpy()

    if pool == "mean":
        pooled = np.mean(outputs, axis=1).squeeze()
    elif pool == "max":
        pooled = np.max(outputs, axis=1).squeeze()
    else:
        pooled = outputs.squeeze()

    return {"wav2vec2": pooled.tolist()}


def _extract_opensmile_features(
    wav_path: str,
    opensmile_bin: str,
    opensmile_config: str
) -> Dict[str, Any]:
    """
    Run openSMILE (eGeMAPS) via subprocess and parse output CSV.
    """
    if not os.path.exists(opensmile_bin):
        print(f"⚠️ openSMILE binary not found at {opensmile_bin}. Skipping eGeMAPS.")
        return {}
    if not os.path.exists(opensmile_config):
        print(f"⚠️ openSMILE config not found at {opensmile_config}. Skipping eGeMAPS.")
        return {}

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmpout:
        tmp_csv = tmpout.name

    cmd = [
        opensmile_bin,
        "-C", opensmile_config,
        "-I", wav_path,
        "-O", tmp_csv,
        "-nologfile",
        "-noconsoleoutput"
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data = np.genfromtxt(tmp_csv, delimiter=';', names=True, dtype=None, encoding='utf-8')
        if data.ndim == 0:  # single row
            feature_dict = {name: float(data[name]) for name in data.dtype.names}
        else:
            feature_dict = {name: float(np.mean(data[name])) for name in data.dtype.names}
    except Exception as e:
        print(f"⚠️ openSMILE extraction failed: {e}")
        feature_dict = {}
    finally:
        try:
            os.remove(tmp_csv)
        except OSError:
            pass

    return {"egemaps": feature_dict}


def _compute_basic_prosody(wav_path: str) -> Dict[str, Any]:
    """
    Compute basic energy, pitch, and zero-crossing features using torchaudio.
    """
    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.mean(dim=0, keepdim=True)  # mono
    energy = torch.mean(waveform ** 2).item()
    zcr = torch.mean((waveform[:, 1:] * waveform[:, :-1]) < 0).item()

    try:
        pitch = torchaudio.functional.detect_pitch_frequency(waveform, sample_rate=sr)
        pitch_mean = torch.mean(pitch[pitch > 0]).item()
    except Exception:
        pitch_mean = 0.0

    return {"energy": energy, "pitch_mean": pitch_mean, "zcr": zcr}


def process_audio_file(
    audio_path: str,
    cfg: Dict[str, Any],
    session_id: str
) -> List[Dict[str, Any]]:
    """
    Extracts multimodal audio features (prosody + wav2vec2 + eGeMAPS).
    Writes encrypted outputs and returns a structured rows list.
    """

    audio_path = str(audio_path)
    print(f"🎧 Processing audio: {audio_path}")
    rows = []

    storage = SecureStore(agent="lda-audio", root=cfg["storage"]["root"])
    rm = CentralReceiptManager(agent="lda-audio")

    features_cfg = cfg["audio_pipe"]["features"]
    derived = {}

    # ---- Prosody ----
    if features_cfg.get("prosody", False):
        derived.update(_compute_basic_prosody(audio_path))

    # ---- eGeMAPS (openSMILE) ----
    egemaps_cfg = features_cfg.get("egemaps", {})
    if egemaps_cfg.get("enabled", False):
        opensmile_bin = egemaps_cfg.get("opensmile_binary")
        opensmile_conf = egemaps_cfg.get("opensmile_config")
        derived.update(_extract_opensmile_features(audio_path, opensmile_bin, opensmile_conf))

    # ---- wav2vec2 ----
    wav2vec_cfg = features_cfg.get("wav2vec2", {})
    if wav2vec_cfg.get("enabled", False):
        derived.update(
            _extract_wav2vec2_features(
                audio_path,
                model_id=wav2vec_cfg.get("model", "facebook/wav2vec2-base-960h"),
                pool=wav2vec_cfg.get("pool", "mean"),
            )
        )

    # ---- Package + Secure Write ----
    record = {
        "session_id": session_id,
        "path": audio_path,
        "features": {"audio": derived},
        "derived": {"num_features": len(derived)},
    }

    fname = Path(audio_path).stem + "_audio_features.json"
    uri = storage.encrypt_write(f"file://{storage.root / session_id / 'audio' / fname}", json.dumps(record).encode())

    receipt = rm.create_receipt(
        session_id=session_id,
        operation="audio_process",
        params={"input": audio_path},
        outputs=[uri],
    )
    receipt_uri = storage.encrypt_write(
        f"file://{storage.root / session_id / 'receipts' / (Path(audio_path).stem + '_audio.json.enc')}",
        json.dumps(receipt).encode(),
    )

    rows.append({"uri": uri, "receipt_uri": receipt_uri, **record})
    print(f"✅ Audio processed and stored for {audio_path}")

    return rows
