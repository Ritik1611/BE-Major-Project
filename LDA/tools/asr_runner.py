#!/usr/bin/env python3
"""
tools/asr_runner.py

Lightweight CLI: run ASR on one or more audio inputs.

Supports:
 - backend=whisper  -> uses openai-whisper (load_model("small"/"base"/...))
 - backend=hf       -> uses transformers pipeline("automatic-speech-recognition", model=...)

Inputs:
 - Local audio files (wav/mp3)
 - file:// URIs pointing inside secure_store (e.g. file:///.../clips/audio/30_630.wav.enc)
   -> will decrypt using app.security.secure_store.SecureStore and run ASR on the decrypted bytes.

Usage examples:
  # whisper (default) on a local wav
  python tools/asr_runner.py --input /path/to/clip.wav --backend whisper --model small --out results.csv

  # whisper on encrypted clip inside secure_store
  python tools/asr_runner.py --input file:///home/me/project/secure_store/sess-.../clips/audio/30_630.wav.enc

  # HF pipeline using a wav2vec2 model
  python tools/asr_runner.py --input /path/to/clip.wav --backend hf --model facebook/wav2vec2-base-960h

  # multiple inputs
  python tools/asr_runner.py --input a.wav --input b.wav --out results.jsonl

Notes:
 - whisper backend requires `openai-whisper` package.
 - hf backend requires `transformers` and a torch backend.
 - For file:// URIs the script tries to find master.key by walking up parent directories.
"""

from __future__ import annotations
import argparse
import csv
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
import traceback
import wave
import contextlib

# Try to import SecureStore; if not available, we'll only support local files
try:
    from app.security.secure_store import SecureStore
    _HAS_SECURESTORE = True
except Exception:
    SecureStore = None
    _HAS_SECURESTORE = False

def find_secure_store_root(start_path: Path) -> Path:
    """
    Walk up from start_path to find a directory that contains 'master.key'.
    Returns the Path to that directory, or raises FileNotFoundError.
    """
    cur = start_path.resolve()
    if cur.is_file():
        cur = cur.parent
    for parent in [cur] + list(cur.parents):
        candidate = parent / "master.key"
        if candidate.exists():
            return parent
    raise FileNotFoundError(f"Could not find 'master.key' walking up from {start_path}.")

def decrypt_uri_to_bytes(uri: str) -> bytes:
    """
    If uri startswith file://, use SecureStore to decrypt and return plaintext bytes.
    Otherwise, treat uri as local path and return its bytes.
    """
    if uri.startswith("file://"):
        if not _HAS_SECURESTORE:
            raise RuntimeError("SecureStore import failed; cannot decrypt file:// URI. Make sure repo root is on PYTHONPATH.")
        # find root by walking up from the file location
        p = Path(uri[len("file://"):])
        root = find_secure_store_root(p)
        store = SecureStore(str(root))
        return store.decrypt_read(uri)
    else:
        p = Path(uri)
        if not p.exists():
            raise FileNotFoundError(f"Local file not found: {p}")
        return p.read_bytes()

def bytes_to_tempfile(bytes_blob: bytes, suffix: str = ".wav") -> Path:
    tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tf.write(bytes_blob)
    tf.flush()
    tf.close()
    return Path(tf.name)

def audio_duration_from_wav_path(wav_path: Path) -> float:
    try:
        with contextlib.closing(wave.open(str(wav_path), 'rb')) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        return 0.0

def run_whisper_on_bytes(bytes_blob: bytes, model_name: str = "small") -> Tuple[str, float]:
    try:
        import whisper
    except Exception as e:
        raise RuntimeError("Whisper backend requested but 'openai-whisper' package is not installed.") from e

    tmp = bytes_to_tempfile(bytes_blob, suffix=".wav")
    try:
        model = whisper.load_model(model_name)
        res = model.transcribe(str(tmp))
        text = res.get("text", "")
        conf = float(res.get("avg_logprob", 0.0)) if "avg_logprob" in res else 0.0
        return text, conf
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass

def run_hf_asr_on_bytes(bytes_blob: bytes, model_name: str = "facebook/wav2vec2-base-960h") -> Tuple[str, float]:
    """
    Use HuggingFace transformers pipeline for automatic-speech-recognition.
    The pipeline accepts a filename or array-like audio. We'll pass a temp file for simplicity.
    Returns (text, confidence) where confidence is aggregated if available.
    """
    try:
        from transformers import pipeline
    except Exception as e:
        raise RuntimeError("HF backend requested but 'transformers' is not installed.") from e

    tmp = bytes_to_tempfile(bytes_blob, suffix=".wav")
    try:
        asr = pipeline("automatic-speech-recognition", model=model_name, chunk_length_s=30)
        out = asr(str(tmp))
        # pipeline result may be {'text': '...', 'score': 0.95}
        text = out.get("text", "") if isinstance(out, dict) else str(out)
        # some pipelines produce 'chunks' with scores; try to estimate
        conf = 0.0
        if isinstance(out, dict):
            if "score" in out and out["score"] is not None:
                try:
                    conf = float(out["score"])
                except Exception:
                    conf = 0.0
            elif "chunks" in out and isinstance(out["chunks"], list) and out["chunks"]:
                # average chunk scores
                scores = [c.get("score", 0.0) for c in out["chunks"] if isinstance(c, dict)]
                if scores:
                    conf = float(sum(scores) / len(scores))
        return text, conf
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass

def run_single_input(uri: str, backend: str, model_name: str) -> Tuple[str, str, str, float]:
    """
    Returns a tuple: (uri, transcript, backend_info, confidence)
    backend_info is e.g. "whisper:small" or "hf:facebook/wav2vec2-base-960h"
    """
    try:
        bytes_blob = decrypt_uri_to_bytes(uri)
    except Exception as e:
        raise RuntimeError(f"Failed to read/decrypt input {uri}: {e}") from e

    # detect likely file type; whisper expects wav-like input; if we have a parquet, abort
    # We'll try to treat whatever bytes as an audio file for ASR.
    backend_info = f"{backend}:{model_name}"
    if backend == "whisper":
        text, conf = run_whisper_on_bytes(bytes_blob, model_name=model_name)
        return uri, text, backend_info, conf
    elif backend == "hf":
        text, conf = run_hf_asr_on_bytes(bytes_blob, model_name=model_name)
        return uri, text, backend_info, conf
    else:
        raise ValueError(f"Unknown backend '{backend}'")

def write_results_csv(rows: List[Tuple[str,str,str,float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input_uri", "transcript", "backend_model", "confidence"])
        for uri, text, backend_info, conf in rows:
            writer.writerow([uri, text, backend_info, conf])

def write_results_jsonl(rows: List[Tuple[str,str,str,float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for uri, text, backend_info, conf in rows:
            f.write(json.dumps({
                "input_uri": uri,
                "transcript": text,
                "backend_model": backend_info,
                "confidence": conf
            }, ensure_ascii=False) + "\n")

def parse_args():
    p = argparse.ArgumentParser(prog="asr_runner", description="Run ASR (Whisper or HF) on audio inputs (local files or file:// secure_store URIs).")
    p.add_argument("--input", "-i", action="append", required=True, help="Input audio path or file:// URI. Can be specified multiple times.")
    p.add_argument("--backend", "-b", choices=["whisper", "hf"], default="whisper", help="ASR backend to use.")
    p.add_argument("--model", "-m", default=None, help="Model name for backend. For whisper: tiny|base|small|medium|large (default small). For hf: HF model id (default facebook/wav2vec2-base-960h).")
    p.add_argument("--out", "-o", default=None, help="Output path (CSV or .jsonl). Defaults to asr_results.csv in cwd.")
    p.add_argument("--format", choices=["csv", "jsonl"], default="csv", help="Output format.")
    return p.parse_args()

def main():
    args = parse_args()
    backend = args.backend
    model = args.model
    if model is None:
        model = "small" if backend == "whisper" else "facebook/wav2vec2-base-960h"

    inputs = args.input
    results = []

    for uri in inputs:
        try:
            print(f"[ASR] Processing: {uri} (backend={backend}, model={model})")
            uri_res, text, backend_info, conf = run_single_input(uri, backend, model)
            results.append((uri_res, text, backend_info, conf))
            print(f"[ASR] Done: {uri} -> {len(text)} chars, conf={conf}")
        except Exception as e:
            print(f"[ASR] ERROR for {uri}: {e}", file=sys.stderr)
            traceback.print_exc()
            results.append((uri, "", f"{backend}:{model}", 0.0))

    out_path = Path(args.out) if args.out else Path.cwd() / "asr_results.csv"
    try:
        if args.format == "csv":
            write_results_csv(results, out_path)
        else:
            write_results_jsonl(results, out_path)
        print(f"[ASR] Results written to {out_path}")
    except Exception as e:
        print(f"[ASR] Failed to write results: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()
