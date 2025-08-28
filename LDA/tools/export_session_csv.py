#!/usr/bin/env python3
"""
tools/export_session_csv.py

Usage:
  python tools/export_session_csv.py --manifest file:///path/to/secure_store/sess-.../manifest/2025-08-25/18.jsonl.enc
  python tools/export_session_csv.py --manifest file:///.../18.jsonl.enc --out /tmp/final.csv --run-asr --asr-model small --scrub-pii

Behavior:
- Decrypts the manifest using SecureStore (locates master.key in the secure_store tree).
- For each manifest record (session_id, modality, uri, row), it decrypts the parquet referenced by `uri`
  (cached per-URI) and extracts the specific row (by integer index).
- Aggregates all extracted rows into a single pandas DataFrame and writes CSV.
- If --run-asr is provided, it will attempt to transcribe using whisper for rows that have an audio_uri but no transcript.
- If --scrub-pii is provided, the script will call TextPreprocessor (app.pipelines.text) to scrub transcripts.
"""

import argparse
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import traceback
from typing import Tuple

# Project imports (SecureStore + TextPreprocessor)
try:
    from app.security.secure_store import SecureStore
except Exception as e:
    print("ERROR: Could not import app.security.secure_store. Ensure your PYTHONPATH includes repo root.", file=sys.stderr)
    raise

# Optional heavy imports will be attempted later (pandas/whisper)
def find_secure_store_root(start_path: Path) -> Path:
    cur = start_path.resolve()
    if cur.is_file():
        cur = cur.parent
    for parent in [cur] + list(cur.parents):
        # master.key is stored in the secure_store root (or under the same tree)
        candidate = parent / "master.key"
        if candidate.exists():
            return parent
    raise FileNotFoundError(f"Could not find 'master.key' walking up from {start_path}. Are you pointing into a secure_store tree?")

def decrypt_manifest_bytes(store: SecureStore, manifest_uri: str) -> bytes:
    try:
        return store.decrypt_read(manifest_uri)
    except AssertionError:
        # decrypt_read asserts scheme; try to coerce
        if manifest_uri.startswith("file://"):
            raise
        else:
            raise AssertionError("Manifest URI must start with file://") from None

def read_manifest_records(manifest_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Manifest is newline-separated JSON lines. Return list of dicts.
    """
    text = manifest_bytes.decode("utf-8")
    lines = [l for l in text.splitlines() if l.strip()]
    records = [json.loads(l) for l in lines]
    return records

def parquet_bytes_to_df(bytes_blob: bytes):
    """
    Read parquet bytes into a pandas DataFrame. Requires pandas + pyarrow or fastparquet.
    """
    try:
        import pandas as pd
        import pyarrow.parquet as pq
        import pyarrow as pa
    except Exception:
        # fallback to pandas.read_parquet
        try:
            import pandas as pd
        except Exception:
            raise RuntimeError("pandas (and preferably pyarrow) are required to load parquet. Install pandas and pyarrow.")
        buf = io.BytesIO(bytes_blob)
        return pd.read_parquet(buf)
    # use pyarrow to read table then convert
    buf = io.BytesIO(bytes_blob)
    table = pq.read_table(buf)
    df = table.to_pandas()
    return df

def safe_get_row(df, idx: int) -> Dict[str, Any]:
    """
    Return row as dict for df.iloc[idx]. If idx out of range, raise IndexError.
    """
    if idx < 0 or idx >= len(df):
        raise IndexError(f"Requested row index {idx} is out of range for parquet with {len(df)} rows.")
    row = df.iloc[idx].to_dict()
    return row

def run_whisper_transcription(audio_bytes: bytes, model_name: str = "small") -> Tuple[str, float]:
    """
    Transcribe audio bytes using whisper if available.
    Returns (text, confidence_estimate).
    Writes bytes to a temp file for whisper to consume.
    """
    try:
        import whisper
    except Exception:
        raise RuntimeError("Whisper is not installed. Install 'openai-whisper' to use --run-asr.")
    # write to temp wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp_path = Path(tf.name)
        tmp_path.write_bytes(audio_bytes)
    try:
        model = whisper.load_model(model_name)
        res = model.transcribe(str(tmp_path))
        text = res.get("text", "")
        conf = float(res.get("avg_logprob", 0.0)) if "avg_logprob" in res else 0.0
        return text, conf
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass

def scrub_pii_via_textpreprocessor(text: str, cfg: Dict[str, Any]) -> str:
    """
    Uses the TextPreprocessor in app.pipelines.text if available to scrub PII.
    Falls back to simple regex if module not importable.
    """
    try:
        from app.pipelines.text import TextPreprocessor
    except Exception:
        # fallback: simple regex-based removal (emails, phones, urls)
        import re
        t = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "<EMAIL>", text)
        t = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b", "<PHONE>", t)
        t = re.sub(r"https?://\S+|www\.\S+", "<URL>", t)
        return t
    # instantiate a TextPreprocessor with a temp dir for output and receipts
    tp = TextPreprocessor(output_dir="tmp_textproc_out", receipt_dir="tmp_textproc_receipts")
    # reuse process_text for PII scrubbing
    out_path, receipt = tp.process_text(text, file_id="asr_tmp")
    try:
        return Path(out_path).read_text(encoding="utf-8")
    except Exception:
        return text

def main():
    p = argparse.ArgumentParser(prog="export_session_csv", description="Decrypt manifest and referenced parquets to produce final CSV.")
    p.add_argument("--manifest", "-m", required=True, help="file:// URI to manifest jsonl.enc inside secure_store")
    p.add_argument("--out", "-o", help="output CSV path (defaults to final_session_<session_id>.csv in cwd)")
    p.add_argument("--run-asr", action="store_true", help="Run Whisper ASR for rows missing transcripts (requires openai-whisper).")
    p.add_argument("--asr-model", default="small", help="Whisper model size to use when --run-asr (small|base|tiny|medium|large).")
    p.add_argument("--scrub-pii", action="store_true", help="Apply PII scrubbing to transcripts using TextPreprocessor if available.")
    args = p.parse_args()

    manifest_uri = args.manifest
    if not manifest_uri.startswith("file://"):
        print("ERROR: manifest must be a file:// URI pointing into your secure_store", file=sys.stderr)
        sys.exit(2)
    manifest_path = Path(manifest_uri[len("file://"):])
    if not manifest_path.exists():
        print(f"ERROR: Manifest file not found: {manifest_path}", file=sys.stderr)
        sys.exit(2)

    # find secure store root
    try:
        root = find_secure_store_root(manifest_path)
    except FileNotFoundError as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(2)

    # instantiate store
    store = SecureStore(str(root))

    # decrypt manifest
    try:
        manifest_bytes = decrypt_manifest_bytes(store, manifest_uri)
    except Exception as e:
        print("ERROR: Failed to decrypt manifest:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(3)

    records = read_manifest_records(manifest_bytes)
    if not records:
        print("No records found in manifest.", file=sys.stderr)
        sys.exit(0)

    # group manifest entries by parquet uri so we only decrypt each parquet once
    from collections import defaultdict
    uri_to_rows = defaultdict(list)
    session_id = records[0].get("session_id", "session")
    for rec in records:
        uri = rec.get("uri")
        row_idx = rec.get("row")
        modality = rec.get("modality")
        sid = rec.get("session_id", session_id)
        # keep session id of the firs
