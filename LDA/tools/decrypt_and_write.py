#!/usr/bin/env python3
"""
tools/decrypt_and_write.py

Usage:
  # Decrypt a single encrypted file URI and write the decrypted output to cwd (defaults)
  python tools/decrypt_and_write.py file:///home/you/project/secure_store/.../18.parquet.enc

  # Specify output path
  python tools/decrypt_and_write.py file:///.../clips/audio/30_630.wav.enc --out /tmp/30_630.wav

  # If the file is a parquet and you want CSV output (requires pandas + pyarrow)
  python tools/decrypt_and_write.py file:///.../18.parquet.enc --to-csv --out /tmp/18.csv

Notes:
- This script locates the SecureStore root by walking up parent directories until it finds a master.key.
- It then instantiates app.security.secure_store.SecureStore(root) and uses decrypt_read() to get the plaintext bytes.
- The default output filename is the original filename with the trailing ".enc" removed, written into the current working directory
  (so `.../18.parquet.enc` -> `./18.parquet`).
"""
import argparse
import io
import sys
from pathlib import Path

# Import SecureStore from your project
try:
    from app.security.secure_store import SecureStore
except Exception as e:
    print("ERROR: Could not import SecureStore from app.security.secure_store. Make sure your PYTHONPATH includes the repo root.")
    raise

def find_secure_store_root(start_path: Path) -> Path:
    """
    Walk up from start_path to find a directory that contains 'master.key'.
    Returns the Path to that directory, or raises FileNotFoundError.
    """
    cur = start_path.resolve()
    # if start_path is a file, start search from its parent
    if cur.is_file():
        cur = cur.parent
    for parent in [cur] + list(cur.parents):
        candidate = parent / "master.key"
        if candidate.exists():
            return parent
    raise FileNotFoundError(f"Could not find 'master.key' walking up from {start_path}. Are you pointing into a secure_store tree?")

def default_out_path_from_enc(uri_path: Path, out_arg: str | None) -> Path:
    if out_arg:
        return Path(out_arg)
    name = uri_path.name
    if name.endswith(".enc"):
        name = name[:-4]
    return Path.cwd() / name

def write_bytes(path: Path, b: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b)

def try_parquet_to_csv(bytes_blob: bytes, csv_out: Path) -> None:
    """Attempt to read parquet bytes and write CSV (optional dependency: pandas + pyarrow)."""
    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError("pandas and pyarrow are required for --to-csv. Install them and try again.") from e

    buf = io.BytesIO(bytes_blob)
    # read_table -> to_pandas
    try:
        import pyarrow as pa
        table = pq.read_table(buf)
        df = table.to_pandas()
    except Exception:
        # fallback to pandas.read_parquet (may use pyarrow/fastparquet)
        buf.seek(0)
        df = pd.read_parquet(buf)
    df.to_csv(csv_out, index=False)

def main():
    p = argparse.ArgumentParser(prog="decrypt_and_write", description="Decrypt a SecureStore file:// URI and write clear output.")
    p.add_argument("uri", help="file:// URI pointing to an encrypted file in secure_store (e.g. file:///.../18.parquet.enc)")
    p.add_argument("--out", "-o", help="Output path to write decrypted bytes. Defaults to cwd/<basename without .enc>")
    p.add_argument("--to-csv", action="store_true", help="If input is a parquet, also convert to CSV (requires pandas+pyarrow).")
    args = p.parse_args()

    uri = args.uri
    if not uri.startswith("file://"):
        print("ERROR: Only file:// URIs are supported (pointing to files inside your secure_store).", file=sys.stderr)
        sys.exit(2)

    path_str = uri[len("file://"):]
    enc_path = Path(path_str)
    if not enc_path.exists():
        print(f"ERROR: Encrypted file does not exist: {enc_path}", file=sys.stderr)
        sys.exit(2)

    try:
        root = find_secure_store_root(enc_path)
    except FileNotFoundError as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(2)

    # Instantiate SecureStore with discovered root
    store = SecureStore(str(root))

    try:
        plaintext = store.decrypt_read(uri)
    except Exception as e:
        print(f"ERROR: Decryption failed for {uri}: {e}", file=sys.stderr)
        sys.exit(3)

    out_path = default_out_path_from_enc(enc_path, args.out)
    try:
        write_bytes(out_path, plaintext)
    except Exception as e:
        print(f"ERROR: Failed to write decrypted file to {out_path}: {e}", file=sys.stderr)
        sys.exit(4)

    print(f"Decrypted file written to: {out_path}")

    # Optional parquet -> csv conversion
    if args.to_csv:
        # only sensible if output looks like parquet
        name_lower = out_path.name.lower()
        if name_lower.endswith(".parquet") or name_lower.endswith(".pq"):
            csv_path = out_path.with_suffix(".csv")
            try:
                try_parquet_to_csv(plaintext, csv_path)
                print(f"Parquet converted to CSV: {csv_path}")
            except Exception as e:
                print(f"ERROR: Failed to convert parquet to CSV: {e}", file=sys.stderr)
                sys.exit(5)
        else:
            print("WARNING: --to-csv specified but output file does not look like a Parquet (.parquet/.pq). Skipping conversion.")

if __name__ == "__main__":
    main()
