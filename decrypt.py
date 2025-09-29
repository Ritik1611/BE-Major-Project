# decrypt.py
import os
import json
import shutil
from centralized_secure_store import SecureStore

def walk_and_decrypt(store: SecureStore, root: str, out_root: str):
    """
    Walk through secure_store and:
      - copy raw files (non .enc) directly
      - decrypt .enc files with SecureStore
    """
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            enc_path = os.path.join(dirpath, fn)
            rel_path = os.path.relpath(enc_path, root)
            out_path = os.path.join(out_root, rel_path)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            if not fn.endswith(".enc"):
                # not encrypted, just copy
                shutil.copy2(enc_path, out_path)
                print(f"{enc_path}\tCOPIED")
                continue

            try:
                data = store.decrypt_read(f"file://{enc_path}")
                out_file = out_path[:-4]  # strip .enc
                with open(out_file, "wb") as f:
                    f.write(data)
                print(f"{enc_path}\tDECRYPTED -> {out_file}")
            except Exception as e:
                print(f"{enc_path}\tDECRYPT_FAILED\t{type(e).__name__}: {e}")

def main():
    store = SecureStore("./secure_store")
    in_root = "./secure_store"
    out_root = "./decrypted_output"

    if not os.path.exists(in_root):
        print(f"Input root not found: {in_root}")
        return

    os.makedirs(out_root, exist_ok=True)
    walk_and_decrypt(store, in_root, out_root)

    print("\n=== Decryption finished ===")

if __name__ == "__main__":
    main()
