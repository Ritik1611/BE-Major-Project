import pyarrow.parquet as pq
import pandas as pd
import io

def read_embeddings_from_parquet(parquet_path):
    df = pq.read_table(parquet_path).to_pandas()

    if 'features' in df.columns:
        embs = df['features'].apply(lambda f: f.get('w2v2_embedding') if isinstance(f, dict) else None)
        embs = [e for e in embs if e is not None]
    else:
        embs = df.select_dtypes(include=['float','int']).values.tolist()

    return embs
