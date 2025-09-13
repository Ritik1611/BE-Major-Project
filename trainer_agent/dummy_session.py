import pandas as pd
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path

# Parameters
n_samples = 10
embedding_dim = 768

# Each row is a dict with key "w2v2_embedding"
features = [{"w2v2_embedding": np.random.rand(embedding_dim).tolist()} for _ in range(n_samples)]

# Create DataFrame
df = pd.DataFrame({"features": features})

# Save parquet in trainer_agent folder
output_path = Path("C:/Users/snick/Desktop/BE-Major-Project/trainer_agent/example_session.parquet")
table = pa.Table.from_pandas(df)
pq.write_table(table, output_path)

print(f"Dummy parquet saved to: {output_path}")
