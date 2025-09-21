import pandas as pd
import numpy as np
import umap
import ast

# --- Load your Parquet file ---
df = pd.read_parquet("full_parq.parquet")

# --- Convert 'vector' column to numpy arrays if stored as strings ---
def parse_vector(v):
    if isinstance(v, str):
        try:
            return np.array(ast.literal_eval(v), dtype=float)
        except:
            return np.array(v, dtype=float)
    elif isinstance(v, (list, np.ndarray)):
        return np.array(v, dtype=float)
    return np.zeros(128, dtype=float)  # fallback length

df['vector'] = df['vector'].apply(parse_vector)

# --- Prepare embeddings array ---
embeddings_array = np.stack(df['vector'].values)

# --- UMAP parameters ---
n_neighbors = 10
min_dist = 0.1
n_components = 2
random_state = 42

# --- Compute UMAP ---
reducer = umap.UMAP(
    n_neighbors=min(n_neighbors, len(embeddings_array)-1),
    min_dist=min_dist,
    n_components=n_components,
    random_state=random_state,
    metric='cosine'
)
umap_coords = reducer.fit_transform(embeddings_array)

# --- Overwrite 'x' and 'y' columns ---
df['x'] = umap_coords[:, 0]
df['y'] = umap_coords[:, 1]

# --- Save back to Parquet ---
df.to_parquet("full_parq_umap.parquet", index=False)

print(f"UMAP recalculated and saved to 'full_parq_umap.parquet' with shape {df.shape}")
