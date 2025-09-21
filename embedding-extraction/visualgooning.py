import pandas as pd
import numpy as np

# --- Load dataset ---
file_path = "umap_arxiv_dataset.parquet"
df = pd.read_parquet(file_path)

# Compute cluster centers
cluster_centers = df.groupby('category')[['x', 'y']].mean()

# Compute the global center
global_center = df[['x', 'y']].mean().values

# Move clusters away from the global center
explosion_factor = 5.0  # how far clusters move from the center
df_exploded = df.copy()

for cat, center in cluster_centers.iterrows():
    # Vector from global center to cluster center
    vector = center.values - global_center
    # Move all points in this category along the vector
    df_exploded.loc[df_exploded['category'] == cat, 'x'] += vector[0] * explosion_factor
    df_exploded.loc[df_exploded['category'] == cat, 'y'] += vector[1] * explosion_factor

# Optionally, add a little random jitter per point for “strands”
jitter = 0.2
df_exploded['x'] += np.random.uniform(-jitter, jitter, size=len(df_exploded))
df_exploded['y'] += np.random.uniform(-jitter, jitter, size=len(df_exploded))

# Save
output_file = "mod_parq_scaled.parquet"
df_exploded.to_parquet(output_file, index=False)

print(f"Exploded clusters saved to {output_file}")
