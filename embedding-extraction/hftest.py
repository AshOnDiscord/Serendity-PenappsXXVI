import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity, NearestNeighbors
import alphashape
from shapely.geometry import Point

# --- Load your Parquet file ---
file_path = "umap_arxiv_dataset.parquet"
df = pd.read_parquet(file_path)

print("Columns:", df.columns.tolist())

# Print all values of the first row
print("First row values:")
print(df.iloc[0])



# Extract x and y coordinates
xy = df[['x', 'y']].values

# --- Step 1: Fit KDE on the data ---
bandwidth = 0.1  # adjust for smoother/rougher density
kde = KernelDensity(bandwidth=bandwidth)
kde.fit(xy)

# --- Step 2: Create a grid of candidate points ---
grid_res = 200
xx, yy = np.meshgrid(
    np.linspace(xy[:, 0].min(), xy[:, 0].max(), grid_res),
    np.linspace(xy[:, 1].min(), xy[:, 1].max(), grid_res)
)
grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

# --- Step 3: Keep only points inside concave hull (alpha shape) ---
alpha = 0.05  # smaller alpha = tighter hull; adjust as needed
hull_shape = alphashape.alphashape(xy, alpha)

inside_mask = np.array([hull_shape.contains(Point(p)) for p in grid_points])
inside_points = grid_points[inside_mask]

# --- Step 4: Compute KDE densities ---
densities = kde.score_samples(inside_points)  # log densities

# --- Step 5: Distance to nearest real point & surroundedness with equal-range check ---
nbrs = NearestNeighbors(n_neighbors=20).fit(xy)  # more neighbors for context
distances, indices = nbrs.kneighbors(inside_points)

# Mean distance to neighbors
mean_dists = distances.mean(axis=1)

# Coefficient of Variation (CV) of neighbor distances
cv_dists = distances.std(axis=1) / mean_dists

# Check surroundedness: angle coverage
vectors = xy[indices[:, 1:]] - inside_points[:, None, :]
angles = np.arctan2(vectors[..., 1], vectors[..., 0])
angle_spread = np.ptp(angles, axis=1)  # peak-to-peak range
surrounded_mask = angle_spread > (np.pi * 1.5)  # must span >270°

# Combine constraints: nearby + surrounded + roughly equal-range
max_allowed_dist = np.percentile(mean_dists, 75)
max_allowed_cv = 0.3  # adjust: smaller = more centered
valid_mask = (mean_dists < max_allowed_dist) & surrounded_mask & (cv_dists < max_allowed_cv)

filtered_points = inside_points[valid_mask]
filtered_densities = densities[valid_mask]


# --- Step 6: Pick top 50 lowest-density surrounded points ---
n_lowdense = 20
lowest_idxs = np.argsort(filtered_densities)[:n_lowdense]
lowdense_points = filtered_points[lowest_idxs]

# --- Step 7: Create LOWDENSE points ---
vector_len = len(df['vector'].iloc[0])
new_points = []
for i, (x_ld, y_ld) in enumerate(lowdense_points, start=1):
    new_points.append({
        'identifier': -i,  # negative IDs to avoid collision
        'x': float(x_ld),
        'y': float(y_ld),
        'category': None,
        'text': f'LOWDENSE_{i}',
        'url': f'LOWDENSE_{i}',
        'vector': [0.0] * vector_len  # placeholder vector
    })

# --- Step 8: Append and save ---
df = pd.concat([df, pd.DataFrame(new_points)], ignore_index=True)

output_file = "mod_parq.parquet"
df.to_parquet(output_file, index=False)

for i, p in enumerate(lowdense_points, start=1):
    print(f"Added LOWDENSE_{i} point at ({p[0]:.3f}, {p[1]:.3f})")

print(f"Modified dataset saved as {output_file}")
