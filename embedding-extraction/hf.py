import pandas as pd
import numpy as np
import umap
from sklearn.cluster import MiniBatchKMeans
from datasets import load_dataset
from tqdm import tqdm  # progress bars

# --- Step 0: Load dataset ---
print("Loading Qdrant arXiv dataset...")
ds = load_dataset("Qdrant/arxiv-titles-instructorxl-embeddings")
ds_subset = ds['train'].select(range(2_000_000))  # full dataset

# Extract embeddings and titles
embeddings = ds_subset['vector']
titles = ds_subset['title']

# Parameters
BATCH_SIZE = 50_000  # adjust based on RAM
N_CLUSTERS = 10
UMAP_SAMPLE_SIZE = 50_000  # small sample to fit UMAP

# --- Step 1: Fit UMAP on a small sample ---
sample_embeddings = np.array(embeddings[:UMAP_SAMPLE_SIZE])

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
print(f"Fitting UMAP on a sample of {UMAP_SAMPLE_SIZE} embeddings...")
reducer.fit(sample_embeddings)

# --- Step 2: Transform all embeddings in batches ---
umap_coords = np.zeros((len(embeddings), 2), dtype=np.float32)
print("Transforming full dataset with UMAP in batches...")
for i in tqdm(range(0, len(embeddings), BATCH_SIZE), desc="UMAP transform"):
    batch_emb = np.array(list(embeddings[i:i+BATCH_SIZE]))

    umap_coords[i:i+BATCH_SIZE] = reducer.transform(batch_emb)

# --- Step 3: Cluster with MiniBatchKMeans ---
print("Clustering embeddings with MiniBatchKMeans...")
kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=BATCH_SIZE, random_state=42)

# Fit MiniBatchKMeans in batches
# Option 1: Convert all embeddings at once (requires more RAM)
# embedding_matrix = np.array([e for e in embeddings])
# categories = kmeans.fit_predict(embedding_matrix)

# Option 2: Incremental fit (memory-friendly)
print("Fitting MiniBatchKMeans incrementally...")
for i in tqdm(range(0, len(embeddings), BATCH_SIZE), desc="KMeans incremental fit"):
    batch_emb = np.array(list(embeddings[i:i+BATCH_SIZE]))

    kmeans.partial_fit(batch_emb)

# Predict clusters in batches
categories = np.zeros(len(embeddings), dtype=int)
for i in tqdm(range(0, len(embeddings), BATCH_SIZE), desc="KMeans predict"):
    batch_emb = np.array(list(embeddings[i:i+BATCH_SIZE]))

    categories[i:i+BATCH_SIZE] = kmeans.predict(batch_emb)

# --- Step 4: Build final dataset in batches ---
dataset_rows = []
print("Building final dataset in batches...")
for i in tqdm(range(0, len(titles), BATCH_SIZE), desc="Building dataset"):
    batch_titles = titles[i:i+BATCH_SIZE]
    batch_coords = umap_coords[i:i+BATCH_SIZE]
    batch_categories = categories[i:i+BATCH_SIZE]
    
    batch_rows = pd.DataFrame({
        "identifier": range(i, i+len(batch_titles)),
        "x": batch_coords[:, 0],
        "y": batch_coords[:, 1],
        "category": batch_categories,
        "text": batch_titles,
        "url": [f"arxiv_paper_{j}" for j in range(i, i+len(batch_titles))],
        "vector": embeddings[i:i+len(batch_titles)]
    })
    dataset_rows.append(batch_rows)

dataset = pd.concat(dataset_rows, ignore_index=True)
del dataset_rows  # free memory

# --- Step 5: Save dataset ---
dataset.to_parquet("umap_arxiv_dataset.parquet", index=False)
print("Saved dataset as umap_arxiv_dataset.parquet")

# --- Step 6: Basic statistics ---
print(f"\nDataset statistics:")
print(f"Total papers: {len(dataset)}")
print(f"Number of clusters: {N_CLUSTERS}")
print(f"UMAP coordinates range:")
print(f"  X: [{dataset['x'].min():.2f}, {dataset['x'].max():.2f}]")
print(f"  Y: [{dataset['y'].min():.2f}, {dataset['y'].max():.2f}]")
print(f"\nCluster distribution:")
print(dataset['category'].value_counts().sort_index())
