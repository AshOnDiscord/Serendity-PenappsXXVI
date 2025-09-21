import pandas as pd
import numpy as np
import umap
from sklearn.cluster import KMeans
from datasets import load_dataset

# --- Step 0: Load dataset ---
print("Loading Qdrant arXiv dataset...")
ds = load_dataset("Qdrant/arxiv-titles-instructorxl-embeddings")
ds_subset = ds['train'].select(range(2_000_000))

# Convert to pandas DataFrame
print("Converting to DataFrame...")
df = ds_subset.to_pandas()

print(df.head())
print("Total elements in DataFrame:", df.size)
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))
print("Column names:", df.columns.tolist())

# Examine structure
print("\nDataset structure:")
for col in df.columns:
    print(f"- {col}: {type(df[col].iloc[0])}")

# --- Step 1: Process embeddings ---
print("\nProcessing embeddings...")
embedding_col = None
for col in df.columns:
    if 'embed' in col.lower() or 'vector' in col.lower():
        embedding_col = col
        break

if not embedding_col:
    raise ValueError("No embedding/vector column found in the dataset")

sample_embedding = df[embedding_col].iloc[0]
if isinstance(sample_embedding, list):
    embedding_matrix = np.array(df[embedding_col].tolist())
else:
    embedding_matrix = np.vstack(df[embedding_col].values)

print(f"Embedding matrix shape: {embedding_matrix.shape}")

# --- Step 2: Compute UMAP projection (2D) ---
print("Computing UMAP projection...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
umap_coords = reducer.fit_transform(embedding_matrix)

df["x"] = umap_coords[:, 0]
df["y"] = umap_coords[:, 1]

# --- Step 3: Assign categories with K-means ---
print("Performing K-means clustering...")
n_clusters = 10  # adjust depending on dataset size
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df["category"] = kmeans.fit_predict(embedding_matrix)

# --- Step 4: Build final dataset ---
print("Building final dataset...")
text_column = 'title' if 'title' in df.columns else df.columns[0]

# Construct URLs pointing directly to arXiv PDFs
if 'url' in df.columns:
    urls = df['url']
elif 'DOI' in df.columns and df['DOI'].notna().any():
    urls = df['DOI'].apply(
        lambda x: f"https://arxiv.org/pdf/{x}.pdf" if pd.notna(x) else f"arxiv_paper_{df.index[df['DOI'] == x].tolist()[0]}"
    )
elif 'id' in df.columns:
    urls = df['id'].apply(lambda x: f"https://arxiv.org/pdf/{x}.pdf")
else:
    urls = [f"arxiv_paper_{i}" for i in range(len(df))]

# Include original embeddings
dataset = pd.DataFrame({
    "identifier": range(len(df)),
    "x": df["x"],
    "y": df["y"],
    "category": df["category"],
    "text": df[text_column],
    "url": urls,
    "vector": df[embedding_col]  # <-- include original embedding
})

# --- Step 5: Save dataset ---
print("Saving dataset...")
dataset.to_parquet("umap_arxiv_dataset.parquet", index=False)
print("Saved dataset as umap_arxiv_dataset.parquet")

# --- Step 6: Statistics ---
print(f"\nDataset statistics:")
print(f"Total papers: {len(dataset)}")
print(f"Number of clusters: {n_clusters}")
print(f"UMAP coordinates range:")
print(f"  X: [{dataset['x'].min():.2f}, {dataset['x'].max():.2f}]")
print(f"  Y: [{dataset['y'].min():.2f}, {dataset['y'].max():.2f}]")

# Sample of final dataset
print(f"\nSample of final dataset:")
print(dataset.head())

# Cluster distribution
print(f"\nCluster distribution:")
print(dataset['category'].value_counts().sort_index())
