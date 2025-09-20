import pandas as pd
import numpy as np
import umap
from sklearn.cluster import KMeans
from datasets import load_dataset

# Load the Qdrant arXiv dataset
print("Loading Qdrant arXiv dataset...")
ds = load_dataset("Qdrant/arxiv-titles-instructorxl-embeddings")

ds_subset = ds['train'].select(range(200_000))


# Convert to pandas DataFrame
print("Converting to DataFrame...")
df = ds_subset.to_pandas()

print(df.head())
print("Total elements in DataFrame:", df.size)
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))
print("Column names:", df.columns.tolist())

# The dataset should have columns like 'title', 'embedding', etc.
# Let's examine the structure
print("\nDataset structure:")
for col in df.columns:
    print(f"- {col}: {type(df[col].iloc[0])}")

# Convert embeddings to numpy array matrix
print("\nProcessing embeddings...")
if 'vector' in df.columns:
    # Check if embeddings are already numpy arrays or lists
    sample_embedding = df['vector'].iloc[0]
    if isinstance(sample_embedding, list):
        embedding_matrix = np.array(df['vector'].tolist())
    else:
        embedding_matrix = np.vstack(df['vector'].values)
elif 'embedding' in df.columns:
    sample_embedding = df['embedding'].iloc[0]
    if isinstance(sample_embedding, list):
        embedding_matrix = np.array(df['embedding'].tolist())
    else:
        embedding_matrix = np.vstack(df['embedding'].values)
elif 'embeddings' in df.columns:  # Alternative column name
    sample_embedding = df['embeddings'].iloc[0]
    if isinstance(sample_embedding, list):
        embedding_matrix = np.array(df['embeddings'].tolist())
    else:
        embedding_matrix = np.vstack(df['embeddings'].values)
else:
    # Find the embedding column
    embedding_col = None
    for col in df.columns:
        if 'embed' in col.lower() or 'vector' in col.lower():
            embedding_col = col
            break
    
    if embedding_col:
        sample_embedding = df[embedding_col].iloc[0]
        if isinstance(sample_embedding, list):
            embedding_matrix = np.array(df[embedding_col].tolist())
        else:
            embedding_matrix = np.vstack(df[embedding_col].values)
    else:
        raise ValueError("No embedding/vector column found in the dataset")

print(f"Embedding matrix shape: {embedding_matrix.shape}")

# --- Step 1: Compute UMAP projection (2D) ---
print("Computing UMAP projection...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
umap_coords = reducer.fit_transform(embedding_matrix)

df["x"] = umap_coords[:, 0]
df["y"] = umap_coords[:, 1]

# --- Step 2: (Optional) Assign categories with clustering ---
print("Performing K-means clustering...")
n_clusters = 10  # adjust depending on dataset size
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df["category"] = kmeans.fit_predict(embedding_matrix)

# --- Step 3: Build final dataset ---
print("Building final dataset...")

# Use appropriate text column (likely 'title' for arXiv dataset)
text_column = 'title' if 'title' in df.columns else df.columns[0]
url_column = 'url' if 'url' in df.columns else None

dataset_dict = {
    "identifier": range(len(df)),
    "x": df["x"],
    "y": df["y"], 
    "category": df["category"],
    "text": df[text_column]
}

# Add URL if available
if url_column and url_column in df.columns:
    dataset_dict["url"] = df[url_column]
else:
    # Create arXiv URLs using the DOI or ID
    if 'DOI' in df.columns and df['DOI'].notna().any():
        dataset_dict["url"] = df['DOI'].apply(lambda x: f"https://doi.org/{x}" if pd.notna(x) else f"arxiv_paper_{dataset_dict['identifier'][df.index[df['DOI'] == x].tolist()[0]]}")
    elif 'id' in df.columns:
        dataset_dict["url"] = df['id'].apply(lambda x: f"https://arxiv.org/abs/{x}")
    else:
        dataset_dict["url"] = [f"arxiv_paper_{i}" for i in range(len(df))]

dataset = pd.DataFrame(dataset_dict)

# Save dataset
print("Saving dataset...")
dataset.to_parquet("umap_arxiv_dataset.parquet", index=False)
print("Saved dataset with UMAP projection as umap_arxiv_dataset.parquet")

# Print some statistics
print(f"\nDataset statistics:")
print(f"Total papers: {len(dataset)}")
print(f"Number of clusters: {n_clusters}")
print(f"UMAP coordinates range:")
print(f"  X: [{dataset['x'].min():.2f}, {dataset['x'].max():.2f}]")
print(f"  Y: [{dataset['y'].min():.2f}, {dataset['y'].max():.2f}]")

# Show sample of final dataset
print(f"\nSample of final dataset:")
print(dataset.head())

# Show cluster distribution
print(f"\nCluster distribution:")
print(dataset['category'].value_counts().sort_index())