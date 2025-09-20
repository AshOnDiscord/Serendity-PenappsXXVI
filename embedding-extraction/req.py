import requests
import pandas as pd
import numpy as np
import ast
import umap
from sklearn.cluster import KMeans

url = "http://localhost:5000/all_data"
response = requests.get(url)

def parse_embedding(x):
    if isinstance(x, list):
        return np.array(x)
    else:
        return np.array(ast.literal_eval(x))

if response.status_code == 200:
    json_data = response.json()
    
    # Extract data
    rows = json_data.get("data", [])
    urls = [row.get("url") for row in rows]
    contents = [row.get("content") for row in rows]
    embeddings = [row.get("embedding") for row in rows]
    
    # Create DataFrame
    df = pd.DataFrame({
        "url": urls,
        "content": contents,
        "embedding": embeddings
    })
    
    print(df.head())
    print("Total elements in DataFrame:", df.size)
    print("Number of rows:", len(df))
    print("Number of columns:", len(df.columns))
    
    # Convert embeddings to numpy arrays
    df['embedding'] = df['embedding'].apply(parse_embedding)
    embedding_matrix = np.vstack(df['embedding'].values)
    
    # --- Step 1: Compute UMAP projection (2D) ---
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_coords = reducer.fit_transform(embedding_matrix)
    
    df["x"] = umap_coords[:, 0]
    df["y"] = umap_coords[:, 1]
    
    # --- Step 2: (Optional) Assign categories with clustering ---
    n_clusters = 10  # adjust depending on dataset size
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["category"] = kmeans.fit_predict(embedding_matrix)
    
    # --- Step 3: Build final dataset ---
    dataset = pd.DataFrame({
        "identifier": range(len(df)),
        "x": df["x"],
        "y": df["y"],
        "category": df["category"],
        "text": df["content"],
        "url": df["url"]
    })
    
    # Save dataset
    dataset.to_parquet("umap_dataset.parquet", index=False)
    print("Saved dataset with UMAP projection as umap_dataset.parquet")

else:
    print(f"Error: {response.status_code}")
    print(response.text)
