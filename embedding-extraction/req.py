import requests
import pandas as pd
import numpy as np
import ast

# Fetch data from your API
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
    x_coords = [row.get("x") for row in rows]  # Use stored UMAP x
    y_coords = [row.get("y") for row in rows]  # Use stored UMAP y
    clusters = [row.get("cluster") for row in rows]  # Use stored cluster
    
    # Create DataFrame
    df = pd.DataFrame({
        "url": urls,
        "content": contents,
        "embedding": embeddings,
        "x": x_coords,
        "y": y_coords,
        "category": clusters  # Use existing cluster assignments
    })
    
    print(df.head())
    print("Total elements in DataFrame:", df.size)
    print("Number of rows:", len(df))
    print("Number of columns:", len(df.columns))
    
    # Convert embeddings to numpy arrays (optional if needed later)
    df['embedding'] = df['embedding'].apply(parse_embedding)
    
    # --- Step 3: Build final dataset ---
    dataset = pd.DataFrame({
        "identifier": range(len(df)),
        "x": df["x"],
        "y": df["y"],
        "category": df["category"],  # Already stored
        "text": df["content"],
        "url": df["url"]
    })
    
    # Save dataset
    dataset.to_parquet("umap_dataset.parquet", index=False)
    print("Saved dataset using existing UMAP coordinates and cluster assignments as umap_dataset.parquet")

else:
    print(f"Error: {response.status_code}")
    print(response.text)
