import requests
import pandas as pd
from embedding_atlas.projection import compute_text_projection
import numpy as np
import ast

url = "http://localhost:5000/all_data"
response = requests.get(url)

def parse_embedding(x):
    if isinstance(x, list):
        return np.array(x)
    else:
        # Safely convert string like "[0.0168, -0.11, ...]" to list
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
    df['embedding'] = df['embedding'].apply(parse_embedding)

    # Export to CSV
    df.to_csv("arxiv_data.csv", index=False)
    print("Saved CSV as arxiv_data.csv")
    
    # Export to JSON (records format)
    df.to_json("arxiv_data.json", orient="records", lines=True)
    print("Saved JSON as arxiv_data.json")

else:
    print(f"Error: {response.status_code}")
    print(response.text)
