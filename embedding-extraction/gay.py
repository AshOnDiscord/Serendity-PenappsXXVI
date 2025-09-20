import pandas as pd
from embedding_atlas.widget import EmbeddingAtlasWidget


# Example
data = [
    {"identifier": 0, "x": 1.2, "y": 0.5, "category": 0, "text": "Apple"},
    {"identifier": 1, "x": -0.8, "y": 1.5, "category": 1, "text": "Banana"},
    {"identifier": 2, "x": 0.3, "y": -0.7, "category": 0, "text": "Cherry"},
]

df = pd.DataFrame(data)
df.to_parquet("sample_dataset.parquet", index=False)

EmbeddingAtlasWidget(df)

