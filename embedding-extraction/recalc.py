import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from InstructorEmbedding import INSTRUCTOR

# --- Load your Parquet file ---
df = pd.read_parquet("full_parq.parquet")

# --- Initialize model ---
model = INSTRUCTOR('hkunlp/instructor-xl')
instruction = "Represent the Research Paper title for retrieval; Input:"

# --- Ensure vector column is proper list ---
def parse_vector(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return None
    elif isinstance(v, (list, np.ndarray)):
        return list(v)
    else:
        return None

df['vector'] = df['vector'].apply(parse_vector)

# --- Recompute embeddings where vector size != 768 ---
fixed_vectors = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fixing vectors"):
    vec = row['vector']
    if vec is None or len(vec) != 768:
        text_input = f"{instruction} {row['title']}" if 'title' in df.columns else row.get('text', '')
        embedding = model.encode([[instruction, text_input]])
        fixed_vectors.append(embedding[0].tolist())
    else:
        fixed_vectors.append(vec)

df['vector'] = fixed_vectors

# --- Export corrected parquet ---
df.to_parquet("full_parq_fixed.parquet", index=False)

print("✅ Exported fixed embeddings to full_parq_fixed.parquet")
