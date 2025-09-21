import pandas as pd
import numpy as np
import umap
import ast
from collections import Counter

# --- Load your Parquet file ---
df = pd.read_parquet("full_parq.parquet")

df.to_csv("full_parq.csv", index=False)