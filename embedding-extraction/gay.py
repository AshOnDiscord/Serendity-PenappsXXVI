import pandas as pd
import numpy as np
import ast

# Load the Parquet files
df_website = pd.read_parquet("website_rows.parquet")
df_umap = pd.read_parquet("umap_arxiv_dataset.parquet")

# # Concatenate along rows (if they have the same columns)

combined_df = pd.concat([df_website, df_umap], ignore_index=True)
combined_df['vector'] = combined_df['vector'].apply(lambda x: np.array(x).tolist() if isinstance(x, np.ndarray) else x)
combined_df['vector'] = combined_df['vector'].apply(lambda x: str(x))
print(combined_df.shape)
combined_df['vector'] = combined_df['vector'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x))
print(combined_df.head())
combined_df.to_parquet("full_parq.parquet", index=False)

print(type(combined_df['vector'].iloc[0]))  # should be <class 'numpy.ndarray'>
print(combined_df.head())
