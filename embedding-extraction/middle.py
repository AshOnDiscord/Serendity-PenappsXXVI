import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from cerebras.cloud.sdk import Cerebras

# --- Initialize Cerebras client ---
client = Cerebras(api_key="")

# --- Load modified dataset ---
file_path = ".parquet"
df = pd.read_parquet(file_path)

# --- Identify LOWDENSE points ---
lowdense_mask = df['identifier'] < 0
lowdense_points = df[lowdense_mask]

# --- Fit nearest neighbors on 2D coordinates instead of embeddings ---
coords = df[['x', 'y']].values
nbrs = NearestNeighbors(n_neighbors=3, metric='euclidean').fit(coords)

# --- Function to generate future research topic using Cerebras chat ---
def suggest_future_topic_chat(neighbor_texts):
    context = " ".join([t[:100] for t in neighbor_texts])  # first 100 chars
    user_message = (
        f"Based on the following texts:\n{context}\n\n"
        "Suggest a related but novel research topic that could be a potential future area of study. "
        "Keep it concise, technical, and specific. Do NOT say 'Here's a suggestion' or similar phrases. "
    )

    chat_completion = client.chat.completions.create(
        model="llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "user", "content": user_message}
        ],
        max_tokens=50
    )

    return chat_completion.choices[0].message.content.strip()

# --- Compute suggested topics for each LOWDENSE point ---
suggested_topics = []
for idx, row in lowdense_points.iterrows():
    distances, indices = nbrs.kneighbors([[row['x'], row['y']]], n_neighbors=3)
    neighbor_texts = df.iloc[indices[0]]['text'].tolist()
    
    # Prepare context
    context = " ".join([t[:250] for t in neighbor_texts])
    print(f"\n--- LOWDENSE point index {idx} ---")
    print("Original text:", row['text'])
    print("Context sent to Cerebras:", context)
    
    # Generate topic
    topic = suggest_future_topic_chat(neighbor_texts)
    suggested_topics.append(topic)
    print("Suggested topic:", topic)

# --- Add the suggested topics to the dataframe ---
df.loc[lowdense_mask, 'suggested_topic'] = suggested_topics

# --- Save updated dataset ---
output_file = "mod_parq_with_suggested_topic.parquet"
df.to_parquet(output_file, index=False)
print(f"Modified dataset with suggested topics saved as {output_file}")
