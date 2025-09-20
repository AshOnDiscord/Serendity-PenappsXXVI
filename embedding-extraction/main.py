import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pprint

df = pd.read_csv("summaries.csv")

paragraphs = df['Summary'].astype(str).tolist()
ids = df['ID'].tolist()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(paragraphs)

best_k = 2
best_score = -1
max_k = min(10, len(paragraphs) - 1)
for k in range(2, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    if score > best_score:
        best_score = score
        best_k = k
        best_labels = labels

print(f"\nOptimal number of clusters: {best_k}")

cluster_map = {}
for idx, label, summary in zip(ids, best_labels, paragraphs):
    cluster_map[idx] = {
        "cluster": int(label),
        "preview": summary[:100] + "..."
    }

for idx, info in cluster_map.items():
    print(f"ID {idx} → Cluster {info['cluster']}, Preview: {info['preview']}")

cos_sim_matrix = cosine_similarity(embeddings)
min_val = cos_sim_matrix.min()
max_val = cos_sim_matrix.max()
normalized_cos_sim = (cos_sim_matrix - min_val) / (max_val - min_val)

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

plt.figure(figsize=(8,6))
for cluster_id in range(best_k):
    cluster_points = reduced_embeddings[best_labels == cluster_id]
    plt.scatter(cluster_points[:,0], cluster_points[:,1], label=f'Cluster {cluster_id}')

plt.title("Summary Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

pprint.pprint(cluster_map)