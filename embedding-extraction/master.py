import requests
from io import BytesIO
from pdfminer.high_level import extract_text
from bs4 import BeautifulSoup
from exa_py import Exa
from cerebras.cloud.sdk import Cerebras
import ast
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import pprint
import csv
import psycopg2
from dotenv import load_dotenv
import os
import umap
from sklearn.cluster import AgglomerativeClustering
import json
from supabase import create_client, Client

client = Cerebras(api_key=os.getenv('CEREBRAS_KEY'))
exa = Exa(os.getenv('EXA_KEY'))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


PARQUET_FILE_PATH = "/home/thinkies/Git/Serendity-PenappsXXVI/embedding-extraction/mod_parq.parquet"  # Update this path
url = ""
key = ""
supabase: Client = create_client(url, key)

MAX_TOKENS = 6800


def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    words = text.split()
    max_words = int(max_tokens * 0.75)
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text

def scrape_url(url: str) -> str:
    resp = requests.get(url)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "").lower()

    text = ""
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        pdf_bytes = resp.content
        text = extract_text(BytesIO(pdf_bytes))
    else:
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text(separator="\n")

    filtered_lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 15]
    return "\n".join(filtered_lines)


def get_top_similar_articles(url: str, top_n: int = 3):
    try:
        results = exa.find_similar_and_contents(
            url=url,
            text=True,
            summary={"query": "Key advancements, details, notes, or applications"},
        )
        filtered_results = [res for res in results.results if "recaptcha" not in res.title.lower()]

        sorted_results = sorted(filtered_results, key=lambda x: x.score, reverse=True)
        top_results = sorted_results[:top_n]

        similar_data = []
        for res in top_results:
            similar_data.extend([
                res.title,
                res.score,
                res.url,
                res.summary,
            ])
        while len(similar_data) < top_n * 4:
            similar_data.extend(["", "", "", ""])
        return similar_data
    except Exception as e:
        print(f"Failed to get similar articles for {url}: {e}")
        return [""] * (top_n * 4)


def find_optimal_clusters(embeddings_array, max_k=14):
    if len(embeddings_array) < 2:
        return 1
    
    max_k = min(max_k, len(embeddings_array) - 1)
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        silhouette_avg = silhouette_score(embeddings_array, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For k={k}, silhouette score = {silhouette_avg:.4f}")
    
    if silhouette_scores:
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    else:
        return 2


def perform_kmeans_clustering(embeddings_array, n_clusters=None):
    if n_clusters is None:
        n_clusters = find_optimal_clusters(embeddings_array)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    
    if len(set(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(embeddings_array, cluster_labels)
        print(f"Average silhouette score: {silhouette_avg:.4f}")
    
    return cluster_labels, kmeans


def perform_agglomerative_clustering(embeddings_array, distance_threshold=0.5):
    clustering = AgglomerativeClustering(
        n_clusters=None,  # Let distance_threshold decide
        metric='cosine',  # <-- replaced 'affinity' with 'metric'
        linkage='average',
        distance_threshold=distance_threshold
    )
    cluster_labels = clustering.fit_predict(embeddings_array)
    print(f"Number of clusters: {len(np.unique(cluster_labels))}")
    return cluster_labels




def visualize_clusters(embeddings_array, cluster_labels, urls):
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings_array)
    
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = cluster_labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[color], label=f'Cluster {label}', alpha=0.7, s=100)
    
    plt.xlabel(f'First Principal Component (explained variance: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'Second Principal Component (explained variance: {pca.explained_variance_ratio_[1]:.2%})')
    plt.title('Website Clustering Visualization (PCA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for i, url in enumerate(urls):
        short_url = url.split('//')[1].split('/')[0] if '//' in url else url[:20]
        plt.annotate(short_url, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('clustering_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    with open("websites.txt", "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    # scrape for content
    contents = []
    for url in urls:
        try:
            print(f"Scraping {url}...")
            content = scrape_url(url)
            contents.append(content)
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            contents.append("")

        
    # get embeddings
    embeddings = []
    for i, content in enumerate(contents):
        try:
            print(f"Getting embeddings for {urls[i]}...")
            embedding = model.encode(truncate_text(content))
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error getting embeddings for {urls[i]}: {e}")
            embeddings.append(np.zeros(384))  # MiniLM-L6-v2 has 384 dimensions

    embeddings_array = np.array(embeddings)
    
    valid_indices = [i for i, emb in enumerate(embeddings) if np.any(emb)]
    if len(valid_indices) != len(embeddings):
        print(f"Removing {len(embeddings) - len(valid_indices)} failed embeddings")
        embeddings_array = embeddings_array[valid_indices]
        urls = [urls[i] for i in valid_indices]
        contents = [contents[i] for i in valid_indices]
        embeddings = [embeddings[i] for i in valid_indices]

    print(f"\nPerforming clustering on {len(embeddings_array)} websites...")
    
    # cluster_labels, kmeans_model = perform_kmeans_clustering(embeddings_array)
    cluster_labels = perform_agglomerative_clustering(embeddings_array)
    
    print("\nCluster assignments:")
    for i, (url, label) in enumerate(zip(urls, cluster_labels)):
        print(f"Cluster {label}: {url}")
    
    df = pd.DataFrame({
        "id": range(len(urls)),
        "url": urls,
        "content": contents,
        "embedding": [emb.tolist() for emb in embeddings],
        "cluster": cluster_labels.tolist(),
        "time": ["00:01:40"] * len(urls)
    })

    df.to_json("embeddings.json", orient='records', indent=2)
    print(f"\nResults exported to embeddings.json")
    
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nCluster statistics:")
    for cluster, count in zip(unique_clusters, counts):
        print(f"Cluster {cluster}: {count} websites")

    cos_sim_matrix = cosine_similarity(embeddings_array)

    print("\nCosine similarity between each pair of websites:\n")
    num_urls = len(urls)
    for i in range(num_urls):
        for j in range(i + 1, num_urls):
            print(f"Similarity between {urls[i]} and {urls[j]}: {cos_sim_matrix[i, j]:.4f}")
    
    if len(embeddings_array) > 1:
        visualize_clusters(embeddings_array, cluster_labels, urls)


    existing_data = supabase.table("website").select("id").execute()
    existing_ids = [row["id"] for row in existing_data.data]

    for i, row_id in enumerate(range(len(urls))):  # Use your own ID logic if different
        row_data = {
            "url": urls[i],
            "content": contents[i],
            "embedding": embeddings[i].tolist(),
            "cluster": int(cluster_labels[i]),
            "time": "00:01:00"
        }

        if row_id in existing_ids:
            # Update existing row
            supabase.table("website").update(row_data).eq("id", row_id).execute()
            print(f"Updated row ID {row_id}")
        else:
            # Insert new row
            row_data["id"] = row_id
            supabase.table("website").insert(row_data).execute()
            print(f"Inserted new row ID {row_id}")



    print("\nClustering complete!")