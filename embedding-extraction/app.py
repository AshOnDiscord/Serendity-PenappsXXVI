from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from io import BytesIO
from pdfminer.high_level import extract_text
from bs4 import BeautifulSoup
import json
import os
import pandas as pd
from exa_py import Exa
from sklearn.metrics.pairwise import cosine_similarity
from cerebras.cloud.sdk import Cerebras
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from supabase import create_client, Client
import umap
import traceback
from sklearn.neighbors import KernelDensity, NearestNeighbors
import alphashape
from shapely.geometry import Point
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize services
client = Cerebras(api_key="csk-k26nnt3e8pkyjmpewjhy462ftm59v2j48p43wtewx3mfyc6h")
exa = Exa('10ae6ddd-08a8-4248-a244-d9cb355352e1')
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


PARQUET_FILE_PATH = "/home/thinkies/Git/Serendity-PenappsXXVI/embedding-extraction/mod_parq.parquet"  # Update this path
# Supabase configuration
url = "https://ewypztrcgtezvdvjmmgt.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImV3eXB6dHJjZ3RlenZkdmptbWd0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgzNjYyNDMsImV4cCI6MjA3Mzk0MjI0M30.nTSX4VIbwDdiYjpzvWNapaw9H1hlJOF91dz8aqrpodg"
supabase: Client = create_client(url, key)

MAX_TOKENS = 6800


def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Truncate text to fit within token limits"""
    words = text.split()
    max_words = int(max_tokens * 0.75)
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text


def scrape_url(url: str) -> str:
    """Scrape content from a URL, handling both web pages and PDFs"""
    try:
        resp = requests.get(url, timeout=30)
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

        # Filter out very short lines
        filtered_lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 15]
        return "\n".join(filtered_lines)
    
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        raise


def get_existing_embeddings():
    """Fetch existing embeddings and clusters from Supabase"""
    try:
        response = supabase.table("website").select("id, url, embedding, cluster, x, y").execute()
        
        # Clean up the embedding data
        cleaned_data = []
        for item in response.data:
            if item.get('embedding'):
                try:
                    # Handle different embedding formats
                    embedding = item['embedding']
                    if isinstance(embedding, str):
                        # Remove numpy string wrapper if present
                        if embedding.startswith("np.str_('") and embedding.endswith("')"):
                            embedding = embedding[9:-2]  # Remove np.str_(' and ')
                        elif embedding.startswith('[') and embedding.endswith(']'):
                            # Parse as list string
                            embedding = eval(embedding)  # Safe here since we control the data
                    
                    # Ensure it's a list of floats
                    if isinstance(embedding, (list, np.ndarray)):
                        item['embedding'] = [float(x) for x in embedding]
                        cleaned_data.append(item)
                    else:
                        logger.warning(f"Skipping invalid embedding format for item {item.get('id')}")
                        
                except Exception as e:
                    logger.error(f"Error processing embedding for item {item.get('id')}: {e}")
                    continue
            else:
                cleaned_data.append(item)
        
        return cleaned_data
    except Exception as e:
        logger.error(f"Error fetching existing data: {e}")
        return []

def clean_text_for_db(text: str) -> str:
    """Remove characters that PostgreSQL cannot store (e.g., null bytes)"""
    return text.replace('\x00', '')  # remove null bytes


def perform_kmeans_clustering(embeddings_array, n_clusters=10, random_state=42):
    """Perform K-means clustering on embeddings array"""
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(embeddings_array)
    logger.info(f"K-means clustering completed with {n_clusters} clusters")
    return cluster_labels, kmeans


def calculate_umap_coordinates(embeddings_array, n_neighbors=10, min_dist=0.1, n_components=2, random_state=42):
    """Calculate UMAP coordinates for the given embeddings array"""
    try:
        # Adjust n_neighbors if we have fewer samples
        actual_n_neighbors = min(n_neighbors, len(embeddings_array) - 1)
        if actual_n_neighbors < 2:
            actual_n_neighbors = 2
        
        logger.info(f"Computing UMAP with n_neighbors={actual_n_neighbors}, min_dist={min_dist}")
        
        reducer = umap.UMAP(
            n_neighbors=actual_n_neighbors, 
            min_dist=min_dist, 
            n_components=n_components,
            random_state=random_state,
            metric='cosine'  # Use cosine distance to match clustering
        )
        
        umap_coords = reducer.fit_transform(embeddings_array)
        return umap_coords, reducer
    
    except Exception as e:
        logger.error(f"Error calculating UMAP coordinates: {e}")
        raise



def update_all_clusters(n_clusters=10):
    """Re-cluster all existing data using K-means and update cluster assignments"""
    try:
        # Get all existing data with embeddings
        existing_data = get_existing_embeddings()
        
        # Filter out items without embeddings
        items_with_embeddings = [item for item in existing_data if item.get('embedding')]
        
        if len(items_with_embeddings) < n_clusters:
            logger.info(f"Not enough items ({len(items_with_embeddings)}) for {n_clusters} clusters")
            return len(items_with_embeddings) if items_with_embeddings else 0
        
        # Prepare embeddings array
        embeddings_array = np.array([item['embedding'] for item in items_with_embeddings])
        
        # Perform K-means clustering
        logger.info(f"Performing K-means clustering on {len(embeddings_array)} items with {n_clusters} clusters...")
        cluster_labels, kmeans_model = perform_kmeans_clustering(embeddings_array, n_clusters)
        
        # Update cluster assignments in database
        for i, item in enumerate(items_with_embeddings):
            new_cluster = int(cluster_labels[i])  # Convert numpy int to Python int
            
            # Update in database
            supabase.table("website").update({
                "cluster": new_cluster
            }).eq("id", item['id']).execute()
            
            logger.info(f"Updated item {item['id']} to cluster {new_cluster}")
        
        logger.info(f"Successfully clustered all items into {n_clusters} clusters")
        return n_clusters
        
    except Exception as e:
        logger.error(f"Error updating clusters: {e}")
        raise



def determine_cluster_kmeans(new_embedding, existing_data, n_clusters=10):
    """Determine cluster for new website using K-means clustering"""

    if not existing_data:
        return 0
    
    # Filter existing data to only include items with embeddings
    items_with_embeddings = [item for item in existing_data if item.get('embedding')]
    
    if not items_with_embeddings:
        return 0
    
    if len(items_with_embeddings) < n_clusters:
        # If we don't have enough existing items, just assign to cluster 0
        return 0
    
    # Combine all embeddings (existing + new)
    all_embeddings = []
    item_ids = []
    
    for item in items_with_embeddings:
        all_embeddings.append(np.array(item['embedding']))
        item_ids.append(item['id'])
    
    # Add the new embedding
    all_embeddings.append(np.array(new_embedding))
    new_item_index = len(all_embeddings) - 1
    
    # Convert to numpy array
    embeddings_array = np.array(all_embeddings)
    
    # Perform K-means clustering
    cluster_labels, kmeans_model = perform_kmeans_clustering(embeddings_array, n_clusters)
    
    # Get the cluster for the new item
    new_cluster = int(cluster_labels[new_item_index])
    
    logger.info(f"Assigned new item to cluster {new_cluster}")
    
    # Update existing items' clusters if they changed
    for i, item in enumerate(items_with_embeddings):
        old_cluster = item.get('cluster')
        new_cluster_label = int(cluster_labels[i])
        
        if old_cluster != new_cluster_label:
            logger.info(f"Updating item {item['id']} cluster from {old_cluster} to {new_cluster_label}")
            supabase.table("website").update({
                "cluster": new_cluster_label
            }).eq("id", item['id']).execute()
    
    return new_cluster


def calculate_umap_for_new_item(new_embedding, existing_data):
    """Calculate UMAP coordinates for a new item given existing data with UMAP coordinates"""
    try:
        # Filter existing data to only include items with embeddings and UMAP coordinates
        items_with_coords = [item for item in existing_data if item.get('embedding') and item.get('x') is not None and item.get('y') is not None]
        
        if len(items_with_coords) < 2:
            # Not enough existing data, calculate fresh UMAP
            all_embeddings = [item['embedding'] for item in existing_data if item.get('embedding')]
            all_embeddings.append(new_embedding)
            embeddings_array = np.array(all_embeddings)
            
            umap_coords, _ = calculate_umap_coordinates(embeddings_array)
            return float(umap_coords[-1, 0]), float(umap_coords[-1, 1])
        
        # Prepare all embeddings (existing + new)
        all_embeddings = [item['embedding'] for item in items_with_coords]
        all_embeddings.append(new_embedding)
        embeddings_array = np.array(all_embeddings)
        
        # Recalculate UMAP for all items (including new one)
        umap_coords, _ = calculate_umap_coordinates(embeddings_array)
        
        # Update existing items' coordinates if they changed significantly
        for i, item in enumerate(items_with_coords):
            new_x = float(umap_coords[i, 0])
            new_y = float(umap_coords[i, 1])
            
            old_x = item.get('x', 0)
            old_y = item.get('y', 0)
            
            # Update if coordinates changed significantly (threshold of 0.1)
            if abs(new_x - old_x) > 0.1 or abs(new_y - old_y) > 0.1:
                logger.info(f"Updating UMAP coordinates for item {item['id']}: ({old_x:.3f}, {old_y:.3f}) -> ({new_x:.3f}, {new_y:.3f})")
                supabase.table("website").update({
                    "x": new_x,
                    "y": new_y
                }).eq("id", item['id']).execute()
        
        # Return coordinates for the new item (last in the array)
        new_x = float(umap_coords[-1, 0])
        new_y = float(umap_coords[-1, 1])
        
        return new_x, new_y
        
    except Exception as e:
        logger.error(f"Error calculating UMAP for new item: {e}")
        # Fallback: return (0, 0) if UMAP calculation fails
        return 0.0, 0.0


def get_next_id():
    """Get the next available ID for the database"""
    try:
        response = supabase.table("website").select("id").order("id", desc=True).limit(1).execute()
        if response.data:
            return response.data[0]["id"] + 1
        return 1
    except Exception as e:
        logger.error(f"Error getting next ID: {e}")
        return 1


@app.route('/add_website', methods=['POST'])
def add_website():
    """API endpoint to add a website to the database"""
    try:
        # Get URL and optional parameters from request
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required'}), 400
        
        url = data['url'].strip()

        n_clusters = data.get('n_clusters', 10)

        
        if not url:
            return jsonify({'error': 'URL cannot be empty'}), 400
        
        logger.info(f"Processing website: {url}")
        
        # Check if URL already exists
        existing_response = supabase.table("website").select("id, url").eq("url", url).execute()
        if existing_response.data:
            return jsonify({
                'error': 'Website already exists',
                'existing_id': existing_response.data[0]['id']
            }), 409
        
        # Step 1: Scrape content
        logger.info("Scraping content...")
        content = scrape_url(url)
        if not content.strip():
            return jsonify({'error': 'No content could be extracted from the URL'}), 400
        content = clean_text_for_db(content)
        
        # Step 2: Generate embedding
        logger.info("Generating embedding...")
        truncated_content = truncate_text(content)
        embedding = model.encode(truncated_content)
        
        # Ensure embedding is a list of Python floats (not numpy types)
        embedding_list = [float(x) for x in embedding.tolist()]
        
        # Step 3: Get existing data and determine cluster using K-means clustering
        logger.info("Determining cluster with K-means clustering...")
        existing_data = get_existing_embeddings()
        cluster = determine_cluster_kmeans(embedding_list, existing_data, n_clusters)
        
        # Step 4: Calculate UMAP coordinates
        logger.info("Calculating UMAP coordinates...")
        umap_x, umap_y = calculate_umap_for_new_item(embedding_list, existing_data)
        
        # Step 5: Get next ID and insert into database
        logger.info("Inserting into database...")
        next_id = get_next_id()
        
        row_data = {
            "id": next_id,
            "url": url,
            "content": content,
            "embedding": embedding_list,
            "cluster": cluster,
            "x": umap_x,
            "y": umap_y,
            "time": "00:01:00"
        }
        
        supabase.table("website").insert(row_data).execute()
        
        logger.info(f"Successfully added website with ID {next_id}, cluster {cluster}, coordinates ({umap_x:.3f}, {umap_y:.3f})")
        
        return jsonify({
            'success': True,
            'id': next_id,
            'url': url,
            'cluster': cluster,
            'x': umap_x,
            'y': umap_y,
            'content_length': len(content),
            'n_clusters': n_clusters,
            'message': 'Website successfully processed and added to database'
        }), 201
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {e}")
        return jsonify({'error': f'Failed to fetch URL: {str(e)}'}), 400
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/calculate_umap', methods=['POST'])
def calculate_umap_for_all():
    """API endpoint to calculate UMAP coordinates for all existing data"""
    try:
        data = request.get_json() or {}
        n_neighbors = data.get('n_neighbors', 10)
        min_dist = data.get('min_dist', 0.1)
        random_state = data.get('random_state', 42)
        
        logger.info(f"Calculating UMAP coordinates for all data with n_neighbors={n_neighbors}, min_dist={min_dist}")
        
        # Get all existing data with embeddings
        existing_data = get_existing_embeddings()
        
        # Filter out items without embeddings
        items_with_embeddings = [item for item in existing_data if item.get('embedding')]
        
        if len(items_with_embeddings) < 2:
            return jsonify({
                'error': 'Not enough items with embeddings to perform UMAP (need at least 2)'
            }), 400
        
        # Prepare embeddings array
        embeddings_array = np.array([item['embedding'] for item in items_with_embeddings])
        
        logger.info(f"Computing UMAP coordinates for {len(embeddings_array)} embeddings...")
        
        # Calculate UMAP coordinates
        umap_coords, reducer = calculate_umap_coordinates(
            embeddings_array, 
            n_neighbors=n_neighbors, 
            min_dist=min_dist,
            random_state=random_state
        )
        
        # Update database with new coordinates
        updated_count = 0
        for i, item in enumerate(items_with_embeddings):
            new_x = float(umap_coords[i, 0])
            new_y = float(umap_coords[i, 1])
            
            # Update in database
            supabase.table("website").update({
                "x": new_x,
                "y": new_y
            }).eq("id", item['id']).execute()
            
            updated_count += 1
            if updated_count % 10 == 0:
                logger.info(f"Updated UMAP coordinates for {updated_count}/{len(items_with_embeddings)} items")
        
        logger.info(f"Successfully calculated and updated UMAP coordinates for {updated_count} items")
        
        # Calculate some statistics about the coordinates
        x_coords = umap_coords[:, 0]
        y_coords = umap_coords[:, 1]
        
        return jsonify({
            'success': True,
            'updated_count': updated_count,
            'parameters': {
                'n_neighbors': n_neighbors,
                'min_dist': min_dist,
                'random_state': random_state
            },
            'coordinate_stats': {
                'x_range': [float(np.min(x_coords)), float(np.max(x_coords))],
                'y_range': [float(np.min(y_coords)), float(np.max(y_coords))],
                'x_mean': float(np.mean(x_coords)),
                'y_mean': float(np.mean(y_coords))
            },
            'message': f'Successfully calculated UMAP coordinates for {updated_count} items'
        }), 200
    
    except Exception as e:
        logger.error(f"Error calculating UMAP: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Failed to calculate UMAP: {str(e)}'}), 500


@app.route('/recluster', methods=['POST'])
def recluster_all():
    """API endpoint to re-cluster all existing data using K-means"""
    try:
        data = request.get_json() or {}

        n_clusters = data.get('n_clusters', 10)

        
        logger.info(f"Re-clustering all data with K-means using {n_clusters} clusters")
        
        num_clusters = update_all_clusters(n_clusters)
        
        return jsonify({
            'success': True,
            'num_clusters': num_clusters,
            'n_clusters_requested': n_clusters,
            'message': f'Successfully re-clustered all data into {num_clusters} clusters using K-means'
        }), 200
    
    except Exception as e:
        logger.error(f"Error re-clustering: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Failed to re-cluster: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'website-processing-api'}), 200


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    try:
        # Get total count
        total_response = supabase.table("website").select("id", count="exact").execute()
        total_count = total_response.count
        
        # Get cluster distribution
        cluster_response = supabase.table("website").select("cluster, x, y").execute()
        clusters = [item['cluster'] for item in cluster_response.data if item['cluster'] is not None]
        
        # Count items with UMAP coordinates
        items_with_coords = [item for item in cluster_response.data if item['x'] is not None and item['y'] is not None]
        
        cluster_counts = {}
        for cluster in clusters:
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        
        return jsonify({
            'total_websites': total_count,
            'total_clusters': len(cluster_counts),
            'items_with_umap_coords': len(items_with_coords),
            'cluster_distribution': cluster_counts
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Failed to get statistics'}), 500


@app.route('/list_websites', methods=['GET'])
def list_websites():
    """List all websites in the database"""
    try:
        response = supabase.table("website").select("id, url, cluster, x, y").order("id").execute()
        return jsonify({
            'websites': response.data,
            'count': len(response.data)
        }), 200
    
    except Exception as e:
        logger.error(f"Error listing websites: {e}")
        return jsonify({'error': 'Failed to list websites'}), 500


@app.route('/summarize_cluster/<int:cluster_id>', methods=['GET'])
def summarize_cluster(cluster_id):
    """Get all websites in a cluster and generate a summary using Cerebras API"""
    try:
        logger.info(f"Fetching websites for cluster {cluster_id}")
        
        # Get all rows for the specified cluster
        response = supabase.table("website").select("id, url, content, cluster").eq("cluster", cluster_id).execute()
        
        if not response.data:
            return jsonify({
                'error': f'No websites found in cluster {cluster_id}'
            }), 404
        
        # Extract first 1000 characters from each row's content
        cluster_contents = []
        website_info = []
        
        for row in response.data:
            content = row.get('content', '')
            if content:
                # Get first 1000 characters
                truncated_content = content[:1000]
                cluster_contents.append(f"Website: {row['url']}\nContent: {truncated_content}\n{'='*50}\n")
                website_info.append({
                    'id': row['id'],
                    'url': row['url'],
                    'content_length': len(content)
                })
        
        if not cluster_contents:
            return jsonify({
                'error': f'No content found for websites in cluster {cluster_id}'
            }), 404
        
        # Combine all content
        combined_content = '\n'.join(cluster_contents)
        
        logger.info(f"Sending {len(combined_content)} characters to Cerebras API for summarization")
        
        # Create prompt for Cerebras API
        prompt = f"""Please analyze and summarize the following content from {len(response.data)} websites that have been clustered together. Provide a very brief summary (maximum 6 words) that captures the essence of what this cluster represents.

Content from Cluster {cluster_id}:

{combined_content}

Do NOT say "Here is a brief summary" or anything similar. Just provide the summary directly."""

        # Call Cerebras API
        try:
            completion = client.chat.completions.create(
                model="llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            summary = completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Cerebras API error: {e}")
            return jsonify({
                'error': f'Failed to generate summary with Cerebras API: {str(e)}'
            }), 500
        
        return jsonify({
            'success': True,
            'cluster_id': cluster_id,
            'website_count': len(response.data),
            'websites': website_info,
            'summary': summary,
            'total_content_chars_analyzed': len(combined_content)
        }), 200
    
    except Exception as e:
        logger.error(f"Error summarizing cluster {cluster_id}: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Failed to summarize cluster: {str(e)}'}), 500



@app.route('/compute_lowdensity', methods=['GET'])
def compute_lowdensity_points():
    """
    Compute lowest-density points (LOWDENSE) from Supabase data
    and insert/update them in the Supabase 'website' table.
    GET request, no parameters exposed.
    """
    try:
        n_lowdense = 50
        bandwidth = 0.1
        alpha = 0.05
        n_neighbors = 20
        max_cv = 0.3

        # Step 1: Fetch existing points from Supabase (ignore id < 1)
        response = supabase.table("website").select("*").execute()
        data = [row for row in response.data if row.get('id', 0) >= 1 and row.get('x') is not None and row.get('y') is not None]

        if not data:
            return jsonify({'error': 'No valid data points found in Supabase'}), 400

        xy = np.array([[row['x'], row['y']] for row in data])
        vector_len = len(data[0].get('vector', [])) if data[0].get('vector') else 128  # fallback length

        # Step 2: Fit KDE
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(xy)

        # Step 3: Grid of candidate points
        grid_res = 200
        xx, yy = np.meshgrid(
            np.linspace(xy[:, 0].min(), xy[:, 0].max(), grid_res),
            np.linspace(xy[:, 1].min(), xy[:, 1].max(), grid_res)
        )
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

        # Step 4: Keep points inside concave hull (alpha shape)
        hull_shape = alphashape.alphashape(xy, alpha)
        inside_mask = np.array([hull_shape.contains(Point(p)) for p in grid_points])
        inside_points = grid_points[inside_mask]

        if len(inside_points) == 0:
            return jsonify({'error': 'No grid points inside alpha shape'}), 400

        # Step 5: Compute KDE densities
        densities = kde.score_samples(inside_points)

        # Step 6: Nearest neighbors metrics
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(xy)
        distances, indices = nbrs.kneighbors(inside_points)
        mean_dists = distances.mean(axis=1)
        cv_dists = distances.std(axis=1) / mean_dists
        vectors = xy[indices[:, 1:]] - inside_points[:, None, :]
        angles = np.arctan2(vectors[..., 1], vectors[..., 0])
        angle_spread = np.ptp(angles, axis=1)
        surrounded_mask = angle_spread > (np.pi * 1.5)
        max_allowed_dist = np.percentile(mean_dists, 75)
        valid_mask = (mean_dists < max_allowed_dist) & surrounded_mask & (cv_dists < max_cv)

        filtered_points = inside_points[valid_mask]
        filtered_densities = densities[valid_mask]

        if len(filtered_points) == 0:
            return jsonify({'error': 'No valid low-density points found'}), 400

        # Step 7: Pick top N lowest-density points
        lowest_idxs = np.argsort(filtered_densities)[:n_lowdense]
        lowdense_points = filtered_points[lowest_idxs]

        # Step 8: Prepare LOWDENSE rows
        new_points = []
        for i, (x_ld, y_ld) in enumerate(lowdense_points, start=1):
            lowdense_id = -i  # negative IDs for LOWDENSE
            row_data = {
                'id': lowdense_id,
                'url': f'LOWDENSE_{i}',
                'content': None,
                'embedding': [0.0] * vector_len,
                'cluster': None,
                'time': None,
                'x': float(x_ld),
                'y': float(y_ld),
            }
            new_points.append(row_data)

        # Step 9: Upsert LOWDENSE points
        for row in new_points:
            supabase.table("website").upsert(row, on_conflict="id").execute()

        return jsonify({
            'success': True,
            'lowdense_count': len(new_points),
            'message': f'Added/Updated {len(new_points)} LOWDENSE points'
        }), 200

    except Exception as e:
        logger.error(f"Error computing LOWDENSE points: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Failed to compute LOWDENSE points: {str(e)}'}), 500




@app.route('/all_data', methods=['GET'])
def get_all_data():
    """Return all website data from Supabase as JSON"""
    try:
        # Fetch all rows from the website table
        response = supabase.table("website").select("*").execute()
        
        if not response.data:
            return jsonify({'data': [], 'message': 'No websites found'}), 200
        
        return jsonify({
            'success': True,
            'count': len(response.data),
            'data': response.data
        }), 200
    
    except Exception as e:
        logger.error(f"Error fetching all data: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to fetch all data'}), 500



@app.route('/all_data_paired', methods=['GET'])
def get_all_data_paired():
    """Return all websites paired with their embeddings as a 2D list"""
    try:
        # Fetch all rows from the website table
        response = supabase.table("website").select("url, embedding").execute()
        
        if not response.data:
            return jsonify({'data': [], 'message': 'No websites found'}), 200
        
        # Create 2D list: [url, embedding]
        paired_list = []
        for row in response.data:
            url = row.get('url')
            embedding = row.get('embedding')
            if embedding:  # Only include if embedding exists
                # Convert numpy strings to float list if needed
                if isinstance(embedding, str):
                    if embedding.startswith("np.str_('") and embedding.endswith("')"):
                        embedding = embedding[9:-2]
                    if embedding.startswith('[') and embedding.endswith(']'):
                        embedding = eval(embedding)
                embedding_list = [float(x) for x in embedding]
                paired_list.append([url, embedding_list])
        
        return jsonify({
            'success': True,
            'count': len(paired_list),
            'data': paired_list
        }), 200
    
    except Exception as e:
        logger.error(f"Error fetching paired data: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to fetch paired data'}), 500


@app.route('/closest_nodes/<int:node_id>', methods=['GET'])
def get_closest_nodes(node_id):
    """
    Given an ID of a specific node, find the two closest nodes by (x, y) coordinates.
    """
    try:
        # Fetch all nodes with valid coordinates
        response = supabase.table("website").select("id, url, x, y").execute()
        data = [row for row in response.data if row.get('x') is not None and row.get('y') is not None]

        # Find the target node
        target = next((row for row in data if row['id'] == node_id), None)
        if not target:
            return jsonify({'error': f'Node with ID {node_id} not found or has no coordinates'}), 404

        target_point = np.array([target['x'], target['y']])

        # Compute distances to all other nodes
        distances = []
        for row in data:
            if row['id'] == node_id:
                continue
            point = np.array([row['x'], row['y']])
            dist = np.linalg.norm(point - target_point)
            distances.append((row, dist))

        # Sort by distance and pick the 2 closest
        distances.sort(key=lambda x: x[1])
        closest_two = [d[0] for d in distances[:2]]

        return jsonify({
            'success': True,
            'target': target,
            'closest_two': closest_two
        }), 200

    except Exception as e:
        logger.error(f"Error finding closest nodes: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Failed to find closest nodes: {str(e)}'}), 500
        

@app.route("/similar_articles", methods=["GET"])
def similar_articles():
    try:
        article_id = request.args.get("id")
        cutoff = float(request.args.get("cutoff", 0.8))  # default cutoff 0.8
        cutoff -= 0.5
        if not article_id:
            return jsonify({"error": "Missing 'id' parameter"}), 400
        if not (0 <= cutoff <= 1):
            return jsonify({"error": "Cutoff must be between 0 and 1"}), 400

        # Fetch the reference article
        ref_resp = supabase.table("website").select("id, embedding").eq("id", article_id).execute()
        if not ref_resp.data:
            return jsonify({"error": f"No article found with ID {article_id}"}), 404

        ref_embedding = ref_resp.data[0]["embedding"]
        if isinstance(ref_embedding, str):
            if ref_embedding.startswith("[") and ref_embedding.endswith("]"):
                ref_embedding = eval(ref_embedding)
        ref_embedding = np.array(ref_embedding).reshape(1, -1)

        # Fetch all other articles
        all_resp = supabase.table("website").select("id, url, embedding").neq("id", article_id).execute()
        if not all_resp.data:
            return jsonify({"error": "No other articles available"}), 404

        similar_items = []
        for row in all_resp.data:
            embedding = row.get("embedding")
            if not embedding:
                continue
            if isinstance(embedding, str):
                if embedding.startswith("[") and embedding.endswith("]"):
                    embedding = eval(embedding)
            embedding = np.array(embedding).reshape(1, -1)

            # Cosine similarity
            sim = cosine_similarity(ref_embedding, embedding)[0][0]
            if sim >= cutoff:
                similar_items.append({
                    "id": row["id"],
                    "url": row.get("url"),
                    "similarity": float(sim)
                })

        # Sort results by similarity
        similar_items.sort(key=lambda x: x["similarity"], reverse=True)

        return jsonify({
            "success": True,
            "reference_id": article_id,
            "cutoff": cutoff,
            "similar_count": len(similar_items),
            "similar_articles": similar_items
        }), 200

    except Exception as e:
        logger.error(f"Error in /similar_articles: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == '__main__':
    print("Starting Website Processing API Server...")
    print("Available endpoints:")
    print("  POST /add_website              - Add a new website (supports n_clusters parameter)")
    print("  POST /recluster                - Re-cluster all existing data using K-means (supports n_clusters parameter)")
    print("  POST /calculate_umap           - Calculate UMAP coordinates for all existing data")
    print("  GET  /health                   - Health check")
    print("  GET  /stats                    - Database statistics (total websites, cluster distribution, UMAP coords)")
    print("  GET  /list_websites            - List all websites (id, url, cluster, x, y)")
    print("  GET  /summarize_cluster/<id>   - Summarize all websites in a specific cluster using Cerebras API")
    print("  GET  /all_data/                - Get all data")
    print("  GET  /all_data_paired/         - Get all websites paired with embeddings")
    print("  GET  /read_parquet             - Read parquet file from predetermined path")

    app.run(debug=True, host='0.0.0.0', port=5000)