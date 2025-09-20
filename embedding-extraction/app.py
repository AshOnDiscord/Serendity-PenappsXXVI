from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from io import BytesIO
from pdfminer.high_level import extract_text
from bs4 import BeautifulSoup
from exa_py import Exa
from cerebras.cloud.sdk import Cerebras
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client
import traceback
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
        response = supabase.table("website").select("id, url, embedding, cluster").execute()
        
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


def perform_agglomerative_clustering(embeddings_array, distance_threshold=0.65):
    """Perform agglomerative clustering on embeddings array"""
    clustering = AgglomerativeClustering(
        n_clusters=None,  # Let distance_threshold decide
        metric='cosine',  # <-- replaced 'affinity' with 'metric'
        linkage='average',
        distance_threshold=distance_threshold
    )
    cluster_labels = clustering.fit_predict(embeddings_array)
    print(f"Number of clusters: {len(np.unique(cluster_labels))}")
    return cluster_labels


def update_all_clusters(distance_threshold=0.65):
    """Re-cluster all existing data and update cluster assignments"""
    try:
        # Get all existing data with embeddings
        existing_data = get_existing_embeddings()
        
        # Filter out items without embeddings
        items_with_embeddings = [item for item in existing_data if item.get('embedding')]
        
        if len(items_with_embeddings) < 2:
            logger.info("Not enough items with embeddings to perform clustering")
            return
        
        # Prepare embeddings array
        embeddings_array = np.array([item['embedding'] for item in items_with_embeddings])
        
        # Perform agglomerative clustering
        logger.info(f"Performing agglomerative clustering on {len(embeddings_array)} items...")
        cluster_labels = perform_agglomerative_clustering(embeddings_array, distance_threshold)
        
        # Update cluster assignments in database
        for i, item in enumerate(items_with_embeddings):
            new_cluster = int(cluster_labels[i])  # Convert numpy int to Python int
            
            # Update in database
            supabase.table("website").update({
                "cluster": new_cluster
            }).eq("id", item['id']).execute()
            
            logger.info(f"Updated item {item['id']} to cluster {new_cluster}")
        
        logger.info(f"Successfully re-clustered all items into {len(np.unique(cluster_labels))} clusters")
        return len(np.unique(cluster_labels))
        
    except Exception as e:
        logger.error(f"Error updating clusters: {e}")
        raise


def determine_cluster_agglomerative(new_embedding, existing_data, distance_threshold=0.65):
    """Determine cluster for new website using agglomerative clustering"""
    if not existing_data:
        return 0
    
    # Filter existing data to only include items with embeddings
    items_with_embeddings = [item for item in existing_data if item.get('embedding')]
    
    if not items_with_embeddings:
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
    
    # Perform clustering
    cluster_labels = perform_agglomerative_clustering(embeddings_array, distance_threshold)
    
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
        distance_threshold = data.get('distance_threshold', 0.65)
        
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
        
        # Step 3: Get existing data and determine cluster using agglomerative clustering
        logger.info("Determining cluster with agglomerative clustering...")
        existing_data = get_existing_embeddings()
        cluster = determine_cluster_agglomerative(embedding_list, existing_data, distance_threshold)
        
        # Step 4: Get next ID and insert into database
        logger.info("Inserting into database...")
        next_id = get_next_id()
        
        row_data = {
            "id": next_id,
            "url": url,
            "content": content,
            "embedding": embedding_list,
            "cluster": cluster,
            "time": "00:01:00"
        }
        
        supabase.table("website").insert(row_data).execute()
        
        logger.info(f"Successfully added website with ID {next_id}, cluster {cluster}")
        
        return jsonify({
            'success': True,
            'id': next_id,
            'url': url,
            'cluster': cluster,
            'content_length': len(content),
            'distance_threshold': distance_threshold,
            'message': 'Website successfully processed and added to database'
        }), 201
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {e}")
        return jsonify({'error': f'Failed to fetch URL: {str(e)}'}), 400
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/recluster', methods=['POST'])
def recluster_all():
    """API endpoint to re-cluster all existing data"""
    try:
        data = request.get_json() or {}
        distance_threshold = data.get('distance_threshold', 0.65)
        
        logger.info(f"Re-clustering all data with distance_threshold={distance_threshold}")
        
        num_clusters = update_all_clusters(distance_threshold)
        
        return jsonify({
            'success': True,
            'num_clusters': num_clusters,
            'distance_threshold': distance_threshold,
            'message': f'Successfully re-clustered all data into {num_clusters} clusters'
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
        cluster_response = supabase.table("website").select("cluster").execute()
        clusters = [item['cluster'] for item in cluster_response.data if item['cluster'] is not None]
        
        cluster_counts = {}
        for cluster in clusters:
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        
        return jsonify({
            'total_websites': total_count,
            'total_clusters': len(cluster_counts),
            'cluster_distribution': cluster_counts
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Failed to get statistics'}), 500


@app.route('/list_websites', methods=['GET'])
def list_websites():
    """List all websites in the database"""
    try:
        response = supabase.table("website").select("id, url, cluster").order("id").execute()
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


if __name__ == '__main__':
    print("Starting Website Processing API Server...")
    print("Available endpoints:")
    print("  POST /add_website              - Add a new website (supports distance_threshold parameter)")
    print("  POST /recluster                - Re-cluster all existing data (supports distance_threshold parameter)")
    print("  GET  /health                   - Health check")
    print("  GET  /stats                    - Database statistics (total websites, cluster distribution)")
    print("  GET  /list_websites            - List all websites (id, url, cluster)")
    print("  GET  /summarize_cluster/<id>   - Summarize all websites in a specific cluster using Cerebras API")
    print("  GET  /all_data/                - Get all")

    app.run(debug=True, host='0.0.0.0', port=5000)