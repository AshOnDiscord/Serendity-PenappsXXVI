import requests
import xml.etree.ElementTree as ET
import time
import sys
from urllib.parse import urlencode

def scrape_arxiv_cs_papers(target_count=500, batch_size=100):
    """
    Scrape ArXiv papers from computer science category
    
    Args:
        target_count: Number of papers to scrape (default: 500)
        batch_size: Number of papers to fetch per request (max 100 for ArXiv API)
    """
    
    base_url = "http://export.arxiv.org/api/query"
    all_links = []
    
    print(f"Starting to scrape {target_count} ArXiv computer science papers...")
    print("=" * 60)
    
    start_index = 0
    
    while len(all_links) < target_count:
        # Calculate how many papers to fetch in this batch
        remaining = target_count - len(all_links)
        current_batch_size = min(batch_size, remaining)
        
        # ArXiv API parameters
        params = {
            'search_query': 'cat:cs.*',  # All computer science categories
            'start': start_index,
            'max_results': current_batch_size,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        print(f"Fetching papers {start_index + 1} to {start_index + current_batch_size}...")
        
        try:
            # Make request to ArXiv API
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Define namespace for ArXiv
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            
            # Find all entry elements (papers)
            entries = root.findall('atom:entry', namespace)
            
            if not entries:
                print("No more papers found. Ending scrape.")
                break
            
            batch_links = []
            
            for entry in entries:
                # Get the paper ID and construct the link
                id_element = entry.find('atom:id', namespace)
                if id_element is not None:
                    paper_id = id_element.text
                    # Extract just the ArXiv ID from the full URL
                    arxiv_id = paper_id.split('/')[-1]
                    paper_link = f"https://arxiv.org/pdf/{arxiv_id}"
                    batch_links.append(paper_link)
                    
                    # Get title for display
                    title_element = entry.find('atom:title', namespace)
                    title = title_element.text.strip().replace('\n', ' ') if title_element is not None else "No title"
                    
                    print(f"  [{len(all_links) + len(batch_links)}] {title}")
                    print(f"      Link: {paper_link}")
            
            all_links.extend(batch_links)
            start_index += current_batch_size
            
            print(f"Batch complete. Total papers collected: {len(all_links)}")
            print("-" * 60)
            
            # Be respectful to ArXiv servers - add a small delay
            time.sleep(1)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    
    return all_links

def save_links_to_file(links, filename="arxiv_cs_papers.txt"):
    """Save links to a text file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for link in links:
                f.write(f"{link}\n")
        
        print(f"\n✅ Successfully saved {len(links)} links to '{filename}'")
        
    except Exception as e:
        print(f"❌ Error saving to file: {e}")

def main():
    """Main function to run the scraper"""
    
    print("ArXiv Computer Science Papers Scraper")
    print("=====================================")
    
    try:
        # Scrape the papers
        links = scrape_arxiv_cs_papers(target_count=500)
        
        if links:
            print(f"\n🎉 Scraping completed! Collected {len(links)} paper links.")
            
            # Save to file
            save_links_to_file(links)
            
            print(f"\nSample of collected links:")
            for link in links[:5]:
                print(f"  {link}")
            
            if len(links) > 5:
                print(f"  ... and {len(links) - 5} more links")
                
        else:
            print("❌ No papers were collected.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()