import requests
import xml.etree.ElementTree as ET
import time
import sys

def scrape_arxiv_papers(categories, target_count=500, batch_size=100):
    """
    Scrape ArXiv papers from multiple categories.
    
    Args:
        categories (list[str]): List of arXiv categories (e.g., ["cs.*", "q-bio.*"])
        target_count (int): Total number of papers to scrape
        batch_size (int): Number of papers per request (max 100 for ArXiv API)
    """
    base_url = "http://export.arxiv.org/api/query"
    all_links = []

    print(f"Starting to scrape {target_count} ArXiv papers from categories {categories}")
    print("=" * 60)

    # Split target count across categories
    per_category = target_count // len(categories)

    for category in categories:
        print(f"\n📂 Scraping category: {category}")
        start_index = 0
        cat_links = []

        while len(cat_links) < per_category:
            remaining = per_category - len(cat_links)
            current_batch_size = min(batch_size, remaining)

            params = {
                "search_query": f"cat:{category}",
                "start": start_index,
                "max_results": current_batch_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }

            print(f"Fetching {category}: papers {start_index + 1} to {start_index + current_batch_size}...")

            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                root = ET.fromstring(response.content)

                namespace = {"atom": "http://www.w3.org/2005/Atom"}
                entries = root.findall("atom:entry", namespace)

                if not entries:
                    print("No more papers found in this category.")
                    break

                for entry in entries:
                    id_element = entry.find("atom:id", namespace)
                    if id_element is not None:
                        paper_id = id_element.text
                        arxiv_id = paper_id.split("/")[-1]
                        paper_link = f"https://arxiv.org/pdf/{arxiv_id}"

                        title_element = entry.find("atom:title", namespace)
                        title = title_element.text.strip().replace("\n", " ") if title_element is not None else "No title"

                        cat_links.append(paper_link)
                        print(f"  [{len(cat_links)}] {title}")
                        print(f"      Link: {paper_link}")

                start_index += current_batch_size
                time.sleep(1)  # respect arXiv API rate

            except Exception as e:
                print(f"⚠️ Error fetching {category}: {e}, retrying in 5s...")
                time.sleep(5)
                continue

        print(f"✅ Category {category} done, collected {len(cat_links)} papers.")
        all_links.extend(cat_links)

    return all_links


def save_links_to_file(links, filename="arxiv_papers.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for link in links:
                f.write(f"{link}\n")
        print(f"\n✅ Saved {len(links)} links to '{filename}'")
    except Exception as e:
        print(f"❌ Error saving to file: {e}")


def main():
    print("ArXiv CS + Bio Papers Scraper")
    print("=============================")

    try:
        links = scrape_arxiv_papers(categories=["cs.*", "q-bio.*"], target_count=500)

        if links:
            print(f"\n🎉 Scraping completed! Collected {len(links)} paper links.")
            save_links_to_file(links)

            print("\nSample of collected links:")
            for link in links[:5]:
                print(f"  {link}")
            if len(links) > 5:
                print(f"  ... and {len(links) - 5} more links")

        else:
            print("❌ No papers were collected.")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
