"""
search_example.py

Examples of how to use the text_searcher module.
"""

from text_searcher import TextSearcher, load_searcher, quick_search, search_and_export, demo_search

def main():
    file_path = "mod_parq.parquet"
    
    # Method 1: Create a searcher instance (best for multiple searches)
    print("=== Method 1: Using TextSearcher class ===")
    searcher = load_searcher(file_path)
    
    # Basic searches
    results1 = searcher.simple_search("machine learning")
    print(f"Simple search results: {len(results1)} matches")
    
    results2 = searcher.word_search("AI")
    print(f"Word search results: {len(results2)} matches")
    
    results3 = searcher.multi_word_search(["deep", "neural"], operator="AND")
    print(f"Multi-word AND search: {len(results3)} matches")
    
    results4 = searcher.multi_word_search(["transformer", "attention"], operator="OR")
    print(f"Multi-word OR search: {len(results4)} matches")
    
    # Method 2: Quick one-off searches
    print("\n=== Method 2: Quick searches ===")
    quick_results1 = quick_search(file_path, "computer vision")
    print(f"Quick search results: {len(quick_results1)} matches")
    
    quick_results2 = quick_search(file_path, "reinforcement learning", search_type="multi")
    print(f"Quick multi-word search: {len(quick_results2)} matches")
    
    # Method 3: Search and export
    print("\n=== Method 3: Search and export ===")
    ml_papers = search_and_export(file_path, "machine learning", "ml_papers.parquet")
    
    cv_papers = search_and_export(file_path, ["computer", "vision"], "cv_papers.parquet", 
                                 search_type="multi", operator="AND")
    
    # Method 4: Advanced searches
    print("\n=== Method 4: Advanced searches ===")
    
    # Regex search for papers from specific years
    year_papers = searcher.regex_search(r"20[12][0-9]")
    print(f"Papers mentioning years 2010-2029: {len(year_papers)} matches")
    
    # Fuzzy search
    fuzzy_results = searcher.fuzzy_search("machien learning", similarity_threshold=0.6)  # Note the typo
    print(f"Fuzzy search results: {len(fuzzy_results)} matches")
    
    # Search near coordinates (useful for UMAP visualization)
    nearby_results = searcher.search_near_coordinates("neural", x_center=0, y_center=0, radius=2.0)
    print(f"Neural papers near origin: {len(nearby_results)} matches")
    
    # Search by category
    if 'cs.AI' in searcher.df['category'].values:
        ai_results = searcher.search_by_category("learning", categories=['cs.AI'])
        print(f"Learning papers in cs.AI: {len(ai_results)} matches")
    
    # Get search statistics
    stats = searcher.get_search_statistics("transformer")
    print(f"\nTransformer statistics:")
    print(f"  Total matches: {stats['total_matches']}")
    print(f"  Match percentage: {stats['match_percentage']:.2f}%")
    print(f"  Categories: {list(stats['categories_found'].keys())[:3]}...")
    
    # Method 5: Demo all features
    print("\n=== Method 5: Full demo ===")
    demo_searcher = demo_search(file_path)
    
    return searcher

if __name__ == "__main__":
    searcher = main()