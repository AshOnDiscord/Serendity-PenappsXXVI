"""
text_searcher.py

A comprehensive text search module for searching through parquet datasets.
Designed for use with datasets containing 'text' columns.

Author: Generated for ArXiv UMAP dataset analysis
"""

import pandas as pd
import numpy as np
import re
from typing import List, Union, Optional, Dict, Any

class TextSearcher:
    """
    A comprehensive text search class for searching through the 'text' column
    of parquet datasets.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the searcher with a DataFrame.
        
        Args:
            df: DataFrame containing the 'text' column to search
        """
        self.df = df.copy()
        # Create a lowercase version for case-insensitive searches
        self.df['text_lower'] = self.df['text'].astype(str).str.lower()
    
    def simple_search(self, query: str, case_sensitive: bool = False) -> pd.DataFrame:
        """
        Simple substring search in the text column.
        
        Args:
            query: Search term
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            DataFrame with matching rows
        """
        if case_sensitive:
            mask = self.df['text'].astype(str).str.contains(query, na=False)
        else:
            mask = self.df['text_lower'].str.contains(query.lower(), na=False)
        
        return self.df[mask].drop('text_lower', axis=1)
    
    def word_search(self, query: str, case_sensitive: bool = False) -> pd.DataFrame:
        """
        Search for complete words (not just substrings).
        
        Args:
            query: Word to search for
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            DataFrame with matching rows
        """
        # Use word boundaries to match complete words only
        pattern = r'\b' + re.escape(query) + r'\b'
        flags = 0 if case_sensitive else re.IGNORECASE
        
        mask = self.df['text'].astype(str).str.contains(pattern, regex=True, flags=flags, na=False)
        return self.df[mask].drop('text_lower', axis=1)
    
    def multi_word_search(self, queries: List[str], operator: str = 'OR', 
                         case_sensitive: bool = False) -> pd.DataFrame:
        """
        Search for multiple words with AND/OR logic.
        
        Args:
            queries: List of search terms
            operator: 'AND' or 'OR' logic
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            DataFrame with matching rows
        """
        if not queries:
            return pd.DataFrame()
        
        masks = []
        text_col = 'text' if case_sensitive else 'text_lower'
        
        for query in queries:
            search_term = query if case_sensitive else query.lower()
            mask = self.df[text_col].str.contains(search_term, na=False)
            masks.append(mask)
        
        if operator.upper() == 'AND':
            final_mask = masks[0]
            for mask in masks[1:]:
                final_mask = final_mask & mask
        else:  # OR
            final_mask = masks[0]
            for mask in masks[1:]:
                final_mask = final_mask | mask
        
        return self.df[final_mask].drop('text_lower', axis=1)
    
    def regex_search(self, pattern: str, case_sensitive: bool = False) -> pd.DataFrame:
        """
        Search using regular expressions.
        
        Args:
            pattern: Regular expression pattern
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            DataFrame with matching rows
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            mask = self.df['text'].astype(str).str.contains(pattern, regex=True, flags=flags, na=False)
            return self.df[mask].drop('text_lower', axis=1)
        except re.error as e:
            print(f"Invalid regex pattern: {e}")
            return pd.DataFrame()
    
    def fuzzy_search(self, query: str, similarity_threshold: float = 0.7) -> pd.DataFrame:
        """
        Fuzzy search using similarity matching.
        
        Args:
            query: Search term
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            DataFrame with matching rows
        """
        from difflib import SequenceMatcher
        
        def similarity_score(text1: str, text2: str) -> float:
            return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        query_lower = query.lower()
        matches = []
        
        for idx, text in enumerate(self.df['text'].astype(str)):
            # Check if query appears as substring (faster check first)
            if query_lower in text.lower():
                matches.append(idx)
            else:
                # Check for words with similar spelling
                words = text.lower().split()
                for word in words:
                    if similarity_score(query_lower, word) > similarity_threshold:
                        matches.append(idx)
                        break
        
        return self.df.iloc[matches].drop('text_lower', axis=1)
    
    def search_by_category(self, query: str, categories: List[str] = None, 
                          case_sensitive: bool = False) -> pd.DataFrame:
        """
        Search within specific categories.
        
        Args:
            query: Search term
            categories: List of categories to search in (None for all)
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            DataFrame with matching rows
        """
        df_filtered = self.df
        
        # Filter by categories first
        if categories:
            df_filtered = df_filtered[df_filtered['category'].isin(categories)]
        
        # Then search in text
        if case_sensitive:
            mask = df_filtered['text'].astype(str).str.contains(query, na=False)
        else:
            mask = df_filtered['text_lower'].str.contains(query.lower(), na=False)
        
        return df_filtered[mask].drop('text_lower', axis=1)
    
    def get_search_statistics(self, query: str, case_sensitive: bool = False) -> Dict[str, Any]:
        """
        Get statistics about search results.
        
        Args:
            query: Search term
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            Dictionary with search statistics
        """
        results = self.simple_search(query, case_sensitive)
        
        stats = {
            'total_matches': len(results),
            'total_documents': len(self.df) - 1,  # Exclude the temp column
            'match_percentage': len(results) / (len(self.df) - 1) * 100,
            'categories_found': results['category'].value_counts().to_dict() if len(results) > 0 else {},
            'sample_matches': results['text'].head(3).tolist() if len(results) > 0 else []
        }
        
        return stats
    
    def search_near_coordinates(self, query: str, x_center: float, y_center: float, 
                               radius: float, case_sensitive: bool = False) -> pd.DataFrame:
        """
        Search for text within a specific radius of coordinates.
        Useful for your UMAP visualization.
        
        Args:
            query: Search term
            x_center: X coordinate center
            y_center: Y coordinate center
            radius: Search radius
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            DataFrame with matching rows within the specified area
        """
        # First filter by location
        distance = np.sqrt((self.df['x'] - x_center)**2 + (self.df['y'] - y_center)**2)
        location_mask = distance <= radius
        
        # Then filter by text
        if case_sensitive:
            text_mask = self.df['text'].astype(str).str.contains(query, na=False)
        else:
            text_mask = self.df['text_lower'].str.contains(query.lower(), na=False)
        
        combined_mask = location_mask & text_mask
        return self.df[combined_mask].drop('text_lower', axis=1)

# Convenience functions for quick usage
def load_searcher(file_path: str) -> TextSearcher:
    """
    Load a parquet file and return a TextSearcher instance.
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        TextSearcher instance
    """
    df = pd.read_parquet(file_path)
    return TextSearcher(df)

def quick_search(file_path: str, query: str, search_type: str = 'simple', **kwargs) -> pd.DataFrame:
    """
    Quick search function for one-off searches.
    
    Args:
        file_path: Path to parquet file
        query: Search term
        search_type: Type of search ('simple', 'word', 'regex', 'fuzzy', 'multi')
        **kwargs: Additional arguments for the search method
        
    Returns:
        DataFrame with search results
    """
    searcher = load_searcher(file_path)
    
    if search_type == 'simple':
        return searcher.simple_search(query, **kwargs)
    elif search_type == 'word':
        return searcher.word_search(query, **kwargs)
    elif search_type == 'regex':
        return searcher.regex_search(query, **kwargs)
    elif search_type == 'fuzzy':
        return searcher.fuzzy_search(query, **kwargs)
    elif search_type == 'multi':
        # For multi-word search, query should be a list
        if isinstance(query, str):
            query = query.split()
        return searcher.multi_word_search(query, **kwargs)
    else:
        raise ValueError(f"Unknown search type: {search_type}")

def search_and_export(file_path: str, query: str, output_path: str, 
                      search_type: str = 'simple', **kwargs):
    """
    Search and export results to a new parquet file.
    
    Args:
        file_path: Path to input parquet file
        query: Search term
        output_path: Path for output parquet file
        search_type: Type of search to perform
        **kwargs: Additional search arguments
    """
    results = quick_search(file_path, query, search_type, **kwargs)
    results.to_parquet(output_path, index=False)
    print(f"Exported {len(results)} search results to {output_path}")
    return results

def get_categories(file_path: str) -> List[str]:
    """
    Get all unique categories from the dataset.
    
    Args:
        file_path: Path to parquet file
        
    Returns:
        List of unique categories
    """
    df = pd.read_parquet(file_path)
    return df['category'].dropna().unique().tolist()

def search_stats(file_path: str, query: str, **kwargs) -> Dict[str, Any]:
    """
    Get search statistics without creating a persistent searcher.
    
    Args:
        file_path: Path to parquet file
        query: Search term
        **kwargs: Additional search arguments
        
    Returns:
        Dictionary with search statistics
    """
    searcher = load_searcher(file_path)
    return searcher.get_search_statistics(query, **kwargs)

# Demo function
def demo_search(file_path: str = "mod_parq.parquet"):
    """
    Demo function showing various search capabilities.
    
    Args:
        file_path: Path to your parquet file
    """
    print(f"Loading dataset from {file_path}...")
    searcher = load_searcher(file_path)
    
    print(f"Dataset loaded: {len(searcher.df)} total records")
    
    # Show available categories
    categories = get_categories(file_path)
    print(f"Available categories: {categories[:5]}..." if len(categories) > 5 else f"Available categories: {categories}")
    
    # Demo searches
    print("\n=== Demo Searches ===")
    
    # Search for LOWDENSE points
    lowdense = searcher.simple_search("LOWDENSE")
    print(f"Found {len(lowdense)} LOWDENSE points")
    
    # Search for neural networks
    neural = searcher.multi_word_search(["neural", "network"], operator="AND")
    print(f"Found {len(neural)} papers about neural networks")
    
    # Get statistics
    stats = searcher.get_search_statistics("deep learning")
    print(f"Deep learning papers: {stats['total_matches']} ({stats['match_percentage']:.1f}%)")
    
    return searcher