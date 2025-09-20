#!/bin/bash

# Check if filename is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename.txt>"
    echo "Example: $0 urls.txt"
    exit 1
fi

FILENAME="$1"

# Check if file exists
if [ ! -f "$FILENAME" ]; then
    echo "Error: File '$FILENAME' not found!"
    exit 1
fi

# Check if file is readable
if [ ! -r "$FILENAME" ]; then
    echo "Error: File '$FILENAME' is not readable!"
    exit 1
fi

echo "Processing URLs from: $FILENAME"
echo "================================"

# Counter for tracking processed URLs
counter=0

# Read file line by line
while IFS= read -r url || [ -n "$url" ]; do
    # Skip empty lines
    if [ -z "$url" ]; then
        continue
    fi
    
    # Trim whitespace
    url=$(echo "$url" | xargs)
    
    # Skip lines that don't look like URLs (basic check)
    if [[ ! "$url" =~ ^https?:// ]]; then
        echo "Skipping invalid URL: $url"
        continue
    fi
    
    counter=$((counter + 1))
    echo "[$counter] Processing: $url"
    
    # Make the curl request
    response=$(curl -s -X POST http://localhost:5000/add_website \
        -H "Content-Type: application/json" \
        -d "{
            \"url\": \"$url\",
            \"distance_threshold\": 0.65
        }")
    
    # Check curl exit status
    if [ $? -eq 0 ]; then
        echo "✓ Success: $response"
    else
        echo "✗ Failed to process: $url"
    fi
    
    echo "---"
    
    # Optional: Add a small delay between requests to avoid overwhelming the server
    # sleep 0.5
    
done < "$FILENAME"

echo "================================"
echo "Finished processing $counter URLs from $FILENAME"


# embedding-atlas umap_arxiv_dataset.parquet --x x --y y --vector vector --text text