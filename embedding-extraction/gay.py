from datasets import load_dataset

# Load dataset
print("Loading Qdrant arXiv dataset...")
ds = load_dataset("Qdrant/arxiv-titles-instructorxl-embeddings")
ds_subset = ds['train'].select(range(2_000_000))

# Extract DOI column and convert to PDF links
pdf_links = [f"https://arxiv.org/pdf/{item['DOI']}.pdf" for item in ds_subset]

# Save to txt file
with open("arxiv_pdf_links.txt", "w") as f:
    for link in pdf_links:
        f.write(link + "\n")

print(f"Saved {len(pdf_links)} PDF links to arxiv_pdf_links.txt")
