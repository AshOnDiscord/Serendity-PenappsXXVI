import requests
from io import BytesIO
from pdfminer.high_level import extract_text
from bs4 import BeautifulSoup
from exa_py import Exa
from cerebras.cloud.sdk import Cerebras
import csv

client = Cerebras(api_key="csk-k26nnt3e8pkyjmpewjhy462ftm59v2j48p43wtewx3mfyc6h")
exa = Exa('10ae6ddd-08a8-4248-a244-d9cb355352e1')

MAX_TOKENS = 6800

def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    words = text.split()
    max_words = int(max_tokens * 0.75)
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text

def summarize_text(text: str) -> str:
    text = truncate_text(text)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Summarize the following content in one paragraph. Do NOT say 'Here is a summary of the content in one paragraph':\n\n{text}",
            }
        ],
        model="llama-4-scout-17b-16e-instruct",
    )
    summary = chat_completion.choices[0].message.content.strip()

    # Remove the unwanted phrase if it appears at the start
    unwanted_phrase = "Here is a summary of the content in one paragraph"
    if summary.lower().startswith(unwanted_phrase.lower()):
        summary = summary[len(unwanted_phrase):].strip(" .\n")
    
    return summary


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
        # Filter out results with "reCAPTCHA" in the title
        filtered_results = [res for res in results.results if "recaptcha" not in res.title.lower()]

        # Sort by score descending
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
        # Pad with empty strings if fewer than top_n results
        while len(similar_data) < top_n * 4:
            similar_data.extend(["", "", "", ""])
        return similar_data
    except Exception as e:
        print(f"Failed to get similar articles for {url}: {e}")
        return [""] * (top_n * 4)


if __name__ == "__main__":
    with open("websites.txt", "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    csv_file = "summaries.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        # Header
        header = ["ID", "URL", "Summary"]
        for i in range(1, 4):
            header.extend([f"Similar_{i}_Title", f"Similar_{i}_Score", f"Similar_{i}_URL", f"Similar_{i}_Summary"])
        writer.writerow(header)

        for i, url in enumerate(urls, start=1):
            print(f"Processing: {url}")
            try:
                text = scrape_url(url)
                summary = summarize_text(text)
                similar_articles = get_top_similar_articles(url)
                writer.writerow([i, url, summary] + similar_articles)
                print(f"Saved row {i}")
            except Exception as e:
                print(f"Failed to process {url}: {e}")
