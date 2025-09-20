from exa_py import Exa

exa = Exa('10ae6ddd-08a8-4248-a244-d9cb355352e1')

results = exa.find_similar_and_contents(
    url="https://arxiv.org/abs/2307.06435",
    text=True,
    summary={"query": "Key advancements, details, notes, or applications"},
)

sorted_results = sorted(results.results, key=lambda x: x.score, reverse=True)

top_3 = sorted_results[:3]

for res in top_3:
    print("Title:", res.title)
    print("Score:", res.score)
    print("URL:", res.url)
    print("Summary:", res.summary)
    print("-" * 80)
