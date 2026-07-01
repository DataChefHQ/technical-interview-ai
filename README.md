# Query Classification Pipeline

## Scenario

We're building a query intent classifier for a search engine. The classifier will be used
in production to understand what users are looking for — enabling downstream task like
result reranking.

We have a dataset of search queries. For each query, we also have the top search results
(title, snippet, and URL) from our search engine. The goal is to build a classifier that
assigns an intent category (`sports`, `technology`, `shopping`) to any incoming
query.

You've been handed this codebase with a first attempt at solving this problem. Your task is
to work with it, understand it, and improve it.

## Input format

Each record in `data/queries_with_results.json` represents a single query with its search results:

```json
{
  "query": "best running shoes 2025",
  "results": [
    {
      "title": "Top 10 Running Shoes for 2025",
      "snippet": "We tested dozens of running shoes to find the best options...",
      "url": "https://www.runnersworld.com/gear/top-running-shoes-2025"
    }
  ]
}
```

## Categories

- `sports` - Queries related to sports, athletics, competitions, fitness
- `technology` - Queries related to software, hardware, programming, IT
- `shopping` - Queries related to buying, deals, price comparisons

## How to run

There are two versions of the pipeline. Use whichever you prefer:

**Python script:**

```bash
uv run main.py
```

**Jupyter notebook:**

```bash
uv run jupyter lab pipeline.ipynb
```
