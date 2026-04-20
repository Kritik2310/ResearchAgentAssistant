import time
from datetime import datetime

import arxiv
import requests


class RetrievalAgent:
    def __init__(self, semantic_api_key=None, google_api_key=None, google_cse_id=None, max_results=25):
        self.semantic_api_key = semantic_api_key
        self.max_results = max_results
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.google_search_url = "https://www.googleapis.com/customsearch/v1"
        self.retrieved_papers = []
        self.retrieved_datasets = []

    def search_recent_papers(self, query, years_back=3, min_results=5, exclude_ids=None):
        print(f" Searching for papers on: '{query}' via ArXiv")
        current_year = datetime.now().year
        from_year = current_year - years_back

        client = arxiv.Client(page_size=self.max_results, delay_seconds=1, num_retries=3)
        search = arxiv.Search(
            query=query,
            max_results=self.max_results * 2,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers = []
        for result in client.results(search):
            year = result.published.year if result.published else 0
            if year >= from_year:
                papers.append(result)

        # Expand year window if not enough results
        if len(papers) < min_results:
            search2 = arxiv.Search(
                query=query,
                max_results=self.max_results * 3,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            papers = list(client.results(search2))

        structured = self._extract_metadata(papers)

        if exclude_ids:
            structured = [p for p in structured if p["paper_id"] not in exclude_ids]

        structured = structured[: self.max_results]
        self.retrieved_papers = structured
        print(f" Retrieved {len(structured)} papers from ArXiv")
        return structured

    def search_datasets(self, query, num_results=5):
        if not self.google_api_key or not self.google_cse_id:
            return []

        print(f" Searching for datasets on: '{query}'")
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": f"{query} dataset kaggle github",
            "num": num_results,
        }
        try:
            response = requests.get(self.google_search_url, params=params, timeout=10)
            if response.status_code != 200:
                return []
            datasets = [
                {
                    "title": item.get("title", "N/A"),
                    "url": item.get("link", "N/A"),
                    "snippet": item.get("snippet", ""),
                    "source": self._identify_source(item.get("link", "")),
                }
                for item in response.json().get("items", [])
            ]
            self.retrieved_datasets = datasets
            return datasets
        except Exception as e:
            print(f" Dataset search error: {e}")
            return []

    def _extract_metadata(self, results):
        structured = []
        for r in results:
            arxiv_id = r.entry_id.split("/")[-1] if r.entry_id else "N/A"
            structured.append({
                "paper_id": arxiv_id,
                "title": r.title or "N/A",
                "url": r.entry_id or "N/A",
                "authors": [a.name for a in r.authors],
                "publication_year": r.published.year if r.published else "N/A",
                "publication_date": r.published.isoformat() if r.published else "N/A",
                "journal": r.journal_ref or "ArXiv Preprint",
                "doi": r.doi or "N/A",
                "arxiv_id": arxiv_id,
                "keywords": list(r.categories),
                "abstract": r.summary or "",
                "tldr": "",
                "citation_count": 0,
                "publication_types": ["JournalArticle"],
                "pdf_url": r.pdf_url or "N/A",
                "retrieved_at": datetime.now().isoformat(),
            })
        return structured

    def _identify_source(self, url):
        sources = {
            "kaggle.com": "Kaggle",
            "github.com": "GitHub",
            "huggingface.co": "Hugging Face",
            "uci.edu": "UCI ML Repository",
            "openml.org": "OpenML",
            "zenodo.org": "Zenodo",
        }
        for key, name in sources.items():
            if key in url.lower():
                return name
        return "Other"
