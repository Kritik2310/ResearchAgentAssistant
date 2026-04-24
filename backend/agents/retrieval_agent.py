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
        self._ss_base = "https://api.semanticscholar.org/graph/v1/paper/search"
        self._ss_fields = (
            "paperId,title,url,abstract,authors,year,citationCount,"
            "tldr,externalIds,publicationTypes,publicationDate,"
            "journal,fieldsOfStudy,openAccessPdf,isOpenAccess"
        )
        self.retrieved_papers = []
        self.retrieved_datasets = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def search_recent_papers(self, query, years_back=3, min_results=5, exclude_ids=None):
        current_year = datetime.now().year
        from_year = current_year - years_back

        papers = []

        # --- Primary: Semantic Scholar (if key available) ---
        if self.semantic_api_key:
            print(f" [SS] Searching Semantic Scholar for: '{query}'")
            papers = self._search_semantic_scholar(query, from_year)
            if len(papers) < min_results:
                print(f" [SS] Only {len(papers)} results — widening to all years")
                papers = self._search_semantic_scholar(query, from_year=None)

        # --- Fallback: ArXiv ---
        if len(papers) < min_results:
            if self.semantic_api_key:
                print(f" [ArXiv] Semantic Scholar returned too few results, falling back to ArXiv")
            else:
                print(f" [ArXiv] No Semantic Scholar key — searching ArXiv for: '{query}'")
            arxiv_papers = self._search_arxiv(query, from_year, min_results)
            # Merge: SS results first, ArXiv fills the gap
            existing_titles = {p["title"].lower() for p in papers}
            for p in arxiv_papers:
                if p["title"].lower() not in existing_titles:
                    papers.append(p)
                    existing_titles.add(p["title"].lower())

        if exclude_ids:
            papers = [p for p in papers if p["paper_id"] not in exclude_ids]

        papers = papers[: self.max_results]
        self.retrieved_papers = papers
        print(f" Retrieved {len(papers)} papers total")
        return papers

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

    # ------------------------------------------------------------------ #
    # Semantic Scholar
    # ------------------------------------------------------------------ #

    def _search_semantic_scholar(self, query, from_year=None):
        headers = {"x-api-key": self.semantic_api_key}
        params = {
            "query": query,
            "fields": self._ss_fields,
            "limit": min(self.max_results, 100),
        }
        if from_year:
            params["year"] = f"{from_year}-{datetime.now().year}"

        papers = []
        for attempt in range(3):
            try:
                resp = requests.get(self._ss_base, params=params, headers=headers, timeout=15)
                time.sleep(1)

                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    papers = [self._structure_ss_paper(p) for p in data if p.get("title")]
                    print(f" [SS] Got {len(papers)} papers")
                    return papers

                if resp.status_code == 429:
                    wait = 10 * (attempt + 1)
                    print(f" [SS] Rate limited — waiting {wait}s...")
                    time.sleep(wait)
                    continue

                if resp.status_code == 403:
                    print(f" [SS] API key rejected (403) — check your key")
                    return []

                print(f" [SS] Error {resp.status_code}")
                return []

            except Exception as e:
                print(f" [SS] Request error: {e}")
                time.sleep(5)

        return papers

    def _structure_ss_paper(self, paper):
        external = paper.get("externalIds") or {}
        open_pdf = paper.get("openAccessPdf") or {}
        arxiv_id = external.get("ArXiv", "")

        pdf_url = "N/A"
        if paper.get("isOpenAccess") and open_pdf.get("url"):
            pdf_url = open_pdf["url"]
        elif arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        elif external.get("DOI"):
            pdf_url = f"https://doi.org/{external['DOI']}"

        return {
            "paper_id": paper.get("paperId", "N/A"),
            "title": paper.get("title", "N/A"),
            "url": paper.get("url", "N/A"),
            "authors": [a.get("name") for a in paper.get("authors", [])],
            "publication_year": paper.get("year", "N/A"),
            "publication_date": paper.get("publicationDate", "N/A"),
            "journal": (paper.get("journal") or {}).get("name", "N/A"),
            "doi": external.get("DOI", "N/A"),
            "arxiv_id": arxiv_id or "N/A",
            "keywords": paper.get("fieldsOfStudy") or [],
            "abstract": paper.get("abstract") or "",
            "tldr": (paper.get("tldr") or {}).get("text", ""),
            "citation_count": paper.get("citationCount", 0),
            "publication_types": paper.get("publicationTypes") or [],
            "pdf_url": pdf_url,
            "source": "semantic_scholar",
            "retrieved_at": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------ #
    # ArXiv fallback
    # ------------------------------------------------------------------ #

    def _search_arxiv(self, query, from_year, min_results):
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
                papers.append(self._structure_arxiv_paper(result))

        if len(papers) < min_results:
            search2 = arxiv.Search(
                query=query,
                max_results=self.max_results * 3,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            papers = [self._structure_arxiv_paper(r) for r in client.results(search2)]

        print(f" [ArXiv] Got {len(papers)} papers")
        return papers

    def _structure_arxiv_paper(self, result):
        arxiv_id = result.entry_id.split("/")[-1] if result.entry_id else "N/A"
        return {
            "paper_id": arxiv_id,
            "title": result.title or "N/A",
            "url": result.entry_id or "N/A",
            "authors": [a.name for a in result.authors],
            "publication_year": result.published.year if result.published else "N/A",
            "publication_date": result.published.isoformat() if result.published else "N/A",
            "journal": result.journal_ref or "ArXiv Preprint",
            "doi": result.doi or "N/A",
            "arxiv_id": arxiv_id,
            "keywords": list(result.categories),
            "abstract": result.summary or "",
            "tldr": "",
            "citation_count": 0,
            "publication_types": ["JournalArticle"],
            "pdf_url": result.pdf_url or "N/A",
            "source": "arxiv",
            "retrieved_at": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

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
