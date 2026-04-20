# ResearchAgentAssistant

## *Automated IEEE Survey Paper Generation using Multi-Agent AI*

> Enter a research topic. Get a fully written, IEEE-formatted survey paper with references — in minutes.

---

## Overview

**ResearchAgentAssistant** is a production-grade multi-agent AI system that automates the complete academic research pipeline. It retrieves real papers from ArXiv, reads and understands them using RAG (Retrieval-Augmented Generation), synthesizes findings using an LLM, evaluates quality, and produces a publication-ready **IEEE-format survey paper** — exported as both Markdown and an Overleaf-ready LaTeX ZIP.

Built with **LangGraph** for orchestration, **LangChain** for structured LLM output, **Groq** as the inference backend, and **Streamlit** as the frontend.

---

## Key Features

- Automated paper retrieval from ArXiv (no API key required)
- RAG pipeline — downloads PDFs, chunks text, embeds with a local SentenceTransformer model
- Structured LLM analysis — LangChain enforces typed output (no regex parsing)
- LangGraph state machine — 6-node pipeline with typed state passing between agents
- IEEE-format LaTeX export — complete `.tex` + `IEEEtran.cls` + `refs.bib` in a ZIP for Overleaf
- Real-time progress stepper — frontend polls pipeline progress every second
- Report history — all generated papers saved and accessible from the UI
- Author information embedded in the LaTeX output (name, institution, email, city)

---

## System Architecture

```text
┌─────────────────────┐         ┌──────────────────────────────────────────┐
│   Streamlit UI      │─ HTTP ─►│            FastAPI Backend               │
│   frontend/app.py   │◄────────│            backend/app.py                │
└─────────────────────┘         │                                          │
                                │  ┌─────────────────────────────────────┐ │
                                │  │       LangGraph State Machine        │ │
                                │  │                                      │ │
                                │  │  retrieve → ingest_rag → summarize   │ │
                                │  │      → evaluate → persist_design     │ │
                                │  │           → write_report → END       │ │
                                │  └─────────────────────────────────────┘ │
                                └──────────────────────────────────────────┘
```

---

## Agent Pipeline

| Step | Agent | Technology | Output |
| ---- | ----- | ---------- | ------ |
| 1 | **RetrievalAgent** | ArXiv API | 10 recent papers with metadata |
| 2 | **ResearchRAG** | SentenceTransformer (local) | PDF chunks + vector embeddings |
| 3 | **SummarizerAgent** | LangChain + Groq LLM | Per-paper analysis + full section synthesis |
| 4 | **EvaluatorAgent** | LangChain + Groq LLM | Quality scores (1–5) per paper and overall |
| 5 | **DesignerAgent** | Pure Python | Experiment plan (hypothesis, datasets, metrics) |
| 6 | **ReportWriterAgent** | Pure Python | Markdown report + IEEE LaTeX report |

---

## Technology Stack

| Layer | Technology |
| ----- | ---------- |
| Orchestration | LangGraph (StateGraph) |
| LLM Framework | LangChain (`with_structured_output`) |
| LLM Provider | Groq — `llama-3.3-70b-versatile` |
| Embeddings | SentenceTransformers — `BAAI/bge-small-en-v1.5` (local, no API key) |
| Paper Source | ArXiv API (free, no key required) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| PDF Parsing | pypdf |
| LaTeX Format | IEEEtran (bundled) |

---

## Project Structure

```text
ResearchAgentAssistant/
├── backend/
│   ├── app.py                     # FastAPI app + LangGraph pipeline
│   ├── agents/
│   │   ├── retrieval_agent.py     # ArXiv paper fetching
│   │   ├── rag_agent.py           # PDF download, chunking, embedding
│   │   ├── summarizer_agent.py    # LangChain structured paper analysis
│   │   ├── evaluator_agent.py     # LangChain quality evaluation
│   │   ├── designer_agent.py      # Experiment design plan
│   │   ├── report_writer_agent.py # Report assembly
│   │   └── memory_agent.py        # JSON persistence
│   ├── tools/
│   │   ├── memory_store.py        # In-process RAM store (topic → data)
│   │   ├── markdown_builder.py    # Sections dict → Markdown
│   │   ├── latex_ieee_builder.py  # Sections dict → IEEE LaTeX + BibTeX
│   │   └── report_history.py      # Report history management
│   ├── assets/
│   │   └── IEEEtran.cls           # Bundled IEEE LaTeX class file
│   ├── Dockerfile
│   ├── .env.example
│   └── requirements.txt
├── frontend/
│   ├── app.py                     # Streamlit UI
│   ├── Dockerfile
│   └── requirements.txt
└── requirements.txt               # Combined root requirements
```

---

## How It Works

1. **User enters** a research topic, sets max papers (5–25), years back (1–8), and author info
2. **RetrievalAgent** queries ArXiv, returns papers filtered by year and relevance
3. **ResearchRAG** downloads PDFs, extracts text, splits into semantic chunks, and embeds them locally
4. **SummarizerAgent** queries the RAG store per paper, then calls Groq LLM via LangChain to extract structured analysis (methods, gaps, results, contribution) — followed by one synthesis call that generates the full abstract, introduction, thematic literature review, and conclusion
5. **EvaluatorAgent** scores each paper summary and the overall analysis using structured LLM output
6. **DesignerAgent** reads from in-memory store and generates a hypothesis, dataset list, and evaluation metrics
7. **ReportWriterAgent** assembles the Markdown and LaTeX reports from stored section content
8. **Frontend** renders the paper preview, offers Markdown download, and packages the LaTeX + IEEEtran.cls + refs.bib into a ZIP for Overleaf

---

## Local Setup

### Prerequisites

- Python 3.10+
- Conda (recommended) or virtualenv
- A free [Groq API key](https://console.groq.com)

### Steps

```bash
# 1. Clone
git clone https://github.com/Kritik2310/ResearchAgentAssistant.git
cd ResearchAgentAssistant

# 2. Create environment
conda create -n research python=3.10
conda activate research

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp backend/.env.example backend/.env
# Edit backend/.env and add your GROQ_API_KEY

# 5. Start backend (Terminal 1)
cd backend
uvicorn app:app --reload --port 8000

# 6. Start frontend (Terminal 2)
cd frontend
streamlit run app.py
# Opens at http://localhost:8501
```

### Environment Variables

```env
# Required
GROQ_API_KEY=your_groq_api_key

# Optional — for dataset search via Google
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id

# Optional — override default LLM model
GROQ_MODEL=llama-3.3-70b-versatile
```

---

## Overleaf Export

The **Download Overleaf ZIP** button packages three files:

| File | Description |
|------|-------------|
| `main.tex` | Full IEEE-format LaTeX paper with your author info |
| `IEEEtran.cls` | IEEE LaTeX class file (bundled — no download needed) |
| `refs.bib` | BibTeX references generated from paper metadata |

Upload the ZIP directly to [Overleaf](https://overleaf.com) → compile → get a PDF.

---

## Deployment

This project is deployed as two separate services on **Hugging Face Spaces**:

- **Backend Space** — Docker runtime, exposes the FastAPI server on port 7860
- **Frontend Space** — Docker runtime, runs the Streamlit app

To connect them, set the `BACKEND_URL` secret in the frontend Space:

```env
BACKEND_URL=https://your-backend-space.hf.space
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/run_pipeline` | Run the full 6-step research pipeline |
| `POST` | `/download` | Download Markdown report |
| `POST` | `/download-zip` | Download Overleaf ZIP (LaTeX package) |
| `GET` | `/history` | List all previously generated reports |
| `GET` | `/progress/{session_id}` | Poll pipeline step (0–6) |
| `GET` | `/health` | Health check |

---

## License

MIT License — free to use, modify, and distribute.
