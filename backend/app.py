import asyncio
import logging
import os
import shutil
import tempfile
import time
import zipfile
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from agents.designer_agent import DesignerAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.memory_agent import MemoryAgent
from agents.rag_agent import ResearchRAG
from agents.report_writer_agent import ReportWriterAgent
from agents.retrieval_agent import RetrievalAgent
from agents.summarizer_agent import SummarizerAgent
from tools import memory_store
from tools.latex_ieee_builder import build_bibtex
from tools.report_history import load_reports, save_report

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY or not SEMANTIC_SCHOLAR_API_KEY:
    logger.warning("Missing required API keys — check your .env file")

# ------------------------------------------------------------------ #
# FastAPI app
# ------------------------------------------------------------------ #

app = FastAPI(title="AI Research Assistant Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Progress tracker: session_id -> pipeline step (0-6)
PROCESS_PIPELINE: Dict[str, int] = {}

# ------------------------------------------------------------------ #
# Request schemas
# ------------------------------------------------------------------ #

class PipelineRequest(BaseModel):
    topic: str
    max_papers: int = 10
    years_back: int = 3
    author_name: str = "Research Author"
    author_institution: str = "University"
    author_email: str = "author@university.edu"
    author_department: str = ""
    author_city: str = ""


class ReportRequest(BaseModel):
    topic: str

# ------------------------------------------------------------------ #
# LangGraph state
# ------------------------------------------------------------------ #

class ResearchState(TypedDict):
    topic: str
    author_info: Dict[str, str]
    max_papers: int
    years_back: int
    session_id: str
    papers: List[Dict]
    datasets: List[Dict]
    rag: Any
    analysis: Dict
    evaluation: Dict
    is_valid: bool
    experiment_plan: Dict
    report_markdown: str
    report_latex: str

# ------------------------------------------------------------------ #
# Helper: sync summarizer output into in-process memory_store
# ------------------------------------------------------------------ #

def _sync_to_memory_store(topic: str, papers: List[Dict], analysis: Dict):
    per_paper = analysis.get("per_paper", [])
    summaries_payload, citations_bucket = [], []

    for paper, pp in zip(papers, per_paper):
        citation = {
            "title": paper.get("title", "Untitled"),
            "authors": paper.get("authors", []),
            "year": paper.get("publication_year"),
            "doi": paper.get("doi", "N/A"),
            "journal": paper.get("journal", "N/A"),
            "url": paper.get("url", "N/A"),
        }
        summaries_payload.append({
            "title": paper.get("title", "Untitled"),
            "summary": pp.get("summary", ""),
            "gaps": pp.get("gaps", []),
            "methods": pp.get("methods", []),
            "key_findings": pp.get("baselines", []),
            "limitations": [],
            "citations": citation,
        })
        citations_bucket.append(citation)

    memory_store.save_summaries(topic, summaries_payload)
    if analysis.get("collected_gaps"):
        memory_store.save_gaps(topic, analysis["collected_gaps"])
    if citations_bucket:
        memory_store.save_citations(topic, citations_bucket)
    if analysis.get("section_content"):
        memory_store.save_section_content(topic, analysis["section_content"])

# ------------------------------------------------------------------ #
# LangGraph nodes
# ------------------------------------------------------------------ #

def _node_retrieve(state: ResearchState) -> dict:
    logger.info("[1/6] Retrieving papers for: %s", state["topic"])
    retrieval = RetrievalAgent(
        semantic_api_key=SEMANTIC_SCHOLAR_API_KEY,
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CSE_ID,
        max_results=state["max_papers"],
    )
    papers = retrieval.search_recent_papers(
        query=state["topic"],
        years_back=state["years_back"],
        min_results=state["max_papers"],
    )
    if not papers:
        raise ValueError("No papers retrieved from Semantic Scholar. Try a different topic or increase years_back.")

    datasets = []
    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        datasets = retrieval.search_datasets(state["topic"], num_results=5)

    PROCESS_PIPELINE[state["session_id"]] = 1
    logger.info("Retrieved %d papers, %d datasets", len(papers), len(datasets))
    return {"papers": papers, "datasets": datasets}


def _node_ingest_rag(state: ResearchState) -> dict:
    logger.info("[2/6] Ingesting papers into RAG vector store")
    rag = ResearchRAG()
    stats = rag.ingest_papers(state["papers"])
    logger.info(
        "RAG: %d full-text / %d abstract-only / %d total chunks",
        stats.get("successful_pdfs", 0),
        stats.get("abstract_only", 0),
        stats.get("final_chunks", 0),
    )
    PROCESS_PIPELINE[state["session_id"]] = 2
    return {"rag": rag}


def _node_summarize(state: ResearchState) -> dict:
    logger.info("[3/6] Summarizing and synthesizing %d papers", len(state["papers"]))
    summarizer = SummarizerAgent(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL)
    analysis = summarizer.summarize(
        topic=state["topic"],
        paper_list=state["papers"],
        rag=state["rag"],
    )
    PROCESS_PIPELINE[state["session_id"]] = 3
    logger.info(
        "Analysis complete: %d gaps | %d methods | %d datasets",
        len(analysis.get("collected_gaps", [])),
        len(analysis.get("collected_methods", [])),
        len(analysis.get("collected_datasets", [])),
    )
    return {"analysis": analysis}


def _node_evaluate(state: ResearchState) -> dict:
    logger.info("[4/6] Evaluating analysis quality")
    evaluator = EvaluatorAgent(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL)
    is_valid, eval_report, _ = evaluator.evaluate_analysis(
        query=state["topic"],
        analysis_result=state["analysis"],
        papers=state["papers"],
    )
    PROCESS_PIPELINE[state["session_id"]] = 4
    return {"evaluation": eval_report, "is_valid": is_valid}


def _node_persist_and_design(state: ResearchState) -> dict:
    logger.info("[5/6] Persisting data and generating experiment design")
    topic = state["topic"]
    session_id = state["session_id"]

    memory_dir = os.path.join("data")
    os.makedirs(memory_dir, exist_ok=True)
    json_memory = MemoryAgent(
        memory_file=os.path.join(memory_dir, "research_memory.json"),
        analysis_file=os.path.join(memory_dir, "research_analysis.json"),
    )
    json_memory.store_papers(session_id, topic, state["papers"])
    json_memory.store_analysis(session_id, state["analysis"])
    json_memory.store_evaluation(session_id, state["evaluation"])

    _sync_to_memory_store(topic, state["papers"], state["analysis"])
    memory_store.save_author_info(topic, state["author_info"])

    designer = DesignerAgent()
    design_result = designer.run({"topic": topic})
    PROCESS_PIPELINE[session_id] = 5
    return {"experiment_plan": design_result}


def _node_write_report(state: ResearchState) -> dict:
    logger.info("[6/6] Writing final Markdown and LaTeX reports")
    topic = state["topic"]
    author_info = state.get("author_info")

    writer = ReportWriterAgent()
    md_result = writer.run({"topic": topic, "format": "markdown", "author_info": author_info})
    latex_result = writer.run({"topic": topic, "format": "latex", "author_info": author_info})

    md_content = md_result.get("content", f"# {topic}\n\nReport generation failed.")
    latex_content = latex_result.get("content", "")

    save_report(topic, md_content)
    PROCESS_PIPELINE[state["session_id"]] = 6
    return {"report_markdown": md_content, "report_latex": latex_content}

# ------------------------------------------------------------------ #
# Build LangGraph pipeline (compiled once at startup)
# ------------------------------------------------------------------ #

def _build_pipeline():
    wf = StateGraph(ResearchState)
    wf.add_node("retrieve", _node_retrieve)
    wf.add_node("ingest_rag", _node_ingest_rag)
    wf.add_node("summarize", _node_summarize)
    wf.add_node("evaluate", _node_evaluate)
    wf.add_node("persist_design", _node_persist_and_design)
    wf.add_node("write_report", _node_write_report)

    wf.set_entry_point("retrieve")
    wf.add_edge("retrieve", "ingest_rag")
    wf.add_edge("ingest_rag", "summarize")
    wf.add_edge("summarize", "evaluate")
    wf.add_edge("evaluate", "persist_design")
    wf.add_edge("persist_design", "write_report")
    wf.add_edge("write_report", END)
    return wf.compile()


pipeline_graph = _build_pipeline()

# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@app.post("/run_pipeline")
async def run_pipeline(req: PipelineRequest):
    topic = req.topic.strip()
    if not topic:
        return {"status": "error", "message": "Topic cannot be empty."}
    if len(topic) > 300:
        return {"status": "error", "message": "Topic must be under 300 characters."}

    max_papers = max(1, min(req.max_papers, 25))
    years_back = max(1, min(req.years_back, 10))
    session_id = f"session-{int(time.time())}"
    PROCESS_PIPELINE[session_id] = 0

    initial_state: ResearchState = {
        "topic": topic,
        "author_info": {
            "name": req.author_name,
            "institution": req.author_institution,
            "email": req.author_email,
            "department": req.author_department,
            "city": req.author_city,
        },
        "max_papers": max_papers,
        "years_back": years_back,
        "session_id": session_id,
        "papers": [],
        "datasets": [],
        "rag": None,
        "analysis": {},
        "evaluation": {},
        "is_valid": True,
        "experiment_plan": {},
        "report_markdown": "",
        "report_latex": "",
    }

    try:
        final_state = await asyncio.to_thread(pipeline_graph.invoke, initial_state)
    except Exception as e:
        logger.error("Pipeline failed for '%s': %s", topic, e, exc_info=True)
        return {"status": "error", "message": str(e)}

    return {
        "status": "ok",
        "topic": topic,
        "session_id": session_id,
        "retrieval": {
            "paper_count": len(final_state["papers"]),
            "dataset_count": len(final_state["datasets"]),
        },
        "analysis": final_state["analysis"],
        "evaluation": final_state["evaluation"],
        "is_valid": final_state["is_valid"],
        "experiment_design": final_state["experiment_plan"],
        "report_markdown": {"status": "ok", "content": final_state["report_markdown"]},
        "report_latex": {"status": "ok", "content": final_state["report_latex"]},
    }


@app.post("/download")
async def download_report(req: ReportRequest):
    writer = ReportWriterAgent()
    result = writer.run({"topic": req.topic, "format": "markdown"})
    if result.get("status") != "ok":
        return result

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, result["filename"])
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(result["content"])
    return FileResponse(filepath, media_type="text/markdown", filename=result["filename"])


@app.post("/download-zip")
async def download_overleaf(req: ReportRequest):
    topic = req.topic
    author_info = memory_store.get_author_info(topic) or None

    writer = ReportWriterAgent()
    result = writer.run({"topic": topic, "format": "latex", "author_info": author_info})
    if result.get("status") != "ok":
        logger.error("LaTeX generation failed for '%s': %s", topic, result.get("message"))
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content=result)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp()

    try:
        main_tex = os.path.join(temp_dir, "main.tex")
        with open(main_tex, "w", encoding="utf-8") as f:
            f.write(result["content"])

        local_cls = os.path.join("assets", "IEEEtran.cls")
        if not os.path.exists(local_cls):
            # fallback: look relative to this file's directory
            local_cls = os.path.join(os.path.dirname(__file__), "assets", "IEEEtran.cls")
        if not os.path.exists(local_cls):
            return {"status": "error", "message": "IEEEtran.cls not found on server."}
        cls_path = os.path.join(temp_dir, "IEEEtran.cls")
        with open(local_cls, "rb") as src, open(cls_path, "wb") as dst:
            dst.write(src.read())

        citations = memory_store.get_citations(topic).get("citations", [])
        bib_path = os.path.join(temp_dir, "refs.bib")
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write(build_bibtex(citations))

        zip_name = f"{topic.replace(' ', '_')}_overleaf.zip"
        zip_path = os.path.join(output_dir, zip_name)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(main_tex, "main.tex")
            zf.write(cls_path, "IEEEtran.cls")
            zf.write(bib_path, "refs.bib")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return FileResponse(zip_path, media_type="application/zip", filename=zip_name)


@app.get("/history")
async def get_history():
    return {"history": load_reports()}


@app.get("/progress/{session_id}")
def get_progress(session_id: str):
    return {"id": session_id, "step": PROCESS_PIPELINE.get(session_id, 0)}


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}
