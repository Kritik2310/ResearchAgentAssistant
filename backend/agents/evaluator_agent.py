import logging
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


# ---------- Pydantic schemas ----------

class PaperEvaluation(BaseModel):
    relevance: int = Field(ge=1, le=5, description="How relevant is this paper to the query (1-5)")
    completeness: int = Field(ge=1, le=5, description="Does the summary capture all key info (1-5)")
    accuracy: int = Field(ge=1, le=5, description="Does the summary accurately reflect the paper (1-5)")
    novelty: int = Field(ge=1, le=5, description="How novel is this paper's contribution (1-5)")
    reasoning: str = Field(description="Brief explanation for the scores")
    issues: List[str] = Field(default_factory=list, description="Any issues found")
    strengths: List[str] = Field(default_factory=list, description="Strengths of this paper/summary")


class OverallEvaluation(BaseModel):
    comprehensiveness: int = Field(ge=1, le=5, description="Are enough papers analyzed with sufficient detail (1-5)")
    coherence: int = Field(ge=1, le=5, description="Do collected findings make sense together (1-5)")
    usefulness: int = Field(ge=1, le=5, description="Is the proposed methodology practical and novel (1-5)")
    overall: int = Field(ge=1, le=5, description="Overall quality of the analysis (1-5)")
    reasoning: str = Field(description="Explanation for these scores")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)


# ---------- Agent ----------

class EvaluatorAgent:
    def __init__(self, groq_api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self._llm = ChatGroq(
            model=model_name,
            groq_api_key=groq_api_key,
            temperature=0.1,
        )
        self._paper_chain = self._llm.with_structured_output(PaperEvaluation)
        self._overall_chain = self._llm.with_structured_output(OverallEvaluation)

        self.min_relevance = 3.0
        self.min_completeness = 3.0
        self.min_novelty = 2.5
        self.min_overall = 3.0

    def evaluate_analysis(self, query: str, analysis_result: dict, papers: list):
        overall_eval = self._evaluate_overall(query, analysis_result)
        paper_evals = [
            self._evaluate_paper(query, paper, summary, idx)
            for idx, (paper, summary) in enumerate(
                zip(papers, analysis_result.get("per_paper", [])), 1
            )
        ]

        avg_relevance = _avg(e["scores"]["relevance"] for e in paper_evals)
        avg_completeness = _avg(e["scores"]["completeness"] for e in paper_evals)
        avg_novelty = _avg(e["scores"]["novelty"] for e in paper_evals)

        is_valid = (
            overall_eval["scores"]["overall"] >= self.min_overall
            and avg_relevance >= self.min_relevance
            and avg_completeness >= self.min_completeness
            and avg_novelty >= self.min_novelty
        )

        recommendations = self._build_recommendations(
            is_valid, overall_eval, paper_evals, avg_relevance, avg_completeness, avg_novelty
        )

        report = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "is_valid": is_valid,
            "overall_evaluation": overall_eval,
            "paper_evaluations": paper_evals,
            "aggregate_scores": {
                "avg_relevance": round(avg_relevance, 2),
                "avg_completeness": round(avg_completeness, 2),
                "avg_novelty": round(avg_novelty, 2),
            },
            "recommendations": recommendations,
        }

        logger.info(
            "Evaluation complete | valid=%s | relevance=%.2f | completeness=%.2f | novelty=%.2f",
            is_valid, avg_relevance, avg_completeness, avg_novelty,
        )
        return is_valid, report, recommendations

    # ------------------------------------------------------------------

    def _evaluate_paper(self, query: str, paper: dict, summary: dict, paper_num: int) -> dict:
        prompt = ChatPromptTemplate.from_template(
            """You are an expert research evaluator.

Query: {query}
Paper Title: {title}
Abstract (excerpt): {abstract}
Generated Summary: {summary}
Methods extracted: {methods}
Gaps identified: {gaps}

Evaluate this paper's summary objectively. Score each dimension 1-5."""
        )
        try:
            chain = prompt | self._paper_chain
            result: PaperEvaluation = chain.invoke({
                "query": query,
                "title": paper.get("title", ""),
                "abstract": (paper.get("abstract") or "")[:400],
                "summary": summary.get("summary", ""),
                "methods": ", ".join(summary.get("methods", [])),
                "gaps": ", ".join(summary.get("gaps", [])),
            })
            return {
                "paper_num": paper_num,
                "paper_title": paper.get("title", "Unknown"),
                "scores": {
                    "relevance": result.relevance,
                    "completeness": result.completeness,
                    "accuracy": result.accuracy,
                    "novelty": result.novelty,
                },
                "reasoning": result.reasoning,
                "issues": result.issues,
                "strengths": result.strengths,
            }
        except Exception as e:
            logger.warning("Paper %d evaluation failed: %s", paper_num, e)
            return _default_paper_eval(paper_num, paper.get("title", "Unknown"))

    def _evaluate_overall(self, query: str, analysis: dict) -> dict:
        prompt = ChatPromptTemplate.from_template(
            """You are an expert research evaluator assessing a full research analysis.

Query: {query}
Papers analyzed: {paper_count}
Gaps found: {gap_count} — sample: {sample_gaps}
Methods collected: {method_count} — sample: {sample_methods}
Proposed methodology (excerpt): {methodology}

Evaluate the overall quality of this analysis."""
        )
        try:
            chain = prompt | self._overall_chain
            result: OverallEvaluation = chain.invoke({
                "query": query,
                "paper_count": len(analysis.get("per_paper", [])),
                "gap_count": len(analysis.get("collected_gaps", [])),
                "sample_gaps": "; ".join(analysis.get("collected_gaps", [])[:3]),
                "method_count": len(analysis.get("collected_methods", [])),
                "sample_methods": "; ".join(analysis.get("collected_methods", [])[:3]),
                "methodology": analysis.get("proposed_methodology", "")[:300],
            })
            return {
                "scores": {
                    "comprehensiveness": result.comprehensiveness,
                    "coherence": result.coherence,
                    "usefulness": result.usefulness,
                    "overall": result.overall,
                },
                "reasoning": result.reasoning,
                "strengths": result.strengths,
                "weaknesses": result.weaknesses,
            }
        except Exception as e:
            logger.warning("Overall evaluation failed: %s", e)
            return _default_overall_eval()

    def _build_recommendations(self, is_valid, overall_eval, paper_evals, avg_rel, avg_comp, avg_nov):
        recs = {
            "action": "ACCEPT" if is_valid else "REFINE",
            "retrieval_refinements": [],
            "summarization_refinements": [],
            "general_improvements": [],
        }
        if not is_valid:
            if avg_rel < self.min_relevance:
                recs["retrieval_refinements"].append({
                    "issue": "Low relevance scores",
                    "action": "Refine search query to retrieve more on-topic papers",
                    "priority": "HIGH",
                })
            if avg_comp < self.min_completeness:
                recs["summarization_refinements"].append({
                    "issue": "Incomplete summaries",
                    "action": "Expand RAG context or add more detailed paper prompts",
                    "priority": "HIGH",
                })
            if avg_nov < self.min_novelty:
                recs["retrieval_refinements"].append({
                    "issue": "Low novelty",
                    "action": "Prioritize papers from last 1-2 years or with novel architectures",
                    "priority": "MEDIUM",
                })
        low_quality = [e for e in paper_evals if any(s < 3 for s in e["scores"].values())]
        if low_quality:
            recs["general_improvements"].append({
                "issue": f"{len(low_quality)} papers scored below threshold",
                "action": f"Review papers: {', '.join(str(p['paper_num']) for p in low_quality)}",
                "priority": "MEDIUM",
            })
        return recs


# ---------- helpers ----------

def _avg(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _default_paper_eval(paper_num: int, title: str) -> dict:
    return {
        "paper_num": paper_num,
        "paper_title": title,
        "scores": {"relevance": 3, "completeness": 3, "accuracy": 3, "novelty": 3},
        "reasoning": "Evaluation unavailable",
        "issues": ["Evaluation failed"],
        "strengths": [],
    }


def _default_overall_eval() -> dict:
    return {
        "scores": {"comprehensiveness": 3, "coherence": 3, "usefulness": 3, "overall": 3},
        "reasoning": "Evaluation unavailable",
        "strengths": [],
        "weaknesses": ["Evaluation failed"],
    }
