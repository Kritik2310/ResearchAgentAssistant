import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


# ---------- Pydantic schemas ----------

class PaperAnalysis(BaseModel):
    summary: str = Field(
        description="4-6 sentence detailed summary: problem addressed, core approach/architecture, "
                    "key results with numbers if available, and what makes it novel vs prior work."
    )
    contribution: str = Field(
        description="1-2 sentence precise statement of the paper's unique scientific contribution."
    )
    methods: List[str] = Field(
        default_factory=list,
        description="Specific named techniques (e.g. 'BERT fine-tuning', 'Graph Attention Network', "
                    "'Monte Carlo Dropout'). No generic terms like 'deep learning'."
    )
    baselines: List[str] = Field(
        default_factory=list,
        description="Named baseline models or methods the paper compares against."
    )
    datasets: List[str] = Field(
        default_factory=list,
        description="Dataset names used in evaluation."
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Specific limitations or open problems the paper itself acknowledges."
    )
    results: str = Field(
        default="",
        description="Key quantitative or qualitative results reported (e.g. 'achieved 94.2% accuracy on X, "
                    "outperforming baseline by 3.1%')."
    )


class SynthesizedSections(BaseModel):
    abstract: str = Field(
        description=(
            "A 200-250 word IEEE-style abstract. Must include: (1) motivation and problem statement, "
            "(2) scope — number of papers reviewed and time period, (3) key themes and methods surveyed, "
            "(4) main findings and identified gaps, (5) proposed methodology summary, "
            "(6) significance to the field. Write in third person, no citations."
        )
    )
    introduction: str = Field(
        description=(
            "A 600-800 word introduction with these parts: "
            "Para 1 — motivate the problem with real-world impact and recent growth of the field. "
            "Para 2 — explain why existing solutions are insufficient (cite specific limitations). "
            "Para 3 — describe the research landscape: major sub-problems, competing approaches, key open questions. "
            "Para 4 — state clearly what this survey covers, how papers were selected, and what the reader will gain. "
            "Para 5 — outline the structure of the paper (Section II covers..., Section III..., etc.). "
            "Use precise technical language. Reference specific paper titles inline where relevant."
        )
    )
    literature_synthesis: str = Field(
        description=(
            "A 1000-1400 word critical literature review organized into THEMATIC SUBSECTIONS (not paper-by-paper). "
            "Each subsection must: name the theme (e.g., 'Retrieval-Augmented Approaches', 'Confidence Estimation'), "
            "discuss 2-4 relevant papers with their specific methods and results, compare their strengths and weaknesses, "
            "and highlight contradictions or complementary findings. "
            "Reference every paper by its title inline. "
            "End with a paragraph synthesizing the overall trajectory of the field — what has improved, what hasn't, "
            "and where the field is converging. Use hedging language where appropriate (e.g., 'however', 'despite', "
            "'in contrast', 'building upon'). Every sentence must carry information — no filler."
        )
    )
    conclusion: str = Field(
        description=(
            "A 300-400 word conclusion with: "
            "Para 1 — restate the survey scope and summarize the 3-4 most important findings. "
            "Para 2 — evaluate the proposed methodology's potential and its limitations honestly. "
            "Para 3 — give 4-5 concrete future research directions, each with a specific actionable suggestion "
            "(not vague statements like 'more research is needed'). "
            "Para 4 — closing statement on the broader impact of solving this problem."
        )
    )
    keywords: List[str] = Field(
        description="7-9 specific IEEE-style keywords. Use domain terms, not generic words like 'research' or 'model'."
    )


# ---------- Agent ----------

class SummarizerAgent:
    def __init__(self, groq_api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self._api_key = groq_api_key
        self._model_name = model_name
        self._llm = ChatGroq(
            model=model_name,
            groq_api_key=groq_api_key,
            temperature=0.3,
        )
        self._analysis_chain = self._llm.with_structured_output(PaperAnalysis)
        self._sections_chain = self._llm.with_structured_output(SynthesizedSections)

    def summarize(self, topic: str, paper_list: list, rag=None) -> dict:
        summaries, all_gaps, all_methods = [], [], []
        all_datasets, all_baselines, all_citations = [], [], []

        for paper in paper_list:
            context = ""
            if rag is not None and paper.get("paper_id"):
                context = rag.get_context(
                    query=f"Key contributions, methods, results, datasets, and limitations for: {paper.get('title', '')}",
                    paper_id=paper.get("paper_id"),
                    top_k=8,
                )
            result = self._analyze_paper(paper, extra_context=context)
            summaries.append(result)
            all_gaps.extend(result.get("gaps", []))
            all_methods.extend(result.get("methods", []))
            all_datasets.extend(result.get("datasets", []))
            all_baselines.extend(result.get("baselines", []))
            all_citations.extend(result.get("citations", []))

        all_gaps = list({g.strip() for g in all_gaps if g.strip()})
        all_methods = list({m.strip() for m in all_methods if m.strip()})
        all_datasets = list({d.strip() for d in all_datasets if d.strip()})
        all_baselines = list({b.strip() for b in all_baselines if b.strip()})
        all_citations = list({c.strip() for c in all_citations if c.strip()})

        proposed_methodology = self._propose_methodology(all_gaps, all_methods, all_baselines, all_datasets)

        llm_sections = self._generate_sections(
            topic=topic,
            summaries=summaries,
            paper_list=paper_list,
            gaps=all_gaps,
            methods=all_methods,
            datasets=all_datasets,
            baselines=all_baselines,
            proposed_methodology=proposed_methodology,
        )

        section_content = {
            "abstract": llm_sections.abstract,
            "introduction": llm_sections.introduction,
            "literature_review": llm_sections.literature_synthesis,
            "research_gaps": all_gaps,
            "methodology": proposed_methodology,
            "conclusion": llm_sections.conclusion,
            "keywords": llm_sections.keywords,
            "datasets": all_datasets,
            "baselines": all_baselines,
        }

        return {
            "per_paper": summaries,
            "collected_gaps": all_gaps,
            "collected_baselines": all_baselines,
            "collected_methods": all_methods,
            "collected_datasets": all_datasets,
            "collected_citations": all_citations,
            "proposed_methodology": proposed_methodology,
            "section_content": section_content,
        }

    # ------------------------------------------------------------------

    def _analyze_paper(self, paper: dict, extra_context: str = "") -> dict:
        context_block = (
            f"\n\nFull-text excerpts from the paper:\n{extra_context}\n"
            if extra_context else ""
        )

        prompt = ChatPromptTemplate.from_template(
            """You are an expert research analyst. Perform a deep analysis of this paper.

Title: {title}
Authors: {authors}
Year: {year}
Abstract: {abstract}
TL;DR: {tldr}
{context_block}

Be specific and technical. Extract exact method names, dataset names, and numerical results where available.
For the summary, go beyond the abstract — explain WHY the approach works, HOW it differs from prior work,
and WHAT concrete improvement it achieves. For gaps, focus on what the authors themselves acknowledge as limitations."""
        )

        try:
            chain = prompt | self._analysis_chain
            result: PaperAnalysis = chain.invoke({
                "title": paper.get("title", ""),
                "authors": ", ".join(paper.get("authors", [])[:5]),
                "year": paper.get("publication_year", ""),
                "abstract": paper.get("abstract", "")[:1200],
                "tldr": paper.get("tldr", ""),
                "context_block": context_block,
            })
            return result.model_dump()
        except Exception as e:
            logger.error(f"Paper analysis failed for '{paper.get('title', 'Unknown')}': {e}")
            return {
                "summary": f"Could not analyze: {paper.get('title', 'Unknown')}",
                "contribution": "",
                "methods": [],
                "baselines": [],
                "datasets": [],
                "gaps": [],
                "results": "",
                "error": str(e),
            }

    def _generate_sections(
        self,
        topic: str,
        summaries: list,
        paper_list: list,
        gaps: List[str],
        methods: List[str],
        datasets: List[str],
        baselines: List[str],
        proposed_methodology: str,
    ) -> SynthesizedSections:

        # Build rich per-paper block with title, authors, year, methods, results, contribution
        paper_details = []
        for p, s in zip(paper_list, summaries):
            authors = ", ".join(p.get("authors", [])[:3])
            if len(p.get("authors", [])) > 3:
                authors += " et al."
            block = (
                f'PAPER: "{p.get("title", "Untitled")}" ({p.get("publication_year", "n.d.")}) by {authors}\n'
                f'  Summary: {s.get("summary", "")}\n'
                f'  Contribution: {s.get("contribution", "")}\n'
                f'  Methods: {", ".join(s.get("methods", []))}\n'
                f'  Results: {s.get("results", "N/A")}\n'
                f'  Gaps: {"; ".join(s.get("gaps", [])[:3])}'
            )
            paper_details.append(block)

        paper_block = "\n\n".join(paper_details)

        prompt = ChatPromptTemplate.from_template(
            """You are a senior IEEE researcher writing a high-quality survey paper for academic publication.
This paper will be read by domain experts and must meet the standard of a top-tier venue.

TOPIC: {topic}

PAPERS REVIEWED ({paper_count} papers):
{paper_details}

RESEARCH GAPS IDENTIFIED:
{gaps}

METHODS ACROSS LITERATURE:
{methods}

DATASETS ENCOUNTERED:
{datasets}

PROPOSED METHODOLOGY:
{methodology}

WRITING REQUIREMENTS:
- Every claim must be grounded in the papers listed above
- Reference papers by their EXACT title inline (e.g., "In 'Self-Evaluating LLMs...', the authors propose...")
- Use precise technical terminology — no vague phrases like "various methods" or "recent advances"
- The literature synthesis MUST have thematic subsections comparing multiple papers per theme
- Quantitative results should be mentioned where available
- Acknowledge contradictions and limitations honestly
- Write at the level of a Nature/IEEE Transactions submission — no generic filler

Generate all four sections now."""
        )

        try:
            chain = prompt | self._sections_chain
            return chain.invoke({
                "topic": topic,
                "paper_count": len(paper_list),
                "paper_details": paper_block[:6000],
                "gaps": "\n".join(f"- {g}" for g in gaps[:12]) or "None identified",
                "methods": ", ".join(methods[:20]) or "None identified",
                "datasets": ", ".join(datasets[:12]) or "None identified",
                "methodology": proposed_methodology[:1200],
            })
        except Exception as e:
            logger.error(f"Section synthesis failed: {e}")
            return SynthesizedSections(
                abstract=f"This survey reviews recent advances in {topic}.",
                introduction=f"The field of {topic} has seen significant growth in recent years.",
                literature_synthesis="See per-paper summaries in the Literature Review section.",
                conclusion=f"This survey of {topic} identifies key trends and open challenges.",
                keywords=[topic] + methods[:4],
            )

    def _propose_methodology(
        self,
        all_gaps: List[str],
        all_methods: List[str],
        all_baselines: List[str],
        all_datasets: List[str],
    ) -> str:
        prompt = (
            "You are a senior researcher designing a novel, publishable research methodology.\n\n"
            "Based on the research context below, propose a SPECIFIC and DETAILED novel methodology "
            "that directly addresses the identified gaps. Be concrete: name exact models, frameworks, "
            "training strategies, evaluation metrics, and datasets you would use. "
            "Explain WHY each component is chosen and HOW it addresses a specific gap. "
            "The methodology should be novel, implementable, and clearly differentiated from existing baselines.\n\n"
            f"Open research gaps:\n{chr(10).join('- ' + g for g in all_gaps[:10]) or '- None identified'}\n\n"
            f"Existing methods in literature: {', '.join(all_methods[:12]) or 'None'}\n"
            f"Baselines to beat: {', '.join(all_baselines[:8]) or 'None'}\n"
            f"Available datasets: {', '.join(all_datasets[:8]) or 'None'}\n\n"
            "Write 3-4 detailed paragraphs. Each paragraph should cover a distinct component of the methodology "
            "(e.g., data pipeline, model architecture, training strategy, evaluation plan). "
            "Be specific enough that another researcher could implement this."
        )
        try:
            response = self._llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Methodology generation failed: {e}")
            return "A structured experimental methodology will be defined based on the identified research gaps."
