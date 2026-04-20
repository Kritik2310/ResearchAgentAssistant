import logging
from tools import memory_store
from tools.markdown_builder import build_markdown_report
from tools.latex_ieee_builder import build_ieee_latex_report

logger = logging.getLogger(__name__)


class ReportWriterAgent:
    def run(self, input_json: dict) -> dict:
        topic = input_json["topic"]
        mode = input_json.get("format", "markdown")
        author_info = input_json.get("author_info")

        section_resp = memory_store.get_section_content(topic)
        if section_resp["status"] != "ok":
            return {
                "status": "error",
                "message": "Section content not found in memory. Run summarizer first.",
            }

        sections = section_resp["section_content"]
        citations = memory_store.get_citations(topic).get("citations", [])

        if mode == "markdown":
            content = build_markdown_report(topic=topic, sections=sections, citations=citations)
            return {
                "status": "ok",
                "topic": topic,
                "format": "markdown",
                "content": content,
                "filename": f"{topic.replace(' ', '_')}.md",
            }

        if mode == "latex":
            content = build_ieee_latex_report(
                topic=topic,
                sections=sections,
                citations=citations,
                author_info=author_info,
            )
            return {
                "status": "ok",
                "topic": topic,
                "format": "latex",
                "content": content,
                "filename": f"{topic.replace(' ', '_')}.tex",
            }

        return {"status": "error", "message": f"Unknown format '{mode}'"}
