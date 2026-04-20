def build_markdown_report(topic: str, sections: dict, citations: list) -> str:
    md = []

    md.append(f"# {topic}\n")
    md.append("---\n")

    if sections.get("abstract"):
        md.append("## Abstract\n")
        md.append(sections["abstract"].strip() + "\n\n")

    if sections.get("keywords"):
        kws = sections["keywords"]
        if isinstance(kws, list):
            kws = ", ".join(kws)
        md.append(f"**Keywords:** {kws}\n\n")

    if sections.get("introduction"):
        md.append("## Introduction\n")
        md.append(sections["introduction"].strip() + "\n\n")

    if sections.get("literature_review"):
        md.append("## Literature Review\n")
        lr = sections["literature_review"]
        if isinstance(lr, list):
            for idx, entry in enumerate(lr, start=1):
                title = entry.get("title", f"Study {idx}")
                summary = entry.get("summary", "")
                md.append(f"### {title}\n")
                md.append(summary.strip() + "\n\n")
        else:
            md.append(lr.strip() + "\n\n")

    if sections.get("research_gaps"):
        md.append("## Research Gaps\n")
        gaps = sections["research_gaps"]
        if isinstance(gaps, list):
            for g in gaps:
                md.append(f"- {g}\n")
        else:
            md.append(gaps.strip() + "\n")
        md.append("\n")

    if sections.get("methodology"):
        md.append("## Proposed Methodology\n")
        md.append(sections["methodology"].strip() + "\n\n")

    if sections.get("results"):
        md.append("## Results and Discussion\n")
        md.append(sections["results"].strip() + "\n\n")

    if sections.get("conclusion"):
        md.append("## Conclusion\n")
        md.append(sections["conclusion"].strip() + "\n\n")

    if citations:
        md.append("## References\n")
        for i, c in enumerate(citations, start=1):
            authors_raw = c.get("authors", [])
            if isinstance(authors_raw, list):
                authors = ", ".join(a for a in authors_raw if a)
            else:
                authors = str(authors_raw)
            title = c.get("title", "Untitled")
            year = c.get("year", "n.d.")
            venue = c.get("journal", c.get("venue", ""))
            doi = c.get("doi", "")
            ref = f"{i}. {authors or 'Unknown'}, *{title}*, {venue}, {year}"
            if doi and doi != "N/A":
                ref += f". DOI: {doi}"
            md.append(ref + "\n")

    return "\n".join(md)
