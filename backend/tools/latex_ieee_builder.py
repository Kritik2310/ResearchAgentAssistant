from typing import List, Dict, Optional


def _escape(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    for k, v in {
        "\\": r"\\",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }.items():
        text = text.replace(k, v)
    return text


def _build_author_block(author_info: Optional[Dict]) -> str:
    if not author_info:
        return (
            r"Author Name\\" + "\n"
            r"Institution\\" + "\n"
            r"{\tt\small author@institution.edu}"
        )
    name = _escape(author_info.get("name", "Author Name"))
    institution = _escape(author_info.get("institution", "Institution"))
    email = author_info.get("email", "author@institution.edu")
    dept = _escape(author_info.get("department", ""))
    city = _escape(author_info.get("city", ""))

    lines = [f"{name}\\\\"]
    if dept:
        lines.append(f"{dept}\\\\")
    lines.append(f"{institution}\\\\")
    if city:
        lines.append(f"{city}\\\\")
    lines.append(r"{\tt\small " + email + r"}")
    return "\n".join(lines)


def build_bibtex(citations: List[Dict]) -> str:
    entries = []
    for i, c in enumerate(citations, start=1):
        authors_raw = c.get("authors", [])
        if isinstance(authors_raw, list):
            authors = " and ".join(a for a in authors_raw if a)
        else:
            authors = str(authors_raw)

        title = c.get("title", "Untitled")
        year = str(c.get("year", "n.d."))
        journal = c.get("journal", c.get("venue", ""))
        doi = c.get("doi", "")
        url = c.get("url", "")

        lines = [
            f"@article{{ref{i},",
            f"  author    = {{{authors or 'Unknown'}}},"
            f"\n  title     = {{{title}}},"
            f"\n  year      = {{{year}}},"
        ]
        if journal and journal != "N/A":
            lines.append(f"  journal   = {{{journal}}},")
        if doi and doi != "N/A":
            lines.append(f"  doi       = {{{doi}}},")
        if url and url != "N/A":
            lines.append(f"  url       = {{{url}}},")
        entry = "\n".join(lines).rstrip(",") + "\n}"
        entries.append(entry)

    return "\n\n".join(entries) if entries else "% No citations available"


def build_ieee_latex_report(
    topic: str,
    sections: Dict,
    citations: List[Dict],
    author_info: Optional[Dict] = None,
) -> str:
    author_block = _build_author_block(author_info)
    tex: List[str] = []

    # Preamble
    tex.append(r"\documentclass[conference]{IEEEtran}")
    tex.append(r"\IEEEoverridecommandlockouts")
    tex.append("")
    tex.append(r"\usepackage{amsmath,amssymb,url,graphicx,hyperref}")
    tex.append(r"\usepackage[utf8]{inputenc}")
    tex.append(r"\usepackage{lmodern}")
    tex.append("")

    # Title & author
    tex.append(r"\title{" + _escape(topic) + r"}")
    tex.append("")
    tex.append(r"\author{" + author_block + r"}")
    tex.append("")
    tex.append(r"\begin{document}")
    tex.append(r"\maketitle")
    tex.append("")

    # Abstract
    tex.append(r"\begin{abstract}")
    tex.append(_escape(sections.get("abstract", "")) + r"\par")
    tex.append(r"\end{abstract}")
    tex.append("")

    # Keywords
    tex.append(r"\begin{IEEEkeywords}")
    keywords = sections.get("keywords", [])
    kw_text = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
    tex.append(_escape(kw_text or topic))
    tex.append(r"\end{IEEEkeywords}")
    tex.append("")

    # Introduction
    tex.append(r"\section{INTRODUCTION}")
    tex.append(_escape(sections.get("introduction", "")) + r"\par")
    tex.append("")

    # Literature Survey
    tex.append(r"\section{LITERATURE SURVEY}")
    literature = sections.get("literature_review", "")
    if isinstance(literature, list):
        for entry in literature:
            title_esc = _escape(entry.get("title", "Related Study"))
            tex.append(r"\subsection{" + title_esc + r"}")
            tex.append(_escape(entry.get("summary", "")) + r"\par")
    elif literature:
        tex.append(_escape(literature) + r"\par")
    tex.append("")

    # Research Gaps
    tex.append(r"\section{RESEARCH GAPS}")
    tex.append(
        _escape("Based on the literature survey, the following open research gaps are identified:") + r"\par"
    )
    gaps = sections.get("research_gaps") or sections.get("gaps")
    tex.append(r"\begin{itemize}")
    if isinstance(gaps, list) and gaps:
        for g in gaps:
            tex.append(r"\item " + _escape(g))
    elif isinstance(gaps, str) and gaps:
        tex.append(r"\item " + _escape(gaps))
    else:
        tex.append(r"\item No explicit gaps were identified in the surveyed literature.")
    tex.append(r"\end{itemize}")
    tex.append("")

    # Proposed Experiment
    tex.append(r"\section{PROPOSED EXPERIMENT}")

    datasets = sections.get("datasets", [])
    tex.append(r"\subsection{Datasets}")
    tex.append(r"\begin{itemize}")
    if datasets:
        for d in datasets:
            tex.append(r"\item " + _escape(d))
    else:
        tex.append(r"\item No specific datasets were identified in the surveyed literature.")
    tex.append(r"\end{itemize}")
    tex.append("")

    baselines = sections.get("baselines", [])
    tex.append(r"\subsection{Baseline Methods}")
    tex.append(r"\begin{itemize}")
    if baselines:
        for b in baselines:
            tex.append(r"\item " + _escape(b))
    else:
        tex.append(r"\item No specific baseline methods were consistently identified.")
    tex.append(r"\end{itemize}")
    tex.append("")

    # Proposed Methodology
    tex.append(r"\section{PROPOSED METHODOLOGY}")
    methodology = sections.get("methodology", "")
    if methodology:
        tex.append(_escape(methodology) + r"\par")
    else:
        tex.append(
            _escape(
                "A structured methodology will be developed based on the identified research gaps, "
                "leveraging the datasets and evaluation metrics described above."
            ) + r"\par"
        )
    tex.append("")

    # Conclusion
    tex.append(r"\section{CONCLUSIONS}")
    conclusion = sections.get("conclusion", "")
    if conclusion:
        tex.append(_escape(conclusion) + r"\par")
    else:
        tex.append(_escape(f"This survey of {topic} identified key trends, open challenges, and a path forward for future research.") + r"\par")
    tex.append("")

    # References (inline thebibliography — bibtex file handled separately in ZIP)
    tex.append(r"\begin{thebibliography}{99}")
    if citations:
        for i, c in enumerate(citations, start=1):
            authors_raw = c.get("authors", [])
            if isinstance(authors_raw, list):
                authors = _escape(", ".join(a for a in authors_raw if a)) or "Unknown Authors"
            else:
                authors = _escape(str(authors_raw))
            title = _escape(c.get("title", "Untitled"))
            year = _escape(str(c.get("year", "n.d.")))
            venue = _escape(c.get("journal", c.get("venue", "")))
            doi = c.get("doi", "")
            line = rf"\bibitem{{ref{i}}} {authors}, ``{title},'' {venue}, {year}."
            if doi and doi != "N/A":
                line += f" DOI: {_escape(doi)}."
            tex.append(line)
    else:
        tex.append(r"\bibitem{placeholder} References will be populated from citation metadata.")
    tex.append(r"\end{thebibliography}")
    tex.append("")
    tex.append(r"\end{document}")

    return "\n".join(tex)
