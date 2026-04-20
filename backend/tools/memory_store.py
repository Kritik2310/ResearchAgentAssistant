from typing import Dict, Any, List

_MEMORY_DB: Dict[str, Dict[str, Any]] = {}


def _get_topic_bucket(topic: str) -> Dict[str, Any]:
    if topic not in _MEMORY_DB:
        _MEMORY_DB[topic] = {
            "summaries": [],
            "gaps": [],
            "citations": [],
            "datasets": [],
            "baselines": [],
            "methodology": "",
            "experiment_plan": None,
            "section_content": {},
            "author_info": {},
        }
    return _MEMORY_DB[topic]


# ---------- WRITE ----------

def save_summaries(topic: str, summaries: List[Dict[str, Any]]):
    bucket = _get_topic_bucket(topic)
    bucket["summaries"] = summaries
    gaps, citations = [], []
    for s in summaries:
        gaps.extend(s.get("gaps", []))
        c = s.get("citations")
        if c:
            citations.append(c)
    if gaps:
        bucket["gaps"] = gaps
    if citations:
        bucket["citations"] = citations


def save_gaps(topic: str, gaps: List[str]):
    _get_topic_bucket(topic)["gaps"] = gaps


def save_citations(topic: str, citations: List[Dict[str, Any]]):
    _get_topic_bucket(topic)["citations"] = citations


def save_experiment_plan(topic: str, plan: Dict[str, Any]):
    _get_topic_bucket(topic)["experiment_plan"] = plan


def save_section_content(topic: str, section_content: Dict[str, Any]):
    _get_topic_bucket(topic)["section_content"] = section_content


def save_datasets(topic: str, datasets: List[str]):
    _get_topic_bucket(topic)["datasets"] = datasets


def save_methodology(topic: str, methodology: str):
    _get_topic_bucket(topic)["methodology"] = methodology


def save_baselines(topic: str, baselines: List[str]):
    _get_topic_bucket(topic)["baselines"] = baselines


def save_author_info(topic: str, author_info: Dict[str, str]):
    _get_topic_bucket(topic)["author_info"] = author_info


# ---------- READ ----------

def get_summaries(topic: str) -> Dict[str, Any]:
    bucket = _MEMORY_DB.get(topic)
    if not bucket or not bucket["summaries"]:
        return {"status": "not_found", "summaries": []}
    return {"status": "ok", "summaries": bucket["summaries"]}


def get_gaps(topic: str) -> Dict[str, Any]:
    bucket = _MEMORY_DB.get(topic)
    if not bucket or not bucket["gaps"]:
        return {"status": "not_found", "gaps": []}
    return {"status": "ok", "gaps": bucket["gaps"]}


def get_citations(topic: str) -> Dict[str, Any]:
    bucket = _MEMORY_DB.get(topic)
    if not bucket or not bucket["citations"]:
        return {"status": "not_found", "citations": []}
    return {"status": "ok", "citations": bucket["citations"]}


def get_experiment_plan(topic: str) -> Dict[str, Any]:
    bucket = _MEMORY_DB.get(topic)
    if not bucket or not bucket["experiment_plan"]:
        return {"status": "not_found", "experiment_plan": None}
    return {"status": "ok", "experiment_plan": bucket["experiment_plan"]}


def get_section_content(topic: str) -> Dict[str, Any]:
    bucket = _MEMORY_DB.get(topic)
    if not bucket or not bucket.get("section_content"):
        return {"status": "not_found", "section_content": {}}
    return {"status": "ok", "section_content": bucket["section_content"]}


def get_datasets(topic: str) -> List[str]:
    bucket = _MEMORY_DB.get(topic)
    return bucket.get("datasets", []) if bucket else []


def get_methodology(topic: str) -> str:
    bucket = _MEMORY_DB.get(topic)
    return bucket.get("methodology", "") if bucket else ""


def get_baselines(topic: str) -> List[str]:
    bucket = _MEMORY_DB.get(topic)
    return bucket.get("baselines", []) if bucket else []


def get_author_info(topic: str) -> Dict[str, str]:
    bucket = _MEMORY_DB.get(topic)
    return bucket.get("author_info", {}) if bucket else {}
