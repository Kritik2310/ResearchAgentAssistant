import json
import os

HISTORY_FILE = "backend/data/report_history.json"
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)


def save_report(topic: str, content: str):
    data = _load_raw()
    data.insert(0, {"topic": topic, "markdown": content})
    _save_raw(data)


def load_reports(limit: int = 10) -> list:
    reports = _load_raw()
    seen, unique = set(), []
    for r in reports:
        topic = r.get("topic")
        if topic and topic not in seen:
            unique.append(r)
            seen.add(topic)
        if limit and len(unique) >= limit:
            break
    return unique


def _load_raw() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_raw(data: list):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        import logging
        logging.getLogger(__name__).error("Failed to save report history: %s", e)
