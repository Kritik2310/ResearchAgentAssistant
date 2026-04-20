import re
from collections import Counter
from typing import List

STOPWORDS = {
    "the", "and", "of", "in", "to", "for", "with", "on", "by",
    "an", "is", "are", "this", "that", "as", "at", "from",
    "study", "paper", "analysis", "results", "using",
    "based", "approach", "method", "methods", "system",
    "work", "show", "shows", "propose", "proposed",
}


def extract_keywords(texts: List[str], top_k: int = 10) -> List[str]:
    tokens = []
    for text in texts:
        if not text:
            continue
        words = re.findall(r"[a-zA-Z]{4,}", text.lower())
        tokens.extend(w for w in words if w not in STOPWORDS)
    freq = Counter(tokens)
    return [word for word, _ in freq.most_common(top_k)]
