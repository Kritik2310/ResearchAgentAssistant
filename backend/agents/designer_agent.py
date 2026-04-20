import logging
from typing import List, Dict

from tools import memory_store

logger = logging.getLogger(__name__)


class DesignerAgent:
    def run(self, input_json: dict) -> dict:
        topic = input_json["topic"]

        mem_summaries = memory_store.get_summaries(topic)
        mem_gaps = memory_store.get_gaps(topic)
        mem_citations = memory_store.get_citations(topic)

        if mem_summaries["status"] != "ok":
            return {
                "status": "error",
                "message": "No summaries found in memory. Run retrieval + summarization first.",
            }

        summaries = mem_summaries["summaries"]
        gaps = mem_gaps["gaps"] if mem_gaps["status"] == "ok" else [
            g for s in summaries for g in s.get("gaps", [])
        ]

        datasets = self._extract_datasets(summaries)
        metrics = self._extract_metrics(summaries)
        baselines = self._extract_baselines(summaries)
        hypothesis = self._generate_hypothesis(topic, gaps)

        plan = {
            "topic": topic,
            "gaps": gaps,
            "hypothesis": hypothesis,
            "datasets": datasets,
            "evaluation_metrics": metrics,
            "baseline_methods": baselines,
            "implementation_notes": {
                "seed": 42,
                "environment": "Python with domain-appropriate ML frameworks",
            },
            "citations_used": mem_citations.get("citations", []),
        }

        memory_store.save_experiment_plan(topic, plan)
        logger.info("Experiment plan generated for topic: %s", topic)

        return {"status": "ok", "experiment_plan": plan}

    # ------------------------------------------------------------------

    def _generate_hypothesis(self, topic: str, gaps: List[str]) -> str:
        if gaps:
            return (
                f"Addressing '{gaps[0][:180]}' will meaningfully advance "
                f"outcomes in {topic} research and real-world applications."
            )
        return (
            f"A comprehensive comparative analysis of recent advances in {topic} "
            "will surface best practices and highlight underexplored optimization opportunities."
        )

    def _extract_datasets(self, summaries: List[Dict]) -> List[Dict]:
        datasets, seen = [], set()
        for s in summaries:
            for text in s.get("methods", []) + s.get("key_findings", []):
                if any(kw in text.lower() for kw in ["dataset", "corpus", "benchmark", "collection"]):
                    if text not in seen and len(text) < 150:
                        datasets.append({
                            "name": text.strip(),
                            "description": f"From: {s.get('title', 'Unknown')[:50]}",
                            "source": "Extracted from literature",
                        })
                        seen.add(text)
        if not datasets:
            datasets.append({
                "name": "Domain-appropriate benchmark dataset",
                "description": "To be selected based on research requirements",
                "source": "Public repositories (Kaggle, HuggingFace, UCI)",
            })
        return datasets[:3]

    def _extract_metrics(self, summaries: List[Dict]) -> List[Dict]:
        metric_map = {
            "accuracy": ("Accuracy", "Higher is better"),
            "precision": ("Precision", "Higher is better"),
            "recall": ("Recall", "Higher is better"),
            "f1": ("F1 Score", "Higher is better"),
            "auc": ("AUC-ROC", "Higher is better"),
            "rmse": ("RMSE", "Lower is better"),
            "mae": ("MAE", "Lower is better"),
            "bleu": ("BLEU Score", "Higher is better"),
            "rouge": ("ROUGE Score", "Higher is better"),
        }
        metrics, seen = [], set()
        for s in summaries:
            combined = " ".join(s.get("key_findings", []) + s.get("methods", [])).lower()
            for kw, (name, interp) in metric_map.items():
                if kw in combined and name not in seen:
                    metrics.append({"name": name, "interpretation": interp})
                    seen.add(name)
        if not metrics:
            metrics = [
                {"name": "Primary Domain Metric", "interpretation": "Domain-specific evaluation"},
                {"name": "Secondary Quality Metric", "interpretation": "Domain-specific evaluation"},
            ]
        return metrics[:5]

    def _extract_baselines(self, summaries: List[Dict]) -> List[Dict]:
        baselines, seen = [], set()
        for s in summaries:
            for text in s.get("methods", []) + s.get("key_findings", []):
                if any(kw in text.lower() for kw in ["baseline", "compared", "benchmark", "existing", "prior", "traditional"]):
                    if text not in seen and len(text) < 150:
                        baselines.append({"name": text.strip(), "reason": "Identified baseline from literature"})
                        seen.add(text)
        if not baselines:
            baselines.append({
                "name": "Current state-of-the-art approach",
                "reason": "Standard comparison point for this domain",
            })
        return baselines[:3]
