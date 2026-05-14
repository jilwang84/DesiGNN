# Copyright (c) 2026 DesiGNN Authors
# License: Apache-2.0 license

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


HUMAN_TO_METRIC = {
    "average shortest path length": "local_average_shortest_path_length",
    "graph diameter": "local_graph_diameter",
    "average closeness centrality": "local_average_closeness_centrality",
    "average betweenness centrality": "local_average_betweenness_centrality",
    "node count": "node_count",
    "edge count": "edge_count",
    "average degree": "average_degree",
    "density": "density",
    "average clustering coefficient": "average_clustering_coefficient",
    "connected components": "connected_components",
    "assortativity": "assortativity",
    "average degree centrality": "average_degree_centrality",
    "average eigenvector centrality": "average_eigenvector_centrality",
    "feature dimensionality": "feature_dimensionality",
    "node feature diversity": "node_feature_diversity",
    "label homophily": "label_homophily",
}


@dataclass
class FormulaSimilarityConfig:
    n_f: int = 8
    eps: float = 1e-12
    ignore_nonpositive_confidence: bool = True


class FormulaDatasetSimilarity:
    """Optional formula-based DCM from the paper.

    This class is separate from the original LLM DCM implementation.
    It reads graph properties from released combined_description.txt files and uses
    LLM-induced property weights w_u^k when requested.
    """

    def __init__(
        self,
        confidence_path: str,
        cfg: Optional[FormulaSimilarityConfig] = None,
    ):
        self.cfg = cfg or FormulaSimilarityConfig()
        with open(confidence_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.metric_order = payload["metrics_order_g1_to_g16"]
        self.confidence = payload["confidence"]

    @staticmethod
    def canonical_name(dataset_identifier: str) -> str:
        return dataset_identifier.split(":", 1)[1] if ":" in dataset_identifier else dataset_identifier

    @staticmethod
    def _parse_float(value: str) -> Optional[float]:
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def load_properties_from_description(self, dataset_identifier: str, dataset_root_dir: str) -> Dict[str, float]:
        dataset_name = self.canonical_name(dataset_identifier)
        dataset_dir = os.path.join(dataset_root_dir, dataset_name)
        description_path = os.path.join(dataset_dir, "combined_description.txt")
        if not os.path.exists(description_path):
            raise FileNotFoundError(
                f"Formula DCM requires {description_path}. Run GDU first or publish the combined description."
            )

        props: Dict[str, float] = {}
        with open(description_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped.startswith("-") or ":" not in stripped:
                    continue
                label, value = stripped[1:].split(":", 1)
                metric = HUMAN_TO_METRIC.get(label.strip().lower())
                parsed = self._parse_float(value)
                if metric and parsed is not None:
                    props[metric] = parsed
        return props

    def selected_metrics(self) -> List[str]:
        ordered = list(self.metric_order)
        if self.cfg.ignore_nonpositive_confidence:
            ordered = [m for m in ordered if float(self.confidence.get(m, 0.0)) > 0.0]
        return ordered[: max(1, int(self.cfg.n_f))]

    def induce_weights(
        self,
        llm,
        unseen_dataset_description: str,
        benchmark_dataset_descriptions: Dict[str, str],
    ) -> Tuple[Dict[str, float], str]:
        metrics = self.selected_metrics()
        metric_lines = "\n".join(
            f"- {m}: average_confidence={float(self.confidence[m]):.8f}" for m in metrics
        )
        benchmark_block = "\n\n".join(
            f"### {name}\n{desc}" for name, desc in benchmark_dataset_descriptions.items()
        )
        prompt = f"""
You are an expert in graph neural network architecture design.
The paper defines an abstract dataset-similarity formula that includes a dataset-specific property importance term w_u^k induced from the unseen dataset. Estimate these non-negative weights for the unseen dataset.

Return STRICT JSON only in this schema:
{{"weights": {{"metric_name": 0.0}}, "rationale": "brief"}}

Rules:
1. Include exactly the metric keys listed below.
2. Weights must be non-negative and should sum to 1.
3. The weights should reflect how important each property is for judging architecture-transfer similarity for the unseen dataset.
4. Use the average confidence values as a prior, but adapt them to the unseen dataset description.

Selected properties:
{metric_lines}

Unseen dataset description:
{unseen_dataset_description}

Benchmark descriptions for context:
{benchmark_block}
""".strip()
        response = llm.invoke(prompt)
        text = response.content
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Could not parse LLM-induced property weights as JSON.")
        payload = json.loads(match.group(0))
        raw_weights = payload.get("weights", {})
        weights = {m: float(raw_weights.get(m, 0.0)) for m in metrics}
        return self._normalize_weights(weights), text

    def uniform_weights(self) -> Dict[str, float]:
        return {m: 1.0 for m in self.selected_metrics()}

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        selected = self.selected_metrics()
        clipped = {m: max(0.0, float(weights.get(m, 0.0))) for m in selected}
        total = sum(clipped.values())
        if total <= 0:
            return {m: 1.0 for m in selected}
        scale = float(len(selected)) / total
        return {m: clipped[m] * scale for m in selected}

    def compute_scores(
        self,
        unseen_properties: Dict[str, float],
        benchmark_properties: Dict[str, Dict[str, float]],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        selected = self.selected_metrics()
        weights = self._normalize_weights(weights)

        ranges: Dict[str, float] = {}
        for metric in selected:
            values = [unseen_properties.get(metric)] + [props.get(metric) for props in benchmark_properties.values()]
            finite = []
            for value in values:
                if value is None:
                    continue
                value = float(value)
                if math.isfinite(value):
                    finite.append(value)
            finite_values = np.array(finite, dtype=np.float64)
            ranges[metric] = float(np.max(finite_values) - np.min(finite_values)) if finite_values.size else 1.0

        scores: Dict[str, float] = {}
        for benchmark_name, props in benchmark_properties.items():
            total = 0.0
            for metric in selected:
                u = unseen_properties.get(metric)
                v = props.get(metric)
                if u is None or v is None:
                    continue
                u = float(u)
                v = float(v)
                if not (math.isfinite(u) and math.isfinite(v)):
                    continue
                distance = abs(u - v) / (ranges[metric] + self.cfg.eps)
                total += weights[metric] * float(self.confidence[metric]) / (1.0 + distance)
            scores[benchmark_name] = float(total / max(1, len(selected)))
        return scores

    @staticmethod
    def rank(scores: Dict[str, float], top_n: int) -> List[str]:
        return [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]]
