from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypedDict

import numpy as np
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor
from tqdm import tqdm

from .data import ARCHIVE_REVISION, CACHE, Example
from .data import load_examples, split_examples
from .methods import UncertaintyEstimator
from .traces import MODEL_ID, MODEL_REVISION, Trace, TransformersExtractor


class LabeledTrace(TypedDict):
    trace: Trace
    labels: Tensor
    lines: Tensor


class TraceStore:
    def __init__(self, quantize: bool = True) -> None:
        precision = "nf4" if quantize else (
            "fp16" if torch.cuda.is_available() else "fp32"
        )
        self.cache = (
            CACHE / "traces-v1" / ARCHIVE_REVISION[:7]
            / MODEL_REVISION[:7] / precision
        )
        self.quantize = quantize
        self._extractor: TransformersExtractor | None = None

    @property
    def extractor(self) -> TransformersExtractor:
        if self._extractor is None:
            self._extractor = TransformersExtractor.from_pretrained(
                MODEL_ID, self.quantize
            )
        return self._extractor

    def load(self, example: Example) -> LabeledTrace:
        name = f"{example.idx}-{example.archived_hallucination}.pt"
        path = self.cache / example.benchmark / name
        if path.exists():
            return torch.load(path, weights_only=True, map_location="cpu")

        extractor = self.extractor
        boundary, labels, lines = example.align(extractor.tokenizer)
        result: LabeledTrace = {
            "trace": extractor.extract_ids(example.input_ids, boundary),
            "labels": torch.tensor(labels),
            "lines": torch.tensor(lines),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            temp_path = Path(temp.name)
        try:
            torch.save(result, temp_path)
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
        return result

    def load_splits(
        self, benchmark: str, limits: Mapping[str, int] | None = None
    ) -> dict[str, list[LabeledTrace]]:
        splits = split_examples(load_examples(benchmark), limits)
        result = {}
        for split, rows in splits.items():
            progress = tqdm(rows, desc=f"{benchmark} {split}", leave=False)
            result[split] = [self.load(example) for example in progress]
        return result


def aggregate_by_level(
    examples: Sequence[LabeledTrace], scores: Sequence[Tensor]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    levels: dict[str, list[tuple[Tensor, Tensor]]] = {
        "token": [], "line": [], "sample": []
    }
    for example, prediction in zip(examples, scores, strict=True):
        labeled = example["labels"] >= 0
        labels = example["labels"][labeled]
        observed = prediction[labeled]
        line_ids = example["lines"][labeled]
        if not len(labels):
            continue
        levels["token"].append((labels, observed))
        levels["sample"].append((labels.amax()[None], observed.amax()[None]))

        lines, indices = line_ids.unique(return_inverse=True)
        line_labels = labels.new_zeros(len(lines))
        line_labels.scatter_reduce_(0, indices, labels, reduce="amax")
        line_scores = observed.new_full((len(lines),), -torch.inf)
        line_scores.scatter_reduce_(0, indices, observed, reduce="amax")
        levels["line"].append((line_labels, line_scores))

    result = {}
    for level, pairs in levels.items():
        labels, predictions = zip(*pairs, strict=True)
        result[level] = (
            torch.cat(labels).numpy(),
            torch.cat(predictions).numpy(),
        )
    return result


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        raise ValueError("Validation requires both classes.")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def compute_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = scores >= threshold
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "pr_auc": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "threshold": threshold,
    }


def evaluate(
    estimator: UncertaintyEstimator,
    splits: Mapping[str, Sequence[LabeledTrace]],
) -> list[dict[str, str | float]]:
    train = splits["train"]
    labels = None
    if estimator.uses_labels:
        labels = [row["labels"] for row in train]
    estimator.fit([row["trace"] for row in train], labels)
    metrics = {}
    for split in ("validation", "test"):
        rows = splits[split]
        scores = estimator.predict_uncertainty([row["trace"] for row in rows])
        metrics[split] = aggregate_by_level(rows, scores)
    results = []
    for level, (test_labels, test_scores) in metrics["test"].items():
        threshold = best_f1_threshold(*metrics["validation"][level])
        results.append(
            {
                "level": level,
                **compute_metrics(test_labels, test_scores, threshold),
            }
        )
    return results
