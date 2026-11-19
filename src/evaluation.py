from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from functools import lru_cache, partial
from io import StringIO
from pathlib import Path
import re
import subprocess
from tempfile import NamedTemporaryFile
import tokenize
from typing import TypedDict
from uuid import uuid4

from langsmith import tracing_context
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

from datasets.benchmarks import (
    ARCHIVE_REVISION,
    CACHE,
    Example,
    load_examples,
    split_examples,
)
from datasets.repair import (
    PYTHON_BENCHMARKS,
    RepairCase,
    python_tests,
    repair_cases,
)

from .agent_loop import (
    AgentState,
    CandidateState,
    Revision,
    code_block,
    create_agent,
    create_agent_candidates,
    extract_code,
    rank_lines,
)
from .methods import (
    SAPLMA,
    EpochEstimator,
    UncertaintyEstimator,
    UncertaintyQuantificationHead,
)
from .traces import MODEL_ID, MODEL_REVISION, Trace, TransformersExtractor


AGENT_METHODS = {
    "UQ Head": UncertaintyQuantificationHead,
    "SAPLMA": SAPLMA,
}
LEXEME = re.compile(
    r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|'
    r"'(?:\\.|[^'\\])*'|[\w$]+|[^\w\s]", re.DOTALL,
)


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


def _fit_estimator(
    estimator: UncertaintyEstimator,
    train: Sequence[LabeledTrace],
    validation: Sequence[LabeledTrace],
) -> None:
    traces = [row["trace"] for row in train]
    labels = None
    if estimator.uses_labels:
        labels = [row["labels"] for row in train]
    if isinstance(estimator, EpochEstimator):
        estimator.fit(
            traces,
            labels,
            validation=(
                [row["trace"] for row in validation],
                [row["labels"] for row in validation],
            ),
        )
    else:
        estimator.fit(traces, labels)


def evaluate(
    estimator: UncertaintyEstimator,
    splits: Mapping[str, Sequence[LabeledTrace]],
) -> list[dict[str, str | float]]:
    _fit_estimator(estimator, splits["train"], splits["validation"])
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


def code_tokens(code: str, benchmark: str) -> list[str]:
    if benchmark in PYTHON_BENCHMARKS:
        ignored = {tokenize.NL, tokenize.COMMENT, tokenize.ENDMARKER}
        structural = {tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE}
        try:
            return [
                tokenize.tok_name[item.type] if item.type in structural
                else item.string
                for item in tokenize.generate_tokens(StringIO(code).readline)
                if item.type not in ignored
            ]
        except (IndentationError, tokenize.TokenError):
            return []
    return [
        item for item in LEXEME.findall(code)
        if not item.startswith(("//", "/*"))
    ]


@lru_cache(maxsize=256)
def passes_python_tests(code: str, tests: str) -> bool:
    container = f"ugar-agent-{uuid4().hex}"
    marker = uuid4().hex
    script = f"{code}\n\n{tests}\nprint({marker!r})\n"
    command = [
        "docker", "run", "--rm", "--pull=never", "--name", container,
        "--network=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--pids-limit=64",
        "--memory=128m", "--cpus=1", "--user=65534:65534", "-i",
        "python:3.11-slim", "python", "-I", "-",
    ]
    try:
        result = subprocess.run(
            command, input=script, capture_output=True, encoding="utf-8",
            errors="replace", timeout=20,
        )
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["docker", "rm", "-f", container], capture_output=True,
            timeout=20, check=False,
        )
        return False
    if result.returncode in (125, 126, 127):
        raise RuntimeError(result.stderr.strip() or "Docker could not run tests.")
    return result.returncode == 0 and marker in result.stdout.splitlines()


def run_benchmark(
    store: TraceStore,
    benchmark: str,
    limit: int = 4,
    max_new_tokens: int = 256,
    check_python: bool = True,
    max_candidates: int | None = None,
    *,
    epochs: int | None = None,
    training_history: list[dict] | None = None,
) -> list[dict]:
    if max_candidates is not None and not 1 <= max_candidates <= 3:
        raise ValueError("Use between one and three candidates.")
    use_python_tests = check_python and benchmark in PYTHON_BENCHMARKS
    tests = python_tests(benchmark) if use_python_tests else {}

    def eligible(case: RepairCase) -> bool:
        if not use_python_tests:
            return True
        checks = tests.get(case.draft.idx)
        return bool(checks and passes_python_tests(
            extract_code(case.reference), checks
        ))

    examples, cases = repair_cases(benchmark, limit, eligible=eligible)
    train = [store.load(example) for example in examples]
    validation = [
        store.load(example)
        for example in split_examples(load_examples(benchmark))["validation"]
    ]
    estimators = {
        name: factory(epochs=epochs) for name, factory in AGENT_METHODS.items()
    }
    for name, estimator in estimators.items():
        _fit_estimator(estimator, train, validation)
        if training_history is not None:
            training_history.extend(
                {"benchmark": benchmark, "method": name, **row}
                for row in estimator.training_history
            )

    generation_limit = max_new_tokens * (max_candidates or 1)

    @lru_cache(maxsize=256)
    def generate(prompt: str) -> str:
        return store.extractor.generate(prompt, max_new_tokens=generation_limit)

    plain_agent = create_agent(generate)
    results = []
    for case in tqdm(cases, desc=benchmark, leave=False):
        example = case.draft
        reference = code_tokens(extract_code(case.reference), benchmark)
        checks = tests.get(example.idx)
        if not reference:
            raise ValueError(f"Empty reference for task {example.task_id}.")

        def measure(
            method: str,
            output: str,
            revision: Revision | None = None,
            selection: CandidateState | None = None,
        ) -> None:
            code = extract_code(output)
            candidate = code_tokens(code, benchmark)
            row = {
                "benchmark": benchmark,
                "task": example.task_id,
                "method": method,
                "similarity": SequenceMatcher(
                    None, reference, candidate, autojunk=False
                ).ratio(),
                "exact": candidate == reference,
                "accepted": revision["accepted"] if revision else True,
                "passed": passes_python_tests(code, checks) if checks else None,
                "hints": revision["hints"] if revision else [],
                "review": revision["review"] if revision else "",
            }
            if selection is not None:
                row.update(
                    accepted=selection["selected"] is not None,
                    variants=len(selection["variants"]),
                    uncertainties=selection["uncertainties"],
                    selected_candidate=selection["selected"],
                )
            results.append(row)

        initial: AgentState = {"task": example.question, "draft": example.reply}
        with tracing_context(enabled=False):
            if max_candidates is None:
                measure("Archived draft", example.reply)
                plain = plain_agent.invoke(initial)
                measure("Plain review", plain["draft"], plain["history"][0])
                original = store.load(example)
                line_ids = original["lines"].tolist()
                for name, estimator in estimators.items():
                    scores, = estimator.predict_uncertainty([original["trace"]])
                    score = partial(rank_lines, scores=scores, line_ids=line_ids)
                    revised = create_agent(generate, score).invoke(initial)
                    measure(name, revised["draft"], revised["history"][0])
                continue

            predict = lru_cache(maxsize=max_candidates)(
                partial(_candidate_trace, store.extractor, example.context)
            )
            for name, estimator in estimators.items():
                def candidate_score(reply: str) -> float:
                    trace, code_mask = predict(reply)
                    scores, = estimator.predict_uncertainty([trace])
                    return float(scores[code_mask].max())

                graph = create_agent_candidates(
                    generate, candidate_score, max_candidates
                )
                selected = graph.invoke(initial)
                measure(name, selected["draft"], selection=selected)
    return results


def _candidate_trace(
    extractor: TransformersExtractor, context: str, reply: str
) -> tuple[Trace, Tensor]:
    block = code_block(reply)
    if block is None:
        raise ValueError("The candidate must contain a complete code block.")
    start, stop = block.span("code")
    tokenizer = extractor.tokenizer
    prefix = tokenizer.encode(context, add_special_tokens=False)
    suffix = tokenizer(
        reply, add_special_tokens=False, return_offsets_mapping=True
    )
    mask = torch.tensor([
        bool(reply[max(start, left):min(stop, right)].strip())
        for left, right in suffix["offset_mapping"]
    ], dtype=torch.bool)
    trace = extractor.extract_ids(prefix + suffix["input_ids"], len(prefix))
    return trace, mask
