from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from pyarrow import ipc
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

ARCHIVE_REVISION = "4681508aeec9c7aa491f42308f6090417118a818"
ARCHIVE_URL = (
    "https://raw.githubusercontent.com/datapaf/CodeHallucinationDetection/"
    f"{ARCHIVE_REVISION}/datasets.zip"
)
CACHE = Path.home() / ".cache" / "ugar"
DATASETS = ("mbpp", "humaneval", "humaneval_java", "defects4j")
SPLITS = ("train", "validation", "test")
PILOT_LIMITS = {"train": 48, "validation": 16, "test": 24}
SEED = 42
IGNORE = -100


@dataclass(frozen=True)
class Example:
    idx: int
    benchmark: str
    task_id: str
    question: str
    reply: str
    context: str
    input_ids: list[int]
    labels: list[int]
    archived_hallucination: int

    @property
    def defect(self) -> int:
        return int(1 in self.labels)

    def align(
        self, tokenizer: PreTrainedTokenizerFast
    ) -> tuple[int, list[int], list[int]]:
        prompt_ids = tokenizer.encode(self.context, add_special_tokens=False)
        reply = tokenizer(
            self.reply, add_special_tokens=False, return_offsets_mapping=True
        )
        if prompt_ids + reply["input_ids"] != self.input_ids:
            raise ValueError(f"Tokenizer mismatch in archived row {self.idx}.")
        boundary = len(prompt_ids)
        if len(self.labels) != len(self.input_ids):
            raise ValueError(f"Label length mismatch in row {self.idx}.")
        newlines = [i for i, char in enumerate(self.reply) if char == "\n"]
        line_ids = []
        for start, end in reply["offset_mapping"]:
            content = self.reply[start:end]
            text = content.lstrip()
            anchor = end - len(text) if text else start
            line_ids.append(bisect_right(newlines, anchor - 1))
        return boundary, self.labels[boundary:], line_ids


def archive_path() -> Path:
    path = CACHE / f"{ARCHIVE_REVISION[:7]}-datasets.zip"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(ARCHIVE_URL, timeout=120) as response:
        content = response.read()
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp.write(content)
    Path(temp.name).replace(path)
    return path


def load_examples(benchmark: str) -> list[Example]:
    if benchmark not in DATASETS:
        raise ValueError(f"Unknown benchmark {benchmark!r}.")
    rows = []
    root = "Datasets/data/line_diff_l38b_instruct_dataset"
    with ZipFile(archive_path()) as archive:
        for split in ("train", "val", "test"):
            member = f"{root}/{split}/data-00000-of-00001.arrow"
            with archive.open(member) as source:
                table = ipc.open_stream(source).read_all()
            table = table.select(
                [
                    "idx", "dataset", "task_id", "question", "reply",
                    "templated_question", "input_ids", "uncertainty_labels",
                    "hallucinated",
                ]
            )
            for row in table.to_pylist():
                if row["dataset"] == benchmark:
                    rows.append(
                        Example(
                            row["idx"],
                            row["dataset"],
                            str(row["task_id"]),
                            row["question"],
                            row["reply"],
                            row["templated_question"],
                            row["input_ids"],
                            row["uncertainty_labels"],
                            row["hallucinated"],
                        )
                    )
    return rows


def split_examples(
    examples: Sequence[Example],
    limits: Mapping[str, int] | None = None,
    seed: int = SEED,
) -> dict[str, list[Example]]:
    labels = [example.defect for example in examples]
    groups = [example.task_id for example in examples]
    kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = [test for _, test in kfold.split(labels, labels, groups)]
    indices = (np.concatenate(folds[2:]), folds[1], folds[0])
    result = {}
    for split, members in zip(SPLITS, indices, strict=True):
        limit = (limits or {}).get(split, len(members))
        if limit < len(members):
            members, _ = train_test_split(
                members,
                train_size=limit,
                stratify=[labels[index] for index in members],
                random_state=seed,
            )
        result[split] = [examples[index] for index in sorted(members)]
    return result


def summarize(splits: Mapping[str, Sequence[Example]]) -> list[dict]:
    rows = []
    for split, examples in splits.items():
        counts = Counter(
            label for example in examples for label in example.labels
        )
        defects = sum(example.defect for example in examples)
        rows.append(
            {
                "split": split,
                "examples": len(examples),
                "tasks": len({example.task_id for example in examples}),
                "defective_examples": defects,
                "archived_hallucinations": sum(
                    example.archived_hallucination for example in examples
                ),
                "clean_tokens": counts[0],
                "defective_tokens": counts[1],
                "ignored_tokens": counts[IGNORE],
            }
        )
    return rows
