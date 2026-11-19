from collections.abc import Callable
from dataclasses import dataclass
from random import Random
from zipfile import ZipFile

from pyarrow import ipc

from .benchmarks import (
    SEED,
    Example,
    archive_path,
    load_examples,
    split_examples,
)

PYTHON_BENCHMARKS = {"mbpp", "humaneval"}


@dataclass(frozen=True)
class RepairCase:
    draft: Example
    reference: str


def repair_cases(
    benchmark: str,
    limit: int,
    seed: int = SEED,
    eligible: Callable[[RepairCase], bool] | None = None,
) -> tuple[list[Example], list[RepairCase]]:
    splits = split_examples(load_examples(benchmark))
    references = {
        row.idx: row.reply for row in splits["test"]
        if not row.archived_hallucination
    }
    tasks = {}
    for row in sorted(splits["test"], key=lambda row: row.idx):
        if row.archived_hallucination and row.defect and row.idx in references:
            tasks.setdefault(row.task_id, RepairCase(row, references[row.idx]))
    if not 1 <= limit <= len(tasks):
        raise ValueError(f"Select between 1 and {len(tasks)} {benchmark} cases.")
    cases = list(tasks.values())
    Random(seed).shuffle(cases)
    selected = []
    for case in cases:
        if eligible is None or eligible(case):
            selected.append(case)
            if len(selected) == limit:
                return splits["train"], selected
    raise ValueError(f"Found only {len(selected)} eligible {benchmark} cases.")


def python_tests(benchmark: str) -> dict[int, str]:
    tests = {}
    root = "Datasets/data/line_diff_l38b_instruct_dataset"
    with ZipFile(archive_path()) as archive:
        for split in ("train", "val", "test"):
            filename = f"{root}/{split}/data-00000-of-00001.arrow"
            with archive.open(filename) as member:
                table = ipc.open_stream(member).read_all()
            for row in table.select(["idx", "dataset", "meta"]).to_pylist():
                if row["dataset"] == benchmark:
                    meta = row["meta"]
                    checks = meta["test"] or meta["test_list"]
                    if checks:
                        setup = meta["test_setup_code"] or ""
                        tests[row["idx"]] = "\n".join([setup, *checks])
    return tests
