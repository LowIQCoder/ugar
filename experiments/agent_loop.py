from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from io import StringIO
from random import Random
import re
import subprocess
import tokenize
from typing import TypedDict
from uuid import uuid4
from zipfile import ZipFile

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langsmith import tracing_context
from pyarrow import ipc
from torch import Tensor
from tqdm import tqdm

from .data import SEED, Example, archive_path, load_examples, split_examples
from .evaluation import TraceStore
from .methods import (
    HaloScope,
    ICRProbe,
    LookbackLens,
    SAPLMA,
    UncertaintyQuantificationHead,
)

METHODS = {
    "Lookback Lens": LookbackLens,
    "UQ Head": UncertaintyQuantificationHead,
    "ICR Probe": ICRProbe,
    "SAPLMA": SAPLMA,
    "HaloScope": HaloScope,
}
FENCE = re.compile(
    r"^[ \t]*```[^\n]*\n(?P<code>.*?)^[ \t]*```[ \t]*\r?(?:\n|$)",
    re.MULTILINE | re.DOTALL,
)
LEXEME = re.compile(
    r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|'
    r"'(?:\\.|[^'\\])*'|[\w$]+|[^\w\s]", re.DOTALL,
)
PYTHON_BENCHMARKS = {"mbpp", "humaneval"}
type Hint = tuple[int, str]
type Scorer = Callable[[str], list[Hint]]


@dataclass(frozen=True)
class RepairCase:
    draft: Example
    reference: str


class Revision(TypedDict):
    review: str
    hints: list[Hint]
    accepted: bool


class AgentState(TypedDict, total=False):
    task: str
    draft: str
    hints: list[Hint]
    history: list[Revision]
    stopped: bool


def code_block(reply: str) -> re.Match[str] | None:
    return next(reversed(list(FENCE.finditer(reply))), None)


def extract_code(reply: str) -> str:
    block = code_block(reply)
    return block["code"] if block else reply


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


def rank_lines(
    reply: str, scores: Tensor, line_ids: Sequence[int], limit: int = 2
) -> list[Hint]:
    block = code_block(reply)
    start = reply.count("\n", 0, block.start("code")) if block else 0
    lines = extract_code(reply).splitlines()
    best: dict[int, float] = {}
    for line, score in zip(line_ids, scores.tolist(), strict=True):
        index = line - start
        if 0 <= index < len(lines) and lines[index].strip():
            best[index] = max(score, best.get(index, -float("inf")))
    ranked = sorted(best, key=lambda index: (-best[index], index))
    return [(index + 1, lines[index].strip()) for index in ranked[:limit]]


def create_agent(
    generate: Callable[[str], str],
    score: Scorer | None = None,
    max_rounds: int = 1,
) -> CompiledStateGraph:
    if max_rounds < 1:
        raise ValueError("Use at least one review round.")

    def inspect(state: AgentState) -> AgentState:
        return {"hints": score(state["draft"]) if score else []}

    def reflect_and_revise(state: AgentState) -> AgentState:
        draft = state["draft"]
        hints = state["hints"]
        feedback = "\n".join(f"Line {line}: {code}" for line, code in hints)
        prompt = (
            f"Task:\n{state['task']}\n\nCurrent draft:\n{draft}\n\n"
            "Review the current draft against the task. Check any suggested "
            "lines and verify the surrounding logic. Give a one-sentence "
            "review, then one fenced code block with the complete requested "
            "solution or replacement fragment. Preserve the draft if correct."
        )
        if feedback:
            prompt += (
                "\n\nEstimated uncertainty, highest first. Line numbers "
                f"refer to the code inside the draft fence.\n{feedback}"
            )
        response = generate(prompt)
        block = code_block(response)
        accepted = block is not None and bool(block["code"].strip())
        revised = block.group().strip() if accepted else draft
        revision: Revision = {
            "review": response[:block.start()].strip() if block else "",
            "hints": hints,
            "accepted": accepted,
        }
        history = [*state.get("history", []), revision]
        stopped = (
            not accepted or len(history) >= max_rounds
            or extract_code(revised).strip() == extract_code(draft).strip()
        )
        return {"draft": revised, "history": history, "stopped": stopped}

    def route(state: AgentState) -> str:
        return END if state["stopped"] else "inspect"

    graph = StateGraph(AgentState)
    graph.add_node("inspect", inspect)
    graph.add_node("reflect_and_revise", reflect_and_revise)
    graph.add_edge(START, "inspect")
    graph.add_edge("inspect", "reflect_and_revise")
    graph.add_conditional_edges("reflect_and_revise", route, ["inspect", END])
    return graph.compile()


def run_benchmark(
    store: TraceStore,
    benchmark: str,
    limit: int = 4,
    max_new_tokens: int = 256,
    check_python: bool = True,
) -> list[dict]:
    tests = (
        python_tests(benchmark)
        if check_python and benchmark in PYTHON_BENCHMARKS else {}
    )

    def eligible(case: RepairCase) -> bool:
        if not (check_python and benchmark in PYTHON_BENCHMARKS):
            return True
        checks = tests.get(case.draft.idx)
        return bool(checks and passes_python_tests(
            extract_code(case.reference), checks
        ))

    train, cases = repair_cases(benchmark, limit, eligible=eligible)
    rows = [store.load(row) for row in train]
    features = [row["trace"] for row in rows]
    labels = [row["labels"] for row in rows]
    estimators = {
        name: factory().fit(features, labels if factory.uses_labels else None)
        for name, factory in METHODS.items()
    }

    @lru_cache(maxsize=256)
    def generate(prompt: str) -> str:
        return store.extractor.generate(prompt, max_new_tokens=max_new_tokens)

    results = []
    for case in tqdm(cases, desc=benchmark, leave=False):
        example = case.draft
        original = store.load(example)
        reference_code = extract_code(case.reference)
        reference = code_tokens(reference_code, benchmark)
        checks = tests.get(example.idx)
        if not reference:
            raise ValueError(f"Empty reference for task {example.task_id}.")

        def measure(
            method: str, output: str, revision: Revision | None = None
        ) -> None:
            code = extract_code(output)
            candidate = code_tokens(code, benchmark)
            results.append({
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
            })

        measure("Archived draft", example.reply)
        initial: AgentState = {"task": example.question, "draft": example.reply}
        with tracing_context(enabled=False):
            plain = create_agent(generate).invoke(initial)
            measure("Plain review", plain["draft"], plain["history"][0])
            for name, estimator in estimators.items():
                def score(draft: str) -> list[Hint]:
                    scores, = estimator.predict_uncertainty([original["trace"]])
                    return rank_lines(draft, scores, original["lines"].tolist())

                revised = create_agent(generate, score).invoke(initial)
                measure(name, revised["draft"], revised["history"][0])
    return results
