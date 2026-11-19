from collections.abc import Callable, Sequence
from math import isfinite
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from torch import Tensor

FENCE = re.compile(
    r"^[ \t]*```[^\n]*\n(?P<code>.*?)^[ \t]*```[ \t]*\r?(?:\n|$)",
    re.MULTILINE | re.DOTALL,
)
type Hint = tuple[int, str]
type Scorer = Callable[[str], list[Hint]]


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


class CandidateState(TypedDict, total=False):
    task: str
    draft: str
    variants: list[str]
    uncertainties: list[float]
    selected: int | None


def create_agent_candidates(
    generate: Callable[[str], str],
    score: Callable[[str], float],
    max_candidates: int = 3,
) -> CompiledStateGraph:
    if not 1 <= max_candidates <= 3:
        raise ValueError("Use between one and three candidates.")

    def propose(state: CandidateState) -> CandidateState:
        prompt = (
            f"Task:\n{state['task']}\n\nCurrent draft:\n{state['draft']}\n\n"
            f"Propose up to {max_candidates} distinct corrected code variants. "
            "Put each complete solution or replacement fragment in its own "
            "fenced code block. Output only the code blocks."
        )
        variants = {}
        for block in FENCE.finditer(generate(prompt)):
            code = block["code"].rstrip()
            if code:
                variants.setdefault(code, block.group().strip())
            if len(variants) == max_candidates:
                break
        return {"variants": list(variants.values())}

    def select(state: CandidateState) -> CandidateState:
        variants = state["variants"]
        uncertainties = [score(reply) for reply in variants]
        usable = [
            index for index, value in enumerate(uncertainties)
            if isfinite(value)
        ]
        selected = min(usable, key=uncertainties.__getitem__, default=None)
        draft = state["draft"] if selected is None else variants[selected]
        return {
            "draft": draft,
            "uncertainties": uncertainties,
            "selected": selected,
        }

    return (
        StateGraph(CandidateState)
        .add_node("propose", propose)
        .add_node("select", select)
        .add_edge(START, "propose")
        .add_edge("propose", "select")
        .add_edge("select", END)
        .compile()
    )
