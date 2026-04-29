from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .models import Plan, State


llm = ChatMistralAI()


def orchestrator(state: State) -> dict:
    print("[orchestrator] starting")
    print(f"[orchestrator] topic={state['topic']}")

    try:
        plan = llm.with_structured_output(Plan).invoke(
            [
                SystemMessage(
                    content=(
                        "You are a senior technical writer and developer advocate. Your job is to produce a "
                        "highly actionable outline for a technical blog post.\n\n"
                        "Hard requirements:\n"
                        "- Create 5–7 sections (tasks) that fit a technical blog.\n"
                        "- Each section must include:\n"
                        "  1) goal (1 sentence: what the reader can do/understand after the section)\n"
                        "  2) 3–5 bullets that are concrete, specific, and non-overlapping\n"
                        "  3) target word count (120–450)\n"
                        "- Include EXACTLY ONE section with section_type='common_mistakes'.\n\n"
                        "Make it technical (not generic):\n"
                        "- Assume the reader is a developer; use correct terminology.\n"
                        "- Prefer design/engineering structure: problem → intuition → approach → implementation → "
                        "trade-offs → testing/observability → conclusion.\n"
                        "- Bullets must be actionable and testable (e.g., 'Show a minimal code snippet for X', "
                        "'Explain why Y fails under Z condition', 'Add a checklist for production readiness').\n"
                        "- Explicitly include at least ONE of the following somewhere in the plan (as bullets):\n"
                        "  * a minimal working example (MWE) or code sketch\n"
                        "  * edge cases / failure modes\n"
                        "  * performance/cost considerations\n"
                        "  * security/privacy considerations (if relevant)\n"
                        "  * debugging tips / observability (logs, metrics, traces)\n"
                        "- Avoid vague bullets like 'Explain X' or 'Discuss Y'. Every bullet should state what "
                        "to build/compare/measure/verify.\n\n"
                        "Ordering guidance:\n"
                        "- Start with a crisp intro and problem framing.\n"
                        "- Build core concepts before advanced details.\n"
                        "- Include one section for common mistakes and how to avoid them.\n"
                        "- End with a practical summary/checklist and next steps.\n\n"
                        "Output must strictly match the Plan schema."
                    )
                ),
                HumanMessage(
                    content=f"Topic: {state['topic']}"
                ),
            ]
        )

        print(f"[orchestrator] plan created: title={plan.blog_title}, tasks={len(plan.tasks)}")
        return {"plan": plan}
    except Exception as exc:
        print(f"[orchestrator] error: {exc!r}")
        raise


def fanout(state: State):
    print(f"[fanout] splitting plan into {len(state['plan'].tasks)} worker tasks")
    return [
        Send("worker", {
            "task": task,
            "topic": state["topic"],
            "plan": state["plan"],
        })
        for task in state["plan"].tasks
    ]


def worker(payload: dict) -> dict:
    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]

    print(f"[worker] generating section for task={task.id} title={task.title}")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    try:
        section_md = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a senior technical writer and developer advocate. Write ONE section of a technical blog post in Markdown.\n\n"
                        "Hard constraints:\n"
                        "- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).\n"
                        "- Stay close to the Target words (±15%).\n"
                        "- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).\n\n"
                        "Technical quality bar:\n"
                        "- Be precise and implementation-oriented (developers should be able to apply it).\n"
                        "- Prefer concrete details over abstractions: APIs, data structures, protocols, and exact terms.\n"
                        "- When relevant, include at least one of:\n"
                        "  * a small code snippet (minimal, correct, and idiomatic)\n"
                        "  * a tiny example input/output\n"
                        "  * a checklist of steps\n"
                        "  * a diagram described in text (e.g., 'Flow: A -> B -> C')\n"
                        "- Explain trade-offs briefly (performance, cost, complexity, reliability).\n"
                        "- Call out edge cases / failure modes and what to do about them.\n"
                        "- If you mention a best practice, add the 'why' in one sentence.\n\n"
                        "Markdown style:\n"
                        "- Start with a '## <Section Title>' heading.\n"
                        "- Use short paragraphs, bullet lists where helpful, and code fences for code.\n"
                        "- Avoid fluff. Avoid marketing language.\n"
                        "- If you include code, keep it focused on the bullet being addressed.\n"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Blog: {plan.blog_title}\n"
                        f"Audience: {plan.audience}\n"
                        f"Tone: {plan.tone}\n"
                        f"Topic: {topic}\n\n"
                        f"Section: {task.title}\n"
                        f"Section type: {task.section_type}\n"
                        f"Goal: {task.goal}\n"
                        f"Target words: {task.target_words}\n"
                        f"Bullets:{bullets_text}\n"
                    )
                ),
            ]
        ).content.strip()

        print(f"[worker] completed section for task={task.id}")
        return {"sections": [section_md]}
    except Exception as exc:
        print(f"[worker] error for task={task.id}: {exc!r}")
        raise


def reducer(state: State) -> dict:
    print("[reducer] assembling final document")
    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"]).strip()

    try:
        final_md = f"# {title}\n\n{body}\n"
        filename = "".join(c if c.isalnum() or c in (" ", "_", "-") else "" for c in title)
        print(f"[reducer] writing output file: {filename}")
        output_path = Path(filename)
        output_path.write_text(final_md, encoding="utf-8")
        return {"final": final_md}
    except Exception as exc:
        print(f"[reducer] error: {exc!r}")
        raise


def build_graph():
    g = StateGraph(State)
    g.add_node("orchestrator", orchestrator)
    g.add_node("worker", worker)
    g.add_node("reducer", reducer)

    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", fanout, ["worker"])
    g.add_edge("worker", "reducer")
    g.add_edge("reducer", END)

    return g.compile()
