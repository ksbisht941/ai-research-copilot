from __future__ import annotations

from src.graph_app import build_graph


def main(topic: str = "Write a blog on Gradient Descent") -> None:
    app = build_graph()

    print("[main] invoking graph")
    try:
        out = app.invoke({"topic": topic, "sections": []})
        print("[main] graph completed")
    except Exception as exc:
        print(f"[main] graph failed: {exc!r}")
        raise

    print(out["final"])
