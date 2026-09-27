"""CLI entry point for the multi-agent research assistant."""

from __future__ import annotations

import argparse
import sys

from langchain_core.messages import HumanMessage

from agents.graph import build_graph


def safe_print(text: str) -> None:
    """Print text safely, handling encoding errors."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Write directly with UTF-8 encoding
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant powered by LangGraph",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="The research query to investigate.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full message contents from every node.",
    )
    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        sys.exit(1)

    safe_print(f"\n{'=' * 60}")
    safe_print(f"  Research query: {args.query}")
    safe_print(f"{'=' * 60}\n")

    graph = build_graph()

    initial_state = {
        "messages": [HumanMessage(content=args.query)],
    }

    # Stream node-by-node so the user sees progress.
    for step in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in step.items():
            safe_print(f"\n--- [{node_name.upper()}] ---")

            # Print latest message from this node
            messages = node_output.get("messages", [])
            for msg in messages:
                content = msg.content if hasattr(msg, "content") else str(msg)
                if args.verbose:
                    safe_print(content)
                else:
                    # Print first 500 chars for brevity
                    preview = content[:500]
                    if len(content) > 500:
                        preview += "..."
                    safe_print(preview)

    # Print final draft
    safe_print(f"\n{'=' * 60}")
    safe_print("  FINAL REPORT")
    safe_print(f"{'=' * 60}\n")

    # Retrieve final state
    final_state = graph.invoke(initial_state)
    draft = final_state.get("draft", "")
    if draft:
        safe_print(draft)
    else:
        safe_print("No draft was produced.")


if __name__ == "__main__":
    main()
