"""LangGraph state-graph definition for the multi-agent research assistant.

Architecture
------------
A **Supervisor** node inspects the current state and routes work to one of
three specialist agents:

* **Researcher** -- gathers information via web search and summarisation.
* **Writer** -- turns research notes into a polished draft.
* **Reviewer** -- critiques the draft; decides *revise* or *accept*.

Conditional edges feed the reviewer's verdict back into the supervisor so
the loop can iterate until the output is accepted or a maximum number of
iterations is reached.
"""

from __future__ import annotations

import time
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from agents.config import get_llm
from agents.tools import summarize, web_search

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

MAX_REVISIONS = 3


class AgentState(BaseModel):
    """Shared state flowing through the graph."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    research_notes: str = ""
    draft: str = ""
    review_feedback: str = ""
    next_agent: str = "researcher"
    revision_count: int = 0


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def supervisor_node(state: AgentState) -> dict:
    """Decide which agent should act next based on the current state."""
    time.sleep(1)  # Delay to prevent throttling
    
    # Deterministic routing based on explicit rules
    if not state.research_notes:
        next_agent = "researcher"
    elif not state.draft:
        next_agent = "writer"
    elif not state.review_feedback:
        next_agent = "reviewer"
    elif "ACCEPT" in state.review_feedback.upper():
        next_agent = "FINISH"
    else:
        next_agent = "writer"

    return {
        "next_agent": next_agent,
        "messages": [AIMessage(content=f"[Supervisor] Routing to: {next_agent}")],
    }


def researcher_node(state: AgentState) -> dict:
    """Use tools to research the user's query."""
    time.sleep(1)  # Delay to prevent throttling
    llm = get_llm()
    query = state.messages[0].content if state.messages else "general research"

    # Step 1 -- web search
    search_results = web_search.invoke(query)

    # Step 2 -- summarise findings
    summary = summarize.invoke(search_results)

    return {
        "research_notes": summary,
        "messages": [AIMessage(content=f"[Researcher] Gathered notes:\n{summary}")],
    }


def writer_node(state: AgentState) -> dict:
    """Produce or revise a written draft from the research notes."""
    time.sleep(1)  # Delay to prevent throttling
    llm = get_llm()

    revision_context = ""
    if state.review_feedback:
        revision_context = (
            f"\n\nThe reviewer provided this feedback on your previous draft -- "
            f"address every point:\n{state.review_feedback}"
        )

    system = SystemMessage(
        content=(
            "You are a skilled technical writer.  Using the research notes "
            "provided, write a clear, well-structured report (3-5 paragraphs). "
            "Use markdown formatting."
            + revision_context
        )
    )
    human = HumanMessage(
        content=f"Research notes:\n{state.research_notes}\n\nPrevious draft:\n{state.draft}"
    )

    response = llm.invoke([system, human])
    draft = response.content

    return {
        "draft": draft,
        "review_feedback": "",  # Clear feedback so supervisor routes to reviewer next
        "messages": [AIMessage(content=f"[Writer] Draft produced ({len(draft)} chars)")],
    }


def reviewer_node(state: AgentState) -> dict:
    """Critique the draft and decide whether to accept or request revision."""
    time.sleep(1)  # Delay to prevent throttling
    llm = get_llm()

    system = SystemMessage(
        content=(
            "You are a meticulous editor.  Review the draft below for accuracy, "
            "clarity, and completeness.  Provide brief, actionable feedback.\n\n"
            "End your review with exactly one of these verdicts on its own line:\n"
            "  ACCEPT -- the draft is ready for publication.\n"
            "  REVISE -- the draft needs further work."
        )
    )
    human = HumanMessage(content=f"Draft:\n{state.draft}")

    response = llm.invoke([system, human])
    feedback = response.content

    verdict = "REVISE"
    if "ACCEPT" in feedback.upper().split("\n")[-1]:
        verdict = "ACCEPT"

    # Enforce maximum revision count
    revision_count = state.revision_count + 1
    if revision_count >= MAX_REVISIONS:
        verdict = "ACCEPT"
        feedback += "\n\n[Auto-accepted after maximum revisions reached.]"

    return {
        "review_feedback": feedback,
        "revision_count": revision_count,
        "messages": [AIMessage(content=f"[Reviewer] Verdict: {verdict}\n{feedback}")],
    }


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


def route_supervisor(state: AgentState) -> Literal["researcher", "writer", "reviewer", "__end__"]:
    """Return the next node name based on the supervisor's decision."""
    agent = state.next_agent
    if agent == "FINISH":
        return END
    if agent in {"researcher", "writer", "reviewer"}:
        return agent
    # Fallback -- shouldn't happen but keeps the graph safe.
    return END


def route_after_review(state: AgentState) -> Literal["supervisor"]:
    """Always return to the supervisor after a review."""
    return "supervisor"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Construct and compile the multi-agent research graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor routes conditionally
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            END: END,
        },
    )

    # After each specialist, return to supervisor
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("reviewer", "supervisor")

    return graph.compile()
