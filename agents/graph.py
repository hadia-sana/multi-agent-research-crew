"""LangGraph state-graph definition for the multi-agent research assistant.

Architecture
------------
A **Supervisor** node inspects the current state and routes work to one of
five specialist agents:

* **Researcher** -- gathers information via web search and summarisation.
* **Writer** -- turns research notes into a polished draft or revision.
* **Fact-Checker** -- cross-checks claims in draft against research notes.
* **Reviewer** -- critiques writing quality; decides *revise* or *accept*.

Conditional edges implement a two-stage review: fact-checking for accuracy,
then writing quality review. Both loop back to Writer with revision_count
shared across both feedback types to prevent infinite loops.
"""

from __future__ import annotations

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
    fact_check_verdict: str = ""
    flagged_claims: list[str] = Field(default_factory=list)
    review_feedback: str = ""
    next_agent: str = "researcher"
    revision_count: int = 0


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def supervisor_node(state: AgentState) -> dict:
    """Decide which agent should act next based on the current state."""
    llm = get_llm()

    if state.revision_count >= MAX_REVISIONS:
        return {
            "next_agent": "FINISH",
            "messages": [AIMessage(content="[Supervisor] Revision cap reached, finishing.")],
        }

    system = SystemMessage(
        content=(
            "You are the supervisor of a research team. Based on the current state, "
            "decide which agent should act next.\n\n"
            "Agents:\n"
            "  - researcher: gathers information from the web\n"
            "  - writer: drafts or revises the report\n"
            "  - fact-checker: verifies claims against research notes\n"
            "  - reviewer: critiques writing quality and structure\n"
            "  - FINISH: the task is complete\n\n"
            "Rules:\n"
            "  1. If there are no research notes yet, pick 'researcher'.\n"
            "  2. If there are research notes but no draft, pick 'writer'.\n"
            "  3. If there is a draft but no fact-check verdict, pick 'fact-checker'.\n"
            "  4. If fact-checker returned FLAGGED, pick 'writer' to fix the claims.\n"
            "  5. If fact-checker returned PASS and there's no review, pick 'reviewer'.\n"
            "  6. If reviewer returned REVISE, pick 'writer' to incorporate feedback.\n"
            "  7. If reviewer returned ACCEPT, pick 'FINISH'.\n\n"
            "Respond with ONLY the agent name (one word)."
        )
    )

    context_parts: list[str] = []
    if state.research_notes:
        context_parts.append(f"Research notes:\n{state.research_notes[:1000]}")
    if state.draft:
        context_parts.append(f"Current draft:\n{state.draft[:1000]}")
    if state.fact_check_verdict:
        context_parts.append(f"Fact-check verdict: {state.fact_check_verdict}")
        if state.flagged_claims:
            flagged_preview = "\n".join(f"  - {claim}" for claim in state.flagged_claims[:2])
            context_parts.append(f"Flagged claims:\n{flagged_preview}")
    if state.review_feedback:
        context_parts.append(f"Review feedback:\n{state.review_feedback[:1000]}")

    human = HumanMessage(
        content=(
            "Current state:\n"
            + ("\n---\n".join(context_parts) if context_parts else "(empty -- no work done yet)")
        )
    )

    response = llm.invoke([system, human])
    next_agent = response.content.strip().lower().replace("'", "").replace('"', "")

    if "finish" in next_agent:
        next_agent = "FINISH"
    elif "research" in next_agent:
        next_agent = "researcher"
    elif "writ" in next_agent:
        next_agent = "writer"
    elif "fact" in next_agent:
        next_agent = "fact-checker"
    elif "review" in next_agent:
        next_agent = "reviewer"

    return {
        "next_agent": next_agent,
        "messages": [AIMessage(content=f"[Supervisor] Routing to: {next_agent}")],
    }


def researcher_node(state: AgentState) -> dict:
    """Use tools to research the user's query."""
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
    llm = get_llm()

    revision_context = ""
    if state.flagged_claims:
        revision_context = (
            f"\n\nThe fact-checker flagged these unsupported or contradicted claims:\n"
            + "\n".join(f"  - {claim}" for claim in state.flagged_claims)
            + "\n\nRevise the draft to either support these claims with information "
            "from the research notes, or remove them."
        )
    elif state.review_feedback:
        revision_context = (
            f"\n\nThe reviewer provided this feedback on your previous draft -- "
            f"address every point:\n{state.review_feedback}"
        )

    system = SystemMessage(
        content=(
            "You are a skilled technical writer. Using the research notes "
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

    result = {
        "draft": draft,
        "messages": [AIMessage(content=f"[Writer] Draft produced ({len(draft)} chars)")],
    }

    if state.flagged_claims:
        result["fact_check_verdict"] = ""
        result["flagged_claims"] = []

    if state.review_feedback:
        result["review_feedback"] = ""

    return result


def fact_checker_node(state: AgentState) -> dict:
    """Cross-check draft claims against research notes for factual accuracy."""
    llm = get_llm()

    system = SystemMessage(
        content=(
            "You are a fact-checker. Your job is to identify specific factual claims "
            "in the draft and verify they are directly supported by the research notes.\n\n"
            "For each major claim in the draft:\n"
            "  - If it's clearly supported by the research notes, it passes.\n"
            "  - If it contradicts the notes or is unsupported, flag it.\n\n"
            "After checking all claims, output exactly one verdict on its own line:\n"
            "  PASS -- all claims are supported by the research notes.\n"
            "  FLAGGED -- some claims are unsupported or contradicted.\n\n"
            "If FLAGGED, list each unsupported/contradicted claim below, starting "
            "each line with '- ' (e.g., '- Claim is contradicted by note X')."
        )
    )

    human = HumanMessage(
        content=(
            f"Research notes:\n{state.research_notes}\n\n"
            f"Draft to fact-check:\n{state.draft}"
        )
    )

    response = llm.invoke([system, human])
    feedback = response.content

    verdict = "FLAGGED"
    if "PASS" in feedback.upper().split("\n")[-1]:
        verdict = "PASS"

    flagged_claims = []
    if verdict == "FLAGGED":
        for line in feedback.split("\n"):
            if line.strip().startswith("- "):
                flagged_claims.append(line.strip()[2:])

    revision_count = state.revision_count
    if verdict == "FLAGGED":
        revision_count += 1

    msg_content = f"[Fact-Checker] Verdict: {verdict}"
    if verdict == "FLAGGED":
        msg_content += f" - {len(flagged_claims)} unsupported claim(s) found"
    else:
        msg_content += " - All claims supported"

    return {
        "fact_check_verdict": verdict,
        "flagged_claims": flagged_claims,
        "revision_count": revision_count,
        "messages": [AIMessage(content=msg_content)],
    }


def reviewer_node(state: AgentState) -> dict:
    """Critique writing quality and decide whether to accept or request revision."""
    llm = get_llm()

    system = SystemMessage(
        content=(
            "You are a meticulous editor. Review the draft below for clarity, "
            "structure, coherence, and writing quality. Provide brief, actionable "
            "feedback.\n\n"
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

    revision_count = state.revision_count
    if verdict == "REVISE":
        revision_count += 1

    return {
        "review_feedback": feedback,
        "revision_count": revision_count,
        "messages": [AIMessage(content=f"[Reviewer] Verdict: {verdict}\n{feedback[:300]}...")],
    }


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


def route_supervisor(
    state: AgentState,
) -> Literal["researcher", "writer", "fact-checker", "reviewer", "__end__"]:
    """Return the next node name based on the supervisor's decision."""
    agent = state.next_agent
    if agent == "FINISH":
        return END
    if agent in {"researcher", "writer", "fact-checker", "reviewer"}:
        return agent
    return END


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
    graph.add_node("fact-checker", fact_checker_node)
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
            "fact-checker": "fact-checker",
            "reviewer": "reviewer",
            END: END,
        },
    )

    # All agents return to supervisor for next routing decision
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("fact-checker", "supervisor")
    graph.add_edge("reviewer", "supervisor")

    return graph.compile()
