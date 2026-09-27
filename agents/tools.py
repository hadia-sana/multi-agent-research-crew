"""Custom tools available to the research assistant agents."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from agents.config import get_llm


@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information using DuckDuckGo (no API key required).

    Parameters
    ----------
    query:
        The search query string.

    Returns
    -------
    str
        Formatted search results with titles, URLs, and content snippets.
    """
    from ddgs import DDGS

    try:
        results = DDGS().text(query, max_results=5)

        if not results:
            return "No results found."

        formatted: list[str] = []
        for idx, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            url = result.get("href", "")
            content = result.get("body", "")
            formatted.append(f"[{idx}] {title}\n    {url}\n    {content}")

        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def summarize(text: str) -> str:
    """Produce a concise summary of the provided text using the configured LLM.

    Parameters
    ----------
    text:
        The text to summarise.

    Returns
    -------
    str
        A concise summary.
    """
    llm = get_llm(temperature=0.0)
    messages = [
        SystemMessage(
            content=(
                "You are a concise summariser. Distil the following text into "
                "its key points using no more than five bullet points."
            )
        ),
        HumanMessage(content=text),
    ]
    response = llm.invoke(messages)
    return response.content
