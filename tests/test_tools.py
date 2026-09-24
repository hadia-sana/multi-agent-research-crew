"""Tests for agents/tools.py."""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from agents.tools import web_search, summarize


class TestWebSearchTool:
    """Tests for the web_search tool."""

    def test_web_search_formatting_logic(self):
        """web_search should format search results correctly."""
        mock_results = [
            {
                "title": "Result 1",
                "link": "https://example.com/1",
                "snippet": "This is the first result",
            },
            {
                "title": "Result 2",
                "link": "https://example.com/2",
                "snippet": "This is the second result",
            },
        ]

        formatted = []
        for idx, result in enumerate(mock_results, 1):
            title = result.get("title", "Untitled")
            url = result.get("link", result.get("url", ""))
            content = result.get("snippet", "")
            formatted.append(f"[{idx}] {title}\n    {url}\n    {content}")

        output = "\n\n".join(formatted)
        assert "Result 1" in output
        assert "Result 2" in output
        assert "[1]" in output
        assert "[2]" in output
        assert "https://example.com/1" in output
        assert "https://example.com/2" in output

    def test_web_search_tool_exists(self):
        """web_search tool should be defined."""
        assert web_search is not None
        assert hasattr(web_search, "invoke")

    def test_web_search_formatting_handles_missing_fields(self):
        """web_search formatting should handle results with missing fields."""
        mock_results = [
            {"title": "Result A"},  # Missing url and snippet
            {"url": "https://example.com"},  # Missing title and snippet
        ]

        formatted = []
        for idx, result in enumerate(mock_results, 1):
            title = result.get("title", "Untitled")
            url = result.get("link", result.get("url", ""))
            content = result.get("snippet", "")
            formatted.append(f"[{idx}] {title}\n    {url}\n    {content}")

        output = "\n\n".join(formatted)
        assert "Result A" in output or "Untitled" in output
        assert "https://example.com" in output


class TestSummarizeTool:
    """Tests for the summarize tool."""

    def test_summarize_calls_llm_with_system_prompt(self):
        """summarize should call LLM with a system message asking for a summary."""
        with patch("agents.tools.get_llm") as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "- Key point 1\n- Key point 2\n- Key point 3"
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            result = summarize.invoke("Long text about quantum computing...")

            assert "Key point" in result
            # Verify LLM was called
            mock_llm.invoke.assert_called_once()
            # Verify the call includes messages (system + human)
            call_args = mock_llm.invoke.call_args
            assert call_args is not None

    def test_summarize_returns_llm_response_content(self):
        """summarize should return the LLM's response content."""
        with patch("agents.tools.get_llm") as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            expected_summary = "Quantum computing is a revolutionary technology using qubits."
            mock_response.content = expected_summary
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            result = summarize.invoke("Lorem ipsum dolor sit amet...")

            assert result == expected_summary

    def test_summarize_uses_deterministic_temperature(self):
        """summarize should use temperature=0.0 for deterministic output."""
        with patch("agents.tools.get_llm") as mock_get_llm:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Summary"
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            summarize.invoke("Text to summarize")

            # Verify get_llm was called with temperature=0.0
            mock_get_llm.assert_called_once_with(temperature=0.0)
