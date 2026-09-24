"""Tests for the supervisor routing logic in agents/graph.py."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from unittest.mock import Mock, patch

from agents.graph import (
    AgentState,
    supervisor_node,
    researcher_node,
    writer_node,
    reviewer_node,
    route_supervisor,
)


@pytest.fixture
def mock_llm():
    """Fixture for a mocked LLM."""
    with patch("agents.graph.get_llm") as mock:
        yield mock


@pytest.fixture
def empty_state():
    """Fixture for an empty initial state."""
    return AgentState(messages=[HumanMessage(content="What is quantum computing?")])


@pytest.fixture
def state_with_research():
    """Fixture for a state with research notes."""
    return AgentState(
        messages=[HumanMessage(content="What is quantum computing?")],
        research_notes="Quantum computing uses qubits...",
    )


@pytest.fixture
def state_with_draft():
    """Fixture for a state with both research and draft."""
    return AgentState(
        messages=[HumanMessage(content="What is quantum computing?")],
        research_notes="Quantum computing uses qubits...",
        draft="# Quantum Computing\n\nQuantum computers...",
    )


@pytest.fixture
def state_with_accept_verdict():
    """Fixture for a state where the reviewer accepted the draft."""
    return AgentState(
        messages=[HumanMessage(content="What is quantum computing?")],
        research_notes="Quantum computing uses qubits...",
        draft="# Quantum Computing\n\nQuantum computers...",
        review_feedback="ACCEPT",
        next_agent="FINISH",
    )


class TestSupervisorRouting:
    """Tests for supervisor_node routing logic."""

    def test_supervisor_routes_to_researcher_when_no_notes(self, empty_state, mock_llm):
        """Supervisor should route to researcher when no research notes exist."""
        mock_response = Mock()
        mock_response.content = "researcher"
        mock_llm.return_value.invoke.return_value = mock_response

        result = supervisor_node(empty_state)

        assert "next_agent" in result
        assert result["next_agent"] == "researcher"
        assert len(result["messages"]) > 0

    def test_supervisor_routes_to_writer_when_notes_exist(
        self, state_with_research, mock_llm
    ):
        """Supervisor should route to writer when research notes exist but no draft."""
        mock_response = Mock()
        mock_response.content = "writer"
        mock_llm.return_value.invoke.return_value = mock_response

        result = supervisor_node(state_with_research)

        assert result["next_agent"] == "writer"

    def test_supervisor_routes_to_reviewer_when_draft_exists(
        self, state_with_draft, mock_llm
    ):
        """Supervisor should route to reviewer when draft exists."""
        mock_response = Mock()
        mock_response.content = "reviewer"
        mock_llm.return_value.invoke.return_value = mock_response

        result = supervisor_node(state_with_draft)

        assert result["next_agent"] == "reviewer"

    def test_supervisor_normalizes_llm_responses(self, empty_state, mock_llm):
        """Supervisor should normalize LLM responses like 'research' to 'researcher'."""
        mock_response = Mock()
        mock_response.content = "research"
        mock_llm.return_value.invoke.return_value = mock_response

        result = supervisor_node(empty_state)

        assert result["next_agent"] == "researcher"

    def test_supervisor_recognizes_finish_verdict(self, state_with_accept_verdict, mock_llm):
        """Supervisor should recognize FINISH when review is accepted."""
        mock_response = Mock()
        mock_response.content = "FINISH"
        mock_llm.return_value.invoke.return_value = mock_response

        result = supervisor_node(state_with_accept_verdict)

        assert result["next_agent"] == "FINISH"


class TestRoutingHelpers:
    """Tests for routing helper functions."""

    def test_route_supervisor_returns_end_on_finish(self, state_with_accept_verdict):
        """route_supervisor should return END when next_agent is FINISH."""
        from langgraph.graph import END

        state_with_accept_verdict.next_agent = "FINISH"
        result = route_supervisor(state_with_accept_verdict)

        assert result == END

    def test_route_supervisor_returns_agent_name(self):
        """route_supervisor should return agent name for valid agents."""
        state = AgentState(messages=[HumanMessage(content="test")])
        state.next_agent = "researcher"

        result = route_supervisor(state)

        assert result == "researcher"

    def test_route_supervisor_fallback_to_end(self):
        """route_supervisor should return END for invalid agent names."""
        from langgraph.graph import END

        state = AgentState(messages=[HumanMessage(content="test")])
        state.next_agent = "invalid_agent"

        result = route_supervisor(state)

        assert result == END


class TestResearcherNode:
    """Tests for the researcher_node."""

    def test_researcher_node_calls_web_search_and_summarize(self, empty_state):
        """Researcher should call web_search and summarize tools."""
        with patch("agents.graph.web_search") as mock_search, patch(
            "agents.graph.summarize"
        ) as mock_summarize, patch("agents.graph.get_llm"):
            mock_search.invoke.return_value = "Search result 1\nSearch result 2"
            mock_summarize.invoke.return_value = "- Key point 1\n- Key point 2"

            result = researcher_node(empty_state)

            assert "research_notes" in result
            assert result["research_notes"] == "- Key point 1\n- Key point 2"
            mock_search.invoke.assert_called_once()
            mock_summarize.invoke.assert_called_once()


class TestWriterNode:
    """Tests for the writer_node."""

    def test_writer_node_produces_draft(self, state_with_research, mock_llm):
        """Writer should produce a draft from research notes."""
        mock_response = Mock()
        mock_response.content = "# Research Report\n\nThis is a report based on the research."
        mock_llm.return_value.invoke.return_value = mock_response

        result = writer_node(state_with_research)

        assert "draft" in result
        assert "Research Report" in result["draft"]

    def test_writer_node_includes_reviewer_feedback(self, state_with_draft, mock_llm):
        """Writer should incorporate reviewer feedback when revising."""
        state_with_draft.review_feedback = "Please add more detail about quantum gates."

        mock_response = Mock()
        mock_response.content = "# Quantum Computing\n\nWith added details about quantum gates..."
        mock_llm.return_value.invoke.return_value = mock_response

        result = writer_node(state_with_draft)

        # Verify the LLM was called with the feedback context
        call_args = mock_llm.return_value.invoke.call_args
        assert "feedback" in str(call_args).lower() or "revision" in str(call_args).lower()


class TestReviewerNode:
    """Tests for the reviewer_node."""

    def test_reviewer_accepts_draft(self, state_with_draft, mock_llm):
        """Reviewer should recognize ACCEPT verdict."""
        mock_response = Mock()
        mock_response.content = "This draft is well-written and accurate.\nACCEPT"
        mock_llm.return_value.invoke.return_value = mock_response

        result = reviewer_node(state_with_draft)

        assert "review_feedback" in result
        assert "ACCEPT" in result["review_feedback"]

    def test_reviewer_requests_revision(self, state_with_draft, mock_llm):
        """Reviewer should recognize REVISE verdict."""
        mock_response = Mock()
        mock_response.content = "This needs more detail about X.\nREVISE"
        mock_llm.return_value.invoke.return_value = mock_response

        result = reviewer_node(state_with_draft)

        assert "review_feedback" in result

    def test_reviewer_enforces_max_revisions(self, state_with_draft, mock_llm):
        """Reviewer should auto-accept after MAX_REVISIONS."""
        from agents.graph import MAX_REVISIONS

        state_with_draft.revision_count = MAX_REVISIONS - 1
        mock_response = Mock()
        mock_response.content = "More work needed.\nREVISE"
        mock_llm.return_value.invoke.return_value = mock_response

        result = reviewer_node(state_with_draft)

        assert result["revision_count"] == MAX_REVISIONS
        assert "Auto-accepted" in result["review_feedback"]
