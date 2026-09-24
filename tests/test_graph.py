"""Tests for the supervisor routing logic in agents/graph.py."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from unittest.mock import Mock, patch

from agents.graph import (
    AgentState,
    supervisor_node,
    researcher_node,
    writer_node,
    fact_checker_node,
    reviewer_node,
    route_supervisor,
    parse_structured_verdict,
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


class TestVerdictExtraction:
    """Tests for verdict extraction from structured JSON responses."""

    def test_parse_structured_verdict_pass(self):
        """Should extract PASS from valid JSON."""
        response = '{"verdict": "PASS", "feedback": "All claims verified."}'
        verdict, feedback = parse_structured_verdict(response, ("PASS", "FLAGGED"))
        assert verdict == "PASS"
        assert "All claims verified" in feedback

    def test_parse_structured_verdict_flagged(self):
        """Should extract FLAGGED from valid JSON."""
        response = '{"verdict": "FLAGGED", "feedback": "- Claim X is unsupported"}'
        verdict, feedback = parse_structured_verdict(response, ("PASS", "FLAGGED"))
        assert verdict == "FLAGGED"
        assert "Claim X" in feedback

    def test_parse_structured_verdict_rejects_negation_in_prose(self):
        """Should NOT be fooled by 'cannot accept' or negation phrases."""
        response = '{"verdict": "REVISE", "feedback": "Cannot accept this claim due to lack of evidence"}'
        verdict, feedback = parse_structured_verdict(response, ("ACCEPT", "REVISE"))
        assert verdict == "REVISE"

    def test_parse_structured_verdict_case_insensitive(self):
        """Should handle verdict values regardless of case."""
        response = '{"verdict": "accept", "feedback": "Ready to publish."}'
        verdict, feedback = parse_structured_verdict(response, ("ACCEPT", "REVISE"))
        assert verdict == "ACCEPT"

    def test_parse_structured_verdict_malformed_json_fails_safe(self):
        """Should default to negative verdict on malformed JSON."""
        response = "This is not JSON at all"
        verdict, feedback = parse_structured_verdict(response, ("ACCEPT", "REVISE"))
        assert verdict == "REVISE"  # Fails safe to negative verdict
        assert feedback == response  # Returns original text

    def test_parse_structured_verdict_missing_verdict_field(self):
        """Should default to negative verdict if verdict field is missing."""
        response = '{"feedback": "Some feedback but no verdict"}'
        verdict, feedback = parse_structured_verdict(response, ("ACCEPT", "REVISE"))
        assert verdict == "REVISE"  # Fails safe to negative verdict

    def test_parse_structured_verdict_wrong_verdict_value(self):
        """Should default to negative verdict if verdict value is unexpected."""
        response = '{"verdict": "UNKNOWN", "feedback": "Not a recognized verdict"}'
        verdict, feedback = parse_structured_verdict(response, ("ACCEPT", "REVISE"))
        assert verdict == "REVISE"  # Fails safe to negative verdict

    def test_parse_structured_verdict_with_markdown_fence_backticks(self):
        """Should parse JSON correctly when wrapped in markdown code fences."""
        response = '```json\n{"verdict": "ACCEPT", "feedback": "Well written."}\n```'
        verdict, feedback = parse_structured_verdict(response, ("ACCEPT", "REVISE"))
        assert verdict == "ACCEPT"
        assert "Well written" in feedback

    def test_parse_structured_verdict_with_generic_markdown_fence(self):
        """Should parse JSON correctly when wrapped in generic ``` fences (no language tag)."""
        response = '```\n{"verdict": "PASS", "feedback": "All claims verified."}\n```'
        verdict, feedback = parse_structured_verdict(response, ("PASS", "FLAGGED"))
        assert verdict == "PASS"
        assert "All claims" in feedback

    def test_parse_structured_verdict_with_whitespace_and_fences(self):
        """Should handle whitespace and fences robustly."""
        response = """
        ```json
        {
          "verdict": "REVISE",
          "feedback": "Needs work"
        }
        ```
        """
        verdict, feedback = parse_structured_verdict(response, ("ACCEPT", "REVISE"))
        assert verdict == "REVISE"
        assert "Needs work" in feedback

    def test_parse_structured_verdict_negation_with_wrapped_json(self):
        """Should correctly parse negation in wrapped JSON format."""
        response = '```json\n{"verdict": "REVISE", "feedback": "Cannot accept: too many unsupported claims"}\n```'
        verdict, feedback = parse_structured_verdict(response, ("ACCEPT", "REVISE"))
        assert verdict == "REVISE"
        assert "Cannot accept" in feedback


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


class TestFactCheckerNode:
    """Tests for the fact_checker_node."""

    def test_fact_checker_passes_when_all_claims_supported(
        self, state_with_draft, mock_llm
    ):
        """Fact-Checker should return PASS when all claims are supported."""
        mock_response = Mock()
        mock_response.content = '{"verdict": "PASS", "feedback": "All claims in the draft are supported by the research notes."}'
        mock_llm.return_value.invoke.return_value = mock_response

        result = fact_checker_node(state_with_draft)

        assert result["fact_check_verdict"] == "PASS"
        assert result["flagged_claims"] == []

    def test_fact_checker_flags_unsupported_claims(self, state_with_draft, mock_llm):
        """Fact-Checker should return FLAGGED with claim list when issues found."""
        mock_response = Mock()
        mock_response.content = (
            '{"verdict": "FLAGGED", "feedback": "- Claim \'X is true\' is not mentioned in research notes\\n'
            '- Claim \'Y contradicts\' contradicts the notes"}'
        )
        mock_llm.return_value.invoke.return_value = mock_response

        result = fact_checker_node(state_with_draft)

        assert result["fact_check_verdict"] == "FLAGGED"
        assert len(result["flagged_claims"]) == 2
        assert "is not mentioned" in result["flagged_claims"][0]

    def test_fact_checker_increments_revision_on_flagged(
        self, state_with_draft, mock_llm
    ):
        """Fact-Checker should increment revision_count only when FLAGGED."""
        state_with_draft.revision_count = 0
        mock_response = Mock()
        mock_response.content = "- Unsupported claim\nFLAGGED"
        mock_llm.return_value.invoke.return_value = mock_response

        result = fact_checker_node(state_with_draft)

        assert result["revision_count"] == 1

    def test_fact_checker_does_not_increment_on_pass(self, state_with_draft, mock_llm):
        """Fact-Checker should not increment revision_count when PASS."""
        state_with_draft.revision_count = 1
        mock_response = Mock()
        mock_response.content = '{"verdict": "PASS", "feedback": "All claims verified."}'
        mock_llm.return_value.invoke.return_value = mock_response

        result = fact_checker_node(state_with_draft)

        assert result["revision_count"] == 1


class TestReviewerNode:
    """Tests for the reviewer_node."""

    def test_reviewer_accepts_draft(self, state_with_draft, mock_llm):
        """Reviewer should recognize ACCEPT verdict from JSON."""
        mock_response = Mock()
        mock_response.content = '{"verdict": "ACCEPT", "feedback": "This draft is well-written and accurate."}'
        mock_llm.return_value.invoke.return_value = mock_response

        result = reviewer_node(state_with_draft)

        assert "review_feedback" in result
        assert "well-written" in result["review_feedback"]

    def test_reviewer_requests_revision(self, state_with_draft, mock_llm):
        """Reviewer should recognize REVISE verdict from JSON."""
        mock_response = Mock()
        mock_response.content = '{"verdict": "REVISE", "feedback": "This needs more detail about X."}'
        mock_llm.return_value.invoke.return_value = mock_response

        result = reviewer_node(state_with_draft)

        assert "review_feedback" in result
        assert "more detail" in result["review_feedback"]

    def test_reviewer_increments_revision_on_revise(self, state_with_draft, mock_llm):
        """Reviewer should increment revision_count when verdict is REVISE."""
        state_with_draft.revision_count = 1
        mock_response = Mock()
        mock_response.content = '{"verdict": "REVISE", "feedback": "More work needed."}'
        mock_llm.return_value.invoke.return_value = mock_response

        result = reviewer_node(state_with_draft)

        assert result["revision_count"] == 2

    def test_reviewer_does_not_increment_on_accept(self, state_with_draft, mock_llm):
        """Reviewer should not increment revision_count when verdict is ACCEPT."""
        state_with_draft.revision_count = 2
        mock_response = Mock()
        mock_response.content = '{"verdict": "ACCEPT", "feedback": "Well written and accurate."}'
        mock_llm.return_value.invoke.return_value = mock_response

        result = reviewer_node(state_with_draft)

        assert result["revision_count"] == 2
