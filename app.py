"""Streamlit UI for the multi-agent research assistant."""

import streamlit as st
from langchain_core.messages import HumanMessage
from agents.graph import build_graph

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 Multi-Agent Research Assistant")
st.markdown("Powered by LangGraph and AWS Bedrock")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    max_revisions = st.slider("Max Revisions", min_value=1, max_value=5, value=3)
    verbose_mode = st.checkbox("Verbose Output", value=False)
    st.divider()
    st.markdown("### About")
    st.markdown(
        """
        This research assistant uses multiple AI agents working together:
        - **Researcher**: Gathers web information
        - **Writer**: Creates polished reports
        - **Reviewer**: Provides quality feedback
        - **Supervisor**: Orchestrates the workflow
        """
    )

# Main content
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Research Query")
    query = st.text_area(
        "Enter your research question:",
        placeholder="e.g., What are the latest breakthroughs in quantum computing?",
        height=100,
        label_visibility="collapsed",
    )

with col2:
    st.subheader("Status")
    run_button = st.button("🚀 Run Research", use_container_width=True, type="primary")

st.divider()

if run_button and query:
    st.subheader("Research Progress")

    with st.spinner("Initializing research workflow..."):
        graph = build_graph()
        initial_state = {"messages": [HumanMessage(content=query)]}

        progress_placeholder = st.empty()
        output_tabs = st.tabs(
            ["Researcher", "Writer", "Reviewer", "Supervisor", "Final Report"]
        )
        tab_outputs = {
            "researcher": output_tabs[0],
            "writer": output_tabs[1],
            "reviewer": output_tabs[2],
            "supervisor": output_tabs[3],
            "final": output_tabs[4],
        }

        step_count = 0
        final_draft = ""

        with progress_placeholder.container():
            st.info("🔄 Research in progress...")

        try:
            for step in graph.stream(initial_state, stream_mode="updates"):
                for node_name, node_output in step.items():
                    step_count += 1
                    progress_placeholder.progress(
                        min(step_count / 20, 1.0), text=f"Step {step_count}..."
                    )

                    messages = node_output.get("messages", [])
                    for msg in messages:
                        content = (
                            msg.content if hasattr(msg, "content") else str(msg)
                        )

                        if node_name.lower() == "supervisor":
                            with tab_outputs["supervisor"]:
                                st.markdown(content)
                        elif node_name.lower() == "researcher":
                            with tab_outputs["researcher"]:
                                st.markdown(content)
                        elif node_name.lower() == "writer":
                            with tab_outputs["writer"]:
                                st.markdown(content)
                                draft_text = node_output.get("draft", "")
                                if draft_text:
                                    with st.expander("View Draft Content", expanded=True):
                                        st.markdown(draft_text)
                            final_draft = (
                                draft_text or final_draft
                            )
                        elif node_name.lower() == "reviewer":
                            with tab_outputs["reviewer"]:
                                st.markdown(content)

            progress_placeholder.empty()

            # Get final report
            final_state = graph.invoke(initial_state)
            final_draft = final_state.get("draft", final_draft)

            with tab_outputs["final"]:
                st.success("✅ Research Complete!")
                st.markdown(final_draft)

            st.balloons()

        except Exception as e:
            progress_placeholder.error(f"❌ Error: {str(e)}")
            st.error(f"An error occurred during research: {str(e)}")

elif run_button:
    st.warning("⚠️ Please enter a research query first!")
