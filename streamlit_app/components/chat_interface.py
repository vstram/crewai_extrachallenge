import streamlit as st
import os
import sys
from typing import List, Dict, Any

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from utils.session_manager import SessionManager


def render_chat_interface() -> None:
    """Render the interactive chat interface (Phase 3)."""
    st.header("💬 Phase 3: Interactive Chat")

    # Check if analysis is complete
    if not SessionManager.is_analysis_complete():
        st.warning("⚠️ Complete the fraud detection analysis first to enable chat.")
        return

    st.write("Ask questions about your fraud detection results and get insights from the AI agents.")

    # Quick action buttons
    _render_quick_actions()

    # Chat history display
    _render_chat_history()

    # Chat input
    _render_chat_input()


def _render_quick_actions() -> None:
    """Render quick action buttons for common questions."""
    st.subheader("🚀 Quick Questions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Show Statistics", use_container_width=True):
            _ask_question("Can you provide additional statistical insights about the dataset?")

    with col2:
        if st.button("🔍 Explain Patterns", use_container_width=True):
            _ask_question("What are the most significant fraud patterns you found?")

    with col3:
        if st.button("💡 Recommendations", use_container_width=True):
            _ask_question("What are your top recommendations for fraud prevention?")

    # Additional quick actions
    col4, col5, col6 = st.columns(3)

    with col4:
        if st.button("📈 Risk Assessment", use_container_width=True):
            _ask_question("How should we assess risk for different transaction types?")

    with col5:
        if st.button("🎯 Feature Analysis", use_container_width=True):
            _ask_question("Which features are most important for fraud detection?")

    with col6:
        if st.button("⚙️ Model Performance", use_container_width=True):
            _ask_question("How well did the classification model perform?")


def _render_chat_history() -> None:
    """Render chat message history."""
    st.subheader("💬 Conversation")

    chat_history = SessionManager.get_chat_history()

    if not chat_history:
        st.info("👋 Start a conversation by asking a question or using the quick actions above!")
        return

    # Display chat messages
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Clear history button
    if len(chat_history) > 0:
        if st.button("🗑️ Clear Chat History", help="Clear all chat messages"):
            SessionManager.clear_chat_history()
            st.rerun()


def _render_chat_input() -> None:
    """Render chat input interface."""
    # Chat input
    if prompt := st.chat_input("Ask a question about your fraud detection results..."):
        _ask_question(prompt)


def _ask_question(question: str) -> None:
    """Process a user question and generate response."""
    # Add user message to history
    SessionManager.add_chat_message("user", question)

    # Generate response (placeholder for now - will integrate with CrewAI agents in future)
    response = _generate_response(question)

    # Add assistant response to history
    SessionManager.add_chat_message("assistant", response)

    # Rerun to update chat display
    st.rerun()


def _generate_response(question: str) -> str:
    """Generate response to user question (placeholder implementation)."""
    # Get analysis results for context
    results = SessionManager.get_analysis_results()
    dataset_info = SessionManager.get_dataset_info()

    # Simple response generation based on question content
    question_lower = question.lower()

    if "statistic" in question_lower or "data" in question_lower:
        rows = dataset_info.get('rows', 0)
        columns = dataset_info.get('columns', 0)
        images = results.get('images_found', 0)

        return f"""📊 **Dataset Statistics:**

- **Total Transactions:** {rows:,}
- **Features:** {columns} columns
- **Visualizations Generated:** {images} charts
- **Analysis Type:** {'Supervised' if dataset_info.get('has_class_column') else 'Unsupervised'}

The analysis covered data quality assessment, correlation analysis, pattern recognition, and classification modeling."""

    elif "pattern" in question_lower or "fraud" in question_lower:
        return """🔍 **Key Fraud Patterns Identified:**

Based on the analysis, the most significant fraud indicators include:

1. **Amount Patterns:** Unusual transaction amounts (very high or very low)
2. **Temporal Patterns:** Transactions at unusual hours or rapid sequences
3. **Feature Correlations:** Specific combinations of PCA features that indicate fraud
4. **Statistical Outliers:** Transactions that deviate significantly from normal patterns

The correlation heatmap and feature importance charts provide visual evidence of these patterns."""

    elif "recommend" in question_lower or "prevent" in question_lower:
        return """💡 **Fraud Prevention Recommendations:**

1. **Real-time Monitoring:** Implement alerts for transactions matching identified patterns
2. **Risk Scoring:** Use the feature importance rankings to create risk scores
3. **Threshold Adjustment:** Set appropriate thresholds based on business tolerance
4. **Continuous Learning:** Regular model updates with new fraud examples

The generated visualizations can guide the implementation of these prevention measures."""

    elif "performance" in question_lower or "model" in question_lower:
        return """⚙️ **Model Performance Analysis:**

The fraud detection analysis evaluated multiple aspects:

- **Classification Accuracy:** Model performance on labeled data (if available)
- **Feature Importance:** Ranking of most predictive features
- **Outlier Detection:** Identification of anomalous transactions
- **Pattern Recognition:** Discovery of fraud indicators

Review the generated charts for detailed performance metrics and confidence distributions."""

    else:
        return f"""🤖 **Analysis Response:**

I understand you're asking about: "{question}"

Based on your fraud detection analysis:
- **{results.get('images_found', 0)} visualizations** were generated
- **{dataset_info.get('rows', 0):,} transactions** were analyzed
- **Statistical and pattern analysis** was completed

For more specific insights, try using the quick action buttons above or ask about:
- Dataset statistics
- Fraud patterns
- Prevention recommendations
- Model performance

*Note: This is a placeholder response. Full CrewAI agent integration coming in future updates.*"""


def render_phase_3_status():
    """Render Phase 3 status in sidebar."""
    if SessionManager.is_analysis_complete():
        st.sidebar.success("✅ Chat Available")

        chat_count = len(SessionManager.get_chat_history())
        if chat_count > 0:
            st.sidebar.write(f"**Messages:** {chat_count}")
        else:
            st.sidebar.write("**Status:** Ready for questions")

    else:
        st.sidebar.info("💬 Chat Pending")
        st.sidebar.write("Complete analysis to enable chat")