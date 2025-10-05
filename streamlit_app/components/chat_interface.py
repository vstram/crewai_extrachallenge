import streamlit as st
import os
import sys
from typing import List, Dict, Any

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from utils.session_manager import SessionManager
from utils.chat_agent import ChatAnalystAgent, QuickResponseHandler


def _get_chat_handler() -> QuickResponseHandler:
    """Get or create the chat handler for this session."""
    if 'chat_handler' not in st.session_state:
        # Initialize chat handler with current analysis results
        dataset_path = SessionManager.get_csv_path()
        analysis_results = SessionManager.get_analysis_results()
        dataset_info = SessionManager.get_dataset_info()

        # Include dataset info in analysis results for context
        if analysis_results and dataset_info:
            analysis_results['dataset_info'] = dataset_info

        st.session_state.chat_handler = QuickResponseHandler(dataset_path, analysis_results or {})

    return st.session_state.chat_handler


def render_chat_interface() -> None:
    """Render the interactive chat interface (Phase 3)."""
    st.header("💬 Phase 3: AI-Powered Interactive Chat")

    # Check if analysis is complete
    if not SessionManager.is_analysis_complete():
        st.warning("⚠️ Complete the fraud detection analysis first to enable chat.")
        return

    st.write("🤖 **Ask questions about your fraud detection results and get detailed insights from AI agents.**")

    # Show enhanced features
    st.info("""
    ✨ **Enhanced AI Features:**
    • Context-aware responses based on your specific dataset
    • Real-time analysis using CrewAI fraud detection agents
    • Access to CSV data exploration and statistical analysis
    • **On-demand visualization generation** - Ask for charts and plots!
    • Actionable insights and recommendations
    """)

    # Quick action buttons
    _render_quick_actions()

    # Chat history display
    _render_chat_history()

    # Chat input
    _render_chat_input()


def _render_quick_actions() -> None:
    """Render quick action buttons for common questions."""
    st.subheader("🚀 Quick Questions")
    st.write("Click any button below to get detailed AI-powered insights:")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Show Statistics", use_container_width=True, help="Get comprehensive dataset statistics and insights"):
            _ask_quick_question("statistics")

    with col2:
        if st.button("🔍 Explain Patterns", use_container_width=True, help="Discover fraud patterns and indicators"):
            _ask_quick_question("patterns")

    with col3:
        if st.button("💡 Recommendations", use_container_width=True, help="Get fraud prevention recommendations"):
            _ask_quick_question("recommendations")

    # Additional quick actions
    col4, col5, col6 = st.columns(3)

    with col4:
        if st.button("📈 Risk Assessment", use_container_width=True, help="Learn about risk factors and scoring"):
            _ask_quick_question("risk")

    with col5:
        if st.button("🎯 Feature Analysis", use_container_width=True, help="Understand feature importance and contributions"):
            _ask_quick_question("features")

    with col6:
        if st.button("📉 Create Visualization", use_container_width=True, help="Generate custom charts and plots"):
            _ask_quick_question("visualization")


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
            # Render markdown content
            content = message["content"]
            st.markdown(content)

            # Detect and render embedded images
            _render_embedded_images(content)

    # Clear history button
    if len(chat_history) > 0:
        if st.button("🗑️ Clear Chat History", help="Clear all chat messages"):
            SessionManager.clear_chat_history()
            st.rerun()


def _render_embedded_images(content: str) -> None:
    """Detect and render images embedded in chat responses."""
    import re

    # Pattern to match markdown images: ![alt text](path)
    image_pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
    matches = re.findall(image_pattern, content)

    if matches:
        for alt_text, image_path in matches:
            # Convert relative path to absolute path
            if image_path.startswith('./images/'):
                # Path relative to reports directory
                abs_path = os.path.join(os.getcwd(), 'reports', 'images', image_path.replace('./images/', ''))
            elif image_path.startswith('reports/images/'):
                # Path from project root
                abs_path = os.path.join(os.getcwd(), image_path)
            else:
                # Assume it's just the filename
                abs_path = os.path.join(os.getcwd(), 'reports', 'images', image_path)

            # Display image if it exists
            if os.path.exists(abs_path):
                st.image(abs_path, caption=alt_text if alt_text else "Generated Visualization", use_container_width=True)


def _render_chat_input() -> None:
    """Render chat input interface."""
    # Chat input
    if prompt := st.chat_input("Ask a question about your fraud detection results..."):
        _ask_question(prompt)


def _ask_question(question: str) -> None:
    """Process a user question and generate response using CrewAI agents."""
    # Add user message to history
    SessionManager.add_chat_message("user", question)

    # Show loading message
    with st.spinner("🤖 AI Agent is analyzing your question..."):
        try:
            # Get chat handler and generate response using CrewAI agents
            chat_handler = _get_chat_handler()
            response = chat_handler.chat_agent.ask_question(question)
        except Exception as e:
            # Fallback to placeholder response if agent fails
            st.error(f"Agent temporarily unavailable: {str(e)}")
            response = _generate_fallback_response(question)

    # Add assistant response to history
    SessionManager.add_chat_message("assistant", response)

    # Rerun to update chat display
    st.rerun()


def _ask_quick_question(question_type: str) -> None:
    """Process a quick action question using specialized handlers."""
    # Map question types to user-friendly messages
    question_map = {
        "statistics": "📊 Show me comprehensive statistics about the dataset",
        "patterns": "🔍 Explain the most significant fraud patterns you found",
        "recommendations": "💡 What are your top recommendations for fraud prevention?",
        "risk": "📈 How should we assess risk for different transaction types?",
        "features": "🎯 Which features are most important for fraud detection?",
        "performance": "⚙️ How well did the classification model perform?",
        "visualization": "📉 Create a custom visualization showing the relationship between transaction amounts and fraud probability"
    }

    question = question_map.get(question_type, question_type)

    # Add user message to history
    SessionManager.add_chat_message("user", question)

    # Show loading message
    with st.spinner("🤖 AI Agent is analyzing your question..."):
        try:
            # Get chat handler and use specialized methods
            chat_handler = _get_chat_handler()

            if question_type == "statistics":
                response = chat_handler.get_statistics_overview()
            elif question_type == "patterns":
                response = chat_handler.explain_fraud_patterns()
            elif question_type == "recommendations":
                response = chat_handler.get_prevention_recommendations()
            elif question_type == "risk":
                response = chat_handler.assess_risk_factors()
            elif question_type == "features":
                response = chat_handler.analyze_feature_importance()
            elif question_type == "performance":
                response = chat_handler.evaluate_model_performance()
            elif question_type == "visualization":
                response = chat_handler.create_custom_visualization()
            else:
                response = chat_handler.chat_agent.ask_question(question)

        except Exception as e:
            # Fallback to placeholder response if agent fails
            st.error(f"Agent temporarily unavailable: {str(e)}")
            response = _generate_fallback_response(question)

    # Add assistant response to history
    SessionManager.add_chat_message("assistant", response)

    # Rerun to update chat display
    st.rerun()


def _generate_fallback_response(question: str) -> str:
    """Generate fallback response when CrewAI agents are unavailable."""
    # Get analysis results for context
    results = SessionManager.get_analysis_results()
    dataset_info = SessionManager.get_dataset_info()

    return f"""⚠️ **AI Agent Temporarily Unavailable**

I'm experiencing temporary issues connecting to the AI analysis agents. Here's basic information about your analysis:

**Analysis Summary:**
- **Dataset:** {dataset_info.get('rows', 0):,} transactions analyzed
- **Features:** {dataset_info.get('columns', 0)} columns processed
- **Visualizations:** {results.get('images_found', 0)} charts generated
- **Analysis Type:** {'Supervised Learning' if dataset_info.get('has_class_column') else 'Unsupervised Pattern Detection'}

**Your Question:** "{question}"

**Suggested Actions:**
1. **Try again** - The agents may be available shortly
2. **Use quick action buttons** - These provide structured queries
3. **Review the generated report** - Check Phase 2 results for detailed insights
4. **Check connectivity** - Ensure CrewAI environment is properly configured

*The AI agents provide much more detailed and context-aware responses when available.*"""


def render_phase_3_status():
    """Render Phase 3 status in sidebar."""
    if SessionManager.is_analysis_complete():
        st.sidebar.success("✅ AI Chat Available")

        chat_count = len(SessionManager.get_chat_history())
        if chat_count > 0:
            st.sidebar.write(f"**Messages:** {chat_count}")
        else:
            st.sidebar.write("**Status:** AI agents ready")

        # Show AI capabilities
        with st.sidebar.expander("🤖 AI Features", expanded=False):
            st.write("**Available AI Agents:**")
            st.write("• Fraud Detection Analyst")
            st.write("• CSV Data Explorer")
            st.write("• Statistical Analyzer")
            st.write("• Visualization Generator")
            st.write("")
            st.write("**Capabilities:**")
            st.write("• Context-aware responses")
            st.write("• Real-time data queries")
            st.write("• Expert fraud insights")
            st.write("• Custom chart generation")
            st.write("")
            st.write("**Try asking:**")
            st.write("• 'Show me a scatter plot'")
            st.write("• 'Create a histogram'")
            st.write("• 'Generate a heatmap'")

    else:
        st.sidebar.info("🤖 AI Chat Pending")
        st.sidebar.write("Complete analysis to enable AI chat")