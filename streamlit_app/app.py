import streamlit as st
import sys
import os

# Add the current directory and parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Import components and utilities
from components.dataset_config import render_dataset_configuration, render_dataset_status
from components.report_generator import render_report_generation, render_phase_2_status
from components.chat_interface import render_chat_interface, render_phase_3_status
from utils.session_manager import SessionManager


def main():
    """Main Streamlit application."""
    # Configure page
    st.set_page_config(
        page_title="CrewAI Fraud Detection",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session
    SessionManager.initialize_session()

    # Main header
    st.title("🔍 CrewAI Fraud Detection System")
    st.markdown("---")

    # Sidebar navigation
    render_sidebar()

    # Main content based on current phase
    current_phase = SessionManager.get_current_phase()

    if current_phase == "dataset_configuration":
        render_phase_1()
    elif current_phase == "report_generation":
        render_phase_2()
    elif current_phase == "interactive_chat":
        render_phase_3()


def render_sidebar():
    """Render sidebar with navigation and status."""
    st.sidebar.title("🔍 Fraud Detection")

    # Phase indicator
    current_phase = SessionManager.get_current_phase()
    phases = {
        "dataset_configuration": "📊 Phase 1: Dataset Configuration",
        "report_generation": "📈 Phase 2: Report Generation",
        "interactive_chat": "💬 Phase 3: Interactive Chat"
    }

    st.sidebar.subheader("Current Phase:")
    st.sidebar.info(phases[current_phase])

    # Phase status
    st.sidebar.subheader("Progress:")

    # Phase 1 status
    if SessionManager.is_dataset_configured():
        st.sidebar.write("✅ Dataset Configured")
    else:
        st.sidebar.write("🔄 Dataset Configuration")

    # Phase 2 status
    if SessionManager.is_analysis_complete():
        st.sidebar.write("✅ Report Generated")
    elif SessionManager.is_analysis_running():
        st.sidebar.write("🔄 Generating Report...")
    else:
        st.sidebar.write("⏳ Report Generation")

    # Phase 3 status
    if SessionManager.is_analysis_complete():
        st.sidebar.write("✅ Chat Available")
    else:
        st.sidebar.write("⏳ Chat Interface")

    st.sidebar.markdown("---")

    # Dataset status
    render_dataset_status()

    # Phase 2 status
    render_phase_2_status()

    # Phase 3 status
    render_phase_3_status()

    st.sidebar.markdown("---")

    # LLM Configuration Info
    with st.sidebar.expander("🤖 LLM Configuration", expanded=False):
        model = os.getenv('MODEL', 'gpt-4-turbo-preview')

        st.markdown(f"""
        **Current Model:**
        - Model: `{model}`

        **Temperature Settings:**
        - Data Analyst: {os.getenv('TEMP_DATA_ANALYST', '0.1')}
        - Pattern Agent: {os.getenv('TEMP_PATTERN_AGENT', '0.3')}
        - Classification: {os.getenv('TEMP_CLASSIFICATION', '0.1')}
        - Reporting: {os.getenv('TEMP_REPORTING', '0.2')}
        """)

    # Reset button
    if st.sidebar.button("🔄 Reset All", help="Clear all data and start over"):
        SessionManager.reset_session()
        st.rerun()

    # Help section
    with st.sidebar.expander("❓ Help & Info"):
        st.markdown("""
        **How to use this app:**

        1. **📊 Configure Dataset:** Upload or specify path to your credit card transaction CSV
        2. **📈 Generate Report:** Run fraud detection analysis
        3. **💬 Ask Questions:** Interactive chat about your results

        **Need help?** Check the validation requirements in Phase 1.
        """)


def render_phase_1():
    """Render Phase 1: Dataset Configuration."""
    st.header("Phase 1: Dataset Configuration")
    st.write("Start by configuring your credit card transaction dataset.")

    # Show next steps info
    if not SessionManager.is_dataset_configured():
        st.info("👆 Configure your dataset above to proceed to report generation.")

    # Render dataset configuration
    dataset_configured = render_dataset_configuration()

    # Show success message and next steps
    if dataset_configured:
        st.success("🎉 Dataset configured successfully!")

        st.subheader("🚀 Next Steps")
        st.write("Your dataset is ready for analysis. You can now proceed to generate the fraud detection report.")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Dataset Status", "✅ Ready")
        with col2:
            if st.button("➡️ Proceed to Report Generation", type="primary"):
                st.info("Click 'Generate Report' in Phase 2 to start the analysis!")


def render_phase_2():
    """Render Phase 2: Report Generation."""
    render_report_generation()


def render_phase_3():
    """Render Phase 3: Interactive Chat."""
    render_chat_interface()


if __name__ == "__main__":
    main()