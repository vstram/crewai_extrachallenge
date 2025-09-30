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

    st.sidebar.markdown("---")

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
    """Render Phase 2: Report Generation (placeholder for now)."""
    st.header("Phase 2: Report Generation")
    st.info("🚧 Phase 2 implementation coming soon...")

    dataset_info = SessionManager.get_dataset_info()
    csv_path = SessionManager.get_csv_path()

    st.write("**Ready for analysis:**")
    st.write(f"- Dataset: {os.path.basename(csv_path)}")
    st.write(f"- Rows: {dataset_info.get('rows', 0):,}")
    st.write(f"- Columns: {dataset_info.get('columns', 0)}")

    # Placeholder button for Phase 2
    if st.button("🚀 Generate Fraud Detection Report (Coming Soon)", disabled=True):
        st.info("Phase 2 implementation in progress...")


def render_phase_3():
    """Render Phase 3: Interactive Chat (placeholder for now)."""
    st.header("Phase 3: Interactive Chat")
    st.info("🚧 Phase 3 implementation coming soon...")

    st.write("**Analysis Complete!**")
    st.write("Interactive chat interface will be available here to ask questions about your fraud detection results.")

    # Placeholder for chat interface
    st.text_input("Ask a question about your results...", disabled=True, placeholder="Coming soon...")


if __name__ == "__main__":
    main()