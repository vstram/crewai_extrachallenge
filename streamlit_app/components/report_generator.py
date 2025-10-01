import streamlit as st
import os
import sys
import time
from typing import Dict, Any

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from utils.session_manager import SessionManager
from utils.crew_runner import StreamlitCrewRunner, validate_crew_environment, estimate_analysis_time


def render_report_generation() -> None:
    """Render the report generation component (Phase 2)."""
    st.header("📈 Phase 2: Report Generation")
    st.write("Generate a comprehensive fraud detection analysis using CrewAI agents.")

    # Check if dataset is configured
    if not SessionManager.is_dataset_configured():
        st.error("❌ No dataset configured. Please complete Phase 1 first.")
        return

    # Get dataset info
    csv_path = SessionManager.get_csv_path()
    dataset_info = SessionManager.get_dataset_info()

    # Display dataset summary
    _display_dataset_summary(csv_path, dataset_info)

    # Check CrewAI environment
    crew_ready, crew_message = validate_crew_environment()
    if not crew_ready:
        st.error(crew_message)
        _display_environment_help()
        return

    st.success(crew_message)

    # Analysis controls
    _render_analysis_controls(csv_path, dataset_info)

    # Show progress if analysis is running
    if SessionManager.is_analysis_running():
        _render_analysis_progress()

    # Display results if analysis is complete
    if SessionManager.is_analysis_complete():
        _render_analysis_results()


def _display_dataset_summary(csv_path: str, dataset_info: Dict[str, Any]) -> None:
    """Display summary of configured dataset."""
    st.subheader("📊 Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📁 File", os.path.basename(csv_path))

    with col2:
        st.metric("📊 Rows", f"{dataset_info.get('rows', 0):,}")

    with col3:
        st.metric("📋 Columns", dataset_info.get('columns', 0))

    with col4:
        st.metric("⏱️ Est. Time", estimate_analysis_time(dataset_info))

    # Additional dataset info
    with st.expander("📋 Dataset Details", expanded=False):
        st.write(f"**Full Path:** `{csv_path}`")
        st.write(f"**File Size:** {dataset_info.get('size_mb', 0)} MB")
        st.write(f"**Memory Usage:** {dataset_info.get('memory_usage_mb', 0)} MB")

        if dataset_info.get('has_class_column', False):
            st.write("**Analysis Type:** Supervised (with fraud labels)")
        else:
            st.write("**Analysis Type:** Unsupervised (pattern detection only)")

        st.write("**Available Columns:**")
        for col in dataset_info.get('column_names', []):
            st.write(f"  • {col}")


def _render_analysis_controls(csv_path: str, dataset_info: Dict[str, Any]) -> None:
    """Render analysis control buttons and options."""
    st.subheader("🚀 Analysis Controls")

    # Show analysis options
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Analysis will include:**")
        st.write("✅ Statistical analysis and data quality assessment")
        st.write("✅ Pattern recognition and correlation analysis")
        st.write("✅ Classification analysis and outlier detection")
        st.write("✅ Visualization generation (6+ charts)")
        st.write("✅ Comprehensive fraud detection report")

    with col2:
        st.write("**Generated outputs:**")
        st.write("📄 Detailed markdown report")
        st.write("📊 Correlation heatmap")
        st.write("📈 Fraud comparison charts")
        st.write("🔍 Feature importance plots")
        st.write("📉 Distribution analysis charts")

    # Main generation button
    st.markdown("---")

    if not SessionManager.is_analysis_running():
        if st.button("🚀 Generate Fraud Detection Report",
                    type="primary",
                    use_container_width=True,
                    help="Start the complete fraud detection analysis"):
            _start_analysis(csv_path)
    else:
        st.warning("⏳ Analysis is currently running. Please wait for completion.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏹️ Stop Analysis", help="Stop the current analysis"):
                _stop_analysis()

        with col2:
            if st.button("🔄 Refresh Status", help="Refresh the current status"):
                st.rerun()


def _start_analysis(csv_path: str) -> None:
    """Start the CrewAI analysis process."""
    try:
        # Mark analysis as started
        SessionManager.set_analysis_running(True, "Initializing")

        # Create crew runner
        crew_runner = StreamlitCrewRunner(csv_path)

        # Store runner in session state for progress tracking
        st.session_state.crew_runner = crew_runner

        # Start analysis in background thread
        def progress_callback(progress):
            st.session_state.analysis_progress = progress

        def status_callback(status):
            st.session_state.analysis_status = status

        # Run analysis
        thread = crew_runner.run_analysis_async(progress_callback, status_callback)
        st.session_state.analysis_thread = thread

        st.success("✅ Analysis started! Refresh the page to see progress.")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Failed to start analysis: {str(e)}")
        SessionManager.set_analysis_running(False)


def _stop_analysis() -> None:
    """Stop the current analysis."""
    SessionManager.set_analysis_running(False)
    if 'crew_runner' in st.session_state:
        del st.session_state.crew_runner
    if 'analysis_thread' in st.session_state:
        del st.session_state.analysis_thread

    st.info("⏹️ Analysis stopped.")
    st.rerun()


def _render_analysis_progress() -> None:
    """Render analysis progress indicators."""
    st.subheader("⏳ Analysis in Progress")

    # Check if we have progress information
    crew_runner = st.session_state.get('crew_runner')
    if crew_runner:
        status_info = crew_runner.get_status()

        # Progress bar
        progress = status_info.get('progress', 0)
        st.progress(progress / 100, text=f"Progress: {progress}%")

        # Current status
        status = status_info.get('status', 'Running...')
        current_agent = status_info.get('current_agent')

        if current_agent:
            st.info(f"🤖 Current Agent: **{current_agent}**")

        st.write(f"**Status:** {status}")

        # Check if analysis is complete
        if not status_info.get('is_running', True) and progress == 100:
            results = status_info.get('results', {})
            SessionManager.set_analysis_complete(results)
            st.rerun()

        # Check for errors
        error = status_info.get('error')
        if error:
            st.error(f"❌ Analysis failed: {error}")
            SessionManager.set_analysis_running(False)

    else:
        # Fallback progress display
        st.progress(0.5, text="Analysis running...")
        st.info("🤖 CrewAI agents are analyzing your dataset...")

    # Refresh button
    if st.button("🔄 Refresh Progress"):
        st.rerun()

    # Estimated time remaining
    st.write("⏱️ **Estimated total time:** 3-8 minutes depending on dataset size")


def _render_analysis_results() -> None:
    """Render completed analysis results."""
    st.subheader("🎉 Analysis Complete!")

    results = SessionManager.get_analysis_results()

    # Success message
    st.success("✅ Fraud detection analysis completed successfully!")

    # Results summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📄 Report", "Generated" if results.get('report_path') else "Missing")

    with col2:
        images_count = results.get('images_found', 0)
        st.metric("📊 Images", f"{images_count} charts")

    with col3:
        execution_time = results.get('execution_time', 0)
        if execution_time:
            elapsed = time.time() - execution_time
            st.metric("⏱️ Duration", f"{elapsed/60:.1f} min")

    # Display report
    _display_generated_report(results)

    # Display images
    _display_generated_images(results)

    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Run New Analysis", help="Start a new analysis with the same dataset"):
            SessionManager.clear_analysis_results()
            st.rerun()

    with col2:
        if st.button("📂 Change Dataset", help="Configure a different dataset"):
            SessionManager.clear_dataset_configuration()
            st.rerun()

    with col3:
        if st.button("💬 Continue to Chat", help="Proceed to interactive chat phase"):
            st.info("✅ Ready for interactive chat! Use the sidebar to navigate.")


def _display_generated_report(results: Dict[str, Any]) -> None:
    """Display the generated markdown report."""
    report_content = results.get('report_content')

    if report_content:
        st.subheader("📄 Generated Report")

        # Report metrics
        lines = report_content.split('\n')
        word_count = len(report_content.split())

        col1, col2 = st.columns(2)
        with col1:
            st.metric("📝 Lines", len(lines))
        with col2:
            st.metric("📖 Words", word_count)

        # Display report in expandable section
        with st.expander("📖 View Full Report", expanded=True):
            st.markdown(report_content)

        # Download button
        st.download_button(
            "⬇️ Download Report",
            data=report_content,
            file_name="fraud_detection_report.md",
            mime="text/markdown",
            help="Download the complete fraud detection report"
        )
    else:
        st.warning("⚠️ Report file not found. The analysis may have encountered an error.")


def _display_generated_images(results: Dict[str, Any]) -> None:
    """Display generated visualization images."""
    image_paths = results.get('image_paths', [])

    if image_paths:
        st.subheader(f"📊 Generated Visualizations ({len(image_paths)} charts)")

        # Display images in a grid
        cols = st.columns(2)

        for i, image_info in enumerate(image_paths):
            with cols[i % 2]:
                if os.path.exists(image_info['path']):
                    st.image(
                        image_info['path'],
                        caption=f"{image_info['filename']} ({image_info['size_kb']} KB)",
                        use_column_width=True
                    )
                else:
                    st.error(f"❌ Image not found: {image_info['filename']}")

        # Images summary
        with st.expander("🗂️ Image Details", expanded=False):
            for image_info in image_paths:
                st.write(f"**{image_info['filename']}**")
                st.write(f"  - Path: `{image_info['relative_path']}`")
                st.write(f"  - Size: {image_info['size_kb']} KB")

    else:
        st.warning("⚠️ No visualization images found. The analysis may not have completed successfully.")


def _display_environment_help() -> None:
    """Display help for setting up CrewAI environment."""
    with st.expander("🔧 Environment Setup Help", expanded=True):
        st.markdown("""
        **CrewAI requires proper environment setup:**

        **Option 1: Using Ollama (Recommended)**
        ```bash
        # Install Ollama
        curl -fsSL https://ollama.ai/install.sh | sh

        # Start Ollama service
        ollama serve

        # Pull required model
        ollama pull llama3.1:8b
        ```

        **Option 2: Using OpenAI**
        ```bash
        # Set your OpenAI API key
        export OPENAI_API_KEY="your-api-key-here"
        ```

        **Environment Variables:**
        - `DATASET_PATH`: Set automatically by the app
        - `OPENAI_API_KEY`: Required if using OpenAI (optional with Ollama)
        - `OLLAMA_BASE_URL`: Usually `http://localhost:11434` (auto-detected)

        **Troubleshooting:**
        - Ensure Ollama is running: `curl http://localhost:11434`
        - Check model availability: `ollama list`
        - Verify environment: `echo $OPENAI_API_KEY`
        """)


def render_phase_2_status():
    """Render Phase 2 status in sidebar."""
    if SessionManager.is_analysis_complete():
        st.sidebar.success("✅ Analysis Complete")

        results = SessionManager.get_analysis_results()
        images_count = results.get('images_found', 0)

        with st.sidebar.expander("📊 Analysis Results", expanded=False):
            st.write(f"**Report:** {'✅ Generated' if results.get('report_path') else '❌ Missing'}")
            st.write(f"**Images:** {images_count} charts generated")

            if results.get('execution_time'):
                elapsed = time.time() - results['execution_time']
                st.write(f"**Duration:** {elapsed/60:.1f} minutes")

    elif SessionManager.is_analysis_running():
        st.sidebar.info("⏳ Analysis Running")

        # Show progress if available
        progress = st.session_state.get('analysis_progress', 0)
        st.sidebar.progress(progress / 100)

        current_status = st.session_state.get('analysis_status', 'Processing...')
        st.sidebar.write(f"**Status:** {current_status}")

    else:
        st.sidebar.info("📈 Ready for Analysis")
        st.sidebar.write("Click 'Generate Report' to start analysis")