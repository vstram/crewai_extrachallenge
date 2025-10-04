import streamlit as st
from typing import Any, Dict, Optional


class SessionManager:
    """Manages Streamlit session state for the fraud detection app."""

    # Session state keys
    CSV_PATH = 'csv_path'
    DATASET_CONFIGURED = 'dataset_configured'
    DATASET_INFO = 'dataset_info'
    DB_CONVERSION_RESULT = 'db_conversion_result'  # NEW: Database conversion info
    USE_DATABASE = 'use_database'  # NEW: Whether to use database mode
    ANALYSIS_RUNNING = 'analysis_running'
    ANALYSIS_COMPLETE = 'analysis_complete'
    ANALYSIS_RESULTS = 'analysis_results'
    CURRENT_AGENT = 'current_agent'
    CHAT_HISTORY = 'chat_history'
    CHAT_CONTEXT = 'chat_context'

    @staticmethod
    def initialize_session():
        """Initialize session state with default values."""
        default_values = {
            SessionManager.CSV_PATH: None,
            SessionManager.DATASET_CONFIGURED: False,
            SessionManager.DATASET_INFO: {},
            SessionManager.DB_CONVERSION_RESULT: None,
            SessionManager.USE_DATABASE: False,
            SessionManager.ANALYSIS_RUNNING: False,
            SessionManager.ANALYSIS_COMPLETE: False,
            SessionManager.ANALYSIS_RESULTS: {},
            SessionManager.CURRENT_AGENT: None,
            SessionManager.CHAT_HISTORY: [],
            SessionManager.CHAT_CONTEXT: {}
        }

        for key, default_value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @staticmethod
    def set_dataset_configured(
        csv_path: str,
        dataset_info: Dict[str, Any],
        db_conversion_result: Optional[Dict[str, Any]] = None
    ):
        """
        Mark dataset as configured with path and info.

        Args:
            csv_path: Path to CSV file
            dataset_info: Dataset information dictionary
            db_conversion_result: Database conversion result (if converted)
        """
        st.session_state[SessionManager.CSV_PATH] = csv_path
        st.session_state[SessionManager.DATASET_INFO] = dataset_info
        st.session_state[SessionManager.DB_CONVERSION_RESULT] = db_conversion_result
        st.session_state[SessionManager.USE_DATABASE] = db_conversion_result is not None
        st.session_state[SessionManager.DATASET_CONFIGURED] = True

    @staticmethod
    def clear_dataset_configuration():
        """Clear dataset configuration."""
        st.session_state[SessionManager.CSV_PATH] = None
        st.session_state[SessionManager.DATASET_INFO] = {}
        st.session_state[SessionManager.DB_CONVERSION_RESULT] = None
        st.session_state[SessionManager.USE_DATABASE] = False
        st.session_state[SessionManager.DATASET_CONFIGURED] = False

        # Also clear analysis results if dataset changes
        SessionManager.clear_analysis_results()

    @staticmethod
    def is_dataset_configured() -> bool:
        """Check if dataset is properly configured."""
        return st.session_state.get(SessionManager.DATASET_CONFIGURED, False)

    @staticmethod
    def get_csv_path() -> Optional[str]:
        """Get the configured CSV path."""
        return st.session_state.get(SessionManager.CSV_PATH)

    @staticmethod
    def get_dataset_info() -> Dict[str, Any]:
        """Get dataset information."""
        return st.session_state.get(SessionManager.DATASET_INFO, {})

    @staticmethod
    def get_db_conversion_result() -> Optional[Dict[str, Any]]:
        """Get database conversion result."""
        return st.session_state.get(SessionManager.DB_CONVERSION_RESULT)

    @staticmethod
    def is_using_database() -> bool:
        """Check if using database mode."""
        return st.session_state.get(SessionManager.USE_DATABASE, False)

    @staticmethod
    def set_analysis_running(running: bool, current_agent: str = None):
        """Set analysis running state."""
        st.session_state[SessionManager.ANALYSIS_RUNNING] = running
        st.session_state[SessionManager.CURRENT_AGENT] = current_agent

    @staticmethod
    def is_analysis_running() -> bool:
        """Check if analysis is currently running."""
        return st.session_state.get(SessionManager.ANALYSIS_RUNNING, False)

    @staticmethod
    def set_analysis_complete(results: Dict[str, Any]):
        """Mark analysis as complete with results."""
        st.session_state[SessionManager.ANALYSIS_COMPLETE] = True
        st.session_state[SessionManager.ANALYSIS_RUNNING] = False
        st.session_state[SessionManager.ANALYSIS_RESULTS] = results
        st.session_state[SessionManager.CURRENT_AGENT] = None

    @staticmethod
    def is_analysis_complete() -> bool:
        """Check if analysis is complete."""
        return st.session_state.get(SessionManager.ANALYSIS_COMPLETE, False)

    @staticmethod
    def get_analysis_results() -> Dict[str, Any]:
        """Get analysis results."""
        return st.session_state.get(SessionManager.ANALYSIS_RESULTS, {})

    @staticmethod
    def clear_analysis_results():
        """Clear analysis results."""
        st.session_state[SessionManager.ANALYSIS_COMPLETE] = False
        st.session_state[SessionManager.ANALYSIS_RESULTS] = {}
        st.session_state[SessionManager.ANALYSIS_RUNNING] = False
        st.session_state[SessionManager.CURRENT_AGENT] = None

    @staticmethod
    def add_chat_message(role: str, content: str):
        """Add message to chat history."""
        if SessionManager.CHAT_HISTORY not in st.session_state:
            st.session_state[SessionManager.CHAT_HISTORY] = []

        st.session_state[SessionManager.CHAT_HISTORY].append({
            "role": role,
            "content": content
        })

    @staticmethod
    def get_chat_history() -> list:
        """Get chat history."""
        return st.session_state.get(SessionManager.CHAT_HISTORY, [])

    @staticmethod
    def clear_chat_history():
        """Clear chat history."""
        st.session_state[SessionManager.CHAT_HISTORY] = []

    @staticmethod
    def get_current_phase() -> str:
        """Determine current phase of the application."""
        if not SessionManager.is_dataset_configured():
            return "dataset_configuration"
        elif not SessionManager.is_analysis_complete():
            return "report_generation"
        else:
            return "interactive_chat"

    @staticmethod
    def reset_session():
        """Reset entire session state."""
        keys_to_clear = [
            SessionManager.CSV_PATH,
            SessionManager.DATASET_CONFIGURED,
            SessionManager.DATASET_INFO,
            SessionManager.DB_CONVERSION_RESULT,
            SessionManager.USE_DATABASE,
            SessionManager.ANALYSIS_RUNNING,
            SessionManager.ANALYSIS_COMPLETE,
            SessionManager.ANALYSIS_RESULTS,
            SessionManager.CURRENT_AGENT,
            SessionManager.CHAT_HISTORY,
            SessionManager.CHAT_CONTEXT
        ]

        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        SessionManager.initialize_session()