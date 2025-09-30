import streamlit as st
import os
import sys
from typing import Optional

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from utils.file_validator import CSVValidator, display_file_info, display_preview_data
from utils.session_manager import SessionManager


def render_dataset_configuration() -> bool:
    """
    Render the dataset configuration component.

    Returns:
        bool: True if dataset is successfully configured, False otherwise
    """
    st.header("📊 Dataset Configuration")
    st.write("Configure your credit card transaction dataset to begin fraud detection analysis.")

    # Input method selection
    input_method = st.radio(
        "Choose how to provide your dataset:",
        ["📁 File Path", "📤 File Upload"],
        help="Select either a file path to an existing CSV or upload a new file"
    )

    csv_path = None

    if input_method == "📁 File Path":
        csv_path = _render_file_path_input()
    else:
        csv_path = _render_file_upload_input()

    # Validate and process the file
    if csv_path:
        return _validate_and_configure_dataset(csv_path)

    return False


def _render_file_path_input() -> Optional[str]:
    """Render file path input interface."""
    st.subheader("📁 File Path Input")

    # Default path suggestion
    default_path = "data/credit_card_transactions.csv"

    csv_path = st.text_input(
        "CSV File Path:",
        value=default_path,
        help="Enter the full path to your credit card transaction CSV file",
        placeholder="/path/to/your/transactions.csv"
    )

    if csv_path:
        # Show file existence check
        if os.path.exists(csv_path):
            st.success(f"✅ File found: {csv_path}")
            return csv_path
        else:
            st.error(f"❌ File not found: {csv_path}")
            st.info("💡 Make sure the file path is correct and the file exists.")

    return None


def _render_file_upload_input() -> Optional[str]:
    """Render file upload interface."""
    st.subheader("📤 File Upload")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type="csv",
        help="Upload your credit card transaction dataset (CSV format only)",
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        # Show file info
        st.info(f"📁 Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

        # Save the uploaded file
        saved_path = CSVValidator.save_uploaded_file(uploaded_file)
        if saved_path:
            st.success("✅ File uploaded successfully")
            return saved_path
        else:
            st.error("❌ Failed to save uploaded file")

    return None


def _validate_and_configure_dataset(csv_path: str) -> bool:
    """Validate the dataset and configure session if valid."""
    st.subheader("🔍 Dataset Validation")

    # Create validation container
    validation_container = st.container()

    with validation_container:
        # Show validation progress
        with st.spinner("Validating dataset..."):
            is_valid, message, file_info = CSVValidator.validate_csv_file(csv_path)

        # Display validation result
        if is_valid:
            st.success(message)

            # Display file information
            st.subheader("📋 Dataset Information")
            display_file_info(file_info)

            # Display data preview
            st.subheader("👀 Data Preview")
            preview_df = CSVValidator.get_preview_data(csv_path)
            if preview_df is not None:
                display_preview_data(preview_df)

                # Configuration confirmation
                st.subheader("✅ Configuration")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("🚀 Use This Dataset", type="primary", use_container_width=True):
                        # Configure session
                        SessionManager.set_dataset_configured(csv_path, file_info)
                        st.success("✅ Dataset configured successfully!")
                        st.balloons()
                        return True

                with col2:
                    if st.button("🔄 Choose Different Dataset", use_container_width=True):
                        SessionManager.clear_dataset_configuration()
                        st.rerun()

            else:
                st.error("❌ Failed to preview dataset")

        else:
            st.error(message)
            _display_validation_help()

    return False


def _display_validation_help():
    """Display help information for dataset validation issues."""
    st.subheader("💡 Need Help?")

    with st.expander("📋 Dataset Requirements", expanded=True):
        st.markdown("""
        Your CSV file must meet these requirements:

        **Required Columns:**
        - `Time`: Transaction time (numeric)
        - `Amount`: Transaction amount (numeric)

        **Optional Columns:**
        - `Class`: Fraud label (0 = legitimate, 1 = fraud)
        - `V1`, `V2`, ..., `V28`: PCA-transformed features (numeric)

        **File Requirements:**
        - Valid CSV format
        - At least 10 rows of data
        - No completely empty columns
        - File must be readable

        **Example CSV structure:**
        ```
        Time,V1,V2,V3,Amount,Class
        0,-1.359,0.876,1.548,149.62,0
        406,1.191,-0.329,-1.468,2.69,0
        ...
        ```
        """)

    with st.expander("🔧 Common Issues & Solutions"):
        st.markdown("""
        **File not found:**
        - Check the file path is correct
        - Ensure the file exists in the specified location
        - Use absolute paths if relative paths don't work

        **Missing required columns:**
        - Ensure your CSV has `Time` and `Amount` columns
        - Column names are case-sensitive
        - Check for extra spaces in column names

        **Invalid CSV format:**
        - Open the file in a text editor to check format
        - Ensure proper comma separation
        - Check for corrupted or truncated files

        **File access issues:**
        - Ensure the file has read permissions
        - Close the file in other applications
        - Try copying the file to a different location
        """)


def render_dataset_status():
    """Render current dataset status in sidebar."""
    if SessionManager.is_dataset_configured():
        st.sidebar.success("✅ Dataset Configured")

        dataset_info = SessionManager.get_dataset_info()
        csv_path = SessionManager.get_csv_path()

        with st.sidebar.expander("📊 Dataset Details", expanded=False):
            st.write(f"**File:** {os.path.basename(csv_path)}")
            st.write(f"**Rows:** {dataset_info.get('rows', 'N/A'):,}")
            st.write(f"**Columns:** {dataset_info.get('columns', 'N/A')}")
            st.write(f"**Size:** {dataset_info.get('size_mb', 'N/A')} MB")

            if dataset_info.get('has_class_column', False):
                st.write("**Type:** Supervised (with labels)")
            else:
                st.write("**Type:** Unsupervised (no labels)")

        if st.sidebar.button("🔄 Change Dataset"):
            SessionManager.clear_dataset_configuration()
            st.rerun()
    else:
        st.sidebar.warning("⚠️ Dataset Not Configured")
        st.sidebar.write("Please configure your dataset first.")