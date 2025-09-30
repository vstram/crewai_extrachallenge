import pandas as pd
import os
import streamlit as st
from typing import Tuple, Optional, Dict, Any


class CSVValidator:
    """Utility class for validating CSV files for fraud detection analysis."""

    REQUIRED_COLUMNS = ['Time', 'Amount']  # Class column is optional
    OPTIONAL_COLUMNS = ['Class']
    MIN_ROWS = 10
    MAX_PREVIEW_ROWS = 10

    @staticmethod
    def validate_csv_file(file_path: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Validate CSV file for fraud detection analysis.

        Returns:
            Tuple[bool, str, Optional[Dict]]: (is_valid, message, file_info)
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return False, f"❌ File not found: {file_path}", None

            # Check if file is readable
            if not os.access(file_path, os.R_OK):
                return False, f"❌ File not readable: {file_path}", None

            # Try to read the CSV
            try:
                df = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                return False, "❌ File is empty", None
            except pd.errors.ParserError as e:
                return False, f"❌ Invalid CSV format: {str(e)}", None
            except Exception as e:
                return False, f"❌ Error reading file: {str(e)}", None

            # Check minimum rows
            if len(df) < CSVValidator.MIN_ROWS:
                return False, f"❌ File must have at least {CSVValidator.MIN_ROWS} rows. Found: {len(df)}", None

            # Check for required columns
            missing_required = [col for col in CSVValidator.REQUIRED_COLUMNS if col not in df.columns]
            if missing_required:
                return False, f"❌ Missing required columns: {', '.join(missing_required)}", None

            # Check for completely empty columns
            empty_cols = [col for col in df.columns if df[col].isna().all()]
            if empty_cols:
                return False, f"❌ Found completely empty columns: {', '.join(empty_cols)}", None

            # Generate file info
            file_info = {
                'rows': len(df),
                'columns': len(df.columns),
                'size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
                'has_class_column': 'Class' in df.columns,
                'column_names': list(df.columns),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            }

            return True, "✅ File validation successful", file_info

        except Exception as e:
            return False, f"❌ Unexpected error: {str(e)}", None

    @staticmethod
    def get_preview_data(file_path: str) -> Optional[pd.DataFrame]:
        """Get preview of first 10 rows of CSV file."""
        try:
            return pd.read_csv(file_path, nrows=CSVValidator.MAX_PREVIEW_ROWS)
        except Exception:
            return None

    @staticmethod
    def save_uploaded_file(uploaded_file) -> Optional[str]:
        """Save uploaded file to temporary location and return path."""
        if uploaded_file is None:
            return None

        try:
            # Create temp directory if it doesn't exist
            temp_dir = "streamlit_app/temp"
            os.makedirs(temp_dir, exist_ok=True)

            # Save file
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            return file_path
        except Exception as e:
            st.error(f"Error saving uploaded file: {str(e)}")
            return None


def display_file_info(file_info: Dict[str, Any]) -> None:
    """Display file information in Streamlit."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Rows", f"{file_info['rows']:,}")

    with col2:
        st.metric("📋 Columns", file_info['columns'])

    with col3:
        st.metric("💾 File Size", f"{file_info['size_mb']} MB")

    with col4:
        st.metric("🧠 Memory Usage", f"{file_info['memory_usage_mb']} MB")

    # Additional info
    st.write("**Column Information:**")
    col_info = st.columns(2)

    with col_info[0]:
        st.write("**Available Columns:**")
        for col in file_info['column_names']:
            if col in CSVValidator.REQUIRED_COLUMNS:
                st.write(f"✅ {col} (required)")
            elif col in CSVValidator.OPTIONAL_COLUMNS:
                st.write(f"🔵 {col} (optional)")
            else:
                st.write(f"ℹ️ {col}")

    with col_info[1]:
        st.write("**Dataset Status:**")
        if file_info['has_class_column']:
            st.write("✅ Class column found (supervised learning available)")
        else:
            st.write("ℹ️ No Class column (unsupervised analysis only)")


def display_preview_data(df: pd.DataFrame) -> None:
    """Display preview of dataset."""
    st.write("**Data Preview (First 10 rows):**")
    st.dataframe(df, use_container_width=True)

    # Show basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        st.write("**Quick Statistics:**")
        stats_df = df[numeric_cols].describe().round(2)
        st.dataframe(stats_df, use_container_width=True)