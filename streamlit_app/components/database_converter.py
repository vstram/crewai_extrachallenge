"""
Database Converter Component

UI component for converting CSV files to SQLite database with progress tracking.
"""

import streamlit as st
import os
import sys
from typing import Optional, Dict, Any

# Add project root to path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.crewai_extrachallenge.utils.csv_to_sqlite import CSVToSQLiteConverter
from src.crewai_extrachallenge.config.database_config import DatabaseConfig


def get_project_root_db_path() -> str:
    """
    Get the absolute path to fraud_detection.db in project root.

    This ensures the database is created in the project root directory,
    not in streamlit_app/ where the Streamlit app runs from.

    Returns:
        Absolute path to fraud_detection.db in project root
    """
    # Navigate from streamlit_app/components/database_converter.py to project root
    current_file = os.path.abspath(__file__)
    streamlit_app_components = os.path.dirname(current_file)  # streamlit_app/components
    streamlit_app = os.path.dirname(streamlit_app_components)  # streamlit_app
    project_root = os.path.dirname(streamlit_app)  # project root

    return os.path.join(project_root, 'fraud_detection.db')


class DatabaseConverterUI:
    """UI component for database conversion with progress tracking."""

    @staticmethod
    def show_conversion_recommendation(file_info: Dict[str, Any], recommendation: str) -> bool:
        """
        Show database conversion recommendation and get user choice.

        Args:
            file_info: File information dictionary
            recommendation: Recommendation message

        Returns:
            True if user wants to convert to database, False otherwise
        """
        st.info(recommendation)

        # Show comparison
        st.write("**Performance Comparison:**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📄 CSV Mode**")
            st.write(f"- Memory: ~{file_info.get('memory_usage_mb', 0):.1f}MB")
            st.write(f"- Load time: ~{file_info.get('size_mb', 0) * 2:.1f}s")
            st.write("- Max file size: ~50MB")

        with col2:
            st.markdown("**💾 Database Mode (Recommended)**")
            st.write(f"- Memory: ~{file_info.get('memory_usage_mb', 0) * 0.05:.1f}MB (95% less)")
            st.write(f"- Load time: <1s (after conversion)")
            st.write("- Max file size: 150MB+")

        # User choice
        convert = st.checkbox(
            "✅ Convert to database for better performance",
            value=True,
            help="Recommended for files over 10MB. Conversion happens once, then database is reused."
        )

        return convert

    @staticmethod
    def convert_csv_to_database(
        csv_path: str,
        db_path: Optional[str] = None,
        table_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Convert CSV to SQLite database with progress display.

        Args:
            csv_path: Path to CSV file
            db_path: Path to database file (default: DatabaseConfig.DB_PATH)
            table_name: Table name (default: DatabaseConfig.DB_TABLE)

        Returns:
            Conversion statistics dict or None if failed
        """
        if db_path is None:
            # Ensure database is created in project root, not streamlit_app/
            db_path = get_project_root_db_path()

        if table_name is None:
            table_name = DatabaseConfig.DB_TABLE

        # Check if database already exists
        if os.path.exists(db_path):
            st.success(f"✅ Database already exists: {db_path}")
            st.info("Using existing database. Delete the database file if you want to reconvert.")
            return {
                'status': 'existing',
                'db_path': db_path,
                'table_name': table_name
            }

        # Show conversion UI
        st.write("### 🔄 Converting CSV to Database")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            converter = CSVToSQLiteConverter()

            # Progress callback
            chunks_processed = [0]
            total_rows_processed = [0]

            def progress_callback(current_chunk: int, total_rows: int):
                chunks_processed[0] = current_chunk
                total_rows_processed[0] = total_rows

                # Update progress (estimate based on chunks)
                # Since we don't know total chunks upfront, show indeterminate progress
                progress = min(0.9, (current_chunk + 1) * 0.1)  # Cap at 90%
                progress_bar.progress(progress)

                if current_chunk == 0:
                    status_text.text("Starting conversion...")
                elif current_chunk % 5 == 0:
                    status_text.text(f"Processing... {total_rows:,} rows converted")

            # Perform conversion
            result = converter.convert(
                csv_path=csv_path,
                db_path=db_path,
                table_name=table_name,
                chunk_size=DatabaseConfig.CHUNK_SIZE,
                progress_callback=progress_callback
            )

            # Update to 100%
            progress_bar.progress(1.0)
            status_text.text("Conversion complete!")

            # Show results
            st.success("✅ Database conversion successful!")

            # Map converter keys to UI expected keys
            # Converter returns: total_rows, columns, compression_pct, duration_seconds
            # UI expects: row_count, column_count, compression_ratio, conversion_time
            row_count = result.get('total_rows', result.get('row_count', 0))
            column_count = result.get('columns', result.get('column_count', 0))
            db_size_mb = result.get('db_size_mb', 0)
            compression_pct = result.get('compression_pct', result.get('compression_ratio', 0))
            conversion_time = result.get('duration_seconds', result.get('conversion_time', 0))

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📊 Rows", f"{row_count:,}")

            with col2:
                st.metric("📋 Columns", column_count)

            with col3:
                st.metric("💾 Size", f"{db_size_mb:.2f}MB")

            with col4:
                # compression_pct is already a percentage (e.g., 40.5), convert to ratio for display
                compression_ratio = compression_pct / 100.0 if compression_pct > 1 else compression_pct
                st.metric("⚡ Compression", f"{compression_ratio:.1%}")

            st.info(f"⏱️ Conversion time: {conversion_time:.2f}s")

            # Normalize result keys for consistency
            result['status'] = 'converted'
            result['db_path'] = db_path
            result['table_name'] = table_name
            result['row_count'] = row_count
            result['column_count'] = column_count
            result['compression_ratio'] = compression_ratio
            result['conversion_time'] = conversion_time

            return result

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Conversion failed: {str(e)}")
            st.warning("Falling back to CSV mode.")
            return None

    @staticmethod
    def show_database_info(db_path: Optional[str] = None, table_name: str = 'transactions') -> None:
        """
        Display information about existing database.

        Args:
            db_path: Path to database file (default: project_root/fraud_detection.db)
            table_name: Table name to inspect
        """
        if db_path is None:
            # Use project root database
            db_path = get_project_root_db_path()

        if not os.path.exists(db_path):
            st.warning(f"Database not found: {db_path}")
            return

        try:
            import sqlite3
            import pandas as pd

            conn = sqlite3.connect(db_path)

            # Get table info
            query = f"SELECT COUNT(*) as count FROM {table_name}"
            result = pd.read_sql(query, conn)
            row_count = result.iloc[0]['count']

            # Get column count
            query = f"PRAGMA table_info({table_name})"
            columns = pd.read_sql(query, conn)
            column_count = len(columns)

            # Get database size
            db_size_mb = os.path.getsize(db_path) / (1024 * 1024)

            conn.close()

            # Display info
            st.write("### 💾 Database Information")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📊 Rows", f"{row_count:,}")

            with col2:
                st.metric("📋 Columns", column_count)

            with col3:
                st.metric("💾 Size", f"{db_size_mb:.2f}MB")

            st.info(f"📍 Location: `{db_path}`")

        except Exception as e:
            st.error(f"Error reading database: {str(e)}")

    @staticmethod
    def get_conversion_status() -> Dict[str, Any]:
        """
        Check database conversion status.

        Returns:
            {
                'db_exists': bool,
                'db_path': str,
                'db_ready': bool
            }
        """
        # Ensure we check in project root
        db_path = get_project_root_db_path()
        db_exists = os.path.exists(db_path)

        return {
            'db_exists': db_exists,
            'db_path': db_path,
            'db_ready': DatabaseConfig.USE_DATABASE and db_exists
        }


# Helper function for easy import
def show_database_converter(
    csv_path: str,
    file_info: Dict[str, Any],
    recommendation: str
) -> Optional[Dict[str, Any]]:
    """
    Show database converter UI and perform conversion if requested.

    Args:
        csv_path: Path to CSV file
        file_info: File information dictionary
        recommendation: Recommendation message

    Returns:
        Conversion result dict or None
    """
    converter = DatabaseConverterUI()

    # Show recommendation and get user choice
    should_convert = converter.show_conversion_recommendation(file_info, recommendation)

    if should_convert:
        # Perform conversion
        return converter.convert_csv_to_database(csv_path)
    else:
        st.info("Continuing with CSV mode (not recommended for large files).")
        return None
