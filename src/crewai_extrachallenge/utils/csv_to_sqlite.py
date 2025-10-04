"""
CSV to SQLite Converter with Chunked Processing

This module provides utilities to convert large CSV files to SQLite databases
without loading the entire file into memory. Designed for fraud detection
datasets with 150MB+ file sizes.

Features:
- Chunked CSV reading (configurable chunk size)
- Progress tracking and reporting
- Automatic index creation for performance
- Compression statistics
- Data validation during conversion
"""

import pandas as pd
import sqlite3
import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from datetime import datetime


class CSVToSQLiteConverter:
    """
    Convert large CSV files to SQLite database with chunked processing.

    This converter reads CSV files in chunks to avoid memory issues with
    large datasets. It creates indexes on key columns and reports conversion
    statistics.
    """

    DEFAULT_CHUNK_SIZE = 10000

    def __init__(self):
        """Initialize the converter."""
        pass

    def convert(
        self,
        csv_path: str,
        db_path: str,
        table_name: str = 'transactions',
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Convert CSV file to SQLite database using chunked processing.

        Args:
            csv_path: Path to the CSV file to convert
            db_path: Path where the SQLite database will be created
            table_name: Name of the table to create in the database
            chunk_size: Number of rows to process at once (default: 10,000)
            progress_callback: Optional callback function(current_rows, total_rows)

        Returns:
            Dictionary with conversion statistics:
            {
                'total_rows': int,
                'duration_seconds': float,
                'db_size_mb': float,
                'csv_size_mb': float,
                'compression_pct': float,
                'table_name': str,
                'columns': int
            }

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            PermissionError: If cannot write to database path
            Exception: For other conversion errors
        """

        # Validate inputs
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Check write permissions
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        print(f"\n{'='*60}")
        print(f"CSV TO SQLITE CONVERSION")
        print(f"{'='*60}")
        print(f"Source CSV:  {csv_path}")
        print(f"Target DB:   {db_path}")
        print(f"Table:       {table_name}")
        print(f"Chunk Size:  {chunk_size:,} rows")
        print(f"{'='*60}\n")

        start_time = datetime.now()

        # Track statistics
        total_rows = 0
        column_count = 0

        # Remove existing database if it exists
        if os.path.exists(db_path):
            print(f"⚠️  Removing existing database: {db_path}")
            os.remove(db_path)

        # Create database connection
        conn = sqlite3.connect(db_path)

        try:
            # First, count total rows for progress tracking (read first chunk to get estimate)
            print("📊 Analyzing CSV file...")
            first_chunk = pd.read_csv(csv_path, nrows=1000)
            column_count = len(first_chunk.columns)

            # Estimate total rows (rough estimate based on file size)
            csv_size_bytes = os.path.getsize(csv_path)
            avg_row_size = csv_size_bytes / len(first_chunk) if len(first_chunk) > 0 else 1000
            estimated_total_rows = int(csv_size_bytes / avg_row_size)

            print(f"   Columns detected: {column_count}")
            print(f"   Estimated rows: ~{estimated_total_rows:,}\n")

            # Process CSV in chunks
            print("🔄 Converting CSV to database...")

            chunk_iterator = pd.read_csv(csv_path, chunksize=chunk_size)

            for chunk_num, chunk in enumerate(chunk_iterator):
                # First chunk: create table with schema
                if chunk_num == 0:
                    chunk.to_sql(table_name, conn, if_exists='replace', index=False)
                    print(f"   ✅ Created table '{table_name}' with {len(chunk.columns)} columns")
                    print(f"   Columns: {', '.join(chunk.columns.tolist()[:5])}{'...' if len(chunk.columns) > 5 else ''}\n")
                else:
                    # Subsequent chunks: append data
                    chunk.to_sql(table_name, conn, if_exists='append', index=False)

                total_rows += len(chunk)

                # Progress reporting
                if chunk_num % 10 == 0 or chunk_num == 0:
                    print(f"   Processed {total_rows:,} rows...")

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(total_rows, estimated_total_rows)

            print(f"\n   ✅ Total rows inserted: {total_rows:,}\n")

            # Create indexes for better query performance
            print("🔍 Creating indexes for faster queries...")
            self._create_indexes(conn, table_name)
            print("   ✅ Indexes created\n")

            # Commit changes
            conn.commit()

            # Calculate statistics
            duration = (datetime.now() - start_time).total_seconds()

            csv_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
            db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
            compression_pct = ((csv_size_mb - db_size_mb) / csv_size_mb * 100) if csv_size_mb > 0 else 0

            stats = {
                'total_rows': total_rows,
                'duration_seconds': duration,
                'db_size_mb': db_size_mb,
                'csv_size_mb': csv_size_mb,
                'compression_pct': compression_pct,
                'table_name': table_name,
                'columns': column_count
            }

            # Print summary
            print(f"{'='*60}")
            print(f"CONVERSION COMPLETE")
            print(f"{'='*60}")
            print(f"Total Rows:      {total_rows:,}")
            print(f"Columns:         {column_count}")
            print(f"Duration:        {duration:.1f} seconds")
            print(f"CSV Size:        {csv_size_mb:.1f} MB")
            print(f"Database Size:   {db_size_mb:.1f} MB")
            print(f"Compression:     {compression_pct:.1f}%")
            print(f"Rows/Second:     {int(total_rows/duration):,}")
            print(f"{'='*60}\n")

            return stats

        except Exception as e:
            print(f"\n❌ Conversion failed: {str(e)}")
            # Rollback - remove incomplete database
            if os.path.exists(db_path):
                os.remove(db_path)
            raise e

        finally:
            conn.close()

    def _create_indexes(self, conn: sqlite3.Connection, table_name: str) -> None:
        """
        Create indexes on key columns for better query performance.

        Args:
            conn: SQLite database connection
            table_name: Name of the table to index
        """
        cursor = conn.cursor()

        # Get list of columns in the table
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]

        # Create index on Class column (fraud indicator) if it exists
        if 'Class' in columns:
            try:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_class ON {table_name}(Class)")
                print(f"      ✓ Index created: Class")
            except Exception as e:
                print(f"      ⚠ Could not create Class index: {e}")

        # Create index on Amount column if it exists
        if 'Amount' in columns:
            try:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_amount ON {table_name}(Amount)")
                print(f"      ✓ Index created: Amount")
            except Exception as e:
                print(f"      ⚠ Could not create Amount index: {e}")

        # Create index on Time column if it exists
        if 'Time' in columns:
            try:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_time ON {table_name}(Time)")
                print(f"      ✓ Index created: Time")
            except Exception as e:
                print(f"      ⚠ Could not create Time index: {e}")

        conn.commit()

    def validate_database(self, db_path: str, table_name: str = 'transactions') -> bool:
        """
        Validate that the database was created correctly.

        Args:
            db_path: Path to the SQLite database
            table_name: Name of the table to validate

        Returns:
            True if database is valid, False otherwise
        """
        if not os.path.exists(db_path):
            print(f"❌ Database file not found: {db_path}")
            return False

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check if table exists
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not cursor.fetchone():
                print(f"❌ Table '{table_name}' not found in database")
                return False

            # Check row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]

            if row_count == 0:
                print(f"⚠️  Warning: Table '{table_name}' is empty")
                return False

            # Check column count
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            print(f"\n✅ Database validation passed:")
            print(f"   Table: {table_name}")
            print(f"   Rows: {row_count:,}")
            print(f"   Columns: {len(columns)}")

            conn.close()
            return True

        except Exception as e:
            print(f"❌ Database validation failed: {str(e)}")
            return False

    def get_statistics(self, db_path: str, table_name: str = 'transactions') -> Dict[str, Any]:
        """
        Get statistics about the database.

        Args:
            db_path: Path to the SQLite database
            table_name: Name of the table to analyze

        Returns:
            Dictionary with database statistics
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            # Row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]

            # Column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]

            # Database file size
            db_size_mb = os.path.getsize(db_path) / (1024 * 1024)

            # Class distribution if Class column exists
            class_distribution = None
            if 'Class' in column_names:
                cursor.execute(f"SELECT Class, COUNT(*) FROM {table_name} GROUP BY Class")
                class_distribution = {row[0]: row[1] for row in cursor.fetchall()}

            stats = {
                'db_path': db_path,
                'table_name': table_name,
                'row_count': row_count,
                'column_count': len(columns),
                'column_names': column_names,
                'db_size_mb': db_size_mb,
                'class_distribution': class_distribution
            }

            return stats

        finally:
            conn.close()

    def query_sample(
        self,
        db_path: str,
        table_name: str = 'transactions',
        limit: int = 5
    ) -> pd.DataFrame:
        """
        Query and return sample data from the database.

        Args:
            db_path: Path to the SQLite database
            table_name: Name of the table to query
            limit: Number of rows to return

        Returns:
            DataFrame with sample data
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")

        conn = sqlite3.connect(db_path)

        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            df = pd.read_sql(query, conn)

            print(f"\nSample data from '{table_name}' (first {limit} rows):")
            print(df.to_string())
            print()

            return df

        finally:
            conn.close()


# CLI usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python csv_to_sqlite.py <csv_file> [db_file] [table_name]")
        print("\nExample:")
        print("  python csv_to_sqlite.py data/transactions.csv")
        print("  python csv_to_sqlite.py data/transactions.csv fraud.db transactions")
        sys.exit(1)

    csv_file = sys.argv[1]
    db_file = sys.argv[2] if len(sys.argv) > 2 else 'fraud_detection.db'
    table_name = sys.argv[3] if len(sys.argv) > 3 else 'transactions'

    # Create converter
    converter = CSVToSQLiteConverter()

    # Convert
    stats = converter.convert(csv_file, db_file, table_name)

    # Validate
    converter.validate_database(db_file, table_name)

    # Show sample
    converter.query_sample(db_file, table_name, limit=5)

    print("✅ Conversion complete! Database ready for use.")
    print(f"   Run queries with: sqlite3 {db_file}")
