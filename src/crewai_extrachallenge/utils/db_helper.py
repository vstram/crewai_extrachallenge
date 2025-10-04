"""
Database Helper Utilities

This module provides helper functions for working with SQLite databases
in the fraud detection system. It includes utilities for connection management,
query execution, schema inspection, and statistics computation.
"""

import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import os


class DatabaseHelper:
    """
    Helper utilities for SQLite database operations.

    Provides convenient methods for common database operations including
    connection management, query execution, schema inspection, and
    statistical computations.
    """

    @staticmethod
    def connect(db_path: str) -> sqlite3.Connection:
        """
        Create and return a SQLite database connection.

        Args:
            db_path: Path to the SQLite database file

        Returns:
            sqlite3.Connection object

        Raises:
            FileNotFoundError: If database file doesn't exist
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")

        conn = sqlite3.connect(db_path)
        # Enable foreign key support
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def execute_query(
        conn: sqlite3.Connection,
        query: str,
        params: Optional[Tuple] = None
    ) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a DataFrame.

        Args:
            conn: SQLite database connection
            query: SQL query to execute
            params: Optional tuple of query parameters for parameterized queries

        Returns:
            DataFrame with query results

        Raises:
            Exception: If query execution fails
        """
        try:
            if params:
                return pd.read_sql_query(query, conn, params=params)
            else:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            raise Exception(f"Query execution failed: {str(e)}\nQuery: {query}")

    @staticmethod
    def get_table_schema(conn: sqlite3.Connection, table_name: str) -> Dict[str, Any]:
        """
        Get schema information for a table.

        Args:
            conn: SQLite database connection
            table_name: Name of the table to inspect

        Returns:
            Dictionary with schema information:
            {
                'table_name': str,
                'columns': List[Dict],
                'indexes': List[str],
                'column_names': List[str]
            }
        """
        cursor = conn.cursor()

        # Get column information
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_raw = cursor.fetchall()

        columns = []
        column_names = []

        for col in columns_raw:
            col_info = {
                'cid': col[0],
                'name': col[1],
                'type': col[2],
                'notnull': bool(col[3]),
                'default_value': col[4],
                'primary_key': bool(col[5])
            }
            columns.append(col_info)
            column_names.append(col[1])

        # Get index information
        cursor.execute(f"PRAGMA index_list({table_name})")
        indexes = [row[1] for row in cursor.fetchall()]

        schema = {
            'table_name': table_name,
            'columns': columns,
            'indexes': indexes,
            'column_names': column_names,
            'column_count': len(columns)
        }

        return schema

    @staticmethod
    def get_table_row_count(conn: sqlite3.Connection, table_name: str) -> int:
        """
        Get the number of rows in a table.

        Args:
            conn: SQLite database connection
            table_name: Name of the table

        Returns:
            Number of rows in the table
        """
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cursor.fetchone()[0]

    @staticmethod
    def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            conn: SQLite database connection
            table_name: Name of the table to check

        Returns:
            True if table exists, False otherwise
        """
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return cursor.fetchone() is not None

    @staticmethod
    def create_indexes(
        conn: sqlite3.Connection,
        table_name: str,
        columns: List[str]
    ) -> None:
        """
        Create indexes on specified columns.

        Args:
            conn: SQLite database connection
            table_name: Name of the table
            columns: List of column names to create indexes on
        """
        cursor = conn.cursor()

        for column in columns:
            index_name = f"idx_{table_name}_{column.lower()}"
            try:
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column})"
                )
                print(f"   ✓ Created index: {index_name}")
            except Exception as e:
                print(f"   ⚠ Could not create index on {column}: {e}")

        conn.commit()

    @staticmethod
    def compute_statistics(
        conn: sqlite3.Connection,
        table_name: str
    ) -> Dict[str, Any]:
        """
        Compute basic statistics for a table.

        Args:
            conn: SQLite database connection
            table_name: Name of the table to analyze

        Returns:
            Dictionary with statistics:
            {
                'row_count': int,
                'column_count': int,
                'numeric_columns': List[str],
                'column_stats': Dict[str, Dict]
            }
        """
        # Get schema
        schema = DatabaseHelper.get_table_schema(conn, table_name)

        # Get row count
        row_count = DatabaseHelper.get_table_row_count(conn, table_name)

        # Identify numeric columns by sampling data
        sample_query = f"SELECT * FROM {table_name} LIMIT 1000"
        sample_df = pd.read_sql(sample_query, conn)
        numeric_columns = sample_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Compute statistics for numeric columns
        column_stats = {}

        for col in numeric_columns:
            try:
                stats_query = f"""
                    SELECT
                        MIN({col}) as min_val,
                        MAX({col}) as max_val,
                        AVG({col}) as mean_val,
                        COUNT({col}) as count_val
                    FROM {table_name}
                """
                stats_df = pd.read_sql(stats_query, conn)

                column_stats[col] = {
                    'min': stats_df['min_val'].iloc[0],
                    'max': stats_df['max_val'].iloc[0],
                    'mean': stats_df['mean_val'].iloc[0],
                    'count': stats_df['count_val'].iloc[0]
                }
            except Exception as e:
                print(f"   ⚠ Could not compute stats for {col}: {e}")

        statistics = {
            'row_count': row_count,
            'column_count': len(schema['columns']),
            'numeric_columns': numeric_columns,
            'column_stats': column_stats,
            'table_name': table_name
        }

        return statistics

    @staticmethod
    def get_sample_data(
        conn: sqlite3.Connection,
        table_name: str,
        limit: int = 10,
        where_clause: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get sample data from a table.

        Args:
            conn: SQLite database connection
            table_name: Name of the table
            limit: Number of rows to return
            where_clause: Optional WHERE clause (without the WHERE keyword)

        Returns:
            DataFrame with sample data
        """
        query = f"SELECT * FROM {table_name}"

        if where_clause:
            query += f" WHERE {where_clause}"

        query += f" LIMIT {limit}"

        return pd.read_sql(query, conn)

    @staticmethod
    def execute_aggregation(
        conn: sqlite3.Connection,
        table_name: str,
        agg_columns: List[str],
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Execute aggregation query.

        Args:
            conn: SQLite database connection
            table_name: Name of the table
            agg_columns: List of aggregation expressions (e.g., ['COUNT(*)', 'AVG(Amount)'])
            group_by: Optional column to group by

        Returns:
            DataFrame with aggregation results
        """
        agg_str = ', '.join(agg_columns)
        query = f"SELECT {agg_str} FROM {table_name}"

        if group_by:
            query += f" GROUP BY {group_by}"

        return pd.read_sql(query, conn)

    @staticmethod
    def get_database_info(db_path: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a database.

        Args:
            db_path: Path to the SQLite database

        Returns:
            Dictionary with database information
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")

        conn = DatabaseHelper.connect(db_path)

        try:
            cursor = conn.cursor()

            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            # Get database size
            db_size_bytes = os.path.getsize(db_path)
            db_size_mb = db_size_bytes / (1024 * 1024)

            # Get table information
            table_info = {}
            for table in tables:
                row_count = DatabaseHelper.get_table_row_count(conn, table)
                schema = DatabaseHelper.get_table_schema(conn, table)

                table_info[table] = {
                    'row_count': row_count,
                    'column_count': len(schema['columns']),
                    'indexes': schema['indexes']
                }

            info = {
                'db_path': db_path,
                'db_size_mb': db_size_mb,
                'db_size_bytes': db_size_bytes,
                'table_count': len(tables),
                'tables': tables,
                'table_info': table_info
            }

            return info

        finally:
            conn.close()

    @staticmethod
    def print_database_summary(db_path: str) -> None:
        """
        Print a formatted summary of database contents.

        Args:
            db_path: Path to the SQLite database
        """
        info = DatabaseHelper.get_database_info(db_path)

        print(f"\n{'='*60}")
        print(f"DATABASE SUMMARY: {os.path.basename(db_path)}")
        print(f"{'='*60}")
        print(f"Path:        {db_path}")
        print(f"Size:        {info['db_size_mb']:.2f} MB")
        print(f"Tables:      {info['table_count']}")
        print()

        for table_name, table_data in info['table_info'].items():
            print(f"📊 Table: {table_name}")
            print(f"   Rows:    {table_data['row_count']:,}")
            print(f"   Columns: {table_data['column_count']}")
            print(f"   Indexes: {len(table_data['indexes'])}")
            if table_data['indexes']:
                print(f"            {', '.join(table_data['indexes'])}")
            print()

        print(f"{'='*60}\n")


# CLI usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python db_helper.py <database_path>")
        print("\nExample:")
        print("  python db_helper.py fraud_detection.db")
        sys.exit(1)

    db_path = sys.argv[1]

    # Print database summary
    DatabaseHelper.print_database_summary(db_path)

    # Connect and show sample data from first table
    info = DatabaseHelper.get_database_info(db_path)
    if info['tables']:
        first_table = info['tables'][0]
        print(f"Sample data from '{first_table}':")

        conn = DatabaseHelper.connect(db_path)
        sample = DatabaseHelper.get_sample_data(conn, first_table, limit=5)
        print(sample.to_string())
        conn.close()
