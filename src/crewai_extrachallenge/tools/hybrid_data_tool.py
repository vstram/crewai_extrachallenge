"""
Hybrid Data Tool

This tool provides intelligent data access that automatically chooses between
database queries (fast, memory-efficient) and CSV sampling (fallback) based
on what's available.
"""

from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import os


class HybridDataInput(BaseModel):
    """Input schema for Hybrid Data Tool."""
    query_type: str = Field(
        ...,
        description="Type of query: 'count', 'stats', 'sample', 'filter'"
    )
    parameters: Optional[str] = Field(
        None,
        description="Optional parameters as JSON string (e.g., '{\"class\": 1, \"limit\": 10}')"
    )


class HybridDataTool(BaseTool):
    """
    Intelligent data access tool that automatically uses the best method.

    - Prefers database queries (fast, memory-efficient)
    - Falls back to CSV sampling if database not available
    - Provides consistent interface regardless of backend
    """

    name: str = "Hybrid Data Tool"
    description: str = (
        "Smart data access tool that uses database when available (fast), "
        "or falls back to CSV sampling. "
        "Can answer questions about transaction data efficiently. "
        "Usage: Specify query_type: 'count', 'stats', 'sample', or 'filter'"
    )
    args_schema: Type[BaseModel] = HybridDataInput

    def _run(self, query_type: str, parameters: Optional[str] = None) -> str:
        """
        Execute data query using best available method.

        Args:
            query_type: Type of query to execute
            parameters: Optional JSON parameters

        Returns:
            Query results as formatted string
        """
        # Check if database mode is enabled and database exists
        use_database = os.getenv('USE_DATABASE', 'true').lower() == 'true'
        db_path = os.getenv('DB_PATH', 'fraud_detection.db')

        if use_database and os.path.exists(db_path):
            return self._query_database(query_type, parameters, db_path)
        else:
            return self._query_csv(query_type, parameters)

    def _query_database(self, query_type: str, parameters: Optional[str], db_path: str) -> str:
        """Execute query using database backend."""
        import sqlite3
        import pandas as pd
        import json

        # Parse parameters
        params = {}
        if parameters:
            try:
                params = json.loads(parameters)
            except:
                params = {}

        try:
            conn = sqlite3.connect(db_path)

            if query_type == 'count':
                # Count transactions
                class_filter = params.get('class', None)

                if class_filter is not None:
                    query = f"SELECT COUNT(*) as count FROM transactions WHERE Class = {class_filter}"
                else:
                    query = "SELECT COUNT(*) as count FROM transactions"

                result = pd.read_sql(query, conn)
                count = result.iloc[0]['count']

                if class_filter is not None:
                    class_label = "fraud" if class_filter == 1 else "normal"
                    return f"Found {count:,} {class_label} transactions (database query)"
                else:
                    return f"Total transactions: {count:,} (database query)"

            elif query_type == 'stats':
                # Get statistics
                column = params.get('column', 'Amount')

                query = f"""
                    SELECT
                        ROUND(MIN({column}), 2) as min_val,
                        ROUND(MAX({column}), 2) as max_val,
                        ROUND(AVG({column}), 2) as avg_val,
                        COUNT({column}) as count_val
                    FROM transactions
                """
                result = pd.read_sql(query, conn).iloc[0]

                return f"""Statistics for {column} (database query):
- Min: {result['min_val']}
- Max: {result['max_val']}
- Average: {result['avg_val']}
- Count: {int(result['count_val']):,}"""

            elif query_type == 'sample':
                # Get sample data
                limit = params.get('limit', 5)
                class_filter = params.get('class', None)

                query = f"SELECT * FROM transactions"

                if class_filter is not None:
                    query += f" WHERE Class = {class_filter}"

                query += f" ORDER BY RANDOM() LIMIT {limit}"

                result = pd.read_sql(query, conn)

                return f"""Sample transactions (database query):
{result.to_string(index=False, max_cols=8)}

Showing {len(result)} of many transactions."""

            elif query_type == 'filter':
                # Filter data
                condition = params.get('condition', 'Amount > 100')
                limit = params.get('limit', 10)

                query = f"SELECT COUNT(*) as count FROM transactions WHERE {condition}"
                count_result = pd.read_sql(query, conn).iloc[0]['count']

                query = f"SELECT * FROM transactions WHERE {condition} LIMIT {limit}"
                result = pd.read_sql(query, conn)

                return f"""Filtered transactions (database query):
Condition: {condition}
Total matching: {count_result:,}

Showing first {len(result)} results:
{result.to_string(index=False, max_cols=8)}"""

            else:
                return f"Unknown query type: {query_type}"

        except Exception as e:
            return f"Database query error: {str(e)}"
        finally:
            conn.close()

    def _query_csv(self, query_type: str, parameters: Optional[str]) -> str:
        """Execute query using CSV backend (fallback)."""
        import pandas as pd
        import json

        # Get CSV path
        csv_path = os.getenv('DATASET_PATH', 'data/credit_card_transactions.csv')

        if not os.path.exists(csv_path):
            return f"❌ Data not available. CSV: {csv_path} not found, Database not configured."

        # Parse parameters
        params = {}
        if parameters:
            try:
                params = json.loads(parameters)
            except:
                params = {}

        try:
            # For CSV, we'll sample to avoid memory issues
            # Read first 10K rows for quick queries
            max_sample = params.get('max_rows', 10000)
            df = pd.read_csv(csv_path, nrows=max_sample)

            if query_type == 'count':
                class_filter = params.get('class', None)

                if class_filter is not None:
                    count = len(df[df['Class'] == class_filter])
                    class_label = "fraud" if class_filter == 1 else "normal"
                    return f"Found ~{count:,} {class_label} transactions (CSV sample of {len(df):,} rows)"
                else:
                    return f"Total transactions: ~{len(df):,} (CSV sample)"

            elif query_type == 'stats':
                column = params.get('column', 'Amount')

                if column not in df.columns:
                    return f"Column '{column}' not found in dataset"

                stats = df[column].describe()

                return f"""Statistics for {column} (CSV sample):
- Min: {stats['min']:.2f}
- Max: {stats['max']:.2f}
- Average: {stats['mean']:.2f}
- Std: {stats['std']:.2f}
- Count: {int(stats['count']):,}"""

            elif query_type == 'sample':
                limit = params.get('limit', 5)
                class_filter = params.get('class', None)

                if class_filter is not None:
                    sample = df[df['Class'] == class_filter].sample(min(limit, len(df)))
                else:
                    sample = df.sample(min(limit, len(df)))

                return f"""Sample transactions (CSV):
{sample.to_string(index=False, max_cols=8)}

Showing {len(sample)} transactions."""

            elif query_type == 'filter':
                # Note: CSV filtering is limited
                return "⚠️ Complex filtering not available in CSV mode. Please use database for advanced queries."

            else:
                return f"Unknown query type: {query_type}"

        except Exception as e:
            return f"CSV query error: {str(e)}"


# Test the tool
if __name__ == "__main__":
    tool = HybridDataTool()

    print("Testing HybridDataTool...\n")

    # Test 1: Count query
    print("=" * 60)
    print("TEST 1: Count all transactions")
    print("=" * 60)
    result = tool._run('count')
    print(result)

    # Test 2: Count fraud
    print("\n" + "=" * 60)
    print("TEST 2: Count fraudulent transactions")
    print("=" * 60)
    result = tool._run('count', '{"class": 1}')
    print(result)

    # Test 3: Statistics
    print("\n" + "=" * 60)
    print("TEST 3: Amount statistics")
    print("=" * 60)
    result = tool._run('stats', '{"column": "Amount"}')
    print(result)

    # Test 4: Sample
    print("\n" + "=" * 60)
    print("TEST 4: Sample fraud transactions")
    print("=" * 60)
    result = tool._run('sample', '{"class": 1, "limit": 3}')
    print(result)

    print("\n✅ All tests complete!")
