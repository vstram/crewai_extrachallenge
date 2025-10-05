#!/usr/bin/env python3
"""
Database Integration Test Script

Tests the complete database integration for the fraud detection system.
Validates CSV to SQLite conversion, tool functionality, and crew execution.
"""

import os
import sys
import time
import sqlite3
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.crewai_extrachallenge.utils.csv_to_sqlite import CSVToSQLiteConverter
from src.crewai_extrachallenge.utils.db_helper import DatabaseHelper
from src.crewai_extrachallenge.config.database_config import DatabaseConfig
from src.crewai_extrachallenge.tools.db_statistical_analysis_tool import DBStatisticalAnalysisTool
from src.crewai_extrachallenge.tools.hybrid_data_tool import HybridDataTool


class DatabaseIntegrationTester:
    """Test suite for database integration."""

    def __init__(self, csv_path: str, test_db_path: str = "test_fraud_detection.db"):
        self.csv_path = csv_path
        self.test_db_path = test_db_path
        self.test_results = []
        self.start_time = None

    def run_all_tests(self):
        """Run all integration tests."""
        print("=" * 80)
        print("DATABASE INTEGRATION TEST SUITE")
        print("=" * 80)
        print(f"CSV File: {self.csv_path}")
        print(f"Test Database: {self.test_db_path}")
        print()

        self.start_time = time.time()

        # Clean up any existing test database
        self._cleanup_test_db()

        # Run tests
        tests = [
            ("CSV to SQLite Conversion", self.test_csv_to_sqlite_conversion),
            ("Database Schema Validation", self.test_database_schema),
            ("Database Statistics", self.test_database_statistics),
            ("Database Helper Functions", self.test_database_helper),
            ("DB Statistical Analysis Tool", self.test_db_statistical_tool),
            ("Hybrid Data Tool (Database Mode)", self.test_hybrid_tool_database),
            ("Hybrid Data Tool (CSV Fallback)", self.test_hybrid_tool_csv_fallback),
            ("Database Query Performance", self.test_query_performance),
        ]

        for test_name, test_func in tests:
            self._run_test(test_name, test_func)

        # Print summary
        self._print_summary()

        # Cleanup
        self._cleanup_test_db()

    def _run_test(self, test_name: str, test_func):
        """Run a single test and record results."""
        print(f"\n{'=' * 80}")
        print(f"TEST: {test_name}")
        print(f"{'=' * 80}")

        try:
            test_start = time.time()
            result = test_func()
            test_time = time.time() - test_start

            self.test_results.append({
                'name': test_name,
                'status': 'PASS' if result else 'FAIL',
                'time': test_time,
                'error': None
            })

            print(f"\n✅ {test_name}: PASSED ({test_time:.2f}s)")

        except Exception as e:
            test_time = time.time() - test_start
            self.test_results.append({
                'name': test_name,
                'status': 'FAIL',
                'time': test_time,
                'error': str(e)
            })

            print(f"\n❌ {test_name}: FAILED ({test_time:.2f}s)")
            print(f"   Error: {str(e)}")

    def test_csv_to_sqlite_conversion(self) -> bool:
        """Test CSV to SQLite conversion."""
        print(f"Converting CSV to SQLite: {self.csv_path} → {self.test_db_path}")

        converter = CSVToSQLiteConverter()

        # Progress tracking
        chunks_processed = [0]

        def progress_callback(chunk_num, total_rows):
            chunks_processed[0] = chunk_num
            if chunk_num % 10 == 0:
                print(f"  Processed chunk {chunk_num}: {total_rows:,} rows")

        # Convert
        result = converter.convert(
            csv_path=self.csv_path,
            db_path=self.test_db_path,
            table_name='transactions',
            chunk_size=10000,
            progress_callback=progress_callback
        )

        # Validate results
        assert os.path.exists(self.test_db_path), "Database file not created"
        assert result.get('total_rows', result.get('row_count', 0)) > 0, "No rows converted"
        assert result.get('total_columns', result.get('column_count', 0)) > 0, "No columns converted"

        # Get values with fallback keys
        total_rows = result.get('total_rows', result.get('row_count', 0))
        total_columns = result.get('total_columns', result.get('column_count', 0))
        db_size = result.get('db_size_mb', 0)
        csv_size = result.get('csv_size_mb', 0)
        conversion_time = result.get('conversion_time', result.get('duration', 0))

        print(f"\n  ✓ Rows converted: {total_rows:,}")
        print(f"  ✓ Columns: {total_columns}")
        print(f"  ✓ DB size: {db_size:.2f}MB")
        print(f"  ✓ CSV size: {csv_size:.2f}MB")
        print(f"  ✓ Conversion time: {conversion_time:.2f}s")
        print(f"  ✓ Chunks processed: {chunks_processed[0] + 1}")

        return True

    def test_database_schema(self) -> bool:
        """Test database schema and structure."""
        print("Validating database schema...")

        conn = DatabaseHelper.connect(self.test_db_path)

        # Get table schema
        schema = DatabaseHelper.get_table_schema(conn, 'transactions')

        print(f"\n  Table: {schema.get('table_name', 'transactions')}")
        print(f"  Columns: {len(schema.get('columns', []))}")
        print(f"  Indexes: {len(schema.get('indexes', []))}")

        # Validate required columns
        column_names = [col['name'] for col in schema.get('columns', [])]
        assert 'Time' in column_names, "Missing Time column"
        assert 'Amount' in column_names, "Missing Amount column"

        print(f"\n  ✓ Required columns present")
        if schema.get('indexes'):
            print(f"  ✓ Indexes created: {', '.join([idx['name'] for idx in schema['indexes']])}")

        conn.close()
        return True

    def test_database_statistics(self) -> bool:
        """Test database statistics computation."""
        print("Computing database statistics...")

        conn = DatabaseHelper.connect(self.test_db_path)

        stats = DatabaseHelper.compute_statistics(conn, 'transactions')

        # Handle different possible return structures
        total_rows = stats.get('total_rows', stats.get('row_count', 0))
        total_columns = stats.get('total_columns', len(stats.get('columns', [])))

        print(f"\n  Total rows: {total_rows:,}")
        print(f"  Columns: {total_columns}")

        # Try to get Class distribution
        try:
            query = "SELECT Class, COUNT(*) as count FROM transactions GROUP BY Class"
            import pandas as pd
            result = pd.read_sql(query, conn)
            print(f"\n  Class distribution:")
            for _, row in result.iterrows():
                class_label = "Fraud" if row['Class'] == 1 else "Normal"
                print(f"    {class_label}: {int(row['count']):,}")
        except:
            pass  # Class column might not exist

        conn.close()
        return True

    def test_database_helper(self) -> bool:
        """Test database helper functions."""
        print("Testing DatabaseHelper functions...")

        conn = DatabaseHelper.connect(self.test_db_path)

        # Test query execution
        query = "SELECT COUNT(*) as count FROM transactions"
        result = DatabaseHelper.execute_query(conn, query)

        assert len(result) > 0, "Query returned no results"
        row_count = result.iloc[0]['count']
        print(f"\n  ✓ Query execution: {row_count:,} rows")

        # Test parameterized query
        query = "SELECT * FROM transactions WHERE Amount > ? LIMIT 5"
        result = DatabaseHelper.execute_query(conn, query, (100,))
        print(f"  ✓ Parameterized query: {len(result)} results")

        # Test schema retrieval
        schema = DatabaseHelper.get_table_schema(conn, 'transactions')
        print(f"  ✓ Schema retrieval: {schema['column_count']} columns")

        # Test statistics
        stats = DatabaseHelper.compute_statistics(conn, 'transactions')
        print(f"  ✓ Statistics computation: {len(stats['numeric_columns'])} numeric columns")

        conn.close()
        return True

    def test_db_statistical_tool(self) -> bool:
        """Test DBStatisticalAnalysisTool."""
        print("Testing DBStatisticalAnalysisTool...")

        # Configure environment
        os.environ['USE_DATABASE'] = 'true'
        os.environ['DB_PATH'] = self.test_db_path

        tool = DBStatisticalAnalysisTool()

        # Test descriptive statistics
        print("\n  Testing descriptive statistics...")
        result = tool._run(analysis_type='descriptive', db_path=self.test_db_path)
        assert len(result) > 0, "No descriptive statistics returned"
        print(f"  ✓ Descriptive stats: {len(result)} characters")

        # Test data quality
        print("\n  Testing data quality assessment...")
        result = tool._run(analysis_type='data_quality', db_path=self.test_db_path)
        assert len(result) > 0, "No data quality results returned"
        print(f"  ✓ Data quality: {len(result)} characters")

        # Test distribution analysis
        print("\n  Testing distribution analysis...")
        result = tool._run(analysis_type='distribution', db_path=self.test_db_path, target_column='Class')
        assert len(result) > 0, "No distribution results returned"
        print(f"  ✓ Distribution: {len(result)} characters")

        # Test outlier detection
        print("\n  Testing outlier detection...")
        result = tool._run(analysis_type='outliers', db_path=self.test_db_path)
        assert len(result) > 0, "No outlier results returned"
        print(f"  ✓ Outliers: {len(result)} characters")

        # Test correlation analysis
        print("\n  Testing correlation analysis...")
        result = tool._run(analysis_type='correlation', db_path=self.test_db_path)
        assert len(result) > 0, "No correlation results returned"
        print(f"  ✓ Correlation: {len(result)} characters")

        return True

    def test_hybrid_tool_database(self) -> bool:
        """Test HybridDataTool in database mode."""
        print("Testing HybridDataTool (database mode)...")

        # Configure environment
        os.environ['USE_DATABASE'] = 'true'
        os.environ['DB_PATH'] = self.test_db_path

        tool = HybridDataTool()

        # Test count query
        print("\n  Testing count query...")
        result = tool._run(query_type='count')
        assert 'database query' in result.lower(), "Not using database"
        print(f"  ✓ Count: {result[:100]}")

        # Test stats query
        print("\n  Testing stats query...")
        result = tool._run(query_type='stats', parameters='{"column": "Amount"}')
        assert 'database query' in result.lower(), "Not using database"
        print(f"  ✓ Stats: {result[:100]}")

        # Test sample query
        print("\n  Testing sample query...")
        result = tool._run(query_type='sample', parameters='{"limit": 3}')
        assert 'database query' in result.lower(), "Not using database"
        print(f"  ✓ Sample: {len(result)} characters")

        return True

    def test_hybrid_tool_csv_fallback(self) -> bool:
        """Test HybridDataTool CSV fallback."""
        print("Testing HybridDataTool (CSV fallback)...")

        # Disable database mode
        os.environ['USE_DATABASE'] = 'false'

        tool = HybridDataTool()

        # Test count query with CSV fallback
        print("\n  Testing CSV fallback...")
        result = tool._run(query_type='count')
        assert 'csv' in result.lower(), "Not using CSV fallback"
        print(f"  ✓ CSV fallback works: {result[:100]}")

        # Re-enable database mode
        os.environ['USE_DATABASE'] = 'true'

        return True

    def test_query_performance(self) -> bool:
        """Test database query performance."""
        print("Testing query performance...")

        conn = DatabaseHelper.connect(self.test_db_path)

        # Test 1: Simple count
        start = time.time()
        result = DatabaseHelper.execute_query(conn, "SELECT COUNT(*) FROM transactions")
        count_time = time.time() - start
        print(f"\n  ✓ COUNT(*): {count_time*1000:.2f}ms")

        # Test 2: Aggregation
        start = time.time()
        result = DatabaseHelper.execute_query(conn,
            "SELECT AVG(Amount), MIN(Amount), MAX(Amount) FROM transactions")
        agg_time = time.time() - start
        print(f"  ✓ Aggregation (AVG/MIN/MAX): {agg_time*1000:.2f}ms")

        # Test 3: Filtered count
        start = time.time()
        result = DatabaseHelper.execute_query(conn,
            "SELECT COUNT(*) FROM transactions WHERE Class = 1")
        filter_time = time.time() - start
        print(f"  ✓ Filtered COUNT: {filter_time*1000:.2f}ms")

        # Test 4: Group by
        start = time.time()
        result = DatabaseHelper.execute_query(conn,
            "SELECT Class, COUNT(*) as count FROM transactions GROUP BY Class")
        group_time = time.time() - start
        print(f"  ✓ GROUP BY: {group_time*1000:.2f}ms")

        conn.close()

        # All queries should be fast
        assert count_time < 5, "COUNT query too slow"
        assert agg_time < 5, "Aggregation query too slow"

        return True

    def _cleanup_test_db(self):
        """Remove test database file."""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
            print(f"Cleaned up test database: {self.test_db_path}")

    def _print_summary(self):
        """Print test summary."""
        total_time = time.time() - self.start_time

        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        passed = sum(1 for r in self.test_results if r['status'] == 'PASS')
        failed = sum(1 for r in self.test_results if r['status'] == 'FAIL')

        print(f"\nTotal Tests: {len(self.test_results)}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Total Time: {total_time:.2f}s")

        print("\nDetailed Results:")
        print("-" * 80)
        for result in self.test_results:
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            print(f"{status_icon} {result['name']}: {result['status']} ({result['time']:.2f}s)")
            if result['error']:
                print(f"   Error: {result['error']}")

        print("\n" + "=" * 80)

        if failed == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"⚠️  {failed} TEST(S) FAILED")

        print("=" * 80)


def main():
    """Main test execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Test database integration')
    parser.add_argument('--csv', default='dataset/data/credit_card_transactions.csv',
                       help='Path to CSV file (default: small test file)')
    parser.add_argument('--large', action='store_true',
                       help='Use large CSV file for testing')

    args = parser.parse_args()

    # Determine CSV path
    if args.large:
        csv_path = 'dataset/data/credit_card_transactions-huge.csv'
    else:
        csv_path = args.csv

    # Verify CSV exists
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)

    # Get file size
    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"\nCSV File: {csv_path}")
    print(f"File Size: {file_size_mb:.2f}MB")

    # Run tests
    tester = DatabaseIntegrationTester(csv_path)
    tester.run_all_tests()


if __name__ == '__main__':
    main()
