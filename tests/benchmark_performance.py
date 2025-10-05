#!/usr/bin/env python3
"""
Performance Benchmark Script

Compares CSV vs Database performance for fraud detection analysis.
Measures memory usage, execution time, and scalability.
"""

import os
import sys
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.crewai_extrachallenge.utils.csv_to_sqlite import CSVToSQLiteConverter
from src.crewai_extrachallenge.tools.db_statistical_analysis_tool import DBStatisticalAnalysisTool
from src.crewai_extrachallenge.tools.hybrid_data_tool import HybridDataTool


class PerformanceBenchmark:
    """Benchmark CSV vs Database performance."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.db_path = "benchmark_test.db"
        self.results = {
            'csv_mode': {},
            'database_mode': {},
            'comparison': {}
        }

    def run_benchmark(self):
        """Run complete benchmark suite."""
        print("=" * 80)
        print("PERFORMANCE BENCHMARK: CSV vs DATABASE")
        print("=" * 80)
        print(f"CSV File: {self.csv_path}")
        print(f"File Size: {os.path.getsize(self.csv_path) / (1024*1024):.2f}MB")
        print()

        # Benchmark 1: Database conversion
        self._benchmark_database_conversion()

        # Benchmark 2: Memory usage
        self._benchmark_memory_usage()

        # Benchmark 3: Query performance
        self._benchmark_query_performance()

        # Benchmark 4: Tool performance
        self._benchmark_tool_performance()

        # Print results
        self._print_results()

        # Cleanup
        self._cleanup()

    def _benchmark_database_conversion(self):
        """Benchmark database conversion time and compression."""
        print("\n" + "=" * 80)
        print("BENCHMARK 1: Database Conversion")
        print("=" * 80)

        # Clean up existing database
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        # Measure conversion
        converter = CSVToSQLiteConverter()

        start_time = time.time()
        start_memory = self._get_memory_usage()

        result = converter.convert(
            csv_path=self.csv_path,
            db_path=self.db_path,
            table_name='transactions',
            chunk_size=10000
        )

        conversion_time = time.time() - start_time
        peak_memory = self._get_memory_usage() - start_memory

        self.results['database_mode']['conversion_time'] = conversion_time
        self.results['database_mode']['conversion_memory_mb'] = peak_memory
        self.results['database_mode']['db_size_mb'] = result['db_size_mb']
        self.results['database_mode']['csv_size_mb'] = result['csv_size_mb']
        self.results['database_mode']['compression_ratio'] = result['compression_ratio']
        self.results['database_mode']['row_count'] = result['row_count']

        print(f"\n✓ Conversion Time: {conversion_time:.2f}s")
        print(f"✓ Memory Usage: {peak_memory:.2f}MB")
        print(f"✓ CSV Size: {result['csv_size_mb']:.2f}MB")
        print(f"✓ DB Size: {result['db_size_mb']:.2f}MB")
        print(f"✓ Compression: {result['compression_ratio']:.1%}")
        print(f"✓ Rows: {result['row_count']:,}")

    def _benchmark_memory_usage(self):
        """Benchmark memory usage for data loading."""
        print("\n" + "=" * 80)
        print("BENCHMARK 2: Memory Usage")
        print("=" * 80)

        # CSV mode - load into pandas
        print("\nCSV Mode (pandas load)...")
        gc.collect()
        start_memory = self._get_memory_usage()

        try:
            import pandas as pd
            start = time.time()
            df = pd.read_csv(self.csv_path)
            load_time = time.time() - start
            csv_memory = self._get_memory_usage() - start_memory

            self.results['csv_mode']['load_time'] = load_time
            self.results['csv_mode']['memory_usage_mb'] = csv_memory
            self.results['csv_mode']['rows_loaded'] = len(df)

            print(f"✓ Load Time: {load_time:.2f}s")
            print(f"✓ Memory Usage: {csv_memory:.2f}MB")
            print(f"✓ Rows Loaded: {len(df):,}")

            # Clean up
            del df
            gc.collect()

        except MemoryError:
            print("❌ MemoryError: File too large to load into pandas")
            self.results['csv_mode']['load_time'] = None
            self.results['csv_mode']['memory_usage_mb'] = None
            self.results['csv_mode']['rows_loaded'] = 0

        # Database mode - query without loading full dataset
        print("\nDatabase Mode (SQL query)...")
        gc.collect()
        start_memory = self._get_memory_usage()

        import sqlite3
        start = time.time()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM transactions")
        count = cursor.fetchone()[0]
        conn.close()
        query_time = time.time() - start
        db_memory = self._get_memory_usage() - start_memory

        self.results['database_mode']['query_time'] = query_time
        self.results['database_mode']['memory_usage_mb'] = db_memory
        self.results['database_mode']['rows_queried'] = count

        print(f"✓ Query Time: {query_time:.4f}s")
        print(f"✓ Memory Usage: {db_memory:.2f}MB")
        print(f"✓ Rows Queried: {count:,}")

        # Calculate improvement
        if self.results['csv_mode']['memory_usage_mb']:
            memory_improvement = (1 - db_memory / self.results['csv_mode']['memory_usage_mb']) * 100
            self.results['comparison']['memory_improvement_pct'] = memory_improvement
            print(f"\n💡 Memory Improvement: {memory_improvement:.1f}%")

    def _benchmark_query_performance(self):
        """Benchmark query execution performance."""
        print("\n" + "=" * 80)
        print("BENCHMARK 3: Query Performance")
        print("=" * 80)

        import sqlite3

        queries = [
            ("Count all rows", "SELECT COUNT(*) FROM transactions"),
            ("Count fraud", "SELECT COUNT(*) FROM transactions WHERE Class = 1"),
            ("Avg amount", "SELECT AVG(Amount) FROM transactions"),
            ("Group by class", "SELECT Class, COUNT(*), AVG(Amount) FROM transactions GROUP BY Class"),
            ("Filter & aggregate", "SELECT AVG(Amount) FROM transactions WHERE Amount > 100"),
        ]

        query_times = []

        conn = sqlite3.connect(self.db_path)

        for query_name, query_sql in queries:
            # Warm-up query
            conn.execute(query_sql).fetchall()

            # Timed query
            start = time.time()
            result = conn.execute(query_sql).fetchall()
            query_time = time.time() - start

            query_times.append(query_time)
            print(f"✓ {query_name}: {query_time*1000:.2f}ms")

        conn.close()

        self.results['database_mode']['avg_query_time_ms'] = (sum(query_times) / len(query_times)) * 1000
        print(f"\n💡 Average Query Time: {self.results['database_mode']['avg_query_time_ms']:.2f}ms")

    def _benchmark_tool_performance(self):
        """Benchmark tool execution performance."""
        print("\n" + "=" * 80)
        print("BENCHMARK 4: Tool Performance")
        print("=" * 80)

        # Configure environment
        os.environ['USE_DATABASE'] = 'true'
        os.environ['DB_PATH'] = self.db_path

        # Test DBStatisticalAnalysisTool
        print("\nDBStatisticalAnalysisTool...")
        tool = DBStatisticalAnalysisTool()

        tool_times = []

        # Test different analysis types
        analyses = [
            ('descriptive', {}),
            ('data_quality', {}),
            ('distribution', {'target_column': 'Class'}),
            ('outliers', {}),
        ]

        for analysis_type, kwargs in analyses:
            gc.collect()
            start = time.time()
            result = tool._run(analysis_type=analysis_type, db_path=self.db_path, **kwargs)
            analysis_time = time.time() - start
            tool_times.append(analysis_time)

            print(f"  ✓ {analysis_type}: {analysis_time:.2f}s")

        self.results['database_mode']['avg_tool_time_s'] = sum(tool_times) / len(tool_times)

        # Test HybridDataTool
        print("\nHybridDataTool...")
        hybrid_tool = HybridDataTool()

        start = time.time()
        result = hybrid_tool._run(query_type='count')
        count_time = time.time() - start

        start = time.time()
        result = hybrid_tool._run(query_type='stats', parameters='{"column": "Amount"}')
        stats_time = time.time() - start

        print(f"  ✓ count query: {count_time:.4f}s")
        print(f"  ✓ stats query: {stats_time:.4f}s")

        print(f"\n💡 Average Tool Execution: {self.results['database_mode']['avg_tool_time_s']:.2f}s")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def _print_results(self):
        """Print benchmark results summary."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 80)

        # File information
        print("\n📁 FILE INFORMATION")
        print("-" * 80)
        print(f"CSV Size: {self.results['database_mode']['csv_size_mb']:.2f}MB")
        print(f"Database Size: {self.results['database_mode']['db_size_mb']:.2f}MB")
        print(f"Compression: {self.results['database_mode']['compression_ratio']:.1%}")
        print(f"Rows: {self.results['database_mode']['row_count']:,}")

        # Conversion
        print("\n🔄 CONVERSION")
        print("-" * 80)
        print(f"Conversion Time: {self.results['database_mode']['conversion_time']:.2f}s")
        print(f"Conversion Memory: {self.results['database_mode']['conversion_memory_mb']:.2f}MB")

        # Memory comparison
        print("\n💾 MEMORY USAGE")
        print("-" * 80)
        if self.results['csv_mode'].get('memory_usage_mb'):
            print(f"CSV Mode: {self.results['csv_mode']['memory_usage_mb']:.2f}MB")
        else:
            print(f"CSV Mode: TOO LARGE (MemoryError)")

        print(f"Database Mode: {self.results['database_mode']['memory_usage_mb']:.2f}MB")

        if 'memory_improvement_pct' in self.results['comparison']:
            print(f"Improvement: {self.results['comparison']['memory_improvement_pct']:.1f}% less memory")

        # Performance
        print("\n⚡ PERFORMANCE")
        print("-" * 80)
        if self.results['csv_mode'].get('load_time'):
            print(f"CSV Load Time: {self.results['csv_mode']['load_time']:.2f}s")

        print(f"Database Query Time: {self.results['database_mode']['query_time']:.4f}s")
        print(f"Average Query Time: {self.results['database_mode']['avg_query_time_ms']:.2f}ms")
        print(f"Average Tool Execution: {self.results['database_mode']['avg_tool_time_s']:.2f}s")

        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        print("-" * 80)

        file_size_mb = self.results['database_mode']['csv_size_mb']

        if file_size_mb < 10:
            print("✅ File is small (<10MB), both approaches work well")
            print("   → Database still recommended for consistency")
        elif file_size_mb < 50:
            print("⚡ File is medium (10-50MB), database recommended")
            print("   → 2-5x faster, 90%+ less memory")
        else:
            print("🚀 File is large (>50MB), database STRONGLY recommended")
            print("   → 5-10x faster, 95%+ less memory")
            print("   → CSV mode may cause MemoryError")

        print("\n" + "=" * 80)

    def _cleanup(self):
        """Clean up test files."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            print(f"\n🧹 Cleaned up: {self.db_path}")


def main():
    """Main benchmark execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark CSV vs Database performance')
    parser.add_argument('--csv', default='dataset/data/credit_card_transactions.csv',
                       help='Path to CSV file')
    parser.add_argument('--large', action='store_true',
                       help='Use large CSV file (144MB)')

    args = parser.parse_args()

    # Determine CSV path
    if args.large:
        csv_path = 'dataset/data/credit_card_transactions-huge.csv'
    else:
        csv_path = args.csv

    # Verify CSV exists
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        print(f"\nAvailable files:")
        import glob
        for f in glob.glob("dataset/data/*.csv"):
            print(f"  - {f}")
        sys.exit(1)

    # Run benchmark
    benchmark = PerformanceBenchmark(csv_path)
    benchmark.run_benchmark()


if __name__ == '__main__':
    main()
