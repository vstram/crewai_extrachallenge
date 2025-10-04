#!/usr/bin/env python
import sys
import warnings
import os

from datetime import datetime

from crewai_extrachallenge.crew import CrewaiExtrachallenge
from crewai_extrachallenge.utils.csv_to_sqlite import CSVToSQLiteConverter
from crewai_extrachallenge.config.database_config import DatabaseConfig

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Credit Card Fraud Detection Crew
# This file runs the fraud detection analysis on credit card transaction datasets
# Usage: Provide dataset path as environment variable or command line argument


def ensure_database_ready(dataset_path: str) -> bool:
    """
    Ensure database is ready for analysis.

    If USE_DATABASE is enabled and CSV file exists but database doesn't,
    automatically convert CSV to SQLite database.

    Args:
        dataset_path: Path to CSV file

    Returns:
        True if database is ready, False if using CSV fallback
    """
    # Check if database mode is enabled
    if not DatabaseConfig.USE_DATABASE:
        print("Database mode disabled (USE_DATABASE=false). Using CSV fallback.")
        return False

    db_path = DatabaseConfig.DB_PATH

    # If database already exists, we're ready
    if os.path.exists(db_path):
        print(f"✓ Database ready: {db_path}")
        return True

    # If CSV doesn't exist, we can't convert
    if not os.path.exists(dataset_path):
        print(f"⚠ CSV file not found: {dataset_path}")
        print("  Database mode enabled but no data source available.")
        return False

    # Convert CSV to database
    print(f"Converting CSV to database for efficient analysis...")
    print(f"  CSV: {dataset_path}")
    print(f"  Database: {db_path}")

    try:
        converter = CSVToSQLiteConverter()

        # Show progress during conversion
        def progress_callback(current_chunk, total_rows):
            if current_chunk == 0:
                print(f"  Starting conversion...")
            elif current_chunk % 5 == 0:  # Update every 5 chunks
                print(f"  Processed {total_rows:,} rows...")

        result = converter.convert(
            csv_path=dataset_path,
            db_path=db_path,
            table_name=DatabaseConfig.DB_TABLE,
            chunk_size=DatabaseConfig.CHUNK_SIZE,
            progress_callback=progress_callback
        )

        print(f"✓ Database conversion complete!")
        print(f"  Rows: {result['row_count']:,}")
        print(f"  Columns: {result['column_count']}")
        print(f"  Compression: {result['compression_ratio']:.1%}")
        print(f"  Time: {result['conversion_time']:.2f}s")

        return True

    except Exception as e:
        print(f"✗ Database conversion failed: {e}")
        print(f"  Falling back to CSV mode.")
        return False

def run():
    """
    Run the fraud detection crew on a credit card transaction dataset.
    Dataset path can be provided via DATASET_PATH environment variable.

    Automatically converts CSV to SQLite database if USE_DATABASE=true.
    """
    # Get dataset path from environment variable or use default
    dataset_path = os.getenv('DATASET_PATH', 'data/credit_card_transactions.csv')

    print(f"Starting fraud detection analysis...")
    print(f"Dataset: {dataset_path}")
    print()

    # Ensure database is ready (will auto-convert CSV if needed)
    ensure_database_ready(dataset_path)
    print()

    inputs = {
        'dataset_path': dataset_path,
        'current_year': str(datetime.now().year)
    }

    try:
        CrewaiExtrachallenge().crew().kickoff(inputs=inputs)
        print()
        print(f"✓ Fraud detection analysis completed!")
        print(f"  Report: reports/fraud_detection_report.md")
        print(f"  Charts: reports/images/")
    except Exception as e:
        raise Exception(f"An error occurred while running the fraud detection crew: {e}")


def train():
    """
    Train the fraud detection crew for a given number of iterations using labeled data.
    Usage: uv run train <n_iterations> <training_file> [dataset_path]

    Automatically converts CSV to SQLite database if USE_DATABASE=true.
    """
    if len(sys.argv) < 3:
        raise ValueError("Usage: train <n_iterations> <training_file> [dataset_path]")

    # Get dataset path from command line or environment variable
    dataset_path = sys.argv[3] if len(sys.argv) > 3 else os.getenv('DATASET_PATH', 'data/labeled_transactions.csv')

    print(f"Training fraud detection crew...")
    print(f"Dataset: {dataset_path}")
    print()

    # Ensure database is ready (will auto-convert CSV if needed)
    ensure_database_ready(dataset_path)
    print()

    inputs = {
        'dataset_path': dataset_path,
        'current_year': str(datetime.now().year)
    }

    try:
        CrewaiExtrachallenge().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs
        )
        print()
        print(f"✓ Training completed after {sys.argv[1]} iterations")
        print(f"  Results: {sys.argv[2]}")
    except Exception as e:
        raise Exception(f"An error occurred while training the fraud detection crew: {e}")

def replay():
    """
    Replay the fraud detection crew execution from a specific task.
    Usage: uv run replay <task_id>
    """
    if len(sys.argv) < 2:
        raise ValueError("Usage: replay <task_id>")

    try:
        CrewaiExtrachallenge().crew().replay(task_id=sys.argv[1])
        print(f"Replayed execution from task: {sys.argv[1]}")
    except Exception as e:
        raise Exception(f"An error occurred while replaying the fraud detection crew: {e}")

def test():
    """
    Test the fraud detection crew execution and return performance results.
    Usage: uv run test <n_iterations> <eval_llm> [dataset_path]

    Automatically converts CSV to SQLite database if USE_DATABASE=true.
    """
    if len(sys.argv) < 3:
        raise ValueError("Usage: test <n_iterations> <eval_llm> [dataset_path]")

    # Get dataset path from command line or environment variable
    dataset_path = sys.argv[3] if len(sys.argv) > 3 else os.getenv('DATASET_PATH', 'data/test_transactions.csv')

    print(f"Testing fraud detection crew...")
    print(f"Dataset: {dataset_path}")
    print()

    # Ensure database is ready (will auto-convert CSV if needed)
    ensure_database_ready(dataset_path)
    print()

    inputs = {
        'dataset_path': dataset_path,
        'current_year': str(datetime.now().year)
    }

    try:
        CrewaiExtrachallenge().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs
        )
        print()
        print(f"✓ Testing completed after {sys.argv[1]} iterations")
        print(f"  Evaluator: {sys.argv[2]}")
    except Exception as e:
        raise Exception(f"An error occurred while testing the fraud detection crew: {e}")

def analyze_dataset():
    """
    Analyze a specific dataset provided as command line argument.
    Usage: uv run analyze_dataset <dataset_path>

    Automatically converts CSV to SQLite database if USE_DATABASE=true.
    """
    if len(sys.argv) < 2:
        raise ValueError("Usage: analyze_dataset <dataset_path>")

    dataset_path = sys.argv[1]

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    print(f"Analyzing dataset: {dataset_path}")
    print()

    # Ensure database is ready (will auto-convert CSV if needed)
    ensure_database_ready(dataset_path)
    print()

    inputs = {
        'dataset_path': dataset_path,
        'current_year': str(datetime.now().year)
    }

    try:
        CrewaiExtrachallenge().crew().kickoff(inputs=inputs)
        print()
        print(f"✓ Analysis completed!")
        print(f"  Report: reports/fraud_detection_report.md")
        print(f"  Charts: reports/images/")
    except Exception as e:
        raise Exception(f"An error occurred while analyzing the dataset: {e}")
