#!/usr/bin/env python
import sys
import warnings
import os

from datetime import datetime

from crewai_extrachallenge.crew import CrewaiExtrachallenge

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Credit Card Fraud Detection Crew
# This file runs the fraud detection analysis on credit card transaction datasets
# Usage: Provide dataset path as environment variable or command line argument

def run():
    """
    Run the fraud detection crew on a credit card transaction dataset.
    Dataset path can be provided via DATASET_PATH environment variable.
    """
    # Get dataset path from environment variable or use default
    dataset_path = os.getenv('DATASET_PATH', 'data/credit_card_transactions.csv')

    inputs = {
        'dataset_path': dataset_path,
        'current_year': str(datetime.now().year)
    }

    print(f"Starting fraud detection analysis on dataset: {dataset_path}")

    try:
        CrewaiExtrachallenge().crew().kickoff(inputs=inputs)
        print(f"Fraud detection analysis completed. Report saved to: reports/fraud_detection_report.md")
        print(f"Charts and images saved to: reports/images/")
    except Exception as e:
        raise Exception(f"An error occurred while running the fraud detection crew: {e}")


def train():
    """
    Train the fraud detection crew for a given number of iterations using labeled data.
    Usage: uv run train <n_iterations> <training_file> [dataset_path]
    """
    if len(sys.argv) < 3:
        raise ValueError("Usage: train <n_iterations> <training_file> [dataset_path]")

    # Get dataset path from command line or environment variable
    dataset_path = sys.argv[3] if len(sys.argv) > 3 else os.getenv('DATASET_PATH', 'data/labeled_transactions.csv')

    inputs = {
        'dataset_path': dataset_path,
        'current_year': str(datetime.now().year)
    }

    print(f"Training fraud detection crew on dataset: {dataset_path}")

    try:
        CrewaiExtrachallenge().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs
        )
        print(f"Training completed after {sys.argv[1]} iterations. Results saved to: {sys.argv[2]}")
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
    """
    if len(sys.argv) < 3:
        raise ValueError("Usage: test <n_iterations> <eval_llm> [dataset_path]")

    # Get dataset path from command line or environment variable
    dataset_path = sys.argv[3] if len(sys.argv) > 3 else os.getenv('DATASET_PATH', 'data/test_transactions.csv')

    inputs = {
        'dataset_path': dataset_path,
        'current_year': str(datetime.now().year)
    }

    print(f"Testing fraud detection crew on dataset: {dataset_path}")

    try:
        CrewaiExtrachallenge().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs
        )
        print(f"Testing completed after {sys.argv[1]} iterations using {sys.argv[2]} evaluator")
    except Exception as e:
        raise Exception(f"An error occurred while testing the fraud detection crew: {e}")

def analyze_dataset():
    """
    Analyze a specific dataset provided as command line argument.
    Usage: uv run analyze_dataset <dataset_path>
    """
    if len(sys.argv) < 2:
        raise ValueError("Usage: analyze_dataset <dataset_path>")

    dataset_path = sys.argv[1]

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    inputs = {
        'dataset_path': dataset_path,
        'current_year': str(datetime.now().year)
    }

    print(f"Analyzing dataset: {dataset_path}")

    try:
        CrewaiExtrachallenge().crew().kickoff(inputs=inputs)
        print(f"Analysis completed. Report saved to: reports/fraud_detection_report.md")
        print(f"Charts and images saved to: reports/images/")
    except Exception as e:
        raise Exception(f"An error occurred while analyzing the dataset: {e}")
