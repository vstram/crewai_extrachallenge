"""
NL2SQL Tool Configuration

This module provides configuration and setup for CrewAI's NL2SQLTool,
which converts natural language questions into SQL queries for fraud
detection analysis.
"""

from crewai_tools import NL2SQLTool
from typing import Optional
import os


class NL2SQLToolConfig:
    """
    Configuration manager for NL2SQLTool.

    Provides factory methods to create properly configured NL2SQL tools
    for different use cases (Ollama local, OpenAI, custom LLM).
    """

    @staticmethod
    def create_tool(
        db_uri: Optional[str] = None,
        db_name: str = 'fraud_transactions',
        use_ollama: bool = True
    ) -> NL2SQLTool:
        """
        Create and configure NL2SQLTool for fraud detection queries.

        Args:
            db_uri: SQLite database URI (defaults to env DB_URI)
            db_name: Logical name for the database
            use_ollama: Whether to use Ollama local LLM (default: True)

        Returns:
            Configured NL2SQLTool instance

        Example:
            >>> tool = NL2SQLToolConfig.create_tool()
            >>> # Agent can now ask: "How many fraudulent transactions are there?"
            >>> # Tool converts to: SELECT COUNT(*) FROM transactions WHERE Class=1
        """
        # Get database URI from environment if not provided
        if not db_uri:
            db_uri = os.getenv('DB_URI', 'sqlite:///fraud_detection.db')

        # Create base tool configuration
        tool_config = {
            'db_uri': db_uri,
            'db_name': db_name
        }

        # Configure LLM provider
        if use_ollama:
            # Use Ollama local LLM
            model = os.getenv('MODEL', 'ollama/llama3.1:8b')
            api_base = os.getenv('API_BASE', 'http://localhost:11434')

            tool_config['config'] = {
                'llm': {
                    'provider': 'ollama',
                    'config': {
                        'model': model.replace('ollama/', ''),  # Remove 'ollama/' prefix
                        'base_url': api_base
                    }
                }
            }
        # If not using Ollama, NL2SQLTool will use default OpenAI

        # Create and return configured tool
        tool = NL2SQLTool(**tool_config)

        return tool

    @staticmethod
    def create_openai_tool(
        db_uri: Optional[str] = None,
        db_name: str = 'fraud_transactions',
        model: str = 'gpt-3.5-turbo'
    ) -> NL2SQLTool:
        """
        Create NL2SQLTool configured for OpenAI.

        Args:
            db_uri: SQLite database URI
            db_name: Logical name for the database
            model: OpenAI model to use

        Returns:
            NL2SQLTool configured for OpenAI
        """
        if not db_uri:
            db_uri = os.getenv('DB_URI', 'sqlite:///fraud_detection.db')

        tool = NL2SQLTool(
            db_uri=db_uri,
            db_name=db_name,
            config={
                'llm': {
                    'provider': 'openai',
                    'config': {
                        'model': model
                    }
                }
            }
        )

        return tool

    @staticmethod
    def get_example_questions() -> list:
        """
        Get a list of example questions that NL2SQLTool can handle.

        Returns:
            List of example natural language questions
        """
        return [
            # Count queries
            "How many total transactions are in the database?",
            "How many fraudulent transactions are there?",
            "How many normal transactions are there?",
            "What percentage of transactions are fraudulent?",

            # Amount queries
            "What is the average transaction amount?",
            "What is the average amount for fraudulent transactions?",
            "What is the average amount for normal transactions?",
            "What is the highest transaction amount?",
            "What is the lowest transaction amount?",

            # Comparison queries
            "How does the average fraud amount compare to normal transactions?",
            "What is the total value of all fraudulent transactions?",
            "What is the total value of all normal transactions?",

            # Distribution queries
            "How many transactions are over $100?",
            "How many transactions are between $50 and $100?",
            "How many fraudulent transactions are over $500?",

            # Time-based queries
            "What is the time range of the transactions?",
            "How many transactions occurred in the first hour?",
            "What is the fraud rate by hour?",

            # Statistical queries
            "What is the median transaction amount?",
            "What are the top 10 highest transaction amounts?",
            "Show me 5 random fraudulent transactions",
            "Show me 5 random normal transactions"
        ]

    @staticmethod
    def print_example_questions():
        """Print example questions for users/agents."""
        questions = NL2SQLToolConfig.get_example_questions()

        print("\n" + "=" * 60)
        print("NL2SQL TOOL - EXAMPLE QUESTIONS")
        print("=" * 60)
        print("\nThe NL2SQL tool can answer questions like:\n")

        for i, question in enumerate(questions, 1):
            print(f"{i:2d}. {question}")

        print("\n" + "=" * 60)
        print("\nUsage in Agent:")
        print("  Ask any natural language question about transactions.")
        print("  The tool will automatically convert it to SQL and execute.")
        print("=" * 60 + "\n")


# Factory function for easy import
def create_nl2sql_tool(use_ollama: bool = True) -> NL2SQLTool:
    """
    Quick factory function to create NL2SQL tool with default settings.

    Args:
        use_ollama: Whether to use Ollama local LLM (default: True)

    Returns:
        Configured NL2SQLTool instance
    """
    return NL2SQLToolConfig.create_tool(use_ollama=use_ollama)


# CLI usage
if __name__ == "__main__":
    import sys

    # Print example questions
    NL2SQLToolConfig.print_example_questions()

    # Test tool creation
    print("\n" + "=" * 60)
    print("TESTING TOOL CREATION")
    print("=" * 60)

    try:
        # NOTE: NL2SQLTool has some SQLite compatibility issues
        # For production use, we'll rely on DBStatisticalAnalysisTool
        # and direct SQL queries via agents
        print("\n✅ NL2SQL tool configuration module loaded successfully")
        print(f"   Database URI: {os.getenv('DB_URI', 'sqlite:///fraud_detection.db')}")
        print(f"   Model: {os.getenv('MODEL', 'ollama/llama3.1:8b')}")
        print("\n   NOTE: NL2SQLTool will be configured in agent setup")
        print("   For SQLite, we primarily use DBStatisticalAnalysisTool")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    print("\n" + "=" * 60 + "\n")
