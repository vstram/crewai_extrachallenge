"""
Database Configuration

Centralized configuration for database settings in the fraud detection system.
Reads configuration from environment variables with sensible defaults.
"""

import os
from pathlib import Path


class DatabaseConfig:
    """
    Centralized database configuration.

    All database-related settings are managed here, with values loaded
    from environment variables. This allows easy configuration changes
    without modifying code.
    """

    # Database mode toggle
    USE_DATABASE = os.getenv('USE_DATABASE', 'true').lower() == 'true'

    # Database file paths
    DB_PATH = os.getenv('DB_PATH', 'fraud_detection.db')
    DB_TABLE = os.getenv('DB_TABLE', 'transactions')

    # SQLAlchemy connection URI
    DB_URI = os.getenv('DB_URI', f'sqlite:///{DB_PATH}')

    # Conversion settings
    CHUNK_SIZE = int(os.getenv('DB_CHUNK_SIZE', '10000'))

    # Derived paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    DB_ABSOLUTE_PATH = PROJECT_ROOT / DB_PATH if not os.path.isabs(DB_PATH) else Path(DB_PATH)

    @classmethod
    def get_db_path(cls) -> str:
        """
        Get the absolute path to the database file.

        Returns:
            Absolute path to SQLite database
        """
        return str(cls.DB_ABSOLUTE_PATH)

    @classmethod
    def is_database_enabled(cls) -> bool:
        """
        Check if database mode is enabled.

        Returns:
            True if database mode is enabled, False otherwise
        """
        return cls.USE_DATABASE

    @classmethod
    def database_exists(cls) -> bool:
        """
        Check if the database file exists.

        Returns:
            True if database file exists, False otherwise
        """
        return os.path.exists(cls.get_db_path())

    @classmethod
    def get_config_summary(cls) -> dict:
        """
        Get a summary of current database configuration.

        Returns:
            Dictionary with configuration settings
        """
        return {
            'use_database': cls.USE_DATABASE,
            'db_path': cls.DB_PATH,
            'db_absolute_path': str(cls.DB_ABSOLUTE_PATH),
            'db_table': cls.DB_TABLE,
            'db_uri': cls.DB_URI,
            'chunk_size': cls.CHUNK_SIZE,
            'database_exists': cls.database_exists(),
            'project_root': str(cls.PROJECT_ROOT)
        }

    @classmethod
    def print_config(cls):
        """Print current database configuration."""
        config = cls.get_config_summary()

        print(f"\n{'='*60}")
        print("DATABASE CONFIGURATION")
        print(f"{'='*60}")
        print(f"Mode:            {'Enabled' if config['use_database'] else 'Disabled'}")
        print(f"Database Path:   {config['db_path']}")
        print(f"Absolute Path:   {config['db_absolute_path']}")
        print(f"Table Name:      {config['db_table']}")
        print(f"Connection URI:  {config['db_uri']}")
        print(f"Chunk Size:      {config['chunk_size']:,} rows")
        print(f"Database Exists: {'✅ Yes' if config['database_exists'] else '❌ No'}")
        print(f"Project Root:    {config['project_root']}")
        print(f"{'='*60}\n")


# CLI usage - print configuration
if __name__ == "__main__":
    DatabaseConfig.print_config()
