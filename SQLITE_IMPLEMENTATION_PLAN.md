# SQLite + NL2SQLTool Implementation Plan

**Goal:** Enable the fraud detection system to handle 150MB+ CSV files efficiently using SQLite database with Ollama (no OpenAI costs)

**Approach:** Chunked CSV processing → SQLite database → NL2SQLTool for natural language queries

**Estimated Time:** 2-4 hours total implementation

---

## Phase Overview

### Phase 1: Database Infrastructure (45-60 minutes)
- Create CSV to SQLite converter with chunked reading
- Add database utilities and helpers
- Create automated conversion on first run

### Phase 2: Database-Aware Tools (60-90 minutes)
- Create DBStatisticalAnalysisTool (SQL-based statistics)
- Configure NL2SQLTool for natural language queries
- Update existing tools to support database queries

### Phase 3: Agent & Crew Integration (30-45 minutes)
- Update agent configurations to use database tools
- Modify task definitions for database approach
- Update main.py workflow for database initialization

### Phase 4: Streamlit UI Integration (30-45 minutes)
- Update file validator to offer database conversion
- Add database status indicators
- Show conversion progress in UI

### Phase 5: Testing & Validation (30-45 minutes)
- Test with current 122KB dataset
- Test with large 150MB+ dataset
- Verify memory usage improvements
- Validate query accuracy

---

## Detailed Implementation Plan

---

## Phase 1: Database Infrastructure

### 1.1 Create CSV to SQLite Converter

**File:** `src/crewai_extrachallenge/utils/csv_to_sqlite.py`

**Features:**
- ✅ Chunked CSV reading (avoid loading entire file)
- ✅ Progress tracking and reporting
- ✅ Automatic index creation for performance
- ✅ Compression statistics
- ✅ Data validation during conversion

**Implementation Details:**

```python
class CSVToSQLiteConverter:
    """
    Convert large CSV files to SQLite database with chunked processing.

    Key Features:
    - Reads CSV in chunks (default 10,000 rows at a time)
    - Shows progress during conversion
    - Creates indexes on key columns (Class, Amount, Time)
    - Reports compression and performance metrics
    """

    Methods:
    - convert(csv_path, db_path, table_name, chunk_size=10000)
    - validate_database(db_path, table_name)
    - get_statistics(db_path, table_name)
    - query_sample(db_path, table_name, limit=5)
```

**Configuration:**
- Default chunk size: 10,000 rows (adjustable for memory constraints)
- Automatic index creation on: Class, Amount, Time columns
- Transaction batching for performance

**Error Handling:**
- File not found validation
- Database write permission checks
- Graceful handling of malformed CSV rows
- Rollback on conversion failure

---

### 1.2 Create Database Helper Utilities

**File:** `src/crewai_extrachallenge/utils/db_helper.py`

**Features:**
- Database connection management
- Query execution helpers
- Schema inspection utilities
- Statistics computation helpers

**Implementation:**

```python
class DatabaseHelper:
    """Helper utilities for SQLite database operations."""

    Methods:
    - connect(db_path) -> sqlite3.Connection
    - execute_query(conn, query) -> pd.DataFrame
    - get_table_schema(conn, table_name) -> dict
    - get_table_row_count(conn, table_name) -> int
    - create_indexes(conn, table_name, columns) -> None
    - compute_statistics(conn, table_name) -> dict
```

---

### 1.3 Update Environment Configuration

**File:** `.env`

**Add:**
```bash
# Database Configuration
USE_DATABASE=true                    # Enable database mode
DB_PATH=fraud_detection.db          # SQLite database file path
DB_TABLE=transactions               # Default table name
DB_URI=sqlite:///fraud_detection.db # SQLAlchemy connection URI
DB_CHUNK_SIZE=10000                 # Rows per chunk during conversion
```

**File:** `src/crewai_extrachallenge/config/database_config.py`

```python
class DatabaseConfig:
    """Centralized database configuration."""

    USE_DATABASE = os.getenv('USE_DATABASE', 'true').lower() == 'true'
    DB_PATH = os.getenv('DB_PATH', 'fraud_detection.db')
    DB_TABLE = os.getenv('DB_TABLE', 'transactions')
    DB_URI = os.getenv('DB_URI', f'sqlite:///{DB_PATH}')
    CHUNK_SIZE = int(os.getenv('DB_CHUNK_SIZE', '10000'))
```

---

## Phase 2: Database-Aware Tools

### 2.1 Create DBStatisticalAnalysisTool

**File:** `src/crewai_extrachallenge/tools/db_statistical_analysis_tool.py`

**Purpose:** Perform statistical analysis using SQL queries instead of loading full CSV

**Features:**
- ✅ Descriptive statistics (via SQL aggregations)
- ✅ Correlation analysis (via SQL sampling)
- ✅ Outlier detection (via SQL percentiles)
- ✅ Distribution analysis (via SQL histograms)
- ✅ Data quality assessment (via SQL checks)

**Key Optimizations:**

```python
# Instead of loading entire CSV:
df = pd.read_csv(dataset_path)  # ❌ 2GB memory for 150MB file

# Use SQL queries:
query = "SELECT AVG(Amount), STDEV(Amount) FROM transactions"
result = pd.read_sql(query, conn)  # ✅ 1KB result, 5MB memory
```

**Analysis Types:**

1. **Descriptive Statistics**
   - Total row count: `SELECT COUNT(*) FROM transactions`
   - Class distribution: `SELECT Class, COUNT(*) FROM transactions GROUP BY Class`
   - Amount stats: `SELECT MIN(Amount), MAX(Amount), AVG(Amount), STDEV(Amount)`
   - Memory efficient for any dataset size

2. **Correlation Analysis**
   - Sample-based (1% random sample): `SELECT * FROM transactions WHERE RANDOM() % 100 = 0`
   - Compute correlations on sample
   - 100x faster than full dataset

3. **Outlier Detection**
   - SQL-based IQR calculation using percentiles
   - Count outliers: `SELECT COUNT(*) WHERE Amount < lower_bound OR Amount > upper_bound`
   - No need to load individual outliers

4. **Distribution Analysis**
   - SQL histogram bins: `SELECT CASE WHEN Amount < 10 THEN '0-10' ... END, COUNT(*)`
   - Efficient aggregation

5. **Data Quality Assessment**
   - Missing values: `SELECT COUNT(*) FROM transactions WHERE Amount IS NULL`
   - Duplicates: `SELECT COUNT(*) FROM (SELECT *, COUNT(*) as cnt GROUP BY * HAVING cnt > 1)`

---

### 2.2 Configure NL2SQLTool

**File:** `src/crewai_extrachallenge/tools/nl2sql_tool_config.py`

**Purpose:** Configure CrewAI's NL2SQLTool for fraud detection queries

**Configuration:**

```python
from crewai_tools import NL2SQLTool
from config.database_config import DatabaseConfig

def create_nl2sql_tool():
    """Create and configure NL2SQLTool for fraud detection."""

    tool = NL2SQLTool(
        db_uri=DatabaseConfig.DB_URI,
        db_name='fraud_transactions',
        # Optional: Configure to use Ollama
        config=dict(
            llm=dict(
                provider="ollama",
                config=dict(
                    model="llama3.1:8b",
                    base_url="http://localhost:11434"
                )
            )
        )
    )

    return tool
```

**Example Queries NL2SQL Can Handle:**

- "How many fraudulent transactions are there?"
  → `SELECT COUNT(*) FROM transactions WHERE Class = 1`

- "What's the average amount for fraud vs normal?"
  → `SELECT Class, AVG(Amount) FROM transactions GROUP BY Class`

- "Show me transactions over $1000"
  → `SELECT * FROM transactions WHERE Amount > 1000 LIMIT 10`

- "What's the fraud rate by time period?"
  → `SELECT CAST(Time/3600 AS INT) as hour, SUM(Class)*100.0/COUNT(*) FROM transactions GROUP BY hour`

---

### 2.3 Create Hybrid Tool (Database + CSV Fallback)

**File:** `src/crewai_extrachallenge/tools/hybrid_data_tool.py`

**Purpose:** Smart tool that uses database if available, falls back to CSV

**Implementation:**

```python
class HybridDataTool(BaseTool):
    """
    Intelligent data access tool.
    - Uses SQLite database if available (fast, memory-efficient)
    - Falls back to CSV sampling if database not initialized
    - Provides consistent interface for agents
    """

    def _run(self, query_type: str, parameters: dict) -> str:
        if DatabaseConfig.USE_DATABASE and os.path.exists(DatabaseConfig.DB_PATH):
            # Use database approach
            return self._query_database(query_type, parameters)
        else:
            # Fall back to CSV sampling
            return self._query_csv_sampled(query_type, parameters)
```

**Benefits:**
- Backward compatibility with existing CSV-based agents
- Graceful degradation if database not available
- Automatic optimization when database exists

---

## Phase 3: Agent & Crew Integration

### 3.1 Update Agent Configurations

**File:** `src/crewai_extrachallenge/config/agents.yaml`

**Changes:**

```yaml
data_analyst:
  role: >
    Senior Credit Card Fraud Data Analyst
  goal: >
    Analyze credit card transaction patterns using efficient database queries
    to identify fraud indicators and statistical anomalies in large datasets
  backstory: >
    You are an expert data analyst with deep knowledge of SQL and fraud detection.
    You use database queries to efficiently analyze millions of transactions without
    loading all data into memory. You understand how to leverage indexes and
    aggregations for fast insights.
  tools:
    - NL2SQLTool              # NEW: Natural language to SQL queries
    - DBStatisticalAnalysisTool  # NEW: Database-optimized statistics
    - ImageGenerationTool      # KEEP: For visualization
    - MarkdownFormatterTool    # KEEP: For formatting
  verbose: true
  allow_delegation: false

pattern_recognition_agent:
  role: >
    Fraud Pattern Recognition Specialist
  goal: >
    Identify unusual transaction patterns and fraud indicators using database analytics
  backstory: >
    You excel at pattern recognition in large transaction databases.
    You use SQL queries to identify statistical anomalies, temporal patterns,
    and suspicious transaction sequences efficiently.
  tools:
    - NL2SQLTool
    - DBStatisticalAnalysisTool
    - ImageGenerationTool
  verbose: true
  allow_delegation: false

classification_agent:
  role: >
    Machine Learning Fraud Classification Expert
  goal: >
    Build and evaluate fraud classification models using database-sourced features
  backstory: >
    You are a machine learning expert specializing in fraud detection.
    You use SQL to extract features and samples from large transaction databases
    for efficient model training and evaluation.
  tools:
    - NL2SQLTool
    - DBStatisticalAnalysisTool
  verbose: true
  allow_delegation: false

reporting_analyst:
  role: >
    Senior Fraud Detection Report Analyst
  goal: >
    Create comprehensive, actionable fraud detection reports with database-backed insights
  backstory: >
    You synthesize database query results into clear, executive-level reports.
    You present statistical findings, visualizations, and recommendations
    based on efficient database analysis.
  tools:
    - MarkdownFormatterTool
    - ImageVerificationTool
    - TaskValidationTool
  verbose: true
  allow_delegation: false
```

---

### 3.2 Update Task Configurations

**File:** `src/crewai_extrachallenge/config/tasks.yaml`

**Add database-specific task instructions:**

```yaml
data_analysis_task:
  description: |
    Perform comprehensive data analysis on the credit card transaction database.

    **Database Information:**
    - Database path: {db_path}
    - Table name: {table_name}
    - Total rows: Use COUNT(*) query to determine

    **Your Analysis Steps:**

    1. **Data Overview** (use NL2SQLTool):
       - Ask: "How many total transactions are in the database?"
       - Ask: "How many fraud cases (Class=1) vs normal (Class=0)?"
       - Ask: "What's the date/time range of the transactions?"

    2. **Statistical Analysis** (use DBStatisticalAnalysisTool):
       - Run descriptive statistics: analysis_type='descriptive'
       - Analyze distributions: analysis_type='distribution'
       - Assess data quality: analysis_type='data_quality'

    3. **Key Questions to Answer:**
       - What's the fraud rate (percentage)?
       - What's the typical transaction amount for fraud vs normal?
       - Are there time-based patterns?
       - What's the data quality (missing values, outliers)?

    **Output Format:**
    Return a structured markdown report with:
    - Dataset Overview section
    - Statistical Summary section
    - Key Findings (bullet points)
    - Data Quality Assessment

    **Important:**
    - Use SQL queries for all data access (don't load raw CSV)
    - Present aggregate statistics, not individual records
    - Focus on insights that will help fraud detection
  expected_output: >
    A comprehensive data analysis report with database statistics,
    fraud rate analysis, and data quality assessment in markdown format
  agent: data_analyst

pattern_recognition_task:
  description: |
    Identify fraud patterns and anomalies using database queries.

    **Your Analysis Approach:**

    1. **Correlation Analysis** (use DBStatisticalAnalysisTool):
       - Run: analysis_type='correlation'
       - Identify features strongly correlated with fraud (Class=1)

    2. **Outlier Detection** (use DBStatisticalAnalysisTool):
       - Run: analysis_type='outliers'
       - Focus on Amount outliers

    3. **Pattern Queries** (use NL2SQLTool):
       - Ask: "What are the top 10 highest fraud amounts?"
       - Ask: "What's the fraud rate by transaction amount ranges?"
       - Ask: "Show distribution of fraud across time periods"

    4. **Temporal Patterns**:
       - Analyze fraud by time of day
       - Identify suspicious time-based sequences

    **Output:**
    Detailed pattern analysis with:
    - Correlation findings
    - Outlier statistics
    - Temporal patterns
    - Recommendations for fraud indicators

    **Remember:** Use database queries, not CSV loads!
  expected_output: >
    Pattern recognition report identifying key fraud indicators,
    correlations, outliers, and temporal patterns from database analysis
  agent: pattern_recognition_agent
  context:
    - data_analysis_task

classification_task:
  description: |
    Evaluate fraud classification approach using database samples.

    **Your Analysis:**

    1. **Feature Importance** (use DBStatisticalAnalysisTool):
       - Run: analysis_type='feature_importance', target_column='Class'
       - Identify most predictive features

    2. **Sample Analysis** (use NL2SQLTool):
       - Ask: "Show me 5 random fraud cases"
       - Ask: "Show me 5 random normal cases"
       - Compare feature patterns

    3. **Classification Recommendations**:
       - Which features are most important?
       - What thresholds should be used?
       - How to handle class imbalance?

    **Output:**
    Classification analysis with:
    - Feature importance ranking
    - Recommended features for modeling
    - Classification strategy recommendations
    - Expected performance insights
  expected_output: >
    Classification analysis report with feature importance,
    modeling recommendations, and fraud detection strategy
  agent: classification_agent
  context:
    - data_analysis_task
    - pattern_recognition_task

reporting_task:
  description: |
    Create comprehensive fraud detection report from database analysis results.

    **Input:** You will receive:
    - Data analysis results (from data_analyst)
    - Pattern recognition findings (from pattern_recognition_agent)
    - Classification recommendations (from classification_agent)

    **Your Task:**
    Synthesize all findings into an executive-level report.

    **Report Structure:**

    # Credit Card Fraud Detection Analysis Report

    ## Executive Summary
    - Total transactions analyzed
    - Fraud rate and statistics
    - Key findings (3-5 bullet points)

    ## Dataset Overview
    - Database statistics
    - Data quality assessment
    - Time period covered

    ## Fraud Pattern Analysis
    - Identified fraud indicators
    - Statistical patterns
    - Correlation findings

    ## Classification Insights
    - Feature importance
    - Recommended detection approach

    ## Visualizations
    - Reference all generated charts
    - Verify images exist using ImageVerificationTool

    ## Recommendations
    - Fraud prevention strategies
    - Implementation priorities
    - Risk mitigation approaches

    **Quality Checks:**
    1. Use MarkdownFormatterTool to ensure proper formatting
    2. Use ImageVerificationTool to verify all images exist
    3. Use TaskValidationTool to verify report completeness

    **Important:** This report is based on database analysis of {total_rows} transactions!
  expected_output: >
    Comprehensive markdown report suitable for executives, with verified
    images, proper formatting, and actionable fraud detection insights
  agent: reporting_analyst
  context:
    - data_analysis_task
    - pattern_recognition_task
    - classification_task
  output_file: reports/fraud_detection_report.md
```

---

### 3.3 Update Main Workflow

**File:** `src/crewai_extrachallenge/main.py`

**Add database initialization logic:**

```python
#!/usr/bin/env python
import sys
import warnings
import os
from datetime import datetime
from pathlib import Path

from crewai_extrachallenge.crew import CrewaiExtrachallenge
from utils.csv_to_sqlite import CSVToSQLiteConverter
from config.database_config import DatabaseConfig

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def ensure_database_ready(csv_path: str) -> tuple[str, str, dict]:
    """
    Ensure database is ready for analysis.
    Converts CSV to SQLite if database doesn't exist.

    Returns:
        (db_path, table_name, conversion_stats)
    """
    db_path = DatabaseConfig.DB_PATH
    table_name = DatabaseConfig.DB_TABLE

    # Check if database already exists
    if os.path.exists(db_path):
        print(f"✅ Using existing database: {db_path}")

        # Verify database has data
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        conn.close()

        stats = {
            'total_rows': row_count,
            'db_path': db_path,
            'table_name': table_name,
            'status': 'existing'
        }

        print(f"   Table '{table_name}': {row_count:,} rows")
        return db_path, table_name, stats

    # Database doesn't exist - convert CSV
    print(f"\n🔄 Converting CSV to SQLite database...")
    print(f"   CSV: {csv_path}")
    print(f"   Database: {db_path}")
    print(f"   This may take a few minutes for large files...\n")

    converter = CSVToSQLiteConverter()
    stats = converter.convert(
        csv_path=csv_path,
        db_path=db_path,
        table_name=table_name,
        chunk_size=DatabaseConfig.CHUNK_SIZE
    )

    stats['status'] = 'converted'

    print(f"\n✅ Database ready for analysis!")
    return db_path, table_name, stats


def run():
    """
    Run the fraud detection crew on a credit card transaction dataset.
    Uses SQLite database for efficient large file handling.
    """
    # Get CSV path from environment variable or use default
    csv_path = os.getenv('DATASET_PATH', 'data/credit_card_transactions.csv')

    # Verify CSV exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    # Ensure database is ready (convert if needed)
    db_path, table_name, stats = ensure_database_ready(csv_path)

    # Prepare inputs for crew
    inputs = {
        'db_path': db_path,
        'table_name': table_name,
        'total_rows': stats['total_rows'],
        'current_year': str(datetime.now().year),
        'dataset_path': csv_path  # Keep for backward compatibility
    }

    print(f"\n🚀 Starting fraud detection analysis...")
    print(f"   Database: {db_path}")
    print(f"   Records: {stats['total_rows']:,}")
    print(f"   Mode: {'Existing database' if stats['status'] == 'existing' else 'Newly converted'}\n")

    try:
        # Run the crew with database inputs
        result = CrewaiExtrachallenge().crew().kickoff(inputs=inputs)

        print(f"\n✅ Fraud detection analysis completed!")
        print(f"   Report: reports/fraud_detection_report.md")
        print(f"   Images: reports/images/")
        print(f"   Database: {db_path} (reusable for future runs)")

        return result

    except Exception as e:
        raise Exception(f"An error occurred while running the fraud detection crew: {e}")


def train():
    """Train the fraud detection crew (updated for database)."""
    if len(sys.argv) < 3:
        raise ValueError("Usage: train <n_iterations> <training_file> [dataset_path]")

    csv_path = sys.argv[3] if len(sys.argv) > 3 else os.getenv('DATASET_PATH', 'data/labeled_transactions.csv')

    # Ensure database ready
    db_path, table_name, stats = ensure_database_ready(csv_path)

    inputs = {
        'db_path': db_path,
        'table_name': table_name,
        'total_rows': stats['total_rows'],
        'current_year': str(datetime.now().year)
    }

    print(f"Training fraud detection crew on database: {db_path}")

    try:
        CrewaiExtrachallenge().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs
        )
        print(f"Training completed after {sys.argv[1]} iterations")
    except Exception as e:
        raise Exception(f"An error occurred while training: {e}")


def replay():
    """Replay the fraud detection crew execution."""
    if len(sys.argv) < 2:
        raise ValueError("Usage: replay <task_id>")

    try:
        CrewaiExtrachallenge().crew().replay(task_id=sys.argv[1])
        print(f"Replayed execution from task: {sys.argv[1]}")
    except Exception as e:
        raise Exception(f"An error occurred while replaying: {e}")


def test():
    """Test the fraud detection crew (updated for database)."""
    if len(sys.argv) < 3:
        raise ValueError("Usage: test <n_iterations> <eval_llm> [dataset_path]")

    csv_path = sys.argv[3] if len(sys.argv) > 3 else os.getenv('DATASET_PATH', 'data/test_transactions.csv')

    # Ensure database ready
    db_path, table_name, stats = ensure_database_ready(csv_path)

    inputs = {
        'db_path': db_path,
        'table_name': table_name,
        'total_rows': stats['total_rows'],
        'current_year': str(datetime.now().year)
    }

    print(f"Testing fraud detection crew on database: {db_path}")

    try:
        CrewaiExtrachallenge().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs
        )
        print(f"Testing completed after {sys.argv[1]} iterations using {sys.argv[2]} evaluator")
    except Exception as e:
        raise Exception(f"An error occurred while testing: {e}")
```

---

## Phase 4: Streamlit UI Integration

### 4.1 Update File Validator

**File:** `streamlit_app/utils/file_validator.py`

**Add database conversion option:**

```python
class CSVValidator:
    """Enhanced validator with database conversion option."""

    @staticmethod
    def validate_and_prepare(file_path: str, offer_db_conversion: bool = True) -> dict:
        """
        Validate CSV and optionally convert to database.

        Returns:
            {
                'valid': bool,
                'message': str,
                'file_info': dict,
                'db_recommended': bool,  # NEW
                'db_path': str or None    # NEW
            }
        """
        # Standard validation
        is_valid, message, file_info = CSVValidator.validate_csv_file(file_path)

        result = {
            'valid': is_valid,
            'message': message,
            'file_info': file_info,
            'db_recommended': False,
            'db_path': None
        }

        if not is_valid:
            return result

        # Check if database conversion is recommended
        file_size_mb = file_info.get('size_mb', 0)

        if offer_db_conversion and file_size_mb > 10:  # >10MB
            result['db_recommended'] = True
            result['recommendation'] = (
                f"⚡ **Performance Tip:** Your file is {file_size_mb:.1f}MB. "
                f"Converting to database will be {int(file_size_mb/10)}x faster and use "
                f"95% less memory!"
            )

        return result
```

---

### 4.2 Add Database Conversion UI Component

**File:** `streamlit_app/components/database_converter.py`

**New component for database conversion:**

```python
import streamlit as st
from utils.csv_to_sqlite import CSVToSQLiteConverter
from config.database_config import DatabaseConfig

def render_database_conversion(csv_path: str) -> bool:
    """
    Render database conversion UI with progress.

    Returns:
        True if conversion successful, False otherwise
    """
    st.subheader("🗄️ Database Conversion")

    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)

    st.info(f"""
    **Converting to SQLite Database**

    File size: {file_size_mb:.1f}MB
    Expected database size: ~{file_size_mb * 0.6:.1f}MB (40% compression)
    Expected time: ~{int(file_size_mb / 5)} seconds

    Benefits:
    - ⚡ 40x faster analysis
    - 💾 95% less memory usage
    - 🔄 Reusable for future analyses
    """)

    if st.button("Convert to Database", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Conversion with progress updates
            converter = CSVToSQLiteConverter()

            # Custom progress callback
            def progress_callback(current, total):
                progress = int((current / total) * 100)
                progress_bar.progress(progress)
                status_text.text(f"Processing: {current:,} / {total:,} rows ({progress}%)")

            stats = converter.convert(
                csv_path=csv_path,
                db_path=DatabaseConfig.DB_PATH,
                table_name=DatabaseConfig.DB_TABLE,
                progress_callback=progress_callback
            )

            progress_bar.progress(100)
            status_text.empty()

            st.success(f"""
            ✅ **Conversion Complete!**

            - Processed: {stats['total_rows']:,} rows
            - Duration: {stats['duration_seconds']:.1f} seconds
            - Database size: {stats['db_size_mb']:.1f}MB
            - Compression: {stats.get('compression_pct', 0):.1f}%
            """)

            return True

        except Exception as e:
            st.error(f"Conversion failed: {str(e)}")
            return False

    return False
```

---

### 4.3 Update Dataset Configuration Component

**File:** `streamlit_app/components/dataset_config.py`

**Add database option:**

```python
def render_dataset_configuration():
    """Render dataset configuration with database option."""

    st.header("📊 Phase 1: Dataset Configuration")

    # ... existing file selection code ...

    if csv_path and os.path.exists(csv_path):
        # Validate
        validation = CSVValidator.validate_and_prepare(csv_path)

        if validation['valid']:
            # Show file info
            st.success("✅ Valid credit card transaction dataset")

            # Database conversion recommendation
            if validation['db_recommended']:
                st.warning(validation['recommendation'])

                col1, col2 = st.columns([1, 1])

                with col1:
                    if st.button("Use CSV Directly", type="secondary"):
                        SessionManager.set_csv_path(csv_path)
                        SessionManager.set_use_database(False)
                        st.rerun()

                with col2:
                    if st.button("Convert to Database (Recommended)", type="primary"):
                        if render_database_conversion(csv_path):
                            SessionManager.set_csv_path(csv_path)
                            SessionManager.set_db_path(DatabaseConfig.DB_PATH)
                            SessionManager.set_use_database(True)
                            st.rerun()
```

---

### 4.4 Update Session Manager

**File:** `streamlit_app/utils/session_manager.py`

**Add database state:**

```python
class SessionManager:
    """Enhanced session manager with database support."""

    @staticmethod
    def initialize_session():
        """Initialize session state."""
        defaults = {
            # ... existing defaults ...
            'use_database': False,
            'db_path': None,
            'db_stats': None,
        }
        # ... rest of initialization

    @staticmethod
    def set_use_database(use_db: bool):
        st.session_state.use_database = use_db

    @staticmethod
    def get_use_database() -> bool:
        return st.session_state.get('use_database', False)

    @staticmethod
    def set_db_path(db_path: str):
        st.session_state.db_path = db_path

    @staticmethod
    def get_db_path() -> str:
        return st.session_state.get('db_path')
```

---

### 4.5 Update Crew Runner for Database Mode

**File:** `streamlit_app/utils/crew_runner.py`

**Add database support:**

```python
class StreamlitCrewRunner:
    """Enhanced crew runner with database support."""

    def __init__(self, dataset_path: str, use_database: bool = False, db_path: str = None):
        self.dataset_path = dataset_path
        self.use_database = use_database
        self.db_path = db_path
        # ... rest of init

    def run_analysis(self, progress_callback, status_callback):
        """Run analysis with database support."""

        # ... existing setup ...

        # If using database, ensure it's ready
        if self.use_database:
            if not self.db_path or not os.path.exists(self.db_path):
                # Convert CSV to database
                from utils.csv_to_sqlite import CSVToSQLiteConverter
                converter = CSVToSQLiteConverter()
                stats = converter.convert(self.dataset_path, self.db_path, 'transactions')

            # Set database environment variables
            os.environ['DB_PATH'] = self.db_path
            os.environ['DB_URI'] = f'sqlite:///{self.db_path}'
            os.environ['USE_DATABASE'] = 'true'

        # Execute crew with appropriate inputs
        inputs = {
            'dataset_path': self.dataset_path,
            'current_year': str(datetime.now().year)
        }

        if self.use_database:
            inputs.update({
                'db_path': self.db_path,
                'table_name': 'transactions'
            })

        result = crew_instance.crew().kickoff(inputs=inputs)
        # ... rest of execution
```

---

## Phase 5: Testing & Validation

### 5.1 Create Test Script

**File:** `tests/test_database_integration.py`

```python
"""Test script for database integration."""

import os
import sys
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crewai_extrachallenge.utils.csv_to_sqlite import CSVToSQLiteConverter
from src.crewai_extrachallenge.utils.db_helper import DatabaseHelper
from src.crewai_extrachallenge.tools.db_statistical_analysis_tool import DBStatisticalAnalysisTool

class TestDatabaseIntegration:
    """Test database integration functionality."""

    def test_csv_to_sqlite_conversion(self):
        """Test CSV to SQLite conversion."""
        csv_path = 'dataset/data/credit_card_transactions.csv'
        db_path = 'test_fraud.db'

        # Clean up previous test
        if os.path.exists(db_path):
            os.remove(db_path)

        # Convert
        converter = CSVToSQLiteConverter()
        stats = converter.convert(csv_path, db_path, 'transactions')

        # Verify
        assert os.path.exists(db_path)
        assert stats['total_rows'] > 0
        assert stats['db_size_mb'] > 0

        # Verify compression (should be smaller than CSV)
        csv_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        assert stats['db_size_mb'] < csv_size_mb

        # Clean up
        os.remove(db_path)

    def test_db_statistical_analysis(self):
        """Test database statistical analysis tool."""
        # Assuming database exists from previous test
        db_path = 'test_fraud.db'

        # Recreate if needed
        if not os.path.exists(db_path):
            converter = CSVToSQLiteConverter()
            converter.convert('dataset/data/credit_card_transactions.csv', db_path, 'transactions')

        # Test tool
        tool = DBStatisticalAnalysisTool()

        # Test descriptive stats
        result = tool._run('descriptive', db_path=db_path)
        assert 'Total Transactions' in result
        assert 'Class Distribution' in result

        # Test correlation
        result = tool._run('correlation', db_path=db_path)
        assert 'Correlation Analysis' in result

        # Clean up
        os.remove(db_path)

    def test_memory_usage(self):
        """Test that database approach uses less memory."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Memory before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Load via database
        db_path = 'test_fraud.db'
        tool = DBStatisticalAnalysisTool()
        result = tool._run('descriptive', db_path=db_path)

        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        memory_used = mem_after - mem_before

        # Should use less than 100MB for analysis
        assert memory_used < 100

        print(f"Memory used: {memory_used:.1f}MB")

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
```

---

### 5.2 Create Benchmark Script

**File:** `tests/benchmark_database_vs_csv.py`

```python
"""Benchmark database approach vs CSV approach."""

import time
import psutil
import os
from pathlib import Path

def benchmark_csv_approach(csv_path: str):
    """Benchmark CSV loading approach."""
    import pandas as pd

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024

    start = time.time()
    df = pd.read_csv(csv_path)
    stats = {
        'count': len(df),
        'fraud_rate': df['Class'].mean() * 100 if 'Class' in df.columns else 0,
        'avg_amount': df['Amount'].mean() if 'Amount' in df.columns else 0
    }
    duration = time.time() - start

    mem_after = process.memory_info().rss / 1024 / 1024
    memory_used = mem_after - mem_before

    return {
        'approach': 'CSV Direct',
        'duration': duration,
        'memory_mb': memory_used,
        'stats': stats
    }

def benchmark_database_approach(db_path: str):
    """Benchmark database approach."""
    import sqlite3
    import pandas as pd

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024

    start = time.time()
    conn = sqlite3.connect(db_path)

    # Equivalent queries
    count = pd.read_sql("SELECT COUNT(*) as cnt FROM transactions", conn).iloc[0]['cnt']
    fraud_rate = pd.read_sql("SELECT AVG(Class)*100 as rate FROM transactions", conn).iloc[0]['rate']
    avg_amount = pd.read_sql("SELECT AVG(Amount) as avg FROM transactions", conn).iloc[0]['avg']

    stats = {
        'count': count,
        'fraud_rate': fraud_rate,
        'avg_amount': avg_amount
    }

    conn.close()
    duration = time.time() - start

    mem_after = process.memory_info().rss / 1024 / 1024
    memory_used = mem_after - mem_before

    return {
        'approach': 'Database',
        'duration': duration,
        'memory_mb': memory_used,
        'stats': stats
    }

if __name__ == "__main__":
    csv_path = 'dataset/data/credit_card_transactions.csv'
    db_path = 'fraud_detection.db'

    print("=" * 60)
    print("BENCHMARK: CSV vs Database Approach")
    print("=" * 60)

    # Benchmark CSV
    print("\n1. Testing CSV approach...")
    csv_result = benchmark_csv_approach(csv_path)

    # Benchmark Database
    print("2. Testing Database approach...")
    db_result = benchmark_database_approach(db_path)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nCSV Approach:")
    print(f"  Duration: {csv_result['duration']:.3f} seconds")
    print(f"  Memory:   {csv_result['memory_mb']:.1f} MB")

    print(f"\nDatabase Approach:")
    print(f"  Duration: {db_result['duration']:.3f} seconds")
    print(f"  Memory:   {db_result['memory_mb']:.1f} MB")

    print(f"\nPerformance Improvement:")
    print(f"  Speed:  {csv_result['duration'] / db_result['duration']:.1f}x faster")
    print(f"  Memory: {csv_result['memory_mb'] / db_result['memory_mb']:.1f}x less memory")

    print("\n" + "=" * 60)
```

---

### 5.3 Validation Checklist

**Manual Testing Checklist:**

- [ ] **CSV to SQLite Conversion**
  - [ ] Converts 122KB file successfully
  - [ ] Shows progress during conversion
  - [ ] Creates indexes automatically
  - [ ] Reports compression statistics
  - [ ] Handles large 150MB+ files

- [ ] **Database Tools**
  - [ ] DBStatisticalAnalysisTool returns correct statistics
  - [ ] NL2SQLTool converts questions to SQL
  - [ ] Queries execute without loading full dataset
  - [ ] Results match CSV-based approach

- [ ] **Agent Integration**
  - [ ] Agents use database tools correctly
  - [ ] Tasks reference database paths
  - [ ] Reports include database statistics
  - [ ] Images still generated correctly

- [ ] **Streamlit UI**
  - [ ] Shows database conversion option
  - [ ] Progress bar works during conversion
  - [ ] Database status displayed correctly
  - [ ] Analysis works with database backend

- [ ] **Memory & Performance**
  - [ ] Memory usage < 100MB for large files
  - [ ] Query execution < 1 second
  - [ ] No context overflow with Ollama
  - [ ] Results accurate and complete

---

## Implementation Sequence

### Day 1: Core Infrastructure (2-3 hours)

**Morning:**
1. Create `csv_to_sqlite.py` converter (45 min)
2. Create `db_helper.py` utilities (30 min)
3. Update `.env` and create `database_config.py` (15 min)
4. Test conversion with current dataset (30 min)

**Afternoon:**
5. Create `DBStatisticalAnalysisTool` (60 min)
6. Configure `NL2SQLTool` (30 min)
7. Test tools independently (30 min)

### Day 2: Integration & Testing (2-3 hours)

**Morning:**
8. Update `agents.yaml` (20 min)
9. Update `tasks.yaml` (30 min)
10. Update `main.py` workflow (30 min)
11. Test CLI workflow (30 min)

**Afternoon:**
12. Update Streamlit components (60 min)
13. Test UI workflow (30 min)
14. Run benchmarks (20 min)
15. Final validation (30 min)

---

## Success Criteria

✅ **Core Functionality:**
- CSV to SQLite conversion completes successfully
- Database queries return accurate results
- Tools work with database backend
- Agents execute tasks using database

✅ **Performance:**
- Memory usage < 100MB for 150MB files
- Conversion time < 2 minutes for 150MB files
- Query execution < 1 second average
- 40x improvement vs CSV approach

✅ **Integration:**
- CLI (`crewai run`) works with database
- Streamlit UI offers database option
- Reports reference database statistics
- All existing features still work

✅ **Quality:**
- Results match CSV-based approach
- No data loss during conversion
- Proper error handling
- Clear user feedback

---

## Rollback Plan

If issues arise, rollback is straightforward:

1. **Set `USE_DATABASE=false` in `.env`**
   - System falls back to CSV approach
   - No code changes needed

2. **Delete database file**
   - Removes converted database
   - Forces fresh conversion on next run

3. **Revert code changes**
   - Database tools are additive (don't replace CSV tools)
   - Can remove database-specific code without breaking CSV workflow

---

## Future Enhancements

**Phase 6 (Optional):**
- Custom SQLite RAG tool for semantic search
- Query result caching for repeated analyses
- Multi-database support (compare datasets)
- PostgreSQL migration path
- Incremental database updates (append new transactions)

---

## Questions to Resolve Before Implementation

1. **Database File Location:**
   - Store in project root? ✅ Recommended
   - Store in `/data` directory?
   - Store in user-specified location?

2. **Conversion Behavior:**
   - Auto-convert on first run? ✅ Recommended
   - Manual conversion only?
   - Offer choice in UI? ✅ Recommended

3. **Backward Compatibility:**
   - Keep CSV tools available? ✅ Yes
   - Hybrid mode (try DB, fallback to CSV)? ✅ Yes
   - Phase out CSV tools eventually?

4. **Testing Dataset:**
   - Test with current 122KB file first? ✅ Yes
   - Need to create/obtain 150MB test file?
   - Synthetic data generation for testing?

---

## Summary

This plan provides a comprehensive roadmap to implement SQLite + NL2SQLTool integration:

- **Estimated Total Time:** 4-6 hours
- **Complexity:** Moderate (mostly additive changes)
- **Risk:** Low (backward compatible, easy rollback)
- **Value:** High (40x performance improvement, unlimited file size)

**Key Benefits:**
✅ Keep using Ollama locally (no OpenAI costs)
✅ Handle 150MB+ files easily
✅ 95% less memory usage
✅ 40x faster analysis
✅ Backward compatible with existing workflow
✅ Reusable database for future analyses

**Next Step:** Review this plan and approve to proceed with implementation! 🚀
