# Credit Card Fraud Detection Crew

Welcome to the Credit Card Fraud Detection Crew project, powered by [crewAI](https://crewai.com). This multi-agent AI system analyzes credit card transaction datasets (supporting files up to 150MB+) using database-optimized tools and collaborative AI agents to detect fraudulent patterns and generate comprehensive analysis reports.

## Project Overview

This application provides both **CLI** and **Streamlit UI** interfaces for fraud detection analysis on CSV datasets containing credit card transactions with the following characteristics:

- **Numerical Data Only**: All features in the dataset are numerical values
- **PCA-Transformed Features**: Features V1 through V28 are principal components obtained via PCA transformation for confidentiality
- **Original Features**:
  - `Time`: Seconds elapsed between each transaction and the first transaction in the dataset
  - `Amount`: Transaction amount (useful for cost analysis)
- **Classification Target**:
  - `Class`: Binary classification where 1 indicates fraud and 0 indicates legitimate transactions

## Key Features

### 🚀 Performance Optimizations
- **Database-Optimized Analysis**: SQLite-backed statistical analysis for memory-efficient processing of large datasets (150MB+)
- **Hybrid Data Tool**: Smart data access that combines database queries with intelligent sampling
- **Guaranteed Visualizations**: Deterministic image generation ensuring all 6 fraud detection charts are created reliably

### 📊 Streamlit UI
- **Phase 1**: Dataset configuration with file upload or path specification
- **Phase 2**: Automated report generation with real-time progress tracking
- **Phase 3**: Interactive AI-powered chat interface for Q&A about analysis results

### 🎯 Analysis Pipeline
The crew of AI agents performs:
1. **Statistical Analysis**: Descriptive statistics, data quality assessment, distribution analysis
2. **Pattern Recognition**: Correlation analysis, temporal clustering, feature importance identification
3. **Classification Analysis**: Outlier detection, risk assessment, fraud probability scoring
4. **Comprehensive Reporting**: Markdown reports with embedded visualizations and actionable recommendations

### 🤖 AI Agent Architecture
- **Data Analyst Agent**: Database-optimized statistical analysis
- **Pattern Recognition Agent**: Fraud pattern identification using correlation and outlier detection
- **Classification Agent**: Transaction risk assessment and classification
- **Reporting Agent**: Professional markdown report generation with verified image references
- **Chat Analyst Agent**: Interactive Q&A using the same AI models

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

### Step 1: Install UV

```bash
pip install uv
```

### Step 2: Install Dependencies

Navigate to your project directory and install the dependencies:

```bash
crewai install
# or
uv sync
```

### Step 3: Configure Environment

Create a `.env` file in the project root with the following configuration:

```bash
# LLM Configuration
MODEL=gpt-4-turbo-preview          # Or your preferred model (gpt-4o, claude-3-5-sonnet, etc.)
OPENAI_API_KEY=your_api_key_here   # Required for OpenAI models
API_BASE=http://localhost:11434    # Optional: For local models (Ollama, etc.)

# Temperature Settings (0.0 = deterministic, 1.0 = creative)
TEMP_DATA_ANALYST=0.1              # Low temp for precise analysis
TEMP_PATTERN_AGENT=0.3             # Medium temp for pattern discovery
TEMP_CLASSIFICATION=0.1            # Low temp for consistent decisions
TEMP_REPORTING=0.2                 # Slight creativity for clear writing

# Dataset Configuration
DATASET_PATH=/path/to/your/dataset.csv

# Database Configuration (for large datasets)
USE_DATABASE=true                  # Enable database mode for 150MB+ files
DB_PATH=fraud_detection.db         # SQLite database file path
DB_TABLE=transactions              # Table name for transaction data

# Optional: Disable telemetry
CREWAI_TRACING_ENABLED=false
```

### Customizing

- Modify `src/crewai_extrachallenge/config/agents.yaml` to define your agents
- Modify `src/crewai_extrachallenge/config/tasks.yaml` to define your tasks
- Modify `src/crewai_extrachallenge/crew.py` to add your own logic, tools and specific args
- Modify `src/crewai_extrachallenge/main.py` to add custom inputs for your agents and tasks

## Running the Project

### Option 1: Streamlit UI (Recommended)

Launch the interactive web interface:

```bash
streamlit run streamlit_app/app.py
```

The UI provides a guided 3-phase workflow:
1. **Phase 1**: Configure your dataset (upload CSV or specify path)
2. **Phase 2**: Generate fraud detection report with visualizations
3. **Phase 3**: Interactive AI chat to ask questions about your results

Features:
- Real-time progress tracking
- Dataset validation with helpful error messages
- Automatic database conversion for large files
- Quick action buttons for common questions
- LLM configuration display in sidebar

### Option 2: Command Line Interface

Run the fraud detection analysis via CLI:

```bash
# Run with default configuration
crewai run

# Alternative commands
uv run crewai_extrachallenge
uv run run_crew
```

This initializes the fraud detection crew and generates a comprehensive analysis report at `reports/fraud_detection_report.md` with 6 visualization charts in `reports/images/`.

## Understanding Your Crew

The Credit Card Fraud Detection Crew is composed of specialized AI agents, each with unique roles in the fraud detection pipeline:

### Agents

1. **Data Analyst Agent** (`data_analyst`)
   - Role: Database-optimized statistical analysis
   - Tools: DB Statistical Analysis Tool, Hybrid Data Tool, Guaranteed Visualizations Tool
   - Output: Descriptive statistics, data quality assessment, distribution analysis
   - Generates: `fraud_comparison.png`, `correlation_heatmap.png`

2. **Pattern Recognition Agent** (`pattern_recognition_agent`)
   - Role: Fraud pattern identification
   - Tools: DB Statistical Analysis Tool, Hybrid Data Tool, Guaranteed Visualizations Tool
   - Output: Correlation analysis, temporal patterns, feature importance
   - Generates: `scatter.png`, `time_series.png`, `feature_importance.png`, `box_plot.png`

3. **Classification Agent** (`classification_agent`)
   - Role: Transaction risk assessment
   - Tools: DB Statistical Analysis Tool, Hybrid Data Tool, Guaranteed Visualizations Tool
   - Output: Outlier detection, risk scoring, confidence metrics
   - Generates: `amount_histogram.png`

4. **Reporting Agent** (`reporting_analyst`)
   - Role: Professional markdown report generation
   - Tools: Image Verification Tool, Markdown Formatter Tool
   - Output: Comprehensive fraud detection report with embedded visualizations
   - File: `reports/fraud_detection_report.md`

5. **Chat Analyst Agent** (Streamlit UI only)
   - Role: Interactive Q&A about analysis results
   - Tools: DB Statistical Analysis Tool, Hybrid Data Tool
   - Output: Context-aware responses to user questions

### Configuration Files

- `src/crewai_extrachallenge/config/agents.yaml`: Agent role definitions and backstories
- `src/crewai_extrachallenge/config/tasks.yaml`: Task descriptions and expected outputs
- `src/crewai_extrachallenge/crew.py`: Agent initialization with LLM and tool configuration

## Custom Tools

This project includes several specialized tools for efficient fraud detection analysis:

### 1. Database Statistical Analysis Tool (`DBStatisticalAnalysisTool`)
Memory-efficient statistical analysis using SQLite for large datasets (150MB+).

**Analysis Types:**
- `descriptive`: Count, mean, std, min/max, percentiles for all numeric columns
- `correlation`: Pearson correlations between features and Class column
- `outliers`: IQR-based outlier detection with statistics
- `distribution`: Class distribution and amount statistics by class
- `data_quality`: Missing values, duplicates, data types

**Benefits:**
- Processes datasets in chunks to avoid memory overflow
- Performs SQL aggregations for fast computation
- Supports datasets from 100KB to 150MB+

### 2. Hybrid Data Tool (`HybridDataTool`)
Smart data access combining database queries with intelligent sampling.

**Features:**
- Automatic database vs. CSV selection based on file size
- Smart sampling strategies for large datasets
- Targeted data retrieval by column or condition
- Row count and basic statistics

### 3. Guaranteed Visualizations Tool (`GuaranteedVisualizationsTool`)
Ensures deterministic generation of all required fraud detection charts.

**Generated Visualizations:**
- Correlation heatmap (`correlation_heatmap.png`)
- Fraud vs legitimate comparison (`fraud_comparison.png`)
- Feature scatter plot (`scatter.png`)
- Time series analysis (`time_series.png`)
- Feature importance chart (`feature_importance.png`)
- Amount histogram (`amount_histogram.png`)

**How it works:**
- Programmatically generates all charts for a task in a single tool call
- Removes dependency on LLM reliably following multi-step instructions
- Works with any LLM model (including o1-mini, gpt-4, etc.)

### 4. Image Verification Tool (`ImageVerificationTool`)
Verifies and lists available images with exact markdown references.

**Actions:**
- `list_available`: Returns table of all generated images with markdown syntax
- `verify_exists`: Checks if a specific image file exists

### 5. Markdown Formatter Tool (`MarkdownFormatterTool`)
Formats and validates markdown reports for proper rendering.

**Features:**
- Automatic image path correction (ensures `./images/` prefix)
- Proper spacing around headers, lists, code blocks
- Trailing newline compliance (MD047)
- Triple backtick removal

### 6. Task Validation Tool (`TaskValidationTool`)
Validates that required images and outputs were generated successfully.

## Output

### Generated Reports

After running the analysis, you'll find:

**Main Report:** `reports/fraud_detection_report.md`
- Executive summary with key findings
- Methodology description
- Data analysis results with statistics
- Pattern recognition insights
- Classification analysis
- Risk assessment and business recommendations

**Visualizations:** `reports/images/`
- `correlation_heatmap.png`: Feature correlation matrix
- `fraud_comparison.png`: Fraud vs legitimate transaction counts
- `scatter.png`: Feature relationship scatter plots
- `time_series.png`: Temporal fraud patterns
- `feature_importance.png`: Most predictive features
- `amount_histogram.png`: Transaction amount distribution

**Database:** `fraud_detection.db` (SQLite)
- Cached transaction data for fast re-analysis
- Automatically created for datasets > 50MB

## Advanced Usage

### Training and Testing

```bash
# Train the crew for N iterations
uv run train <n_iterations> <filename>

# Test with evaluation LLM
uv run test <n_iterations> <eval_llm>

# Replay specific task execution
uv run replay <task_id>
```

### LLM Model Selection

The system supports multiple LLM providers:

**OpenAI Models:**
```bash
MODEL=gpt-4-turbo-preview
MODEL=gpt-4o
MODEL=gpt-4
```

**Anthropic Claude (via LiteLLM):**
```bash
MODEL=claude-3-5-sonnet-20241022
```

**Local Models (Ollama):**
```bash
MODEL=ollama/llama3.1:8b
API_BASE=http://localhost:11434
```

**Important:** Models like `o1-mini` and `o1-preview` have limited tool-calling capabilities and may not generate images reliably. Use GPT-4 variants for best results.

### Dataset Size Recommendations

- **< 50MB**: Works with both CSV and database mode
- **50MB - 150MB**: Automatically uses database mode for efficiency
- **> 150MB**: May require chunking or sampling strategies

## Troubleshooting

### Images Not Generated
- **Cause:** Using o1-mini or models with limited tool-calling
- **Solution:** Switch to `gpt-4-turbo-preview` or `gpt-4o` in `.env`

### Chat Interface 404 Error
- **Cause:** Chat agent using different LLM configuration than main crew
- **Solution:** Fixed in latest version - chat agent now inherits LLM config

### Memory Issues with Large Files
- **Cause:** CSV file too large to load into memory
- **Solution:** Set `USE_DATABASE=true` in `.env` for automatic database conversion

### Markdown Not Rendering
- **Cause:** Missing trailing newline (MD047)
- **Solution:** Fixed in Markdown Formatter Tool - reports now end with proper newline

## Project Structure

```
crewai_extrachallenge/
├── src/crewai_extrachallenge/
│   ├── config/
│   │   ├── agents.yaml          # Agent definitions
│   │   └── tasks.yaml           # Task definitions
│   ├── tools/                   # Custom CrewAI tools
│   │   ├── db_statistical_analysis_tool.py
│   │   ├── hybrid_data_tool.py
│   │   ├── guaranteed_visualizations.py
│   │   ├── image_verification_tool.py
│   │   ├── markdown_formatter_tool.py
│   │   └── visualization_tool.py
│   ├── crew.py                  # Crew configuration
│   └── main.py                  # Entry points
├── streamlit_app/
│   ├── app.py                   # Main Streamlit app
│   ├── components/              # UI components
│   └── utils/                   # Utilities and chat agent
├── reports/                     # Generated reports and images
├── dataset/                     # Sample datasets
├── .env                         # Environment configuration
└── README.md
```

## Support

For support, questions, or feedback:
- **CrewAI Documentation**: [docs.crewai.com](https://docs.crewai.com)
- **CrewAI GitHub**: [github.com/joaomdmoura/crewai](https://github.com/joaomdmoura/crewai)
- **CrewAI Discord**: [discord.com/invite/X4JWnZnxPb](https://discord.com/invite/X4JWnZnxPb)
- **Chat with Docs**: [chatg.pt/DWjSBZn](https://chatg.pt/DWjSBZn)

---

**Built with ❤️ using CrewAI - Let's build intelligent fraud detection systems together!**
