# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CrewAI-based credit card fraud detection system using multiple AI agents to analyze transaction datasets and classify them as fraudulent or legitimate. The project uses numerical features (V1-V28 from PCA transformation, Time, Amount) to train on labeled data and classify unlabeled transactions.

The system leverages CrewAI tools, particularly the CSV RAG Search tool, to perform semantic analysis of large CSV datasets containing credit card transactions.

## Development Commands

### Running the Application
```bash
# Run the crew with default configuration
crewai run

# Alternative command
uv run crewai_extrachallenge

# Or using the script entry point
uv run run_crew
```

### Training and Testing
```bash
# Train the crew for N iterations with a filename
uv run train <n_iterations> <filename>

# Test the crew with N iterations and evaluation LLM
uv run test <n_iterations> <eval_llm>

# Replay crew execution from a specific task ID
uv run replay <task_id>
```

### Dependency Management
```bash
# Install dependencies using CrewAI CLI
crewai install

# Or use uv directly
uv sync
```

## Architecture Overview

### Core Structure
- **Entry Point**: `src/crewai_extrachallenge/main.py` - Contains run(), train(), test(), and replay() functions
- **Crew Definition**: `src/crewai_extrachallenge/crew.py` - Defines the CrewaiExtrachallenge class with agents and tasks
- **Configuration**: `src/crewai_extrachallenge/config/` - YAML files defining agents and tasks
- **Tools**: `src/crewai_extrachallenge/tools/` - Custom tools for agents
- **Knowledge**: `knowledge/user_preference.txt` - User context for agents

### Agent Architecture
The system uses a CrewBase decorator pattern with:
- **Agents**: Defined via @agent decorators, configured through `config/agents.yaml`
- **Tasks**: Defined via @task decorators, configured through `config/tasks.yaml`
- **Crew**: Sequential process by default, configurable to hierarchical

### Current Agents (Template - needs adaptation for fraud detection)
- `researcher`: Data research and analysis agent
- `reporting_analyst`: Report generation agent

### Configuration Pattern
- Agents and tasks use YAML configuration with variable interpolation (`{topic}`, `{current_year}`)
- Input variables are passed through the `inputs` parameter in main functions
- Output files are generated based on task configuration (e.g., `report.md`)

### Key Dependencies
- `crewai[tools]`: Core framework and tools
- `crewai-tools`: Additional tool library
- Python 3.10-3.13 supported

## Development Notes

### Environment Setup
- Requires `OPENAI_API_KEY` in `.env` file
- Uses UV for dependency management
- Entry points defined in `pyproject.toml`

### Code Organization
- Agent definitions use the `@agent` decorator and reference YAML configs
- Task definitions use the `@task` decorator with output file specifications
- Custom tools inherit from `BaseTool` with Pydantic input schemas

## CrewAI Tools Integration

### CSV RAG Search Tool
This project uses the CSV RAG Search tool for semantic analysis of credit card transaction datasets:

**Installation**: Already included with `crewai[tools]` dependency

**Usage Patterns**:
```python
from crewai_tools import CSVSearchTool

# For specific CSV file
csv_tool = CSVSearchTool(csv='path/to/transactions.csv')

# For runtime CSV file selection
csv_tool = CSVSearchTool()
```

**Key Features**:
- Semantic search within CSV file contents using RAG
- Supports large transaction datasets
- Customizable LLM providers (OpenAI, Google, Anthropic)
- Configurable embedding models

**Use Cases for Fraud Detection**:
- Query transaction patterns: "Find transactions with unusual amounts"
- Identify anomalies: "Show transactions that deviate from normal patterns"
- Feature analysis: "Analyze V1-V28 PCA components for fraud indicators"
- Time-based queries: "Find suspicious transactions in specific time windows"

### Tool Integration in Agents
Tools are assigned to agents in the YAML configuration:
```yaml
agent_name:
  tools:
    - CSVSearchTool
```

### Extending the System
To adapt for fraud detection:
1. Modify `config/agents.yaml` to define fraud detection agents with CSV RAG Search tool
2. Update `config/tasks.yaml` for fraud detection tasks using CSV analysis
3. Create additional CSV processing tools in `tools/` directory if needed
4. Update input parameters in `main.py` to accept dataset paths instead of topic strings