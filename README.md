# Credit Card Fraud Detection Crew

Welcome to the Credit Card Fraud Detection Crew project, powered by [crewAI](https://crewai.com). This multi-agent AI system is specifically designed to analyze large CSV datasets containing credit card transactions and classify them as fraudulent or legitimate. The project leverages the collaborative intelligence of specialized AI agents to perform statistical and mathematical analysis on transaction data for accurate fraud detection.

## Project Overview

This application analyzes CSV datasets containing credit card transactions with the following characteristics:

- **Numerical Data Only**: All features in the dataset are numerical values
- **PCA-Transformed Features**: Features V1 through V28 are principal components obtained via PCA transformation for confidentiality
- **Original Features**:
  - `Time`: Seconds elapsed between each transaction and the first transaction in the dataset
  - `Amount`: Transaction amount (useful for cost analysis)
- **Classification Target**:
  - `Class`: Binary classification where 1 indicates fraud and 0 indicates legitimate transactions

## Mission

The crew of AI agents is designed to:
1. Train on labeled datasets (with `Class` feature) to understand fraud patterns
2. Analyze unlabeled datasets (without `Class` feature) using statistical and mathematical methods
3. Classify transactions as fraudulent or legitimate based on learned patterns
4. Provide comprehensive analysis reports with confidence metrics and insights

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```
### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/crewai_extrachallenge/config/agents.yaml` to define your agents
- Modify `src/crewai_extrachallenge/config/tasks.yaml` to define your tasks
- Modify `src/crewai_extrachallenge/crew.py` to add your own logic, tools and specific args
- Modify `src/crewai_extrachallenge/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```

This command initializes the crewai_extrachallenge Crew, assembling the agents and assigning them tasks as defined in your configuration.

This will initialize the fraud detection crew and generate analysis reports on your credit card transaction datasets.

## Understanding Your Crew

The Credit Card Fraud Detection Crew is composed of specialized AI agents, each with unique roles in the fraud detection pipeline. These agents collaborate on analyzing transaction data, identifying patterns, and classifying transactions. The crew includes:

- **Data Analyst Agent**: Performs statistical analysis and data preprocessing
- **Pattern Recognition Agent**: Identifies fraud patterns and anomalies
- **Classification Agent**: Makes final fraud/legitimate classifications
- **Reporting Agent**: Generates comprehensive analysis reports

Tasks are defined in `config/tasks.yaml`, and agent configurations are outlined in `config/agents.yaml`.

## Support

For support, questions, or feedback regarding the CrewaiExtrachallenge Crew or crewAI.
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's build intelligent fraud detection systems together with the power and simplicity of crewAI.
