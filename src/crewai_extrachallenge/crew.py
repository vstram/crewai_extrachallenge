from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import CSVSearchTool
from typing import List
from .tools.visualization_tool import VisualizationTool
from .tools.statistical_analysis_tool import StatisticalAnalysisTool
from .tools.image_verification_tool import ImageVerificationTool
from .tools.markdown_formatter_tool import MarkdownFormatterTool
from .tools.task_validation_tool import TaskValidationTool
# Database-optimized tools
from .tools.db_statistical_analysis_tool import DBStatisticalAnalysisTool
from .tools.hybrid_data_tool import HybridDataTool
import os
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class CrewaiExtrachallenge():
    """Credit Card Fraud Detection Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # Fraud detection agents with specialized tools
    @agent
    def data_analyst(self) -> Agent:
        # LLM configuration for precise analytical work
        analytical_llm = LLM(
            model="ollama/llama3.1:8b",
            temperature=0.1,  # Low randomness for accurate calculations
            base_url="http://localhost:11434"
        )

        return Agent(
            config=self.agents_config['data_analyst'], # type: ignore[index]
            llm=analytical_llm,
            tools=[
                DBStatisticalAnalysisTool(),  # Database-optimized statistics
                HybridDataTool(),              # Smart data access (DB or CSV)
                VisualizationTool(),
                TaskValidationTool()
            ],
            verbose=True
        )

    @agent
    def pattern_recognition_agent(self) -> Agent:
        # LLM configuration for creative pattern discovery
        pattern_discovery_llm = LLM(
            model="ollama/llama3.1:8b",
            temperature=0.3,  # Medium randomness for pattern discovery
            base_url="http://localhost:11434"
        )

        return Agent(
            config=self.agents_config['pattern_recognition_agent'], # type: ignore[index]
            llm=pattern_discovery_llm,
            tools=[
                DBStatisticalAnalysisTool(),  # Database-optimized correlation & outliers
                HybridDataTool(),              # Smart data sampling
                VisualizationTool(),
                TaskValidationTool()
            ],
            verbose=True
        )

    @agent
    def classification_agent(self) -> Agent:
        # LLM configuration for consistent classification decisions
        classification_llm = LLM(
            model="ollama/llama3.1:8b",
            temperature=0.1,  # Low randomness for reliable classification
            base_url="http://localhost:11434"
        )

        return Agent(
            config=self.agents_config['classification_agent'], # type: ignore[index]
            llm=classification_llm,
            tools=[
                DBStatisticalAnalysisTool(),  # Database-optimized outliers & distribution
                HybridDataTool(),              # Smart data sampling
                VisualizationTool(),
                TaskValidationTool()
            ],
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        # LLM configuration for clear communication and reporting
        reporting_llm = LLM(
            model="ollama/llama3.1:8b",
            temperature=0.2,  # Slight creativity for clear explanations
            base_url="http://localhost:11434"
        )

        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
            llm=reporting_llm,
            tools=[ImageVerificationTool(), MarkdownFormatterTool()],
            verbose=True
        )

    # Fraud detection task pipeline
    @task
    def data_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_analysis_task'], # type: ignore[index]
        )

    @task
    def pattern_recognition_task(self) -> Task:
        return Task(
            config=self.tasks_config['pattern_recognition_task'], # type: ignore[index]
        )

    @task
    def classification_task(self) -> Task:
        return Task(
            config=self.tasks_config['classification_task'], # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            output_file='reports/fraud_detection_report.md',
            context=[self.data_analysis_task(), self.pattern_recognition_task(), self.classification_task()]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CrewaiExtrachallenge crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            max_iter=2,  # Further limit iterations to prevent infinite loops
            max_rpm=10,  # Limit requests per minute
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
