from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import CSVSearchTool
from typing import List
from .tools.visualization_tool import VisualizationTool
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
        return Agent(
            config=self.agents_config['data_analyst'], # type: ignore[index]
            tools=[CSVSearchTool(), VisualizationTool()],
            verbose=True
        )

    @agent
    def pattern_recognition_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['pattern_recognition_agent'], # type: ignore[index]
            tools=[CSVSearchTool(), VisualizationTool()],
            verbose=True
        )

    @agent
    def classification_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['classification_agent'], # type: ignore[index]
            tools=[VisualizationTool()],
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
            tools=[VisualizationTool()],
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
            output_file='reports/fraud_detection_report.md'
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
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
