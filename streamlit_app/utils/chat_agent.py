import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent directory to import CrewAI classes
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

from crewai import Agent, Task, Crew
from crewai.tools.base_tool import BaseTool
from crewai_tools import CSVSearchTool


class ChatAnalystAgent:
    """
    CrewAI agent specialized for interactive chat and Q&A about fraud detection analysis.
    This agent can answer questions about datasets, patterns, and recommendations.
    """

    def __init__(self, dataset_path: str, analysis_results: Dict[str, Any]):
        self.dataset_path = dataset_path
        self.analysis_results = analysis_results
        self.csv_tool = CSVSearchTool(csv=dataset_path) if os.path.exists(dataset_path) else None

        # Create the chat agent
        self.agent = self._create_chat_agent()

    def _create_chat_agent(self) -> Agent:
        """Create the chat analyst agent with appropriate tools and configuration."""

        tools = []
        if self.csv_tool:
            tools.append(self.csv_tool)

        return Agent(
            role="Fraud Detection Chat Analyst",
            goal="Provide insightful, context-aware answers about fraud detection analysis results and dataset patterns",
            backstory="""You are an expert fraud detection analyst with deep knowledge of credit card transaction patterns,
            statistical analysis, and machine learning techniques. You have just completed a comprehensive fraud detection
            analysis and can answer detailed questions about the results, patterns found, and recommendations for fraud prevention.

            You have access to the analyzed dataset and can perform additional queries as needed to answer user questions.
            Your responses should be accurate, informative, and actionable.""",
            tools=tools,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            memory=True
        )

    def ask_question(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Ask a question to the chat agent and get a detailed response.

        Args:
            question: The user's question
            context: Additional context from the analysis results

        Returns:
            Formatted response from the agent
        """
        try:
            # Prepare context information
            context_info = self._prepare_context(context)

            # Create a task for answering the question
            task = Task(
                description=f"""
                Answer the following question about the fraud detection analysis:

                **User Question:** {question}

                **Analysis Context:**
                {context_info}

                **Instructions:**
                1. Provide a comprehensive, accurate answer based on the analysis results
                2. Use specific data points and insights from the dataset when relevant
                3. If the question requires dataset exploration, use the CSV search tool
                4. Structure your response with clear headings and bullet points
                5. Include actionable insights or recommendations when appropriate
                6. If you cannot answer definitively, explain what additional analysis would be needed

                **Response Format:**
                - Use markdown formatting for clarity
                - Include relevant statistics and numbers
                - Provide practical, actionable insights
                - Keep the tone professional but accessible
                """,
                agent=self.agent,
                expected_output="A detailed, well-structured answer to the user's question with specific insights and recommendations"
            )

            # Create and run a single-agent crew for this question
            crew = Crew(
                agents=[self.agent],
                tasks=[task],
                verbose=False,
                process="sequential"
            )

            # Execute the task
            result = crew.kickoff()

            return self._format_response(str(result))

        except Exception as e:
            # Fallback to a helpful error message
            return f"""❌ **Unable to process question at the moment.**

**Error:** {str(e)}

**Suggested Actions:**
- Try rephrasing your question
- Check if the analysis completed successfully
- Ensure the dataset is accessible

**Alternative:** Use the quick action buttons above for common questions."""

    def _prepare_context(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Prepare comprehensive context information for the agent."""

        context_parts = []

        # Dataset information
        if hasattr(self, 'analysis_results') and self.analysis_results:
            dataset_info = self.analysis_results.get('dataset_info', {})
            if dataset_info:
                context_parts.append(f"""
**Dataset Information:**
- Path: {self.dataset_path}
- Total Transactions: {dataset_info.get('rows', 'Unknown'):,}
- Features/Columns: {dataset_info.get('columns', 'Unknown')}
- File Size: {dataset_info.get('size_mb', 'Unknown')} MB
- Memory Usage: {dataset_info.get('memory_usage_mb', 'Unknown')} MB
- Has Fraud Labels: {dataset_info.get('has_class_column', False)}
- Analysis Type: {'Supervised Learning' if dataset_info.get('has_class_column') else 'Unsupervised Pattern Detection'}
- Available Columns: {', '.join(dataset_info.get('column_names', [])[:10])}{'...' if len(dataset_info.get('column_names', [])) > 10 else ''}
""")

        # Analysis results and report content
        if self.analysis_results:
            report_content = self.analysis_results.get('report_content', '')
            if report_content:
                # Extract key insights from the report
                report_lines = report_content.split('\n')

                # Find important sections
                executive_summary = []
                key_findings = []
                recommendations = []

                current_section = None
                for line in report_lines[:100]:  # Process first 100 lines
                    line = line.strip()
                    if not line:
                        continue

                    if 'executive summary' in line.lower() or 'summary' in line.lower():
                        current_section = 'summary'
                    elif 'key findings' in line.lower() or 'findings' in line.lower():
                        current_section = 'findings'
                    elif 'recommendations' in line.lower():
                        current_section = 'recommendations'
                    elif line.startswith('#') or line.startswith('*'):
                        continue
                    else:
                        if current_section == 'summary' and len(executive_summary) < 5:
                            executive_summary.append(line)
                        elif current_section == 'findings' and len(key_findings) < 5:
                            key_findings.append(line)
                        elif current_section == 'recommendations' and len(recommendations) < 5:
                            recommendations.append(line)

                context_parts.append(f"""
**Analysis Results Summary:**
{chr(10).join(executive_summary[:3]) if executive_summary else 'Comprehensive fraud detection analysis completed'}

**Key Findings:**
{chr(10).join(key_findings[:3]) if key_findings else 'Pattern recognition and statistical analysis performed'}

**Available Recommendations:**
{chr(10).join(recommendations[:3]) if recommendations else 'Fraud prevention strategies identified'}
""")

            # Visualization and output information
            images_found = self.analysis_results.get('images_found', 0)
            image_paths = self.analysis_results.get('image_paths', [])

            if image_paths:
                image_types = [img.get('filename', '') for img in image_paths[:5]]
                context_parts.append(f"""
**Generated Visualizations ({images_found} total):**
- Available Charts: {', '.join(image_types)}
- Chart Types: Correlation heatmaps, distribution plots, fraud pattern analysis
""")

            # Execution metadata
            execution_time = self.analysis_results.get('execution_time', 0)
            if execution_time:
                import time
                duration = time.time() - execution_time
                context_parts.append(f"""
**Analysis Metadata:**
- Analysis Duration: {duration/60:.1f} minutes
- Report Generated: {'Yes' if self.analysis_results.get('report_path') else 'No'}
- Tools Used: Statistical analysis, pattern recognition, classification modeling
""")

        # CSV tool availability
        if self.csv_tool:
            context_parts.append("""
**Available Tools:**
- CSV Search Tool: Can query and explore the dataset for specific patterns
- Statistical Analysis: Can perform additional calculations on demand
- Pattern Recognition: Can identify specific fraud indicators
""")

        # Additional context if provided
        if context:
            context_parts.append(f"""
**Additional Context:**
{context}
""")

        return '\n'.join(context_parts) if context_parts else "No additional context available."

    def _format_response(self, response: str) -> str:
        """Format the agent response for display in Streamlit."""

        # Add emoji and formatting for better readability
        if not response.startswith('🤖'):
            response = f"🤖 **AI Analyst Response:**\n\n{response}"

        # Ensure proper markdown formatting
        if not response.endswith('\n'):
            response += '\n'

        return response


class QuickResponseHandler:
    """Handler for quick action responses using CrewAI agents."""

    def __init__(self, dataset_path: str, analysis_results: Dict[str, Any]):
        self.chat_agent = ChatAnalystAgent(dataset_path, analysis_results)

    def get_statistics_overview(self) -> str:
        """Get detailed statistics overview."""
        return self.chat_agent.ask_question(
            "Provide a comprehensive statistical overview of the dataset including key metrics, distribution patterns, and data quality insights."
        )

    def explain_fraud_patterns(self) -> str:
        """Explain fraud patterns found in the analysis."""
        return self.chat_agent.ask_question(
            "What are the most significant fraud patterns and indicators identified in this dataset? Include specific examples and statistical evidence."
        )

    def get_prevention_recommendations(self) -> str:
        """Get fraud prevention recommendations."""
        return self.chat_agent.ask_question(
            "Based on the analysis results, what are your top recommendations for preventing fraud? Include specific strategies and implementation guidance."
        )

    def assess_risk_factors(self) -> str:
        """Assess risk factors and scoring."""
        return self.chat_agent.ask_question(
            "How should we assess and score risk for different transaction types? What factors are most predictive of fraud?"
        )

    def analyze_feature_importance(self) -> str:
        """Analyze feature importance."""
        return self.chat_agent.ask_question(
            "Which features and variables are most important for fraud detection? How do the PCA components and other features contribute to identifying fraudulent transactions?"
        )

    def evaluate_model_performance(self) -> str:
        """Evaluate model performance."""
        return self.chat_agent.ask_question(
            "How well did the fraud detection models perform? What are the accuracy metrics, and what do they mean for practical implementation?"
        )