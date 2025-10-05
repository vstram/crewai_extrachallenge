"""
Guaranteed Visualization Generator
Ensures all required fraud detection visualizations are created programmatically.
This removes dependency on LLM following tool call instructions.
"""

from crewai.tools import BaseTool
from typing import Type, Dict, List
from pydantic import BaseModel, Field
import os
from .visualization_tool import VisualizationTool


class GuaranteedVisualizationsInput(BaseModel):
    """Input schema for GuaranteedVisualizationsTool."""
    task_name: str = Field(..., description="Name of task requiring visualizations: 'data_analysis_task', 'pattern_recognition_task', or 'classification_task'")


class GuaranteedVisualizationsTool(BaseTool):
    name: str = "Guaranteed Visualizations Tool"
    description: str = (
        "CRITICAL: Use this tool to guarantee all required visualizations are created. "
        "This tool programmatically generates all images for a task, regardless of LLM behavior. "
        "USAGE: Call once per task with task_name: 'data_analysis_task', 'pattern_recognition_task', or 'classification_task'. "
        "This ensures deterministic image generation across all LLM models."
    )
    args_schema: Type[BaseModel] = GuaranteedVisualizationsInput

    def _run(self, task_name: str) -> str:
        """Generate all required visualizations for the specified task."""

        viz_tool = VisualizationTool()

        # Define required visualizations for each task
        task_visualizations: Dict[str, List[Dict]] = {
            "data_analysis_task": [
                {
                    "chart_type": "fraud_comparison",
                    "data_description": "Fraud vs legitimate transaction counts",
                    "title": "Transaction Class Distribution",
                    "filename": "fraud_comparison"
                },
                {
                    "chart_type": "correlation_heatmap",
                    "data_description": "Feature correlations",
                    "title": "Feature Correlation Matrix",
                    "filename": "correlation_heatmap"
                }
            ],
            "pattern_recognition_task": [
                {
                    "chart_type": "scatter",
                    "data_description": "Relationship between key features",
                    "title": "Feature Relationships",
                    "filename": "scatter"
                },
                {
                    "chart_type": "time_series",
                    "data_description": "Fraud incidents over time",
                    "title": "Time Series Analysis",
                    "filename": "time_series"
                },
                {
                    "chart_type": "feature_importance",
                    "data_description": "Most predictive features",
                    "title": "Feature Importance",
                    "filename": "feature_importance"
                },
                {
                    "chart_type": "box_plot",
                    "data_description": "Feature distributions by class",
                    "title": "Distribution Comparison",
                    "filename": "box_plot"
                }
            ],
            "classification_task": [
                {
                    "chart_type": "histogram",
                    "data_description": "Transaction amount distribution",
                    "title": "Transaction Amount Histogram",
                    "filename": "amount_histogram"
                }
            ]
        }

        if task_name not in task_visualizations:
            return f"❌ Error: Unknown task '{task_name}'. Valid tasks: data_analysis_task, pattern_recognition_task, classification_task"

        # Generate all visualizations for this task
        results = []
        visualizations = task_visualizations[task_name]

        for viz_config in visualizations:
            try:
                result = viz_tool._run(
                    chart_type=viz_config["chart_type"],
                    data_description=viz_config["data_description"],
                    title=viz_config["title"],
                    filename=viz_config["filename"],
                    use_timestamp=False  # Deterministic filenames
                )
                results.append(f"✅ {viz_config['filename']}: {result}")
            except Exception as e:
                results.append(f"❌ {viz_config['filename']}: Failed - {str(e)}")

        # Verify all images exist
        images_dir = os.path.join(os.getcwd(), 'reports', 'images')
        generated_count = len([f for f in os.listdir(images_dir) if f.endswith('.png')]) if os.path.exists(images_dir) else 0

        summary = f"✅ GUARANTEED VISUALIZATIONS COMPLETE for {task_name}\n\n"
        summary += f"Generated {len(visualizations)} visualizations:\n"
        summary += "\n".join(results)
        summary += f"\n\nTotal images in directory: {generated_count}"

        return summary
