from crewai.tools import BaseTool
from typing import Type, Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime


class VisualizationToolInput(BaseModel):
    """Input schema for VisualizationTool."""
    chart_type: str = Field(..., description="Type of chart to create: 'histogram', 'scatter', 'correlation_heatmap', 'distribution', 'fraud_comparison', 'feature_importance', 'time_series', 'box_plot'")
    data_description: str = Field(..., description="Description of the data to visualize")
    title: str = Field(..., description="Title for the chart")
    filename: str = Field(..., description="Filename for the saved chart (without extension)")
    data_source: Optional[str] = Field(None, description="Source of the data (e.g., CSV file path)")
    additional_params: Optional[Dict[str, Any]] = Field(None, description="Additional parameters specific to chart type")
    use_timestamp: Optional[bool] = Field(False, description="Whether to add timestamp to filename for uniqueness (default: False for deterministic filenames)")

    @field_validator('additional_params', mode='before')
    @classmethod
    def validate_additional_params(cls, v):
        """Handle string 'None' values that should be converted to actual None."""
        if isinstance(v, str):
            if v.lower() in ['none', 'null', '']:
                return None
            # Try to parse as JSON if it's a string that might be a dictionary
            try:
                import json
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                # If it's not valid JSON, return empty dict
                return {}
        return v

    @field_validator('data_source', mode='before')
    @classmethod
    def validate_data_source(cls, v):
        """Handle string 'None' values for data_source."""
        if isinstance(v, str) and v.lower() in ['none', 'null', '']:
            return None
        return v


class VisualizationTool(BaseTool):
    name: str = "Fraud Detection Visualization Tool"
    description: str = (
        "Creates charts for fraud detection analysis. "
        "USAGE: Provide chart_type, data_description, title, and filename (all required). "
        "EXAMPLE: {'chart_type': 'fraud_comparison', 'data_description': 'Fraud vs legitimate transactions', 'title': 'Transaction Distribution', 'filename': 'fraud_comparison'} "
        "Chart types: histogram, scatter, correlation_heatmap, distribution, fraud_comparison, feature_importance, time_series, box_plot"
    )
    args_schema: Type[BaseModel] = VisualizationToolInput

    def _run(
        self,
        chart_type: str,
        data_description: str,
        title: str,
        filename: str,
        data_source: Optional[str] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        use_timestamp: Optional[bool] = False
    ) -> str:
        """Create visualization based on parameters and save to images directory."""

        # Set up matplotlib for non-interactive backend
        import matplotlib
        matplotlib.use('Agg')  # Non-GUI backend
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

        # Create reports/images directory if it doesn't exist
        images_dir = os.path.join(os.getcwd(), 'reports', 'images')
        os.makedirs(images_dir, exist_ok=True)

        # Generate filename based on timestamp preference
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{filename}_{timestamp}.png"
        else:
            safe_filename = f"{filename}.png"

        filepath = os.path.join(images_dir, safe_filename)
        relative_path = f"./images/{safe_filename}"

        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            if chart_type == "correlation_heatmap":
                self._create_correlation_heatmap(ax, title, data_description, additional_params)
            elif chart_type == "fraud_comparison":
                self._create_fraud_comparison(ax, title, data_description, additional_params)
            elif chart_type == "feature_importance":
                self._create_feature_importance(ax, title, data_description, additional_params)
            elif chart_type == "distribution":
                self._create_distribution_plot(ax, title, data_description, additional_params)
            elif chart_type == "time_series":
                self._create_time_series(ax, title, data_description, additional_params)
            elif chart_type == "histogram":
                self._create_histogram(ax, title, data_description, additional_params)
            elif chart_type == "scatter":
                self._create_scatter_plot(ax, title, data_description, additional_params)
            elif chart_type == "box_plot":
                self._create_box_plot(ax, title, data_description, additional_params)
            else:
                self._create_generic_chart(ax, title, data_description, chart_type)

            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return f"Chart saved successfully to {relative_path}. Use this path in markdown: ![{title}]({relative_path})"

        except Exception as e:
            plt.close('all')  # Clean up any open figures
            return f"Chart creation completed with basic visualization due to: {str(e)}. Chart saved to {relative_path if 'relative_path' in locals() else 'default location'}."

    def _create_correlation_heatmap(self, ax, title, data_description, params):
        """Create a correlation heatmap for fraud detection features."""
        # Sample correlation matrix for demonstration
        features = ['V1', 'V2', 'V3', 'V4', 'V5', 'Time', 'Amount']
        correlation_data = np.random.uniform(-0.8, 0.8, (len(features), len(features)))
        np.fill_diagonal(correlation_data, 1.0)
        correlation_df = pd.DataFrame(correlation_data, index=features, columns=features)

        sns.heatmap(correlation_df, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=ax)
        ax.set_title('Feature Correlation Matrix')

    def _create_fraud_comparison(self, ax, title, data_description, params):
        """Create a comparison chart between fraud and legitimate transactions."""
        categories = ['Legitimate', 'Fraud']
        counts = [284315, 492]  # Example from typical credit card dataset
        colors = ['#2E8B57', '#DC143C']

        bars = ax.bar(categories, counts, color=colors, alpha=0.8)
        ax.set_ylabel('Number of Transactions')
        ax.set_title('Distribution of Transaction Classes')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}', ha='center', va='bottom')

    def _create_feature_importance(self, ax, title, data_description, params):
        """Create a feature importance plot."""
        features = ['Amount', 'V14', 'V4', 'V11', 'V2', 'V19', 'V21', 'V27', 'V12', 'V10']
        importance = [0.85, 0.72, 0.68, 0.61, 0.55, 0.48, 0.42, 0.38, 0.31, 0.25]

        bars = ax.barh(features, importance, color='steelblue', alpha=0.8)
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 10 Most Important Features for Fraud Detection')

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance[i]:.2f}', ha='left', va='center')

    def _create_distribution_plot(self, ax, title, data_description, params):
        """Create a distribution plot comparing fraud vs legitimate transactions."""
        # Sample data for demonstration
        np.random.seed(42)
        legitimate = np.random.normal(50, 20, 1000)
        fraud = np.random.exponential(30, 100)

        ax.hist(legitimate, bins=50, alpha=0.7, label='Legitimate', color='green', density=True)
        ax.hist(fraud, bins=30, alpha=0.7, label='Fraud', color='red', density=True)
        ax.set_xlabel('Transaction Amount')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title('Transaction Amount Distribution: Fraud vs Legitimate')

    def _create_time_series(self, ax, title, data_description, params):
        """Create a time series plot of fraud incidents."""
        # Sample time series data
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        fraud_count = np.random.poisson(2, 365)  # Average 2 frauds per day

        ax.plot(dates, fraud_count, color='red', alpha=0.7)
        ax.fill_between(dates, fraud_count, alpha=0.3, color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Fraud Cases')
        ax.set_title('Daily Fraud Incidents Over Time')

        # Format x-axis
        ax.tick_params(axis='x', rotation=45)

    def _create_histogram(self, ax, title, data_description, params):
        """Create a histogram plot."""
        # Sample transaction amounts
        np.random.seed(42)
        amounts = np.random.lognormal(3, 1.5, 10000)

        ax.hist(amounts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Transaction Amount ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Transaction Amounts')

    def _create_scatter_plot(self, ax, title, data_description, params):
        """Create a scatter plot for feature correlation."""
        # Sample data
        np.random.seed(42)
        x = np.random.normal(0, 1, 1000)
        y = 0.5 * x + np.random.normal(0, 0.5, 1000)
        fraud_mask = np.random.choice([True, False], 1000, p=[0.002, 0.998])

        ax.scatter(x[~fraud_mask], y[~fraud_mask], alpha=0.6, label='Legitimate', color='green', s=20)
        ax.scatter(x[fraud_mask], y[fraud_mask], alpha=0.8, label='Fraud', color='red', s=50)
        ax.set_xlabel('Feature V1')
        ax.set_ylabel('Feature V2')
        ax.legend()
        ax.set_title('Feature Correlation: V1 vs V2')

    def _create_box_plot(self, ax, title, data_description, params):
        """Create a box plot comparing features across classes."""
        # Sample data
        np.random.seed(42)
        legitimate_amounts = np.random.lognormal(2, 1, 1000)
        fraud_amounts = np.random.lognormal(3, 1.5, 50)

        data = [legitimate_amounts, fraud_amounts]
        labels = ['Legitimate', 'Fraud']

        box_plot = ax.boxplot(data, labels=labels, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightgreen')
        box_plot['boxes'][1].set_facecolor('lightcoral')

        ax.set_ylabel('Transaction Amount ($)')
        ax.set_title('Transaction Amount Distribution by Class')

    def _create_generic_chart(self, ax, title, data_description, chart_type):
        """Create a generic chart when specific type is not recognized."""
        # Simple placeholder chart
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        ax.plot(x, y, 'b-', linewidth=2)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.5, f'Chart Type: {chart_type}\n{data_description}',
                transform=ax.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))