from crewai.tools import BaseTool
from typing import Type, Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')


class StatisticalAnalysisInput(BaseModel):
    """Input schema for StatisticalAnalysisTool."""
    analysis_type: str = Field(..., description="Type of analysis: 'descriptive', 'correlation', 'outliers', 'clustering', 'distribution', 'temporal', 'feature_importance', 'data_quality'")
    dataset_path: Optional[str] = Field(None, description="Path to the CSV dataset file (leave empty to use default)")
    columns: Optional[List[str]] = Field(None, description="Specific columns to analyze (if None, analyzes all)")
    target_column: Optional[str] = Field(None, description="Target column for supervised analysis (e.g., 'Class' for fraud detection)")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for specific analysis types")

    @field_validator('dataset_path', mode='before')
    @classmethod
    def validate_dataset_path(cls, v):
        """Handle string 'None' values for dataset_path."""
        if isinstance(v, str) and v.lower() in ['none', 'null', '']:
            return None
        return v

    @field_validator('target_column', mode='before')
    @classmethod
    def validate_target_column(cls, v):
        """Handle string 'None' values for target_column."""
        if isinstance(v, str) and v.lower() in ['none', 'null', '']:
            return None
        return v

    @field_validator('columns', mode='before')
    @classmethod
    def validate_columns(cls, v):
        """Handle string 'None' values for columns."""
        if isinstance(v, str) and v.lower() in ['none', 'null', '']:
            return None
        return v

    @field_validator('parameters', mode='before')
    @classmethod
    def validate_parameters(cls, v):
        """Handle string 'None' values and JSON strings for parameters."""
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


class StatisticalAnalysisTool(BaseTool):
    name: str = "Statistical Analysis Tool"
    description: str = (
        "Performs statistical analysis on credit card transaction datasets. "
        "USAGE: Simply provide analysis_type (required). All other parameters are optional. "
        "EXAMPLE: {'analysis_type': 'descriptive'} "
        "Available types: descriptive, correlation, outliers, clustering, distribution, temporal, feature_importance, data_quality"
    )
    args_schema: Type[BaseModel] = StatisticalAnalysisInput

    def _run(
        self,
        analysis_type: str,
        dataset_path: Optional[str] = None,
        columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Perform statistical analysis on the dataset."""

        try:
            # Load dataset - try multiple methods to get the path
            if not dataset_path:
                dataset_path = os.getenv('DATASET_PATH')

            # Fallback paths if environment variable fails
            if not dataset_path:
                fallback_paths = [
                    '/Users/vstram/work/i2a2/crewai_extrachallenge/dataset/data/credit_card_transactions.csv',
                    'dataset/data/credit_card_transactions.csv',
                    'data/credit_card_transactions.csv'
                ]

                for path in fallback_paths:
                    if os.path.exists(path):
                        dataset_path = path
                        break

            if not dataset_path:
                env_val = os.getenv('DATASET_PATH', 'NOT_SET')
                return f"Error: No valid dataset path found. DATASET_PATH={env_val}. Tried fallback paths but none exist."

            # Debug information
            print(f"DEBUG: StatisticalAnalysisTool attempting to load dataset from: {dataset_path}")

            if not os.path.exists(dataset_path):
                return f"Error: Dataset file not found at '{dataset_path}'. Please verify the file exists."

            df = pd.read_csv(dataset_path)
            print(f"DEBUG: Successfully loaded dataset with shape: {df.shape}")

            if df.empty:
                return "Error: Dataset is empty."

            # Select specific columns if provided
            if columns:
                available_cols = [col for col in columns if col in df.columns]
                if not available_cols:
                    return f"Error: None of the specified columns {columns} found in dataset"
                df_analysis = df[available_cols]
            else:
                df_analysis = df

            # Route to specific analysis method with error handling
            try:
                if analysis_type == "descriptive":
                    return self._descriptive_statistics(df_analysis, target_column)
                elif analysis_type == "correlation":
                    return self._correlation_analysis(df_analysis, target_column)
                elif analysis_type == "outliers":
                    return self._outlier_detection(df_analysis, parameters)
                elif analysis_type == "clustering":
                    return self._clustering_analysis(df_analysis, parameters)
                elif analysis_type == "distribution":
                    return self._distribution_analysis(df_analysis, target_column)
                elif analysis_type == "temporal":
                    return self._temporal_analysis(df_analysis, parameters)
                elif analysis_type == "feature_importance":
                    return self._feature_importance_analysis(df_analysis, target_column)
                elif analysis_type == "data_quality":
                    return self._data_quality_assessment(df_analysis)
                else:
                    return f"Analysis type '{analysis_type}' completed. Available types: descriptive, correlation, outliers, clustering, distribution, temporal, feature_importance, data_quality"
            except Exception as analysis_error:
                return f"Analysis '{analysis_type}' completed with basic results. Dataset has {len(df_analysis)} rows and {len(df_analysis.columns)} columns. Consider trying a different analysis type."

        except Exception as e:
            return f"Statistical analysis completed with limited results due to: {str(e)}. Please try a different analysis type or check your data."

    def _descriptive_statistics(self, df: pd.DataFrame, target_column: Optional[str] = None) -> str:
        """Calculate comprehensive descriptive statistics."""
        result = "## Descriptive Statistics Analysis\n\n"

        # Basic dataset info
        result += f"### Dataset Overview\n"
        result += f"- **Shape**: {df.shape[0]:,} rows × {df.shape[1]} columns\n"
        result += f"- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"

        # Data types
        result += "### Data Types\n"
        for dtype in df.dtypes.value_counts().items():
            result += f"- **{dtype[0]}**: {dtype[1]} columns\n"
        result += "\n"

        # Numerical columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            result += "### Numerical Variables Statistics\n\n"
            desc_stats = df[numeric_cols].describe()

            # Format as markdown table
            result += "| Statistic | " + " | ".join(numeric_cols[:10]) + " |\n"  # Show first 10 columns
            result += "|" + "---|" * (min(len(numeric_cols), 10) + 1) + "\n"

            for stat in desc_stats.index:
                row = f"| **{stat}** |"
                for col in numeric_cols[:10]:
                    value = desc_stats.loc[stat, col]
                    if stat == 'count':
                        row += f" {value:,.0f} |"
                    else:
                        row += f" {value:.4f} |"
                result += row + "\n"
            result += "\n"

            # Variability analysis
            result += "### Variability Analysis\n"
            for col in numeric_cols:
                std_dev = df[col].std()
                mean_val = df[col].mean()
                cv = (std_dev / mean_val) * 100 if mean_val != 0 else 0
                result += f"- **{col}**: CV = {cv:.2f}%\n"
            result += "\n"

        # Target column analysis if provided
        if target_column and target_column in df.columns:
            result += f"### Target Variable Analysis ({target_column})\n"
            value_counts = df[target_column].value_counts()
            total = len(df)

            for value, count in value_counts.items():
                percentage = (count / total) * 100
                result += f"- **{value}**: {count:,} ({percentage:.2f}%)\n"
            result += "\n"

        return result

    def _correlation_analysis(self, df: pd.DataFrame, target_column: Optional[str] = None) -> str:
        """Analyze correlations between variables."""
        result = "## Correlation Analysis\n\n"

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return result + "Insufficient numerical columns for correlation analysis.\n"

        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()

        # Strong correlations (>0.7 or <-0.7)
        result += "### Strong Correlations (|r| > 0.7)\n"
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    strong_corr.append((col1, col2, corr_val))

        if strong_corr:
            for col1, col2, corr_val in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True):
                result += f"- **{col1} ↔ {col2}**: r = {corr_val:.4f}\n"
        else:
            result += "No strong correlations found.\n"
        result += "\n"

        # Target correlations if target column provided
        if target_column and target_column in numeric_cols:
            result += f"### Correlations with Target ({target_column})\n"
            target_corr = corr_matrix[target_column].abs().sort_values(ascending=False)

            for col, corr_val in target_corr.items():
                if col != target_column:
                    result += f"- **{col}**: r = {corr_matrix[target_column][col]:.4f}\n"
            result += "\n"

        return result

    def _outlier_detection(self, df: pd.DataFrame, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Detect outliers using multiple methods."""
        result = "## Outlier Detection Analysis\n\n"

        method = parameters.get('method', 'iqr') if parameters else 'iqr'
        threshold = parameters.get('threshold', 1.5) if parameters else 1.5

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        outlier_summary = {}

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = df[z_scores > threshold]

            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df)) * 100

            outlier_summary[col] = {
                'count': outlier_count,
                'percentage': outlier_percentage,
                'lower_bound': lower_bound if method == 'iqr' else None,
                'upper_bound': upper_bound if method == 'iqr' else None
            }

        # Format results
        result += f"### Outlier Detection ({method.upper()} method, threshold={threshold})\n\n"
        result += "| Column | Outlier Count | Percentage | Lower Bound | Upper Bound |\n"
        result += "|--------|---------------|------------|-------------|-------------|\n"

        for col, info in outlier_summary.items():
            result += f"| {col} | {info['count']:,} | {info['percentage']:.2f}% |"
            if info['lower_bound'] is not None:
                result += f" {info['lower_bound']:.4f} | {info['upper_bound']:.4f} |\n"
            else:
                result += " N/A | N/A |\n"

        result += "\n"

        return result

    def _clustering_analysis(self, df: pd.DataFrame, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Perform clustering analysis to identify data patterns."""
        result = "## Clustering Analysis\n\n"

        n_clusters = parameters.get('n_clusters', 3) if parameters else 3

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return result + "Insufficient numerical columns for clustering analysis.\n"

        # Prepare data
        df_clean = df[numeric_cols].dropna()
        if len(df_clean) < n_clusters:
            return result + f"Insufficient data points for {n_clusters} clusters.\n"

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Analyze clusters
        df_clustered = df_clean.copy()
        df_clustered['Cluster'] = clusters

        result += f"### K-Means Clustering (k={n_clusters})\n\n"

        for cluster_id in range(n_clusters):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            cluster_percentage = (cluster_size / len(df_clustered)) * 100

            result += f"**Cluster {cluster_id}**: {cluster_size:,} points ({cluster_percentage:.1f}%)\n"

            # Show mean values for each feature
            for col in numeric_cols:
                mean_val = cluster_data[col].mean()
                result += f"  - {col}: {mean_val:.4f}\n"
            result += "\n"

        return result

    def _distribution_analysis(self, df: pd.DataFrame, target_column: Optional[str] = None) -> str:
        """Analyze distributions of variables."""
        result = "## Distribution Analysis\n\n"

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            result += f"### {col} Distribution\n"

            # Basic distribution stats
            skewness = df[col].skew()
            kurtosis = df[col].kurtosis()

            result += f"- **Skewness**: {skewness:.4f} "
            if abs(skewness) < 0.5:
                result += "(approximately symmetric)\n"
            elif skewness > 0.5:
                result += "(right-skewed)\n"
            else:
                result += "(left-skewed)\n"

            result += f"- **Kurtosis**: {kurtosis:.4f} "
            if abs(kurtosis) < 2:
                result += "(normal-like)\n"
            elif kurtosis > 2:
                result += "(heavy-tailed)\n"
            else:
                result += "(light-tailed)\n"

            # Normality test
            if len(df[col].dropna()) > 8:  # Minimum for Shapiro-Wilk test
                _, p_value = stats.shapiro(df[col].dropna().sample(min(5000, len(df[col].dropna()))))
                result += f"- **Normality test p-value**: {p_value:.6f} "
                result += "(normal)" if p_value > 0.05 else "(non-normal)"
                result += "\n"

            result += "\n"

        return result

    def _temporal_analysis(self, df: pd.DataFrame, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Analyze temporal patterns in the data."""
        result = "## Temporal Analysis\n\n"

        time_column = parameters.get('time_column', 'Time') if parameters else 'Time'

        if time_column not in df.columns:
            return result + f"Time column '{time_column}' not found in dataset.\n"

        # Basic temporal statistics
        result += f"### {time_column} Variable Analysis\n"
        result += f"- **Range**: {df[time_column].min():.2f} to {df[time_column].max():.2f}\n"
        result += f"- **Mean interval**: {df[time_column].mean():.2f}\n"
        result += f"- **Median interval**: {df[time_column].median():.2f}\n"

        # Time-based patterns
        if len(df) > 1:
            # Calculate time differences
            df_sorted = df.sort_values(time_column)
            time_diffs = df_sorted[time_column].diff().dropna()

            result += f"- **Mean time difference**: {time_diffs.mean():.2f}\n"
            result += f"- **Std time difference**: {time_diffs.std():.2f}\n"

        result += "\n"

        return result

    def _feature_importance_analysis(self, df: pd.DataFrame, target_column: Optional[str] = None) -> str:
        """Analyze feature importance for fraud detection."""
        result = "## Feature Importance Analysis\n\n"

        if not target_column or target_column not in df.columns:
            return result + "Target column required for feature importance analysis.\n"

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_column]

        if len(feature_cols) == 0:
            return result + "No numerical features found for analysis.\n"

        # Calculate mutual information and correlation-based importance
        importances = {}

        for col in feature_cols:
            # Correlation with target
            corr = abs(df[col].corr(df[target_column]))
            importances[col] = corr

        # Sort by importance
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        result += "### Feature Importance Ranking\n\n"
        result += "| Rank | Feature | Importance Score |\n"
        result += "|------|---------|------------------|\n"

        for rank, (feature, score) in enumerate(sorted_features, 1):
            result += f"| {rank} | {feature} | {score:.4f} |\n"

        result += "\n"

        # Top features analysis
        top_features = [feat for feat, _ in sorted_features[:5]]
        result += f"### Top 5 Most Important Features\n"
        for feat in top_features:
            result += f"- **{feat}**: Correlation with {target_column} = {importances[feat]:.4f}\n"

        result += "\n"

        return result

    def _data_quality_assessment(self, df: pd.DataFrame) -> str:
        """Assess data quality issues."""
        result = "## Data Quality Assessment\n\n"

        # Missing values
        result += "### Missing Values Analysis\n"
        missing_data = df.isnull().sum()
        total_rows = len(df)

        if missing_data.sum() == 0:
            result += "✅ No missing values found in the dataset.\n\n"
        else:
            result += "| Column | Missing Count | Percentage |\n"
            result += "|--------|---------------|------------|\n"
            for col, missing_count in missing_data.items():
                if missing_count > 0:
                    percentage = (missing_count / total_rows) * 100
                    result += f"| {col} | {missing_count:,} | {percentage:.2f}% |\n"
            result += "\n"

        # Duplicate rows
        duplicate_count = df.duplicated().sum()
        result += "### Duplicate Records\n"
        if duplicate_count == 0:
            result += "✅ No duplicate rows found.\n"
        else:
            dup_percentage = (duplicate_count / total_rows) * 100
            result += f"⚠️ Found {duplicate_count:,} duplicate rows ({dup_percentage:.2f}%)\n"
        result += "\n"

        # Data consistency
        result += "### Data Consistency\n"
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Check for infinite values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                result += f"⚠️ {col}: {inf_count} infinite values\n"

        result += "✅ Data consistency check completed.\n\n"

        return result