"""Streamlit configuration settings for the fraud detection app."""

# App configuration
APP_CONFIG = {
    "title": "CrewAI Fraud Detection System",
    "icon": "🔍",
    "layout": "wide",
    "sidebar_state": "expanded"
}

# File upload limits
FILE_UPLOAD_CONFIG = {
    "max_file_size_mb": 200,
    "allowed_types": ["csv"],
    "temp_dir": "streamlit_app/temp"
}

# Dataset validation config
DATASET_CONFIG = {
    "required_columns": ["Time", "Amount"],
    "optional_columns": ["Class"],
    "min_rows": 10,
    "max_preview_rows": 10
}

# UI styling
UI_CONFIG = {
    "primary_color": "#1f77b4",
    "success_color": "#2ca02c",
    "error_color": "#d62728",
    "warning_color": "#ff7f0e",
    "info_color": "#17becf"
}

# Phase configuration
PHASES = {
    "dataset_configuration": {
        "name": "Dataset Configuration",
        "icon": "📊",
        "description": "Configure your credit card transaction dataset"
    },
    "report_generation": {
        "name": "Report Generation",
        "icon": "📈",
        "description": "Generate fraud detection analysis report"
    },
    "interactive_chat": {
        "name": "Interactive Chat",
        "icon": "💬",
        "description": "Ask questions about your analysis results"
    }
}