# CrewAI Fraud Detection - Streamlit GUI

A user-friendly Streamlit interface for the CrewAI fraud detection system.

## Features

### Phase 1: Dataset Configuration ✅
- File path input or file upload
- CSV validation and preview
- Dataset statistics and column verification
- Interactive configuration confirmation

### Phase 2: Report Generation ✅
- CrewAI integration with progress tracking
- Real-time agent status updates during analysis
- Generated report display with markdown formatting
- Visualization gallery with downloadable charts
- Error handling and environment validation

### Phase 3: Interactive Chat ✅
- Chat interface for follow-up questions
- Quick action buttons for common queries (statistics, patterns, recommendations)
- Chat history with conversation memory
- Context-aware responses based on analysis results
- *Note: Uses placeholder responses; full CrewAI agent integration available for enhancement*

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

### Phase 1: Configure Dataset ✅

1. **Choose Input Method:**
   - **File Path**: Enter path to existing CSV file
   - **File Upload**: Upload CSV file directly

2. **File Validation:**
   - Required columns: `Time`, `Amount`
   - Optional columns: `Class` (for supervised learning)
   - Minimum 10 rows of data

3. **Preview & Confirm:**
   - Review dataset statistics
   - Preview first 10 rows
   - Confirm configuration to proceed

### Phase 2: Generate Report ✅

1. **Environment Check:**
   - Validates CrewAI environment setup
   - Checks for Ollama or OpenAI API configuration
   - Provides setup guidance if needed

2. **Analysis Execution:**
   - Real-time progress tracking (0-100%)
   - Agent status updates (Data Analyst → Pattern Recognition → Classification → Reporting)
   - Estimated completion time based on dataset size

3. **Results Display:**
   - Generated markdown report with proper formatting
   - Visualization gallery (correlation heatmap, fraud comparison, etc.)
   - Download buttons for report and images
   - Analysis summary metrics

### Phase 3: Interactive Chat ✅

1. **Quick Actions:**
   - Statistics overview
   - Pattern explanations
   - Risk assessment
   - Feature analysis
   - Prevention recommendations

2. **Chat Interface:**
   - Natural language questions about results
   - Context-aware responses based on analysis
   - Chat history with conversation memory
   - Clear and restart options

### Dataset Requirements

Your CSV file must contain:

- **Time**: Transaction timestamp (numeric)
- **Amount**: Transaction amount (numeric)
- **Class** (optional): Fraud label (0=legitimate, 1=fraud)
- **V1-V28** (optional): PCA-transformed features

Example CSV structure:
```csv
Time,V1,V2,V3,Amount,Class
0,-1.359,0.876,1.548,149.62,0
406,1.191,-0.329,-1.468,2.69,0
```

## File Structure

```
streamlit_app/
├── app.py                 # Main Streamlit application
├── components/
│   └── dataset_config.py  # Dataset configuration component
├── utils/
│   ├── file_validator.py  # CSV validation utilities
│   └── session_manager.py # Session state management
├── config/
│   └── streamlit_config.py# App configuration
└── requirements.txt       # Dependencies
```

## Environment Setup

### For CrewAI Integration (Phase 2)

**Option 1: Using Ollama (Recommended)**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull required model
ollama pull llama3.1:8b
```

**Option 2: Using OpenAI**
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Development

### Running in Development Mode

```bash
# From the streamlit_app directory
./run_app.sh

# Or directly:
uv run streamlit run app.py
```

### Adding New Components

1. Create component file in `components/`
2. Import and use in `app.py`
3. Update session state management as needed

## Next Steps

### Potential Enhancements
- **Enhanced Chat**: Full CrewAI agent integration for chat responses
- **Custom Analysis**: User-configurable analysis parameters
- **Batch Processing**: Multiple dataset analysis
- **Export Options**: PDF report generation, data export
- **Performance Optimization**: Async processing, caching
- **Deployment**: Docker containerization, cloud deployment

## Troubleshooting

### Common Issues

**File not found:**
- Check file path is correct
- Use absolute paths if needed

**Validation errors:**
- Ensure CSV has required columns
- Check for proper CSV formatting
- Verify file has minimum 10 rows

**Upload issues:**
- Check file size (max 200MB)
- Ensure file is CSV format
- Try different browser if upload fails