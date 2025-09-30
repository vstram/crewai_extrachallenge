# CrewAI Fraud Detection - Streamlit GUI

A user-friendly Streamlit interface for the CrewAI fraud detection system.

## Features

### Phase 1: Dataset Configuration ✅
- File path input or file upload
- CSV validation and preview
- Dataset statistics and column verification
- Interactive configuration confirmation

### Phase 2: Report Generation 🚧
- Progress tracking during analysis
- Real-time agent status updates
- Generated report display
- *Coming soon in next implementation phase*

### Phase 3: Interactive Chat 🚧
- Chat interface for follow-up questions
- LLM integration with fraud detection context
- Quick action buttons for common queries
- *Coming soon in next implementation phase*

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

### Phase 1: Configure Dataset

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

## Development

### Running in Development Mode

```bash
# From the streamlit_app directory
streamlit run app.py --server.runOnSave true
```

### Adding New Components

1. Create component file in `components/`
2. Import and use in `app.py`
3. Update session state management as needed

## Next Steps

- **Phase 2**: Implement CrewAI integration for report generation
- **Phase 3**: Add interactive chat interface
- **Enhancements**: Progress tracking, error handling, result export

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