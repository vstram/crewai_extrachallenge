# Quick Start Guide

Get up and running with the Credit Card Fraud Detection Crew in minutes.

## Prerequisites

- Python 3.10 - 3.13
- An OpenAI API key (or local LLM setup)

## Installation

1. **Install UV:**
   ```bash
   pip install uv
   ```

2. **Install Dependencies:**
   ```bash
   crewai install
   ```

3. **Configure Environment:**

   Create a `.env` file in the project root:
   ```bash
   # Required
   MODEL=gpt-4-turbo-preview
   OPENAI_API_KEY=your_api_key_here

   # For large datasets (optional)
   USE_DATABASE=true
   ```

## Run Analysis

### Option 1: Streamlit UI (Recommended)

```bash
streamlit run streamlit_app/app.py
```

Then:
1. Upload or specify your CSV dataset path
2. Click "Generate Report"
3. Ask questions in the interactive chat

### Option 2: Command Line

```bash
crewai run
```

Report generated at: `reports/fraud_detection_report.md`

## Dataset Format

Your CSV should include:
- **Features:** V1-V28 (PCA components), Time, Amount
- **Target:** Class (0 = legitimate, 1 = fraud)

## Troubleshooting

**Images not generated?**
- Switch to `MODEL=gpt-4-turbo-preview` (avoid o1-mini)

**Memory issues?**
- Set `USE_DATABASE=true` in `.env`

**Need help?**
- See [README.md](README.md) for detailed documentation

---

**That's it! You're ready to detect fraud.** 🚀
