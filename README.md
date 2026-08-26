# AutoDS — Autonomous Data Scientist

AutoDS is an agentic AutoML and Auto-EDA platform that aims to autonomously inspect unseen CSV datasets, identify data-quality issues, perform EDA, detect the ML problem type, train candidate models, evaluate them, and generate an explainable report.

## Current MVP

- CSV upload
- Dataset profiling
- Missing-value and duplicate detection
- Numeric/categorical detection
- Target selection
- Classification/regression detection
- Baseline preprocessing
- Multiple model training
- Best-model selection

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```

## Planned agent loop

Observe → Analyze → Decide → Act → Evaluate → Reflect → Improve → Report
