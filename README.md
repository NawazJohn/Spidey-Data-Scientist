# 🕷️ Spidy DATA SCIENTIST — Autonomous Data Scientist (AutoDS)

> **Agentic Autonomous ML Pipeline, Exploratory Data Analysis & Intelligent AutoML Platform**

Spidy DATA SCIENTIST (AutoDS) is an end-to-end, LLM-powered autonomous data science platform built with **Python**, **Streamlit**, **scikit-learn**, **Plotly**, and **Groq (Qwen 3.8)**. Given any unseen tabular CSV or Excel dataset, AutoDS autonomously validates data health, profiles statistical distributions, recommends and applies data transformations, detects task type (Classification vs. Regression), trains candidate machine learning pipelines, ranks models on a benchmark leaderboard, and writes plain-English AI dataset intelligence reports.

---

## 📌 1. Problem Statement & Mission

Building tabular Machine Learning pipelines manually requires tedious repetition: data ingestion, missing value and sentinel checks, column type detection, outlier identification, encoding, scaling, model selection, hyperparameter evaluation, and report generation.

**AutoDS** automates this entire loop using an **Agentic AI Workflow**:
$$\text{GOAL} \longrightarrow \text{OBSERVE} \longrightarrow \text{VALIDATE} \longrightarrow \text{PROFILE} \longrightarrow \text{TRANSFORM} \longrightarrow \text{AUTOML} \longrightarrow \text{AI REFLECT}$$

Key design goals:
* ⚡ **Zero-Code Automation**: Ingest raw datasets and get trained models + reports in seconds.
* 🧠 **LLM-Driven Reasoning**: Powered by Groq's ultra-fast inference (Qwen 3.8 model) to generate plain-English dataset reports and cleaning strategies.
* 🛡️ **Robust Error Handling & Heuristic Fallbacks**: Graceful degradation if Groq API keys are not provided; the platform continues seamlessly using rule-based heuristics.
* 🎨 **Glossy Emerald UI**: State-of-the-art visual design built with custom glassmorphism, responsive Tailwind CSS styling, and reactive Plotly dark-mode charts.

---

## 🚀 2. Core Capabilities & Pipeline Modules

AutoDS is structured into 4 sequential Mission Control modules:

### 📋 Module 01 — Data Validation & Health
* **File Health Audit**: Evaluates rows, columns, memory usage, and empty file conditions.
* **Sentinel Dummy Detection**: Scans for placeholder values (`?`, `-999`, `null`, `n/a`, `unknown`, empty strings).
* **Duplicate & High Missing Alert**: Flags full and feature-level duplicates, as well as columns with >50% missing data.
* **Data Quality Score**: Calculates an overall 0–100 quality score badge (`Badge Good`, `Badge Warning`, `Badge Error`).

### 📊 Module 02 — Exploratory Profiling & AI Intelligence
* **Column Meta Breakdown**: Data types, missing counts, unique values, sample values, skewness, and IQR outlier counts.
* **Distribution Visualizations**: Interactive histograms, box plots, and missing value distribution bar charts.
* **Correlation Heatmaps**: Multi-variate numeric correlation matrix formatted with Plotly dark themes.
* **⚡ Groq AI Report**: LLM-generated executive summary analyzing data patterns, quality risks, and preprocessing recommendations.

### 🧹 Module 03 — Data Transformation & Filtering
* **Interactive Transformations**: Drop duplicate rows, drop fully-null rows/columns, drop constant (zero-variance) columns, drop high-cardinality ID columns, IQR outlier filtering.
* **Automated Imputation**: Median imputation for numeric features, mode imputation for categorical features.
* **Audit Trail**: Real-time changelog tracking exact row/column shape changes before and after cleaning.
* **Export**: Instant single-click download of the processed `cleaned_dataset.csv`.

### 🚀 Module 04 — AutoML Model Engine
* **Automatic Task Detection**: Inspects target variable properties to auto-classify as **Classification** (F1 score metric) or **Regression** (R² metric).
* **Target Visualizations**: Dynamic user-selected target charts (Bar Chart, Pie Chart, Histogram, Box Plot, Line Trend).
* **Pipeline Preprocessor**: Automated `ColumnTransformer` with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder(handle_unknown='ignore')`.
* **Model Benchmark Leaderboard**: Simultaneously trains and evaluates candidate models (Logistic Regression, Decision Trees, Random Forests, Linear Regression) and highlights the top-performing model.

---

## 📐 3. System Architecture & Agent Flow

```
Raw CSV/XLSX Upload
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  Module 01: Data Validation & Health Check             │
│  - Sentinel value check (?, -999, null)                │
│  - Duplicate detection & Quality Score (0-100)          │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Module 02: Exploratory Profiling & AI Agent            │
│  - Column metadata, histograms, box plots, correlations │
│  - Groq AI (Qwen 3.8) Dataset Intelligence Report       │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Module 03: Data Filtering & Preprocessing Pipeline    │
│  - Interactive cleaning options & Median/Mode Imputation│
│  - Change log tracking & CSV exporter                   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Module 04: AutoML Model Engine                         │
│  - Automatic task detection (Classification/Regression) │
│  - Multi-model training (Random Forest, Decision Tree)  │
│  - Leaderboard ranking & Best Model metric selection    │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ 4. Tech Stack

| Domain | Technology / Library | Description |
|---|---|---|
| **Frontend UI** | Streamlit, Custom HTML/CSS (Tailwind CSS, Glassmorphism) | Interactive web dashboard with glossy dark emerald theme |
| **Data Visualization** | Plotly Express & Plotly Graph Objects | Dark-mode interactive histograms, correlation heatmaps, box plots |
| **Data Manipulation** | pandas, numpy | Dataset loading, filtering, imputation, and statistics |
| **Machine Learning** | scikit-learn | Pipelines, preprocessing, ColumnTransformer, metrics, models |
| **Agent / LLM** | Groq API (`qwen/qwen3.8-27b`), python-dotenv | Fast LLM inference for data analysis and cleaning strategy |
| **Environment** | Python 3.9+ | Core programming language |

---

## 📂 5. Project Repository Structure

```
AutoDS/
├── autods_app.py            # 🚀 Standalone, single-file version of the complete app
├── app/
│   └── main.py              # Modular Streamlit Mission Control interface
├── agent/
│   └── agent.py             # AutoDSAgent — Groq LLM Orchestrator
├── tools/
│   ├── data_validator.py    # Health check & validation engine
│   ├── dataset_profiler.py   # Statistical profiling engine
│   ├── data_filter.py       # Data cleaning & preprocessing pipeline
│   ├── auto_eda.py          # Auto-EDA summary helper
│   └── model_trainer.py     # AutoML training & evaluation pipeline
├── config/
│   └── settings.py          # Directory & path configuration
├── dummy_dataset.csv        # Pre-generated sample dataset for quick testing
├── generate_dataset.py      # Script to regenerate synthetic benchmark CSV
├── requirements.txt         # Python package dependencies
├── .env.example             # Template for environment configuration
└── README.md                # Project documentation
```

---

## ⚡ 6. Getting Started

### Step 1: Clone & Setup Virtual Environment
```bash
git clone https://github.com/NawazJohn/Spidey-Data-Scientist.git
cd AutoDS

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Create a `.env` file in the root directory (or copy `.env.example`):
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```
> *Note: If `GROQ_API_KEY` is omitted, AutoDS will gracefully fall back to heuristic rule-based recommendations without breaking.*

### Step 4: Launch Mission Control
You can run the single-file edition or the modular app:

```bash
# Option A: Run Standalone Single-File App (Recommended)
streamlit run autods_app.py

# Option B: Run Modular App
streamlit run app/main.py
```

Open **http://localhost:8501** in your browser.

---

## 🧪 7. Testing with Sample Data

AutoDS includes a built-in synthetic dataset generator:

```bash
python generate_dataset.py
```
This generates `dummy_dataset.csv` containing missing values, dummy sentinels (`?`, `-999`), and explicit duplicates—perfect for testing validation, cleaning, profiling, and AutoML features out of the box!

---

<p center>
🚀 <b>Spidy DATA SCIENTIST (AutoDS)</b> · Powered by Groq AI & Streamlit
</p>
