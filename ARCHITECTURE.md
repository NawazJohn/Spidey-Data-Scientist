# 🏗️ Spidey DATA SCIENTIST (AutoDS) — Architecture & System Design

> **Detailed Architecture Specification, Component Breakdown, and Agent Control Flow**

---

## 1. High-Level System Architecture

AutoDS is designed as a **decoupled 4-tier autonomous architecture**:
1. **User Interface (Streamlit Mission Control Launchpad)**
2. **LLM Orchestration Layer (AutoDSAgent via Groq API Qwen 3.8)**
3. **Analytical Tools & Preprocessing Engines (Validation, Profiling, Filtering)**
4. **AutoML Training Engine (scikit-learn Pipeline Evaluation)**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Streamlit Web Interface                            │
│                 (Glossy Emerald Theme + Plotly Charts)                  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
               ┌─────────────────────┴─────────────────────┐
               ▼                                           ▼
┌─────────────────────────────┐             ┌─────────────────────────────┐
│  Statistical Tools Engine   │             │   LLM Orchestration Agent   │
│  - Data Validator           │             │   - AutoDSAgent             │
│  - Dataset Profiler         │             │   - Groq Qwen 3.8 LLM Model  │
│  - Data Filter / Cleaner    │             │   - AI Insight Report Generator│
│  - Model Trainer (AutoML)   │             │   - Cleaning Strategy Recommender│
└──────────────┬──────────────┘             └──────────────┬──────────────┘
               │                                           │
               └─────────────────────┬─────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Data Store & Model Memory                          │
│        (Uploads, Cleaned CSVs, Execution Logs, Benchmark Metrics)       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. End-to-End Pipeline Execution Flow

```
[Raw File Upload (.csv / .xlsx)]
             │
             ▼
[Module 01: Data Health Validation]
   ├── File Emptiness & Column Count Audit
   ├── Sentinel Dummy Check ('?', '-999', 'null', 'n/a')
   ├── Full & Subset Duplicate Calculation
   └── Quality Score (0–100) Calculation
             │
             ▼
[Module 02: Exploratory Profiling & AI Analysis]
   ├── Column Type Inference & Summary Statistics (Mean, Std, Skewness, IQR Outliers)
   ├── Interactive Plotly Visualizations (Histograms, Box Plots, Correlation Heatmaps)
   └── Groq LLM API Call: Executive AI Dataset Intelligence Synthesis
             │
             ▼
[Module 03: Data Filtering & Transformation]
   ├── Preprocessing Options (Drop duplicates, constant/ID cols, null rows/cols, IQR outliers)
   ├── Imputation Engine (Median for numeric, Mode for categorical)
   ├── Transformation Changelog Generator
   └── Export Cleaned Dataset CSV
             │
             ▼
[Module 04: AutoML Baseline Engine]
   ├── Target Variable Validation & High-Cardinality Safeguards
   ├── Task Type Auto-Detection (Classification vs. Regression)
   ├── Dynamic Target Charts (Bar, Pie, Histogram, Box Plot, Line Trend)
   ├── ColumnTransformer Pipeline Construction (SimpleImputer + StandardScaler + OneHotEncoder)
   ├── Multi-Model Candidate Training (Random Forest, Decision Tree, Logistic/Linear Regression)
   └── Model Benchmark Leaderboard & Best Model Selection (F1 / R² Metric)
```

---

## 3. Detailed Component Breakdown

### 🛠️ 1. Data Validator (`tools/data_validator.py`)
- **Input**: `df: pd.DataFrame`, `filename: str`
- **Output**: JSON validation dictionary containing validity boolean, severity-categorized issues (Error, Warning, Info), and computed 0–100 Data Quality Score.
- **Rule Engine**:
  - `Error`: Empty dataset, >30% missing cells, fully null rows.
  - `Warning`: Single column dataset, >5% missing cells, constant columns, duplicate rows, >50% missing column rate.
  - `Info`: High-cardinality ID-like columns, minor missing placeholders.

### 📊 2. Dataset Profiler (`tools/dataset_profiler.py`)
- **Input**: `df: pd.DataFrame`
- **Output**: Enriched profiling dictionary.
- **Metrics Computed**:
  - Rows, columns, cell count, total missing percentage.
  - Memory usage in MB (`df.memory_usage(deep=True)`).
  - Per-column metadata: Dtype, missing count/pct, unique values count, sample values.
  - Numeric metrics: Mean, standard deviation, min, max, skewness, IQR outlier counts.

### 🧹 3. Data Filter (`tools/data_filter.py`)
- **Input**: `df: pd.DataFrame`, `options: dict`
- **Output**: `(cleaned_df: pd.DataFrame, changelog: list[str])`
- **Operations**:
  1. `drop_null_rows`: Removes rows where all values are NaN.
  2. `drop_duplicates`: Removes duplicate rows.
  3. `drop_constant_cols`: Removes columns with $\le 1$ unique value.
  4. `drop_null_cols`: Removes columns where all values are NaN.
  5. `drop_id_cols`: Drops high-cardinality categorical text columns ($>90\%$ unique).
  6. `impute_missing`: Fills numeric NaNs with median; categorical NaNs with mode.
  7. `remove_outliers`: Filters out rows beyond $Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$.

### 🚀 4. Model Trainer (`tools/model_trainer.py`)
- **Input**: `df: pd.DataFrame`, `target: str`
- **Output**: Dictionary containing task type, top model name, metric name, best score, and benchmark DataFrame.
- **Task Classifier**:
  - `Classification`: If `y` is categorical, string, or has $\le 10$ unique values. Uses **F1 Score (Weighted)**.
  - `Regression`: If `y` is continuous numeric. Uses **R² Score**.
- **Model Suite**:
  - Classification: Logistic Regression, Decision Tree Classifier, Random Forest Classifier ($n\_estimators=150$).
  - Regression: Linear Regression, Decision Tree Regressor, Random Forest Regressor ($n\_estimators=150$).

### 🧠 5. Agent Orchestrator (`agent/agent.py`)
- **Engine**: Groq API using `qwen/qwen3.8-27b` model.
- **Key Methods**:
  - `analyze(profile, validation)`: Generates plain-English executive analysis under 200 words.
  - `suggest_filters(profile, validation)`: Recommends specific cleaning steps with justifications.
  - `_call_llm()`: Handles API execution and provides heuristic fallback warnings if API key is missing.

---

## 4. Repository File Structure

```
AutoDS/
│
├── autods_app.py            # 🚀 Unified single-file edition (Run: streamlit run autods_app.py)
├── generate_dataset.py      # Synthetic benchmark CSV dataset generator script
├── dummy_dataset.csv        # Pre-generated sample dataset for immediate testing
├── eda_report.html          # Sample exported HTML report
├── requirements.txt         # Dependencies (streamlit, pandas, numpy, plotly, scikit-learn, groq, python-dotenv)
├── README.md                # General project overview & quickstart guide
├── ARCHITECTURE.md          # Comprehensive architecture & design specification
├── .env                     # Local environment variables (GROQ_API_KEY)
├── .env.example             # Template environment configuration
│
├── app/
│   └── main.py              # Modular Streamlit dashboard frontend
│
├── agent/
│   ├── __init__.py
│   └── agent.py             # Groq LLM Agent Orchestrator (AutoDSAgent)
│
├── config/
│   └── settings.py          # Global path configurations (BASE_DIR, UPLOAD_DIR, REPORT_DIR)
│
├── data_store/
│   ├── reports/             # Generated analytical reports directory
│   └── uploads/             # File ingestion upload directory
│
└── tools/
    ├── __init__.py
    ├── data_validator.py    # Health validation & sentinel check engine
    ├── dataset_profiler.py   # Statistical profiling engine
    ├── data_filter.py       # Data cleaning & preprocessing pipeline
    ├── auto_eda.py          # Summary EDA helper
    └── model_trainer.py     # AutoML training & evaluation pipeline
```

---

## 5. Deployment & Execution Instructions

```bash
# Clone Repository
git clone https://github.com/NawazJohn/Spidey-Data-Scientist.git
cd AutoDS

# Create & Activate Virtual Environment
python -m venv .venv
.venv\Scripts\activate       # On Windows

# Install Dependencies
pip install -r requirements.txt

# Run Standalone App
streamlit run autods_app.py
```
