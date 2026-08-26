import pandas as pd
import numpy as np


def validate_dataset(df: pd.DataFrame, filename: str = "unknown") -> dict:
    """
    Validate an uploaded dataset and return a structured report
    with severity levels: error, warning, info.
    """
    issues = []

    # --- File-level checks ---
    if df.empty:
        issues.append({
            "severity": "error",
            "category": "File",
            "message": "The uploaded file is empty (0 rows).",
        })
        return {"valid": False, "issues": issues, "score": 0}

    if len(df.columns) < 2:
        issues.append({
            "severity": "warning",
            "category": "File",
            "message": f"Only {len(df.columns)} column detected. Most ML tasks need at least 2.",
        })

    # --- Row-level checks & Duplicates ---
    full_duplicates = int(df.duplicated().sum())
    non_id_cols = [c for c in df.columns if c.lower() not in ['id', 'passengerid', 'uuid', 'index']]
    feature_duplicates = int(df.duplicated(subset=non_id_cols).sum()) if non_id_cols else full_duplicates
    duplicate_count = max(full_duplicates, feature_duplicates)

    if duplicate_count > 0:
        pct = round(duplicate_count / len(df) * 100, 1)
        issues.append({
            "severity": "warning",
            "category": "Duplicates",
            "message": f"{duplicate_count} duplicate rows detected ({pct}% of dataset).",
        })

    fully_null_rows = int(df.isna().all(axis=1).sum())
    if fully_null_rows > 0:
        issues.append({
            "severity": "error",
            "category": "Null Rows",
            "message": f"{fully_null_rows} rows are entirely null and should be removed.",
        })

    # --- Column-level checks & Dummy Values ---
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        issues.append({
            "severity": "warning",
            "category": "Constant Columns",
            "message": f"{len(constant_cols)} constant column(s) detected: {', '.join(constant_cols)}. They carry no information.",
        })

    # High cardinality (potential ID columns)
    id_like_cols = []
    for c in df.select_dtypes(include=["object", "category"]).columns:
        if df[c].nunique() > 0.9 * len(df) and len(df) > 50:
            id_like_cols.append(c)
    if id_like_cols:
        issues.append({
            "severity": "info",
            "category": "ID-like Columns",
            "message": f"Possible ID columns (very high cardinality): {', '.join(id_like_cols)}.",
        })

    # Missing values & sentinel dummy values (?, -999, null)
    dummy_mask = df.isin(['?', 'null', 'none', 'n/a', 'na', 'unknown', 'missing', '', -999])
    total_missing = int((df.isna() | dummy_mask).sum().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    if total_missing > 0:
        pct = round(total_missing / total_cells * 100, 1)
        severity = "error" if pct > 30 else ("warning" if pct > 5 else "info")
        issues.append({
            "severity": severity,
            "category": "Missing & Dummy Values",
            "message": f"{total_missing} missing/dummy placeholder cells ({pct}% of all data).",
        })

    # Columns with very high missing rate
    for col in df.columns:
        col_missing_pct = df[col].isna().mean() * 100
        if col_missing_pct > 50:
            issues.append({
                "severity": "warning",
                "category": "Missing Values",
                "message": f"Column '{col}' has {col_missing_pct:.1f}% missing values. Consider dropping it.",
            })

    # --- Compute quality score ---
    error_count = sum(1 for i in issues if i["severity"] == "error")
    warning_count = sum(1 for i in issues if i["severity"] == "warning")
    score = max(0, 100 - (error_count * 20) - (warning_count * 10))

    return {
        "valid": error_count == 0,
        "issues": issues,
        "score": score,
        "error_count": error_count,
        "warning_count": warning_count,
        "info_count": sum(1 for i in issues if i["severity"] == "info"),
    }
