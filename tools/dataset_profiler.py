import pandas as pd
import numpy as np


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generate an enriched dataset profile for use by the UI and LLM agent.
    """
    dummy_mask = df.isin(['?', 'null', 'none', 'n/a', 'na', 'unknown', 'missing', '', -999])
    missing_by_column = (df.isna() | dummy_mask).sum()
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(missing_by_column.sum())

    full_duplicates = int(df.duplicated().sum())
    non_id_cols = [c for c in df.columns if c.lower() not in ['id', 'passengerid', 'uuid', 'index']]
    feature_duplicates = int(df.duplicated(subset=non_id_cols).sum()) if non_id_cols else full_duplicates
    duplicate_count = max(full_duplicates, feature_duplicates)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    # Column-level details
    column_details = []
    for col in df.columns:
        col_missing = int((df[col].isna() | df[col].isin(['?', 'null', 'none', 'n/a', 'na', 'unknown', 'missing', '', -999])).sum())
        detail = {
            "name": col,
            "dtype": str(df[col].dtype),
            "missing": col_missing,
            "missing_pct": round(col_missing / len(df) * 100, 1),
            "unique": int(df[col].nunique()),
            "sample_values": [str(v) for v in df[col].dropna().head(3).tolist()],
        }

        if col in numeric_cols:
            clean_num = df[col].replace(-999, np.nan)
            detail["mean"] = round(float(clean_num.mean()), 2) if not clean_num.isna().all() else None
            detail["std"] = round(float(clean_num.std()), 2) if not clean_num.isna().all() else None
            detail["min"] = round(float(clean_num.min()), 2) if not clean_num.isna().all() else None
            detail["max"] = round(float(clean_num.max()), 2) if not clean_num.isna().all() else None
            try:
                detail["skewness"] = round(float(clean_num.skew()), 2)
            except Exception:
                detail["skewness"] = None
            try:
                Q1 = clean_num.quantile(0.25)
                Q3 = clean_num.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((clean_num < Q1 - 1.5 * IQR) | (clean_num > Q3 + 1.5 * IQR)).sum()
                detail["outliers"] = int(outliers)
            except Exception:
                detail["outliers"] = 0

        column_details.append(detail)

    # Compute memory usage
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": total_missing,
        "missing_pct": round(total_missing / total_cells * 100, 1) if total_cells > 0 else 0,
        "duplicates": duplicate_count,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_by_column": missing_by_column[missing_by_column > 0].to_dict(),
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
        "column_details": column_details,
        "memory_mb": memory_mb,
        "dtype_counts": df.dtypes.astype(str).value_counts().to_dict(),
    }
