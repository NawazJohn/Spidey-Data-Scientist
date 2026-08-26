import pandas as pd
import numpy as np


def filter_dataset(df: pd.DataFrame, options: dict) -> tuple:
    """
    Apply selected cleaning operations to a DataFrame.
    Returns (cleaned_df, changelog) where changelog is a list of actions taken.
    """
    changelog = []
    cleaned = df.copy()
    original_shape = cleaned.shape

    # 1. Drop fully-null rows
    if options.get("drop_null_rows", False):
        before = len(cleaned)
        cleaned = cleaned.dropna(how="all")
        dropped = before - len(cleaned)
        if dropped > 0:
            changelog.append(f"🗑️ Dropped {dropped} fully-null rows.")

    # 2. Drop duplicate rows
    if options.get("drop_duplicates", False):
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        dropped = before - len(cleaned)
        if dropped > 0:
            changelog.append(f"🗑️ Dropped {dropped} duplicate rows.")

    # 3. Drop constant (zero-variance) columns
    if options.get("drop_constant_cols", False):
        constant_cols = [c for c in cleaned.columns if cleaned[c].nunique(dropna=False) <= 1]
        if constant_cols:
            cleaned = cleaned.drop(columns=constant_cols)
            changelog.append(f"🗑️ Dropped {len(constant_cols)} constant column(s): {', '.join(constant_cols)}.")

    # 4. Drop fully-null columns
    if options.get("drop_null_cols", False):
        null_cols = [c for c in cleaned.columns if cleaned[c].isna().all()]
        if null_cols:
            cleaned = cleaned.drop(columns=null_cols)
            changelog.append(f"🗑️ Dropped {len(null_cols)} fully-null column(s): {', '.join(null_cols)}.")

    # 5. Drop ID-like columns (high cardinality categorical)
    if options.get("drop_id_cols", False):
        id_cols = []
        for c in cleaned.select_dtypes(include=["object", "category"]).columns:
            if cleaned[c].nunique() > 0.9 * len(cleaned) and len(cleaned) > 50:
                id_cols.append(c)
        if id_cols:
            cleaned = cleaned.drop(columns=id_cols)
            changelog.append(f"🗑️ Dropped {len(id_cols)} ID-like column(s): {', '.join(id_cols)}.")

    # 6. Impute missing values
    if options.get("impute_missing", False):
        numeric_cols = cleaned.select_dtypes(include="number").columns
        categorical_cols = cleaned.select_dtypes(exclude="number").columns

        num_imputed = 0
        for c in numeric_cols:
            missing = cleaned[c].isna().sum()
            if missing > 0:
                cleaned[c] = cleaned[c].fillna(cleaned[c].median())
                num_imputed += missing

        cat_imputed = 0
        for c in categorical_cols:
            missing = cleaned[c].isna().sum()
            if missing > 0:
                cleaned[c] = cleaned[c].fillna(cleaned[c].mode().iloc[0] if not cleaned[c].mode().empty else "Unknown")
                cat_imputed += missing

        if num_imputed > 0:
            changelog.append(f"🔧 Imputed {num_imputed} missing numeric values (median).")
        if cat_imputed > 0:
            changelog.append(f"🔧 Imputed {cat_imputed} missing categorical values (mode).")

    # 7. Remove outliers (IQR method)
    if options.get("remove_outliers", False):
        before = len(cleaned)
        numeric_cols = cleaned.select_dtypes(include="number").columns
        for c in numeric_cols:
            Q1 = cleaned[c].quantile(0.25)
            Q3 = cleaned[c].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            cleaned = cleaned[(cleaned[c] >= lower) & (cleaned[c] <= upper) | cleaned[c].isna()]
        dropped = before - len(cleaned)
        if dropped > 0:
            changelog.append(f"📊 Removed {dropped} outlier rows (IQR method).")

    if not changelog:
        changelog.append("✅ No changes applied.")

    return cleaned, changelog
