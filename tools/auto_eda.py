import pandas as pd

def generate_eda_summary(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(exclude="number").columns.tolist()

    summary = (
        f"The dataset contains {len(df)} rows and {len(df.columns)} columns. "
        f"It has {len(numeric)} numeric and {len(categorical)} categorical columns. "
        f"There are {int(df.isna().sum().sum())} missing cells and "
        f"{int(df.duplicated().sum())} duplicate rows."
    )

    return {
        "summary": summary,
        "numeric_columns": numeric,
        "categorical_columns": categorical,
    }
