import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def detect_task(y):
    if y.dtype == "object" or str(y.dtype).startswith("category") or y.nunique() <= 10:
        return "classification"
    return "regression"


def run_baseline_ml(df: pd.DataFrame, target: str) -> dict:
    data = df.dropna(subset=[target]).copy()
    if data.empty:
        return {"error": f"Target column '{target}' has no non-null values."}

    X = data.drop(columns=[target])
    y = data[target]

    # Validate target cardinality
    if y.dtype == "object" and y.nunique() > 0.5 * len(data) and len(data) > 30:
        return {
            "error": f"Target column '{target}' contains high-cardinality text ({y.nunique()} unique values out of {len(data)} rows). "
                     f"Text/ID columns like 'Name' or 'Ticket' cannot be used as ML target variables. Please select a valid target column like 'Survived' or 'Age'."
        }

    task = detect_task(y)

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    if task == "classification":
        # Label encode string targets if needed
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=150, random_state=42, n_jobs=-1
            ),
        }
        scoring = "F1"
        metric_fn = f1_score
        
        # Only stratify if all classes have at least 2 samples
        class_counts = y.value_counts()
        stratify = y if (class_counts.min() >= 2 and len(class_counts) > 1) else None
    else:
        # Convert non-numeric regression targets if possible
        if y.dtype == "object":
            y = pd.to_numeric(y, errors="coerce")
            valid_idx = y.dropna().index
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(
                n_estimators=150, random_state=42, n_jobs=-1
            ),
        }
        scoring = "R2"
        metric_fn = r2_score
        stratify = None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except Exception as e:
        # Fallback without stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )

    results = []
    fitted = {}

    for name, model in models.items():
        try:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            if task == "classification":
                score = metric_fn(y_test, pred, average="weighted", zero_division=0)
            else:
                score = metric_fn(y_test, pred)

            results.append({"model": name, scoring: float(score)})
            fitted[name] = pipe
        except Exception as ex:
            results.append({"model": name, scoring: 0.0})

    results_df = pd.DataFrame(results).sort_values(scoring, ascending=False)
    best_name = results_df.iloc[0]["model"]

    return {
        "task": task,
        "best_model": best_name,
        "metric": scoring,
        "best_score": float(results_df.iloc[0][scoring]),
        "model_results": results_df,
    }

