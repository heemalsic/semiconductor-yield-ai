import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

DATA_PATH = "data/semiconductor_yield_forecasting_data.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def preprocess_data(df):

    df = df.copy()

    # drop identifiers
    df = df.drop(columns=["lot_id", "wafer_id", "process_date"])

    target = "yield"

    y = df[target]
    X = df.drop(columns=[target])

    categorical_cols = [
        "product_type",
        "technology_node",
        "etch_tool",
        "litho_tool",
        "deposition_tool",
        "implant_tool"
    ]

    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    os.makedirs("models", exist_ok=True)

    joblib.dump(preprocessor, "models/preprocessor.pkl")

    return X_processed, y


if __name__ == "__main__":
    df = load_data()
    X, y = preprocess_data(df)
    print("Preprocessing complete")