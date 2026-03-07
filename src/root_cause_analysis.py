import shap
import joblib
import pandas as pd

from preprocess import load_data, preprocess_data

def analyze():

    df = load_data()

    # load preprocessor
    preprocessor = joblib.load("models/preprocessor.pkl")

    # prepare data
    df = df.drop(columns=["lot_id", "wafer_id", "process_date"])

    y = df["yield"]
    X = df.drop(columns=["yield"])

    # transform data
    X_processed = preprocessor.transform(X)

    # get feature names
    feature_names = preprocessor.get_feature_names_out()

    # load trained model
    model = joblib.load("models/yield_model.pkl")

    # compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)

    # summary plot with real feature names
    shap.summary_plot(
        shap_values,
        X_processed,
        feature_names=feature_names
    )


if __name__ == "__main__":
    analyze()