import shap
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt

from preprocess import load_data, preprocess_data


def analyze():

    os.makedirs("plots", exist_ok=True)

    df = load_data()

    preprocessor = joblib.load("models/preprocessor.pkl")

    df = df.drop(columns=["lot_id", "wafer_id", "process_date"])

    y = df["yield"]
    X = df.drop(columns=["yield"])

    X_processed = preprocessor.transform(X)

    feature_names = [f.split("__")[-1] for f in preprocessor.get_feature_names_out()]

    model = joblib.load("models/yield_model.pkl")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_processed)

    shap.summary_plot(
        shap_values,
        X_processed,
        feature_names=feature_names,
        show=False
    )

    plt.savefig("plots/shap_summary.png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    analyze()