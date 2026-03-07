import shap
import joblib
import pandas as pd

from preprocess import load_data

def recommend():

    df = load_data()

    model = joblib.load("models/yield_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")

    X = df.drop(columns=["yield","lot_id","wafer_id","process_date"])

    X_processed = preprocessor.transform(X)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_processed)

    importance = abs(shap_values).mean(axis=0)

    feature_names = [f.split("__")[-1] for f in preprocessor.get_feature_names_out()]

    ranking = sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop Yield Drivers:\n")

    for feature, score in ranking[:5]:
        print(f"{feature}: {score:.4f}")

if __name__ == "__main__":
    recommend()