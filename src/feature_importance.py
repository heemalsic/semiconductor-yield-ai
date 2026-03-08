import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

from preprocess import load_data


def plot_importance():

    os.makedirs("plots", exist_ok=True)

    df = load_data()

    model = joblib.load("models/yield_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")

    feature_names = [f.split("__")[-1] for f in preprocessor.get_feature_names_out()]

    importance = model.feature_importances_

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    imp_df = imp_df.sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(10,6))

    plt.barh(imp_df["feature"], imp_df["importance"])

    plt.gca().invert_yaxis()

    plt.title("Top Process Parameters Affecting Yield")

    plt.tight_layout()

    # save plot
    plt.savefig("plots/feature_importance.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    plot_importance()