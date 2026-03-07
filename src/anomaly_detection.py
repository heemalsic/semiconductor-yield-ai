import pandas as pd
from sklearn.ensemble import IsolationForest

DATA_PATH = "data/semiconductor_yield_forecasting_data.csv"


def detect_anomalies():

    df = pd.read_csv(DATA_PATH)

    features = df[
        [
            "etch_rate",
            "pressure",
            "temperature",
            "defect_density",
            "critical_dimension",
            "oxide_thickness",
            "resistance"
        ]
    ]

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    df["anomaly_score"] = model.fit_predict(features)

    anomalies = df[df["anomaly_score"] == -1]

    print("Yield excursions detected:", len(anomalies))

    return anomalies


if __name__ == "__main__":
    anomalies = detect_anomalies()
    print(anomalies.head())