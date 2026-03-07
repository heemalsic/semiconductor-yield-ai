import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/semiconductor_yield_forecasting_data.csv"


def control_chart(parameter):

    df = pd.read_csv(DATA_PATH)

    values = df[parameter]

    mean = values.mean()
    std = values.std()

    ucl = mean + 3 * std
    lcl = mean - 3 * std

    plt.figure(figsize=(10,5))

    plt.plot(values.values)
    plt.axhline(mean)
    plt.axhline(ucl)
    plt.axhline(lcl)

    plt.title(f"SPC Chart for {parameter}")

    plt.show()


if __name__ == "__main__":
    control_chart("pressure")