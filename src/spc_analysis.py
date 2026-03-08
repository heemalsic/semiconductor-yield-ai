import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_PATH = "data/semiconductor_yield_forecasting_data.csv"


def control_chart(parameter):

    os.makedirs("plots", exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    values = df[parameter]

    mean = values.mean()
    std = values.std()

    ucl = mean + 3 * std
    lcl = mean - 3 * std

    plt.figure(figsize=(10,5))

    plt.plot(values.values)
    plt.axhline(mean, label="Mean")
    plt.axhline(ucl, linestyle="--", label="UCL")
    plt.axhline(lcl, linestyle="--", label="LCL")

    plt.title(f"SPC Chart for {parameter}")

    plt.legend()

    plt.tight_layout()

    # save chart
    plt.savefig(f"plots/spc_{parameter}.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    control_chart("pressure")