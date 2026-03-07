import pandas as pd

DATA_PATH = "data/semiconductor_yield_forecasting_data.csv"


def tool_yield():

    df = pd.read_csv(DATA_PATH)

    tool_columns = [
        "etch_tool",
        "litho_tool",
        "deposition_tool",
        "implant_tool"
    ]

    for tool in tool_columns:

        result = df.groupby(tool)["yield"].mean().sort_values()

        print("\nTool Yield Analysis:", tool)
        print(result)


if __name__ == "__main__":
    tool_yield()