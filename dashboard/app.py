import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

DATA_PATH = "data/semiconductor_yield_forecasting_data.csv"
model = joblib.load("models/yield_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

df = pd.read_csv(DATA_PATH)

st.title("Semiconductor Yield Analytics Dashboard")

st.subheader("Yield Distribution")

fig = px.histogram(df, x="yield")

st.plotly_chart(fig)


st.subheader("Yield vs Defect Density")

fig = px.scatter(
    df,
    x="defect_density",
    y="yield",
    color="technology_node"
)

st.plotly_chart(fig)


st.subheader("Yield by Lithography Tool")

fig = px.box(
    df,
    x="litho_tool",
    y="yield"
)

st.plotly_chart(fig)


st.subheader("Process Parameter Correlation")

corr = df.corr(numeric_only=True)

fig = px.imshow(corr)

st.plotly_chart(fig)

st.header("Predict Yield for Custom Process Parameters")

st.write("Enter semiconductor process parameters to estimate wafer yield.")

# numeric parameters
etch_rate = st.number_input("Etch Rate", value=float(df["etch_rate"].mean()))
pressure = st.number_input("Pressure", value=float(df["pressure"].mean()))
temperature = st.number_input("Temperature", value=float(df["temperature"].mean()))
dose = st.number_input("Dose", value=float(df["dose"].mean()))
implant_energy = st.number_input("Implant Energy", value=float(df["implant_energy"].mean()))
defect_density = st.number_input("Defect Density", value=float(df["defect_density"].mean()))
critical_dimension = st.number_input("Critical Dimension", value=float(df["critical_dimension"].mean()))
oxide_thickness = st.number_input("Oxide Thickness", value=float(df["oxide_thickness"].mean()))
resistance = st.number_input("Resistance", value=float(df["resistance"].mean()))

# categorical parameters
product_type = st.selectbox("Product Type", df["product_type"].unique())
technology_node = st.selectbox("Technology Node", df["technology_node"].unique())
etch_tool = st.selectbox("Etch Tool", df["etch_tool"].unique())
litho_tool = st.selectbox("Lithography Tool", df["litho_tool"].unique())
deposition_tool = st.selectbox("Deposition Tool", df["deposition_tool"].unique())
implant_tool = st.selectbox("Implant Tool", df["implant_tool"].unique())

if st.button("Predict Yield"):

    # create a default row using dataset averages
    input_data = df.mean(numeric_only=True).to_dict()

    # update with user inputs
    input_data.update({
        "product_type": product_type,
        "technology_node": technology_node,
        "etch_tool": etch_tool,
        "litho_tool": litho_tool,
        "deposition_tool": deposition_tool,
        "implant_tool": implant_tool,
        "etch_rate": etch_rate,
        "pressure": pressure,
        "temperature": temperature,
        "dose": dose,
        "implant_energy": implant_energy,
        "defect_density": defect_density,
        "critical_dimension": critical_dimension,
        "oxide_thickness": oxide_thickness,
        "resistance": resistance
    })

    input_df = pd.DataFrame([input_data])

    # drop target column if present
    if "yield" in input_df.columns:
        input_df = input_df.drop(columns=["yield"])

    # transform
    X_processed = preprocessor.transform(input_df)

    prediction = model.predict(X_processed)[0]

    st.success(f"Predicted Wafer Yield: {prediction:.3f}")