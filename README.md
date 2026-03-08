# AI-Driven Semiconductor Yield Analytics Platform

An interactive machine learning system for **predicting wafer yield, identifying yield limiters, and analyzing semiconductor manufacturing processes**.

This project simulates the workflow of **yield engineers in semiconductor fabs**, combining statistical analysis, machine learning, explainable AI, and interactive dashboards to understand how fabrication parameters influence chip yield.

# Analytics Dashboard

![UI](https://github.com/heemalsic/semiconductor-yield-ai/blob/main/plots/dashboard_1.png)
---

# Overview

Semiconductor manufacturing involves hundreds of tightly controlled process parameters. Small variations in parameters such as **etch rate, temperature, pressure, or defect density** can significantly impact wafer yield.

This platform provides tools to:

* Predict wafer yield using machine learning
* Identify key process parameters impacting yield
* Detect abnormal wafers and process excursions
* Analyze fabrication tool performance
* Visualize relationships between process variables
* Interactively simulate how process changes affect yield

The system mimics how **yield engineering analytics platforms** are used inside semiconductor fabs.

---

# Key Features

## 1. Yield Prediction (Machine Learning)

A supervised learning model predicts wafer yield using manufacturing process data.

Model used:

* **XGBoost Regressor**

Inputs include:

* Process parameters (pressure, temperature, etch rate)
* Electrical measurements
* Physical measurements
* Defect statistics
* Manufacturing tool information

The model allows early estimation of yield before expensive wafer testing is completed.

---

## 2. Root Cause Analysis (Explainable AI)

The project uses **SHAP (SHapley Additive exPlanations)** to interpret the trained model.

SHAP identifies which process parameters most influence predicted yield.

Example insights:

* High defect density reduces yield
* Pressure instability correlates with yield loss
* Oxide thickness variation impacts device performance

This enables engineers to quickly identify **yield limiters**.

---

## 3. Interactive Yield Prediction

The Streamlit dashboard allows users to enter custom process parameters such as:

* Etch rate
* Pressure
* Temperature
* Defect density
* Tool selections

The system then predicts wafer yield using the trained model.

This simulates a **process tuning tool used by yield engineers**.

---

## 4. Tool Performance Analysis

The system analyzes yield by fabrication tool:

* Lithography tools
* Etch tools
* Deposition tools
* Implantation tools

This helps identify tools that may be causing yield degradation.

Example output:

Lithography Tool L3 → lower average yield compared to other tools.

---

## 5. Yield Excursion Detection

Anomaly detection is used to identify wafers with abnormal process behavior.

Algorithm used:

* **Isolation Forest**

This helps detect potential **process excursions**, where the manufacturing process deviates from normal operating conditions.

---

## 6. Statistical Process Control (SPC)

The project implements classic manufacturing monitoring techniques.

SPC charts are used to detect:

* Process drift
* Out-of-control process parameters
* Process instability

Example monitored parameters:

* Pressure
* Temperature
* Etch rate

---

## 7. Process Parameter Correlation Analysis

A correlation heatmap visualizes statistical relationships between fabrication variables.

This helps engineers quickly identify relationships such as:

* Defect density vs yield
* Temperature vs oxide thickness
* Pressure vs defect formation

---

## 8. Interactive Dashboard

The project includes a **Streamlit dashboard** for interactive analytics.

The dashboard visualizes:

* Yield distribution
* Yield vs defect density
* Yield by lithography tool
* Process parameter correlations
* Interactive yield prediction

---

# Project Architecture

```
Manufacturing Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Machine Learning Model
(Yield Prediction)
        │
        ▼
Explainability (SHAP)
        │
        ▼
Manufacturing Analytics
        │
        ▼
Interactive Streamlit Dashboard
```

---

# Tech Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Explainable AI

* SHAP

### Visualization

* Plotly
* Matplotlib

### Dashboard

* Streamlit

---

# Project Structure

```
semiconductor-yield-ai
│
├ data
│   semiconductor_yield_forecasting_data.csv
│
├ src
│   preprocess.py
│   train_model.py
│   root_cause_analysis.py
│   anomaly_detection.py
│   spc_analysis.py
│   tool_analysis.py
│
├ dashboard
│   app.py
│
├ models
│   yield_model.pkl
│   preprocessor.pkl
│
├ requirements.txt
└ README.md
```

---

# Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/semiconductor-yield-ai.git
cd semiconductor-yield-ai
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Training the Model

Run the training script:

```
python src/train_model.py
```

This trains the yield prediction model and saves it to:

```
models/yield_model.pkl
```

---

# Root Cause Analysis

To analyze feature importance:

```
python src/root_cause_analysis.py
```

This generates SHAP plots identifying the key parameters impacting yield.

---

# Running the Dashboard

Start the Streamlit dashboard:

```
streamlit run dashboard/app.py
```

Then open the browser at:

```
http://localhost:8501
```

---

# Example Use Cases

This platform can simulate common semiconductor engineering tasks:

* Early wafer yield prediction
* Identifying yield limiters
* Investigating process excursions
* Monitoring fab process stability
* Evaluating tool performance
* Process optimization experiments

---

# Future Improvements

Possible extensions include:

* Real-time manufacturing data streaming
* Bayesian process optimization
* Automated yield improvement recommendations
* Digital twin simulations of fab processes

---

# Author

Himal Sharma
MS Applied Machine Learning
University of Maryland
