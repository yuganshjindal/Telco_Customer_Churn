import os
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Telco Churn Classifier", layout="wide")
st.title("Telco Customer Churn Prediction")

@st.cache_data
def load_metrics_table():
    return pd.read_csv("model/metrics_table.csv")

@st.cache_resource
def load_models():
    model_dir = "model/saved_models"
    files = [f for f in os.listdir(model_dir) if f.endswith(".joblib") and f != "all_models.joblib"]

    models = {}
    for f in files:
        key = f.replace(".joblib", "").replace("_", " ").title()
        models[key] = joblib.load(os.path.join(model_dir, f))

    return models

metrics_df = load_metrics_table()
models = load_models()

st.sidebar.header("Controls")

model_names = sorted(models.keys())
selected_model_name = st.sidebar.selectbox("Select Model", model_names)

uploaded_file = st.sidebar.file_uploader("Upload CSV (test data)", type=["csv"])

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Model Performance Comparison (Training Data)")
    st.dataframe(metrics_df, use_container_width=True)

with col2:
    st.subheader("Selected Model")
    st.write(f"**{selected_model_name}**")

if uploaded_file is None:
    st.warning("Upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("Uploaded Data Preview")
st.dataframe(df.head(15), use_container_width=True)

model = models[selected_model_name]

st.subheader("Model Predictions")

if "Churn" in df.columns:
    X = df.drop(columns=["Churn"])
    y_true = df["Churn"]
    has_labels = True
else:
    X = df.copy()
    has_labels = False

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

pred_df = X.copy()
pred_df["Predicted_Churn"] = y_pred
pred_df["Churn_Probability"] = y_prob

st.dataframe(pred_df.head(20), use_container_width=True)

if has_labels:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    st.write(pd.DataFrame(
        cm,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"]
    ))

    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred))
else:
    st.info("No ground-truth labels found in uploaded data. Showing prediction insights only.")

    st.subheader("Prediction Distribution")
    st.bar_chart(pd.Series(y_pred).value_counts())

    st.subheader("Average Churn Probability")
    st.write(round(y_prob.mean(), 4))
