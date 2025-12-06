import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------ UI SETUP ------------------------------ #
st.set_page_config(page_title="Sales Predictor", layout="wide")
st.title("📊 Sales Prediction Dashboard")
st.write("Upload a CSV file with sales features to generate predictions.")

# ------------------------------ LOAD MODEL ---------------------------- #
pipeline = joblib.load("sales_pipeline.pkl")

# ------------------------------ FILE UPLOAD --------------------------- #
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(df.head())

    # ------------------ Prediction ------------------ #
    try:
        predictions = pipeline.predict(df)
        df["Predicted_Sales"] = predictions
    except Exception as e:
        st.error(f"🚫 Error during prediction: {e}")
        st.stop()

    # ------------------ Results Table ------------------ #
    st.subheader("📈 Prediction Output")
    st.dataframe(df.head())

    # ------------------ Stats ------------------ #
    st.subheader("📊 Summary Statistics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Max Predicted Sales", f"{df['Predicted_Sales'].max():.2f}")
    col2.metric("Min Predicted Sales", f"{df['Predicted_Sales'].min():.2f}")
    col3.metric("Avg Predicted Sales", f"{df['Predicted_Sales'].mean():.2f}")

    # ------------------ Visualizations ------------------ #
    st.subheader("📊 Visual Analysis")

    colA, colB = st.columns(2)

    # Distribution Plot
    with colA:
        st.write("📌 **Prediction Distribution**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(df["Predicted_Sales"], kde=True, ax=ax1)
        st.pyplot(fig1)

    # Scatter Plot
    with colB:
        st.write("📌 **Sequential Predictions (Trend)**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=df["Predicted_Sales"], marker="o", ax=ax2)
        ax2.set_xlabel("Record Index")
        ax2.set_ylabel("Predicted Sales")
        st.pyplot(fig2)

    # ------------------ Download Button ------------------ #
    csv_output = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=csv_output,
        file_name="sales_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("📂 Please upload a CSV file to begin.")
