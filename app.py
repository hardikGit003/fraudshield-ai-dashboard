import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


feature_columns = joblib.load("feature_columns.pkl")

# CUSTOM CSS

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: white;
}

div[data-testid="metric-container"] {
    background-color: #262730;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #444;
    text-align: center;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    background-color: #FF4B4B;
    color: white;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #ff1a1a;
    color: white;
}

</style>
""", unsafe_allow_html=True)
# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():

    url = "https://drive.google.com/uc?export=download&id=1PnD1yJM1EpEjt1o9h2U1MTxEPRkURNQk"

    df = pd.read_csv(url)

    return df

df = load_data()
st.write(df.head())
st.write(df.columns)
# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------


st.sidebar.markdown("""
# 💳 FraudShield AI

### Fraud Analytics Platform
""")

page = st.sidebar.radio(
    "Navigation",
    [
    "Overview",
    "Data Visualization",
    "Model Analytics",
    "Fraud Prediction",
    "Live Monitoring",
    "CSV Fraud Scanner"
   ]
)

# ---------------------------------------------------
# OVERVIEW PAGE
# ---------------------------------------------------

if page == "Overview":

    st.markdown("""
    # 💳 Credit Card Fraud Detection Dashboard

    ### AI-Powered Financial Fraud Monitoring System

    Analyze transactions, detect fraudulent activity, monitor live transaction risk, and generate fraud intelligence reports using Machine Learning.
    """)

    st.markdown("""
    Detect fraudulent credit card transactions using Machine Learning and Random Forest Classifier.
    """)

    total_transactions = len(df)
    fraud_cases = df[df["Class"] == 1].shape[0]
    normal_cases = df[df["Class"] == 0].shape[0]

    # KPI CARDS
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Fraud Cases", fraud_cases)
    col3.metric("Normal Cases", normal_cases)

    st.markdown("---")

    # Dataset Preview
    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.markdown("---")

    # Fraud Ratio
    fraud_ratio = (fraud_cases / total_transactions) * 100

    st.info(f"Fraudulent transactions represent only {fraud_ratio:.4f}% of total transactions.")

# ---------------------------------------------------
# DATA VISUALIZATION PAGE
# ---------------------------------------------------

elif page == "Data Visualization":

    st.title("📊 Data Visualization")

    # PIE CHART
    st.subheader("Fraud vs Non-Fraud Distribution")

    fig1 = px.pie(
        df,
        names="Class",
        title="Fraud Distribution",
        color="Class",
        color_discrete_map={
            0: "blue",
            1: "red"
        }
    )

    st.plotly_chart(fig1, use_container_width=True)

    # HISTOGRAM
    st.subheader("Transaction Amount Distribution")

    fig2 = px.histogram(
        df,
        x="Amount",
        color="Class",
        nbins=50,
        title="Amount Distribution by Transaction Type"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # CORRELATION HEATMAP
    st.subheader("Correlation Heatmap")

    corr = df.corr()

    fig3 = px.imshow(
        corr,
        aspect="auto",
        title="Feature Correlation Matrix"
    )

    st.plotly_chart(fig3, use_container_width=True)




# ---------------------------------------------------
# MODEL ANALYTICS PAGE
# ---------------------------------------------------

elif page == "Model Analytics":

    st.title("📈 Model Analytics Dashboard")

    st.markdown("""
    Analyze Random Forest model performance using classification metrics and evaluation charts.
    """)

    # SAMPLE DATA
      # PREPARE DATA

    df_model = df.copy()

    # CREATE SCALED COLUMNS
    df_model["scaled_amount"] = scaler.transform(
        df_model[["Amount"]]
    )

    df_model["scaled_time"] = scaler.transform(
        df_model[["Time"]]
    )

    # DROP ORIGINAL COLUMNS
    df_model = df_model.drop(["Amount", "Time"], axis=1)

    # FEATURES + TARGET
    X = df_model.drop("Class", axis=1)
    # MATCH TRAINING COLUMN ORDER
    X = X[feature_columns]

    y = df_model["Class"]

    # PREDICTIONS
    y_pred = model.predict(X)

    # PROBABILITIES
    y_prob = model.predict_proba(X)[:, 1]

    # ---------------------------------------------------
    # METRICS
    # ---------------------------------------------------

    col1, col2, col3 = st.columns(3)

    accuracy = model.score(X, y)

    cm = confusion_matrix(y, y_pred)

    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")

    st.markdown("---")

    # ---------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(3.5,3))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Reds',
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    st.markdown("---")

    # ---------------------------------------------------
    # ROC CURVE
    # ---------------------------------------------------

    st.subheader("ROC Curve")

    fpr, tpr, thresholds = roc_curve(y, y_prob)

    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots(figsize=(4,3))

    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")

    ax2.plot([0,1], [0,1], linestyle='--')

    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")

    ax2.legend()

    st.pyplot(fig2)

    st.markdown("---")

    # ---------------------------------------------------
    # CLASSIFICATION REPORT
    # ---------------------------------------------------

    st.subheader("Classification Report")

    report = classification_report(y, y_pred)

    st.text(report)

    st.markdown("---")



    # ---------------------------------------------------
    # FEATURE COEFFICIENTS
    # ---------------------------------------------------

    st.subheader("Feature Impact Analysis")

    coefficients = model.coef_[0]

    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": coefficients
    })

    feature_df["Absolute"] = feature_df["Coefficient"].abs()

    feature_df = feature_df.sort_values(
        by="Absolute",
        ascending=False
    ).head(10)

    fig3 = px.bar(
        feature_df,
        x="Coefficient",
        y="Feature",
        orientation='h',
        title="Top Features Influencing Fraud Prediction"
    )

    st.plotly_chart(fig3, use_container_width=True)

                        

# ---------------------------------------------------
# FRAUD PREDICTION PAGE
# ---------------------------------------------------



elif page == "Fraud Prediction":

    st.title("🔍 Fraud Prediction System")

    st.markdown("""
    Enter transaction details to analyze fraud risk using the Random Forest model.
    """)

    # USER INPUTS
    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0
    )

    time = st.number_input(
        "Transaction Time",
        min_value=0.0,
        value=1.0
    )

    st.markdown("---")

    # CREATE INPUT ARRAY
    input_data = np.zeros((1, 30))

    # SET VALUES
    input_data[0][0] = time
    input_data[0][29] = amount

    # SCALE ONLY AMOUNT
    try:
        scaled_amount = scaler.transform([[amount]])[0][0]
        input_data[0][29] = scaled_amount
    except:
        pass

    # PREDICT BUTTON
    if st.button("Analyze Transaction"):

        

        prediction = model.predict(input_data)

        probability = model.predict_proba(input_data)[0][1]

        st.markdown("---")

        st.subheader("🧠 Fraud Risk Analysis")

        # RISK SCORE
        risk_score = probability * 100

        # METRIC CARDS
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Fraud Probability",
                f"{risk_score:.2f}%"
            )

        with col2:
            if risk_score > 80:
                risk_level = "HIGH RISK"
            elif risk_score > 50:
                risk_level = "MEDIUM RISK"
            else:
                risk_level = "LOW RISK"

            st.metric(
                "Risk Level",
                risk_level
            )

        st.markdown("---")

        # PROGRESS BAR
        st.subheader("Fraud Confidence Meter")

        st.progress(float(probability))

        st.write(
            f"Model confidence for fraud detection: {risk_score:.2f}%"
        )

        st.markdown("---")

        # FINAL RESULT
        if probability > 0.80:

            st.error("🚨 Fraudulent Transaction Detected")

        elif probability > 0.50:

            st.warning("⚠ Suspicious Transaction Pattern")

        else:

            st.success("✅ Legitimate Transaction")

        st.markdown("---")

        # AI INSIGHTS
        st.subheader("🤖 AI Transaction Insights")

        insights = []

        if amount > 10000:
            insights.append(
                "High-value transaction detected."
            )

        if amount > 50000:
            insights.append(
                "Unusually large transaction amount."
            )

        if time < 1000:
            insights.append(
                "Transaction occurred during unusual activity window."
            )

        if probability > 0.70:
            insights.append(
                "Model detected patterns similar to historical fraud cases."
            )

        if probability < 0.30:
            insights.append(
                "Transaction behavior appears consistent with legitimate activity."
            )

        if len(insights) == 0:
            insights.append(
                "No major fraud indicators detected."
            )

        for insight in insights:
            st.info(insight)

        st.markdown("---")

        # TRANSACTION SUMMARY
        st.subheader("📋 Transaction Summary")

        summary_df = pd.DataFrame({
            "Metric": [
                "Transaction Amount",
                "Transaction Time",
                "Fraud Probability",
                "Risk Classification"
            ],
            "Value": [
                amount,
                time,
                f"{risk_score:.2f}%",
                risk_level
            ]
        })

        st.table(summary_df)


# ---------------------------------------------------
# LIVE MONITORING PAGE
# ---------------------------------------------------

elif page == "Live Monitoring":

    st.title("📡 Live Fraud Monitoring System")

    st.markdown("""
    Simulate real-time transaction monitoring using live transaction samples from the dataset.
    """)

    st.markdown("---")

    # RANDOM TRANSACTION BUTTON
    if st.button("Generate Live Transaction"):

        # RANDOM SAMPLE
        sample = df.sample(1)

        # DISPLAY TRANSACTION
        st.subheader("💳 Incoming Transaction")

        st.dataframe(sample)

        st.markdown("---")

        # PREPARE DATA
        # PREPARE SAMPLE FEATURES

        sample_features = sample.copy()

        # CREATE SCALED COLUMNS
        sample_features["scaled_amount"] = scaler.transform(
            sample_features[["Amount"]]
        )

        sample_features["scaled_time"] = scaler.transform(
            sample_features[["Time"]]
        )

        # DROP ORIGINAL COLUMNS
        sample_features = sample_features.drop(
            ["Amount", "Time", "Class"],
            axis=1
        )

        # MATCH TRAINING COLUMN ORDER
        sample_features = sample_features[feature_columns]

        # PREDICT
        prediction = model.predict(sample_features)[0]

        probability = model.predict_proba(sample_features)[0][1]

        risk_score = probability * 100

        # DISPLAY RESULTS
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Fraud Probability",
                f"{risk_score:.2f}%"
            )

        with col2:

            if risk_score > 80:
                risk = "HIGH RISK"

            elif risk_score > 50:
                risk = "MEDIUM RISK"

            else:
                risk = "LOW RISK"

            st.metric(
                "Risk Level",
                risk
            )

        st.markdown("---")

        # ALERTS
        if prediction == 1:

            st.error("🚨 Fraudulent Transaction Alert")

        else:

            st.success("✅ Legitimate Transaction")

        st.markdown("---")

        # LIVE INSIGHTS
        st.subheader("🤖 Monitoring Insights")

        amount_value = float(sample["Amount"].values[0])

        if amount_value > 10000:
            st.info("High-value transaction detected.")

        if probability > 0.70:
            st.info("Transaction pattern similar to historical fraud behavior.")

        if prediction == 0:
            st.info("Transaction behavior appears normal.")


# ---------------------------------------------------
# CSV FRAUD SCANNER
# ---------------------------------------------------

elif page == "CSV Fraud Scanner":

    st.title("📂 CSV Fraud Scanner")

    st.markdown("""
    Upload transaction datasets to scan for potentially fraudulent transactions.
    """)

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        uploaded_df = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Dataset")

        st.dataframe(
        uploaded_df.head(),
        use_container_width=True
        )

        st.markdown("---")

        with st.spinner("Scanning uploaded transactions..."):

            try:

                # COPY DATA
                prediction_df = uploaded_df.copy()

                # CREATE SCALED COLUMNS
                prediction_df["scaled_amount"] = scaler.transform(
                    prediction_df[["Amount"]]
                )

                prediction_df["scaled_time"] = scaler.transform(
                    prediction_df[["Time"]]
                )

                # DROP ORIGINAL COLUMNS
                prediction_df = prediction_df.drop(
                    ["Amount", "Time"],
                    axis=1
                )

                # REMOVE TARGET IF EXISTS
                if "Class" in prediction_df.columns:
                    prediction_df = prediction_df.drop(
                        "Class",
                        axis=1
                    )

                # MATCH TRAINING ORDER
                prediction_df = prediction_df[feature_columns]

                # PREDICTIONS
                predictions = model.predict(prediction_df)

                probabilities = model.predict_proba(
                    prediction_df
                )[:,1]

                # OUTPUT RESULTS
                uploaded_df["Fraud Prediction"] = predictions

                uploaded_df["Fraud Probability"] = (
                    probabilities * 100
                ).round(2)

                st.subheader("🧠 Fraud Scan Results")
                st.success("Fraud analysis completed successfully.")

                st.dataframe(
                    uploaded_df,
                    use_container_width=True
                )

                st.markdown("---")

                # SUMMARY
                fraud_count = (predictions == 1).sum()

                normal_count = (predictions == 0).sum()

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Fraudulent Transactions",
                        fraud_count
                    )

                with col2:
                    st.metric(
                        "Legitimate Transactions",
                        normal_count
                    )

                st.markdown("---")

                # DOWNLOAD BUTTON
                csv = uploaded_df.to_csv(index=False)

                st.download_button(
                    label="⬇ Download Results",
                    data=csv,
                    file_name="fraud_scan_results.csv",
                    mime="text/csv"
                )

            except Exception as e:

                st.error(f"Error processing file: {e}")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>

💳 FraudShield AI | Built with Streamlit, Machine Learning & Python

</div>
""", unsafe_allow_html=True)