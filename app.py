import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1450px;}
    h1, h2, h3 {letter-spacing: -0.4px;}
    .hero {
        padding: 1.35rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554 0%, #1e3a8a 55%, #0f766e 100%);
        color: white;
        margin-bottom: 1.25rem;
    }
    .hero p {color: #dbeafe; margin-bottom: 0;}
    div[data-testid="stMetric"] {
        background: #171b26;
        border: 1px solid #2c3445;
        padding: 18px 20px;
        border-radius: 14px;
    }
    div[data-testid="stMetricLabel"] {font-size: 0.92rem;}
    div[data-testid="stMetricValue"] {font-size: 1.8rem;}
    .section-note {color: #aab4c5; font-size: 0.92rem;}
    .result-card {
        border: 1px solid #344054;
        border-radius: 16px;
        padding: 1.25rem;
        background: #171b26;
    }
    .small-muted {color: #aab4c5; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA / MODEL ----------------
@st.cache_data
def load_data():
    data = pd.read_csv("Telco-Customer-Churn.csv")
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0)
    return data

@st.cache_resource
def load_model():
    bundle = joblib.load("churn_model.pkl")
    # Supports either a bundle dictionary or a model saved directly.
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"], float(bundle.get("threshold", 0.5))
    return bundle, 0.5

try:
    df = load_data()
    model, threshold = load_model()
except Exception as e:
    st.error("I couldn't load the dataset or trained model. Make sure app.py, Telco-Customer-Churn.csv, and churn_model.pkl are in the same folder.")
    st.exception(e)
    st.stop()

required_columns = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 📊 Churn Analytics")
st.sidebar.caption("Customer retention intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Churn Analysis", "Predict Churn"],
    index=0
)
st.sidebar.divider()
st.sidebar.markdown("**Model**")
st.sidebar.caption("Logistic Regression")
st.sidebar.caption(f"Prediction threshold: {threshold:.2f}")
st.sidebar.markdown("---")
st.sidebar.caption("Telco Customer Churn dataset")


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
  <h1 style="margin:0;color:white;">Customer Churn Analytics</h1>
  <p>Understand customer behavior, explore churn patterns, and identify customers who may leave.</p>
</div>
""", unsafe_allow_html=True)

total_customers = len(df)
churn_count = int((df["Churn"] == "Yes").sum())
churn_rate = (churn_count / total_customers * 100) if total_customers else 0
retained_count = total_customers - churn_count
avg_monthly = df["MonthlyCharges"].mean()
avg_tenure = df["tenure"].mean()

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.subheader("Business overview")
    st.markdown('<p class="section-note">A high-level view of customer retention and churn in the dataset.</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{total_customers:,}")
    c2.metric("Customers churned", f"{churn_count:,}")
    c3.metric("Churn rate", f"{churn_rate:.1f}%")
    c4.metric("Average monthly charge", f"${avg_monthly:,.2f}")

    st.write("")
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("### Customer status")
        status = df["Churn"].value_counts().rename_axis("Status").reset_index(name="Customers")
        status["Status"] = status["Status"].map({"No": "Stayed", "Yes": "Churned"})
        fig = px.pie(status, names="Status", values="Customers", hole=0.62,
                     color="Status", color_discrete_map={"Stayed": "#14b8a6", "Churned": "#f97373"})
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Churn by contract")
        contract_churn = df.groupby("Contract", as_index=False).agg(
            Customers=("customerID", "count"),
            Churned=("Churn", lambda x: (x == "Yes").sum())
        )
        contract_churn["Churn rate (%)"] = contract_churn["Churned"] / contract_churn["Customers"] * 100
        fig = px.bar(contract_churn, x="Contract", y="Churn rate (%)", text=contract_churn["Churn rate (%)"].map(lambda x: f"{x:.1f}%"),
                     color="Contract", color_discrete_sequence=["#60a5fa", "#34d399", "#fbbf24"])
        fig.update_layout(showlegend=False, yaxis_title="Churn rate (%)", xaxis_title="", margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Key observations")
    month_to_month = df[df["Contract"] == "Month-to-month"]
    mtm_rate = (month_to_month["Churn"] == "Yes").mean() * 100 if len(month_to_month) else 0
    churn_tenure = df.loc[df["Churn"] == "Yes", "tenure"].mean()
    stay_tenure = df.loc[df["Churn"] == "No", "tenure"].mean()
    o1, o2, o3 = st.columns(3)
    o1.info(f"**Month-to-month contracts:** {mtm_rate:.1f}% churn rate.")
    o2.info(f"**Average tenure of churned customers:** {churn_tenure:.1f} months.")
    o3.info(f"**Average tenure of retained customers:** {stay_tenure:.1f} months.")


# ---------------- CHURN ANALYSIS ----------------
elif page == "Churn Analysis":
    st.subheader("Explore churn patterns")
    st.markdown('<p class="section-note">Use the filters to investigate which customer groups have higher churn.</p>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        contract_filter = st.multiselect("Contract type", sorted(df["Contract"].dropna().unique()), default=sorted(df["Contract"].dropna().unique()))
    with f2:
        internet_filter = st.multiselect("Internet service", sorted(df["InternetService"].dropna().unique()), default=sorted(df["InternetService"].dropna().unique()))

    filtered = df[df["Contract"].isin(contract_filter) & df["InternetService"].isin(internet_filter)].copy()
    if filtered.empty:
        st.warning("No customers match these filters. Select at least one contract and internet service.")
        st.stop()

    a, b, c = st.columns(3)
    a.metric("Filtered customers", f"{len(filtered):,}")
    b.metric("Filtered churn rate", f"{(filtered['Churn'].eq('Yes').mean() * 100):.1f}%")
    c.metric("Average tenure", f"{filtered['tenure'].mean():.1f} months")

    left, right = st.columns(2)
    with left:
        st.markdown("### Churn rate by contract")
        by_contract = filtered.groupby("Contract", as_index=False).agg(
            Customers=("customerID", "count"),
            Churned=("Churn", lambda x: x.eq("Yes").sum())
        )
        by_contract["Churn rate (%)"] = by_contract["Churned"] / by_contract["Customers"] * 100
        fig = px.bar(by_contract, x="Contract", y="Churn rate (%)", text=by_contract["Churn rate (%)"].map(lambda x: f"{x:.1f}%"),
                     color="Contract", color_discrete_sequence=["#60a5fa", "#34d399", "#fbbf24"])
        fig.update_layout(showlegend=False, xaxis_title="", margin=dict(t=15, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Monthly charges vs tenure")
        fig = px.scatter(filtered, x="tenure", y="MonthlyCharges", color="Churn",
                         color_discrete_map={"No": "#14b8a6", "Yes": "#f97373"},
                         labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly charges ($)", "Churn": "Churned"})
        fig.update_layout(margin=dict(t=15, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Churn by tenure group")
    tenure_bins = [0, 12, 24, 48, 72]
    tenure_labels = ["0–12 months", "13–24 months", "25–48 months", "49–72 months"]
    filtered["Tenure group"] = pd.cut(filtered["tenure"], bins=tenure_bins, labels=tenure_labels, include_lowest=True)
    tenure_data = filtered.groupby("Tenure group", observed=False).agg(
        Customers=("customerID", "count"),
        Churned=("Churn", lambda x: x.eq("Yes").sum())
    ).reset_index()
    tenure_data["Churn rate (%)"] = tenure_data["Churned"] / tenure_data["Customers"].replace(0, pd.NA) * 100
    fig = px.bar(tenure_data, x="Tenure group", y="Churn rate (%)", text=tenure_data["Churn rate (%)"].map(lambda x: "" if pd.isna(x) else f"{x:.1f}%"),
                 color="Churn rate (%)", color_continuous_scale="Teal")
    fig.update_layout(xaxis_title="", margin=dict(t=15, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View filtered customer data"):
        st.dataframe(filtered.drop(columns=["Tenure group"], errors="ignore"), use_container_width=True, hide_index=True)


# ---------------- PREDICT CHURN ----------------
else:
    st.subheader("Predict an individual customer's churn risk")
    st.markdown('<p class="section-note">Enter customer details, then select Predict to estimate churn risk using the trained model.</p>', unsafe_allow_html=True)

    with st.form("customer_prediction_form"):
        st.markdown("#### Customer profile")
        p1, p2, p3 = st.columns(3)
        with p1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Has partner?", ["Yes", "No"])
            dependents = st.selectbox("Has dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with p2:
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        with p3:
            support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

        st.markdown("#### Billing details")
        b1, b2 = st.columns(2)
        with b1:
            monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        with b2:
            total = st.number_input("Total charges ($)", min_value=0.0, max_value=20000.0,
                                    value=float(monthly * tenure), step=10.0)

        submitted = st.form_submit_button("🔍 Predict churn risk", type="primary", use_container_width=True)

    if submitted:
        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }], columns=required_columns)

        try:
            probability = float(model.predict_proba(customer)[0][1])
            prediction = probability >= threshold
            st.divider()
            st.markdown("### Prediction result")
            r1, r2 = st.columns([1, 1.4])
            with r1:
                if prediction:
                    st.error("⚠️ Higher churn risk")
                    st.metric("Estimated churn probability", f"{probability * 100:.1f}%")
                    st.write("This customer may be more likely to leave. Consider reviewing their service experience and retention options.")
                else:
                    st.success("✅ Lower churn risk")
                    st.metric("Estimated churn probability", f"{probability * 100:.1f}%")
                    st.write("The model estimates a lower likelihood of churn, though the customer could still leave.")
            with r2:
                # Plotly Express needs a DataFrame when names/values/color are column names.
                gauge_data = pd.DataFrame({
                    "Status": ["Churn probability", "Remaining probability"],
                    "Probability": [probability * 100, 100 - probability * 100],
                })
                gauge = px.pie(
                    gauge_data,
                    names="Status",
                    values="Probability",
                    hole=0.72,
                    color="Status",
                    color_discrete_map={
                        "Churn probability": "#f97373",
                        "Remaining probability": "#293244",
                    },
                )
                gauge.update_traces(textinfo="none", hovertemplate="%{label}: %{value:.1f}%<extra></extra>")
                gauge.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10), annotations=[
                    dict(text=f"{probability * 100:.0f}%", x=0.5, y=0.5, font_size=30, showarrow=False)
                ])
                st.plotly_chart(gauge, use_container_width=True)
            st.caption(f"Decision threshold: {threshold:.2f}. This is a model estimate, not a guarantee of customer behavior.")
        except Exception as e:
            st.error("Prediction failed. Check that the model was trained using the same customer columns as this form.")
            st.exception(e)

st.markdown("---")
st.caption("Customer Churn Analytics • Built with Python, Streamlit, Plotly and scikit-learn")
