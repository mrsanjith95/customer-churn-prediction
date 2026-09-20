import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Custom CSS for Professional Dashboard Styling
# --------------------------------------------------

st.markdown("""
<style>
    /* Global Container Adjustments for Single Viewport */
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 0.3rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] {
        display: none !important;
    }
    footer {
        display: none !important;
    }
    
    /* Layout Spacing */
    div[data-testid="stVerticalBlock"] {
        gap: 0.35rem !important;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 0.8rem !important;
    }
    
    /* Typography & Labels */
    div[data-testid="stMarkdownContainer"] p {
        font-size: 0.82rem !important;
        margin-bottom: 0.1rem !important;
        font-weight: 500;
        color: #334155;
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 0px !important;
    }
    .stNumberInput input {
        padding-top: 0.15rem !important;
        padding-bottom: 0.15rem !important;
        font-size: 0.82rem !important;
        height: 30px !important;
        border-radius: 5px !important;
    }
    div[data-baseweb="select"] > div {
        min-height: 30px !important;
        font-size: 0.82rem !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
        border-radius: 5px !important;
    }

    /* Dashboard Header */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 0.4rem;
        margin-bottom: 0.4rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .main-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        line-height: 1.1;
    }
    .main-subtitle {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.1rem;
    }
    .header-chip {
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        white-space: nowrap;
    }

    /* Container Cards */
    .dashboard-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .section-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.35rem;
        padding-bottom: 0.2rem;
        border-bottom: 1px solid #f1f5f9;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }

    /* Prediction Badges */
    .prediction-box-stay {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 6px;
        padding: 0.45rem 0.8rem;
        text-align: center;
        margin-bottom: 0.35rem;
    }
    .prediction-box-churn {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 6px;
        padding: 0.45rem 0.8rem;
        text-align: center;
        margin-bottom: 0.35rem;
    }
    .prediction-title {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .prediction-status-stay {
        font-size: 1.15rem;
        font-weight: 700;
        color: #166534;
    }
    .prediction-status-churn {
        font-size: 1.15rem;
        font-weight: 700;
        color: #991b1b;
    }

    /* Probability Metric Cards */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.35rem 0.5rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.68rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-val {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* Progress Bar Box */
    .viz-container {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.45rem 0.65rem;
        margin-top: 0.35rem;
    }
    .viz-label-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.15rem;
    }
    .bar-bg {
        width: 100%;
        background-color: #e2e8f0;
        border-radius: 4px;
        height: 10px;
        overflow: hidden;
        margin-bottom: 0.35rem;
    }
    .bar-fill-stay {
        background-color: #16a34a;
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s;
    }
    .bar-fill-churn {
        background-color: #dc2626;
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s;
    }

    /* Risk Insight Box */
    .insight-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.4rem 0.65rem;
        margin-top: 0.35rem;
        font-size: 0.78rem;
    }
    .insight-title {
        font-weight: 600;
        color: #1e293b;
        font-size: 0.78rem;
        margin-bottom: 0.15rem;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    .insight-desc {
        color: #475569;
        line-height: 1.3;
    }

    /* Project Summary Row */
    .summary-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.3rem 0.5rem;
        text-align: center;
    }
    .summary-label {
        font-size: 0.68rem;
        color: #64748b;
        font-weight: 600;
    }
    .summary-val {
        font-size: 0.82rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* Expander Compact styling */
    .streamlit-expanderHeader {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        padding-top: 0.15rem !important;
        padding-bottom: 0.15rem !important;
        background-color: #f8fafc !important;
        border-radius: 4px !important;
        border: 1px solid #e2e8f0 !important;
    }

    /* Button Styling */
    div.stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.3rem 0.8rem;
        height: auto;
        border: none;
        width: 100%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    div.stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/customer_churn_model.pkl")

model = load_model()

# --------------------------------------------------
# 1. HEADER WITH BADGE
# --------------------------------------------------

st.markdown("""
<div class="header-container">
    <div>
        <div class="main-header">📊 Customer Churn Prediction</div>
        <div class="main-subtitle">Predict whether a customer is likely to churn using banking and demographic information.</div>
    </div>
    <div class="header-chip">ML-Powered • Gradient Boosting</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 2. MAIN DASHBOARD (Two-Column Layout)
# --------------------------------------------------

col_left, col_right = st.columns([1.12, 0.88])

with col_left:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Customer Information</div>', unsafe_allow_html=True)
    
    # 10 input fields in 2 compact sub-columns
    sub_col1, sub_col2 = st.columns(2)
    
    with sub_col1:
        credit_score = st.number_input(
            "Credit Score (300–850)",
            min_value=300,
            max_value=850,
            value=650
        )
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )
        tenure = st.number_input(
            "Tenure (0–10 yrs)",
            min_value=0,
            max_value=10,
            value=5
        )
        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=4,
            value=2
        )
        is_active_member = st.selectbox(
            "Active Member",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        
    with sub_col2:
        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"]
        )
        age = st.number_input(
            "Age (18–100 yrs)",
            min_value=18,
            max_value=100,
            value=35
        )
        balance = st.number_input(
            "Account Balance ($)",
            min_value=0.0,
            value=75000.0,
            step=1000.0
        )
        has_credit_card = st.selectbox(
            "Has Credit Card",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        estimated_salary = st.number_input(
            "Estimated Salary ($)",
            min_value=0.0,
            value=75000.0,
            step=1000.0
        )
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Prediction</div>', unsafe_allow_html=True)
    
    predict_clicked = st.button("🔮 Predict Churn", use_container_width=True)
    
    # Compute Model Prediction
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_credit_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    stay_prob = probabilities[0] * 100
    churn_prob = probabilities[1] * 100
    
    # Risk Assessment
    if churn_prob < 30.0:
        risk_level = "Low"
        risk_insight = "The model estimates a relatively low likelihood of churn."
    elif churn_prob < 60.0:
        risk_level = "Medium"
        risk_insight = "The model estimates a moderate likelihood of churn."
    else:
        risk_level = "High"
        risk_insight = "The model estimates a high likelihood of churn."
    
    # Prediction Result Display
    if prediction == 1:
        st.markdown("""
        <div class="prediction-box-churn">
            <div class="prediction-title">Prediction</div>
            <div class="prediction-status-churn">🔴 Likely to Churn</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="prediction-box-stay">
            <div class="prediction-title">Prediction</div>
            <div class="prediction-status-stay">🟢 Likely to Stay</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability Cards
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stay Probability</div>
            <div class="metric-val">{stay_prob:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Churn Probability</div>
            <div class="metric-val">{churn_prob:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Horizontal Comparison Visualization (Utilizing Right Side Area)
    st.markdown(f"""
    <div class="viz-container">
        <div class="section-title" style="margin-bottom:0.2rem; font-size:0.8rem; border-bottom:none;">📊 Churn Probability Comparison</div>
        <div class="viz-label-row">
            <span>Stay</span>
            <span>{stay_prob:.2f}%</span>
        </div>
        <div class="bar-bg">
            <div class="bar-fill-stay" style="width: {stay_prob:.1f}%;"></div>
        </div>
        <div class="viz-label-row">
            <span>Churn</span>
            <span>{churn_prob:.2f}%</span>
        </div>
        <div class="bar-bg" style="margin-bottom:0px;">
            <div class="bar-fill-churn" style="width: {churn_prob:.1f}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Risk Insight Section (Utilizing Right Side Area)
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-title">💡 Risk Insight</div>
        <div class="insight-desc">{risk_insight}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# 3. PROJECT SUMMARY ROW
# --------------------------------------------------

st.markdown("<div style='margin-top: 0.35rem;'></div>", unsafe_allow_html=True)

s_col1, s_col2, s_col3 = st.columns(3)
with s_col1:
    st.markdown("""
    <div class="summary-card">
        <div class="summary-label">📁 Dataset</div>
        <div class="summary-val">10,000 customers</div>
    </div>
    """, unsafe_allow_html=True)
with s_col2:
    st.markdown("""
    <div class="summary-card">
        <div class="summary-label">🤖 Algorithm</div>
        <div class="summary-val">Gradient Boosting</div>
    </div>
    """, unsafe_allow_html=True)
with s_col3:
    st.markdown("""
    <div class="summary-card">
        <div class="summary-label">🎯 Target</div>
        <div class="summary-val">Customer Churn</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# 4. MODEL INFORMATION (Expander at bottom)
# --------------------------------------------------

st.markdown("<div style='margin-top: 0.25rem;'></div>", unsafe_allow_html=True)

with st.expander("🤖 Model Information", expanded=False):
    info_col1, info_col2, info_col3, info_col4, info_col5, info_col6 = st.columns(6)
    with info_col1:
        st.markdown("<div class='metric-label'>Model</div><div style='font-size:0.78rem; font-weight:600; color:#1e293b;'>Gradient Boosting</div>", unsafe_allow_html=True)
    with info_col2:
        st.markdown("<div class='metric-label'>Accuracy</div><div style='font-size:0.78rem; font-weight:600; color:#1e293b;'>87.05%</div>", unsafe_allow_html=True)
    with info_col3:
        st.markdown("<div class='metric-label'>Precision</div><div style='font-size:0.78rem; font-weight:600; color:#1e293b;'>79.37%</div>", unsafe_allow_html=True)
    with info_col4:
        st.markdown("<div class='metric-label'>Recall</div><div style='font-size:0.78rem; font-weight:600; color:#1e293b;'>49.14%</div>", unsafe_allow_html=True)
    with info_col5:
        st.markdown("<div class='metric-label'>F1-Score</div><div style='font-size:0.78rem; font-weight:600; color:#1e293b;'>60.70%</div>", unsafe_allow_html=True)
    with info_col6:
        st.markdown("<div class='metric-label'>ROC-AUC</div><div style='font-size:0.78rem; font-weight:600; color:#1e293b;'>86.97%</div>", unsafe_allow_html=True)

# --------------------------------------------------
# 5. FOOTER
# --------------------------------------------------

st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.72rem; margin-top: 0.25rem;">
    Customer Churn Prediction • Machine Learning Project
</div>
""", unsafe_allow_html=True)