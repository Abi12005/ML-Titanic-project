"""
app.py  –  Titanic Survival Predictor  |  Streamlit Web Application
Run:  streamlit run app.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import streamlit as st
warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "model_meta.json")
OUT_DIR    = os.path.join(BASE_DIR, "outputs")
SRC_DIR    = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0a0e1a; }
/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(160deg, #0f1b35 0%, #1a0a2e 100%); }
/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 12px; padding: 16px; margin: 4px 0;
}
/* Headings */
h1, h2, h3 { color: #e0d7ff !important; }
/* Predict button */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    color: white; border: none; border-radius: 12px;
    padding: 14px 40px; font-size: 1.1rem; font-weight: 700;
    width: 100%; cursor: pointer;
    transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(124,58,237,0.5);
}
div.stButton > button:first-child:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(124,58,237,0.7);
}
/* Result box */
.result-survive {
    background: linear-gradient(135deg, #065f46, #059669);
    border: 2px solid #10b981; border-radius: 16px;
    padding: 28px; text-align: center; margin: 16px 0;
}
.result-died {
    background: linear-gradient(135deg, #7f1d1d, #dc2626);
    border: 2px solid #ef4444; border-radius: 16px;
    padding: 28px; text-align: center; margin: 16px 0;
}
.result-text { font-size: 2rem; font-weight: 900; color: white; }
.prob-text   { font-size: 1.1rem; color: rgba(255,255,255,0.8); margin-top: 8px; }
/* Tab styling */
.stTabs [data-baseweb="tab"] {
    font-size: 1rem; font-weight: 600; padding: 12px 24px;
    color: #a78bfa !important;
}
/* Divider */
hr { border-color: rgba(124,58,237,0.3); }
/* Info box */
.info-card {
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 12px; padding: 16px; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Load model & metadata ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)


pipeline = load_model()
meta     = load_meta()

FEATURES = meta["features"]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚢 Titanic ML Project")
    st.markdown("---")

    st.markdown("### 🏆 Best Model")
    st.success(f"**{meta['best_model']}**")

    c1, c2 = st.columns(2)
    c1.metric("Accuracy",  f"{meta['accuracy']*100:.1f}%")
    c2.metric("ROC-AUC",   f"{meta['roc_auc']:.3f}")
    c1.metric("CV Score",  f"{meta['cv_mean']*100:.1f}%")

    st.markdown("---")
    st.markdown("### 📊 All Models")
    rows = []
    for m, v in meta["all_results"].items():
        rows.append({"Model": m,
                     "Acc": f"{v['accuracy']*100:.1f}%",
                     "AUC": f"{v['roc_auc']:.3f}"})
    st.dataframe(pd.DataFrame(rows).set_index("Model"),
                 use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div class="info-card">
    <b>📁 Stack</b><br>
    Python · scikit-learn · pandas · Streamlit · Joblib
    </div>
    """, unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="text-align:center; font-size:2.8rem; margin-bottom:0;">
    🚢 Titanic Survival Predictor
</h1>
<p style="text-align:center; color:#a78bfa; font-size:1.1rem; margin-top:4px;">
    Machine Learning · 7 Algorithms · 13 Features · Interactive Prediction
</p>
""", unsafe_allow_html=True)
st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Survival",
    "📊 EDA & Insights",
    "🤖 Model Performance",
    "ℹ️ About Project",
])


# ════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🎫 Enter Passenger Details")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🎭 Demographics**")
            sex = st.selectbox("Sex", ["male", "female"],
                               help="Passenger's gender")
            age = st.slider("Age", 0, 80, 28,
                            help="Passenger's age in years")
            pclass = st.selectbox(
                "Passenger Class",
                [1, 2, 3],
                index=2,
                format_func=lambda x: f"Class {x} – "
                                      + ("First" if x==1 else
                                         "Second" if x==2 else "Third"),
                help="Ticket class"
            )

        with col2:
            st.markdown("**👨‍👩‍👧 Family**")
            sibsp = st.number_input(
                "Siblings / Spouses aboard", 0, 8, 0,
                help="Number of siblings or spouses on the Titanic")
            parch = st.number_input(
                "Parents / Children aboard", 0, 6, 0,
                help="Number of parents or children on the Titanic")

        with col3:
            st.markdown("**🎟️ Ticket & Embarkation**")
            fare = st.number_input(
                "Fare (£)", 0.0, 600.0, 14.45,
                step=0.5, format="%.2f",
                help="Ticket price paid")
            embarked = st.selectbox(
                "Port of Embarkation",
                ["S", "C", "Q"],
                format_func=lambda x: {
                    "S": "S – Southampton",
                    "C": "C – Cherbourg",
                    "Q": "Q – Queenstown"
                }[x],
                help="Where the passenger boarded"
            )
            has_cabin = st.checkbox("Has Cabin Record", value=False,
                                    help="Was a cabin number recorded?")

        st.markdown("---")
        submitted = st.form_submit_button("🔮 Predict Survival",
                                          use_container_width=True)

    # ── Prediction logic ─────────────────────────────────────────────────────
    if submitted:
        sex_enc      = 0 if sex == "male" else 1
        emb_enc      = {"S": 0, "C": 1, "Q": 2}[embarked]
        family_size  = sibsp + parch + 1
        is_alone     = int(family_size == 1)
        fare_pp      = fare / family_size
        age_class    = age * pclass

        # Derive title (default to Mr / Miss)
        title_enc = 1 if sex == "male" else 2

        row = pd.DataFrame([{
            "Pclass":        pclass,
            "Sex":           sex_enc,
            "Age":           age,
            "SibSp":         sibsp,
            "Parch":         parch,
            "Fare":          fare,
            "Embarked":      emb_enc,
            "Title":         title_enc,
            "FamilySize":    family_size,
            "IsAlone":       is_alone,
            "FarePerPerson": fare_pp,
            "HasCabin":      int(has_cabin),
            "Age*Class":     age_class,
        }])[FEATURES]

        pred  = pipeline.predict(row)[0]
        proba = pipeline.predict_proba(row)[0]
        surv_prob  = proba[1] * 100
        death_prob = proba[0] * 100

        # ── Result card ───────────────────────────────────────────────────────
        st.markdown("### 🎯 Prediction Result")
        res_col, gauge_col = st.columns([1.2, 1])

        with res_col:
            if pred == 1:
                st.markdown(f"""
                <div class="result-survive">
                    <div class="result-text">✅ SURVIVED</div>
                    <div class="prob-text">Survival Probability: <b>{surv_prob:.1f}%</b></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-died">
                    <div class="result-text">❌ DID NOT SURVIVE</div>
                    <div class="prob-text">Death Probability: <b>{death_prob:.1f}%</b></div>
                </div>""", unsafe_allow_html=True)

            # Probability bar
            st.markdown("**Probability Breakdown**")
            st.progress(surv_prob / 100,
                        text=f"Survival: {surv_prob:.1f}%")
            c1, c2 = st.columns(2)
            c1.metric("🟢 Survived",     f"{surv_prob:.1f}%")
            c2.metric("🔴 Not Survived", f"{death_prob:.1f}%")

        with gauge_col:
            # Mini gauge chart
            fig, ax = plt.subplots(figsize=(4, 4),
                                   subplot_kw={"aspect": "equal"})
            fig.patch.set_facecolor("#0a0e1a")
            ax.set_facecolor("#0a0e1a")
            colors_pie = ["#dc2626", "#10b981"]
            wedges, _ = ax.pie(
                [death_prob, surv_prob],
                startangle=90,
                colors=colors_pie,
                wedgeprops=dict(width=0.45, edgecolor="#0a0e1a",
                                linewidth=2),
                counterclock=False,
            )
            ax.text(0, 0, f"{surv_prob:.0f}%\nSurvival",
                    ha="center", va="center",
                    fontsize=16, fontweight="bold", color="white")
            ax.set_title("Survival Probability",
                         color="white", fontsize=11, pad=10)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # ── Passenger summary ────────────────────────────────────────────────
        st.markdown("**📋 Input Summary**")
        summary = pd.DataFrame({
            "Feature": ["Sex", "Age", "Class", "Fare", "Family Size",
                        "Alone", "Embarkation"],
            "Value": [sex.capitalize(), str(age), f"Class {pclass}",
                      f"£{fare:.2f}", str(family_size),
                      "Yes" if is_alone else "No",
                      {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}[embarked]]
        })
        st.dataframe(summary.set_index("Feature"),
                     use_container_width=True)

        # ── Key insights ─────────────────────────────────────────────────────
        st.markdown("**💡 Key Factors**")
        insights = []
        if sex == "female":
            insights.append("🟢 Female passengers had a significantly higher survival rate.")
        else:
            insights.append("🔴 Male passengers had a significantly lower survival rate.")
        if pclass == 1:
            insights.append("🟢 First class passengers had priority access to lifeboats.")
        elif pclass == 3:
            insights.append("🔴 Third class passengers faced greater difficulties reaching lifeboats.")
        if is_alone:
            insights.append("🟡 Solo travelers showed mixed outcomes.")
        else:
            insights.append("🟡 Traveling with family had mixed effects on survival.")
        for ins in insights:
            st.markdown(ins)


# ════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Exploratory Data Analysis")

    eda_path  = os.path.join(OUT_DIR, "eda_overview.png")
    heat_path = os.path.join(OUT_DIR, "correlation_heatmap.png")
    feat_path = os.path.join(OUT_DIR, "feature_importance.png")

    if os.path.exists(eda_path):
        st.image(eda_path, caption="EDA Overview", use_container_width=True)
    if os.path.exists(heat_path):
        c1, c2 = st.columns(2)
        with c1:
            st.image(heat_path, caption="Correlation Heatmap",
                     use_container_width=True)
        with c2:
            if os.path.exists(feat_path):
                st.image(feat_path, caption="Feature Importance (Random Forest)",
                         use_container_width=True)

    # Key stats
    st.markdown("---")
    st.markdown("### 📌 Key Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Passengers", "891")
    col2.metric("Survivors",        "342  (38.4%)")
    col3.metric("Female Survival",  "~74%")
    col4.metric("1st Class Surv.",  "~63%")


# ════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🤖 Model Performance Comparison")

    comp_path = os.path.join(OUT_DIR, "model_comparison.png")
    roc_path  = os.path.join(OUT_DIR, "roc_curves.png")
    cm_path   = os.path.join(OUT_DIR, "confusion_matrices.png")

    if os.path.exists(comp_path):
        st.image(comp_path, caption="Model Comparison", use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curves", use_container_width=True)
    with c2:
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrices",
                     use_container_width=True)

    # Detailed results table
    st.markdown("### 📊 Detailed Results")
    df_res = pd.DataFrame([
        {"Model": k,
         "Test Accuracy": f"{v['accuracy']*100:.2f}%",
         "ROC-AUC": f"{v['roc_auc']:.4f}",
         "CV Score": f"{v['cv_mean']*100:.2f}%",
         "Best": "⭐" if k == meta["best_model"] else ""}
        for k, v in meta["all_results"].items()
    ]).set_index("Model")
    st.dataframe(df_res, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ℹ️ About This Project")
    st.markdown("""
    <div class="info-card">
    <h4>🚢 Titanic Survival Predictor — ML Workflow Project</h4>
    <p>
    This project implements a <b>complete end-to-end Machine Learning pipeline</b>
    on the classic Titanic dataset. It demonstrates data preprocessing,
    feature engineering, model training, evaluation, and deployment.
    </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🔧 Workflow Steps**
        1. Data loading & exploration (EDA)
        2. Missing value imputation
        3. Feature engineering (Title, FamilySize, etc.)
        4. Feature scaling (StandardScaler)
        5. Model training (7 algorithms)
        6. Cross-validation & evaluation
        7. Best model selection & Joblib save
        8. Prediction on test set
        9. Streamlit deployment
        """)
    with col2:
        st.markdown("""
        **🤖 Algorithms Trained**
        - Logistic Regression
        - Decision Tree
        - Random Forest
        - K-Nearest Neighbors (KNN)
        - Support Vector Machine (SVM)
        - Naive Bayes
        - Gradient Boosting ⭐ (Best)

        **📦 Tech Stack**
        Python · scikit-learn · pandas · numpy
        matplotlib · seaborn · Streamlit · Joblib
        """)

    st.markdown("---")
    st.markdown("""
    **📁 Project Structure**
    ```
    titanic_ml_project/
    ├── data/               # Raw CSV files
    ├── src/
    │   ├── preprocessing.py    # EDA + feature engineering
    │   └── train_models.py     # Training + evaluation
    ├── models/
    │   ├── best_model.pkl      # Saved pipeline (Joblib)
    │   └── model_meta.json     # Metrics + metadata
    ├── outputs/                # Charts + submission.csv
    ├── streamlit_app/
    │   └── app.py              # This application
    ├── requirements.txt
    └── README.md
    ```
    """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#6b7280; font-size:0.85rem;">
    Built with ❤️ using Python, scikit-learn & Streamlit &nbsp;|&nbsp;
    Titanic Dataset – Kaggle Competition
</p>
""", unsafe_allow_html=True)