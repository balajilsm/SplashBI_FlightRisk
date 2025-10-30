import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

st.set_page_config(page_title="Employee Flight Risk — FAST", layout="wide")
st.title("✈️ Employee Flight Risk Prediction (Optimized)")

# -------------------------
# Config
# -------------------------
DEFAULT_FEATURES = [
    'gender', 'age', 'age_group', 'years_of_experience',
    'department_name', 'position', 'job_group', 'business_division_name',
    'hourly_flag', 'ft_pt', 'tenure_years', 'promotion_count', 'move_count',
    'annual_salary', 'percentage_change', 'previous_salary',
    'rating', 'engagement_level', 'appraisal_flag', 'location_state'
]
DEFAULT_PATH = "emp_history_data2.csv"

# -------------------------
# Caching helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_csv(path_or_buf):
    return pd.read_csv(path_or_buf, low_memory=False)

@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame, features, target):
    X = df[features].copy()
    y = df[target].copy()

    # Identify categoricals (object or category)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Ordinal encode categoricals in one vectorized shot (fast)
    enc = None
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = enc.fit_transform(X[cat_cols].astype("string"))

    # Downcast numerics to float32/int32 to save memory and speed up training
    for c in num_cols:
        if pd.api.types.is_integer_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], downcast="integer")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.astype(np.float32)

    # Clean target to 0/1 if it looks categorical
    if y.dtype == "O" or pd.api.types.is_categorical_dtype(y):
        y = y.astype("category").cat.codes

    return X, y

def band_from_prob(p):
    if p < 0.30:
        return "Low"
    if p < 0.60:
        return "Medium"
    return "High"

# -------------------------
# Data input (embedded + upload)
# -------------------------
if os.path.exists(DEFAULT_PATH):
    df_default = load_csv(DEFAULT_PATH)
else:
    df_default = pd.DataFrame()

uploaded = st.file_uploader("📂 Upload Employee CSV (optional)", type=["csv"])
if uploaded is not None:
    df = load_csv(uploaded)
elif not df_default.empty:
    st.info("Using embedded dataset: `emp_history_data2.csv`")
    df = df_default.copy()
else:
    st.error("No data found. Upload a CSV or include `emp_history_data2.csv` next to the app.")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(10), use_container_width=True)

# Target + features
target_col = st.selectbox("🎯 Target column (e.g., flight_risk)", df.columns, index=min(0, len(df.columns)-1))
feature_cols = st.multiselect("🧩 Feature columns", df.columns.tolist(), default=[c for c in DEFAULT_FEATURES if c in df.columns])

if not feature_cols:
    st.warning("Select at least one feature.")
    st.stop()

# Speed toggles
colA, colB, colC = st.columns(3)
with colA:
    fast_mode = st.checkbox("⚡ Fast Mode (use sample)", value=True,
                            help="Use a stratified sample for training on large datasets.")
with colB:
    sample_size = st.number_input("Sample size (rows) when Fast Mode is ON",
                                  min_value=1000, max_value=200000, value=30000, step=1000)
with colC:
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

# XGBoost hyperparams tuned for speed
st.markdown("#### ⚙️ Training Parameters (speed-optimized)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    n_estimators = st.number_input("n_estimators", 50, 1000, 400, step=50)
with col2:
    max_depth = st.number_input("max_depth", 2, 12, 6, step=1)
with col3:
    learning_rate = st.select_slider("learning_rate", options=[0.05, 0.07, 0.1, 0.15, 0.2], value=0.1)
with col4:
    early_stopping_rounds = st.number_input("early_stopping_rounds", 10, 200, 50, step=10)

go = st.button("🚀 Train Model")

if go:
    with st.spinner("Preprocessing…"):
        X, y = preprocess(df, feature_cols, target_col)

        # Optional sampling for speed
        if fast_mode and len(X) > sample_size:
            # stratified sample by target
            X_tmp, _, y_tmp, _ = train_test_split(
                X, y, train_size=sample_size, random_state=42, stratify=y
            )
            X, y = X_tmp, y_tmp

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

    # Fast XGB params
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",        # fast histogram algorithm (CPU)
        max_bin=256,               # fewer bins = faster
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,                 # use all cores
        random_state=42,
    )

    with st.spinner("Training XGBoost (early stopping active)…"):
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
            early_stopping_rounds=early_stopping_rounds
        )

    # Evaluate
    y_pred = model.predict(X_valid)
    y_prob = model.predict_proba(X_valid)[:, 1]
    acc = accuracy_score(y_valid, y_pred)

    st.subheader("📊 Performance")
    st.write(f"**Accuracy:** {acc*100:.2f}%  •  Best Iteration: {getattr(model, 'best_iteration', 'n/a')}")
    st.text("Classification Report")
    st.code(classification_report(y_valid, y_pred), language="text")

    cm = confusion_matrix(y_valid, y_pred)
    st.write("**Confusion Matrix**")
    st.dataframe(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

    # Feature Importance
    st.subheader("🔥 Feature Importance")
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(order)), importances[order])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([X.columns[i] for i in order], rotation=45, ha="right")
    ax.set_title("XGBoost Feature Importance")
    st.pyplot(fig, clear_figure=True)

    # Predict full dataset (not just sample) for output
    with st.spinner("Scoring full dataset…"):
        X_full, y_full = preprocess(df, feature_cols, target_col)
        probs = model.predict_proba(X_full)[:, 1]
        preds = (probs >= 0.5).astype(int)
        out = df.copy()
        out["Predicted_Risk"] = preds
        out["Predicted_Risk_Prob"] = probs.astype(np.float32)
        out["Risk_Band"] = [band_from_prob(p) for p in probs]

    st.subheader("📋 Top 10 Predictions")
    st.dataframe(out[["Predicted_Risk","Predicted_Risk_Prob","Risk_Band"]].head(10), use_container_width=True)

    st.download_button(
        "⬇️ Download predictions (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="flight_risk_predictions.csv",
        mime="text/csv"
    )

    st.success("Done.")
