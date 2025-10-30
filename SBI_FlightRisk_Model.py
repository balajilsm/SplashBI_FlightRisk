# SBI_FlightRisk_Model.py
import os
import io
import json
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas.api.types as ptypes

# ---- try to import xgboost (fallback to sklearn GBDT if missing) ----
XGB_OK = True
try:
    from xgboost import XGBClassifier
except Exception:
    XGB_OK = False
    from sklearn.ensemble import GradientBoostingClassifier as GBDT

# ---- optional chardet for better encoding detection ----
TRY_CHARDET = True
try:
    import chardet
except Exception:
    TRY_CHARDET = False

st.set_page_config(page_title="✈️ Employee Flight Risk — Optimized", layout="wide")
st.title("✈️ Employee Flight Risk Prediction (Fast, Robust, CSV-friendly)")

# -------------------------
# Defaults (your feature list)
# -------------------------
DEFAULT_FEATURES = [
    'gender', 'age', 'age_group', 'years_of_experience',
    'department_name', 'position', 'job_group', 'business_division_name',
    'hourly_flag', 'ft_pt', 'tenure_years', 'promotion_count', 'move_count',
    'annual_salary', 'percentage_change', 'previous_salary',
    'rating', 'engagement_level', 'appraisal_flag', 'location_state'
]
DEFAULT_PATH = "emp_history_data2.csv"  # put your CSV next to this file

def band_from_prob(p: float) -> str:
    if p < 0.30:
        return "Low"
    if p < 0.60:
        return "Medium"
    return "High"

# -------------------------
# Encoding-safe CSV loaders
# -------------------------
def _detect_encoding_from_bytes(b: bytes) -> str:
    if TRY_CHARDET:
        enc = (chardet.detect(b) or {}).get("encoding") or "utf-8"
        return enc
    # simple heuristic fallback
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            b.decode(enc)
            return enc
        except Exception:
            continue
    return "latin1"

def _csv_hash_key(b: bytes, extra: str = "") -> str:
    h = hashlib.sha256()
    h.update(b)
    if extra:
        h.update(extra.encode("utf-8"))
    return h.hexdigest()

@st.cache_data(show_spinner=False)
def load_csv_from_bytes(b: bytes, hint_name: str = "") -> pd.DataFrame:
    enc = _detect_encoding_from_bytes(b)
    try:
        return pd.read_csv(io.BytesIO(b), encoding=enc, low_memory=False)
    except UnicodeDecodeError:
        # fallback chain
        for alt in ("cp1252", "latin1"):
            try:
                return pd.read_csv(io.BytesIO(b), encoding=alt, low_memory=False)
            except Exception:
                pass
        # final attempt with errors='ignore'
        return pd.read_csv(io.BytesIO(b.decode(enc, errors="ignore").encode()), low_memory=False)

@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    # read bytes to detect encoding
    with open(path, "rb") as f:
        b = f.read()
    return load_csv_from_bytes(b, hint_name=os.path.basename(path))

# -------------------------
# Preprocess (safe for mixed/corrupt categoricals)
# -------------------------
def _safe_stringify(x):
    """Stable, hashable string for any cell value (handles NaN, lists, dicts, timestamps)."""
    if x is None:
        return "NA"
    try:
        if pd.isna(x):
            return "NA"
    except Exception:
        pass
    if isinstance(x, (list, dict, set, tuple)):
        try:
            return json.dumps(x, sort_keys=True, default=str)
        except Exception:
            return str(x)
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        try:
            return pd.to_datetime(x).isoformat()
        except Exception:
            return str(x)
    return str(x)

@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame, features, target):
    # Keep existing columns
    features = [c for c in features if c in df.columns]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data.")

    X = df[features].copy()
    y = df[target].copy()

    # Detect categorical columns (object/category or mixed)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in X.columns:
        if c not in cat_cols and X[c].dtype == "O":
            cat_cols.append(c)
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Sanitize & encode categoricals
    if cat_cols:
        # normalize datetimes if any
        for c in cat_cols:
            if ptypes.is_datetime64_any_dtype(X[c]):
                X[c] = X[c].astype("datetime64[ns]")
        X[cat_cols] = X[cat_cols].applymap(_safe_stringify)

        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float32
        )
        X[cat_cols] = enc.fit_transform(X[cat_cols])

    # Numeric clean
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X[num_cols] = X[num_cols].fillna(0)
    X = X.astype(np.float32)

    # Target → numeric labels if categorical
    if y.dtype == "O" or ptypes.is_categorical_dtype(y):
        y = y.astype("category").cat.codes
    y = pd.Series(y).fillna(0).astype(int).to_numpy()

    return X, y

# -------------------------
# Data input (embedded + upload with robust encoding)
# -------------------------
uploaded = st.file_uploader("📂 Upload Employee CSV (optional)", type=["csv"])

if uploaded is not None:
    raw = uploaded.read()
    if not raw:
        st.error("Uploaded file is empty.")
        st.stop()
    df = load_csv_from_bytes(raw, hint_name=getattr(uploaded, "name", "upload.csv"))
    st.success("Loaded uploaded CSV.")
elif os.path.exists(DEFAULT_PATH):
    df = load_csv_from_path(DEFAULT_PATH)
    st.info(f"Using embedded dataset: `{DEFAULT_PATH}`")
else:
    st.error("No data found. Upload a CSV or include `emp_history_data2.csv` next to the app.")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(10), use_container_width=True)
st.markdown("---")

# -------------------------
# Target & features
# -------------------------
if 'flight_risk' in df.columns:
    default_target_idx = list(df.columns).index('flight_risk')
else:
    default_target_idx = 0

target_col = st.selectbox("🎯 Target column (e.g., flight_risk)", df.columns, index=default_target_idx)
feature_cols = st.multiselect(
    "🧩 Feature columns",
    df.columns.tolist(),
    default=[c for c in DEFAULT_FEATURES if c in df.columns]
)

if not feature_cols:
    st.warning("Select at least one feature.")
    st.stop()

# -------------------------
# Speed controls
# -------------------------
c1, c2, c3 = st.columns(3)
with c1:
    fast_mode = st.checkbox("⚡ Fast Mode (use stratified sample)", value=True)
with c2:
    sample_size = st.number_input("Sample size (if Fast Mode ON)", min_value=1000, max_value=200000, value=30000, step=1000)
with c3:
    test_size = st.slider("Test size (validation split)", 0.1, 0.4, 0.2, 0.05)

st.markdown("#### ⚙️ Training Parameters")
d1, d2, d3, d4 = st.columns(4)
with d1:
    n_estimators = st.number_input("n_estimators", 50, 1000, 400, step=50)
with d2:
    max_depth = st.number_input("max_depth", 2, 12, 6, step=1)
with d3:
    learning_rate = st.select_slider("learning_rate", options=[0.05, 0.07, 0.1, 0.15, 0.2], value=0.1)
with d4:
    early_stopping_rounds = st.number_input("early_stopping_rounds (XGB only)", 10, 200, 50, step=10)

go = st.button("🚀 Train Model")

if go:
    with st.spinner("Preprocessing…"):
        X_all, y_all = preprocess(df, feature_cols, target_col)

        # Optional sampling for speed
        X_use, y_use = X_all, y_all
        if fast_mode and len(X_all) > sample_size:
            X_use, _, y_use, _ = train_test_split(
                X_all, y_all,
                train_size=sample_size,
                random_state=42,
                stratify=y_all
            )

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_use, y_use,
            test_size=test_size,
            random_state=42,
            stratify=y_use
        )

    # -------------------------
    # Model
    # -------------------------
    if XGB_OK:
        model = XGBClassifier(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",   # FAST
            max_bin=256,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
        with st.spinner("Training XGBoost (early stopping active)…"):
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
                early_stopping_rounds=int(early_stopping_rounds)
            )
        best_iter = getattr(model, "best_iteration", None)
    else:
        st.warning("xgboost not available — using GradientBoostingClassifier fallback.")
        model = GBDT(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate)
        )
        with st.spinner("Training Gradient Boosting…"):
            model.fit(X_train, y_train)
        best_iter = "n/a"

    # -------------------------
    # Validation metrics
    # -------------------------
    y_pred = model.predict(X_valid)
    y_prob = model.predict_proba(X_valid)[:, 1] if hasattr(model, "predict_proba") else (y_pred * 1.0)

    acc = accuracy_score(y_valid, y_pred)
    st.subheader("📊 Performance")
    st.write(f"**Accuracy:** {acc*100:.2f}%  •  Best Iteration: {best_iter if best_iter is not None else 'n/a'}")
    st.text("Classification Report")
    st.code(classification_report(y_valid, y_pred), language="text")

    cm = confusion_matrix(y_valid, y_pred)
    st.write("**Confusion Matrix**")
    st.dataframe(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

    # -------------------------
    # Feature Importance
    # -------------------------
    st.subheader("🔥 Feature Importance")
    try:
        importances = model.feature_importances_
    except Exception:
        importances = np.zeros(X_use.shape[1], dtype=np.float32)
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(order)), importances[order])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([pd.Index(feature_cols)[i] for i in order], rotation=45, ha="right")
    ax.set_title("Feature Importance")
    st.pyplot(fig, clear_figure=True)

    # -------------------------
    # Score full dataset & download
    # -------------------------
    with st.spinner("Scoring full dataset…"):
        probs_full = model.predict_proba(X_all)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_all)
        preds_full = (probs_full >= 0.5).astype(int)
        out = df.copy()
        out["Predicted_Risk"] = preds_full
        out["Predicted_Risk_Prob"] = probs_full.astype(np.float32)
        out["Risk_Band"] = [band_from_prob(p) for p in probs_full]

    st.subheader("📋 Top 10 Predictions")
    st.dataframe(out[["Predicted_Risk", "Predicted_Risk_Prob", "Risk_Band"]].head(10), use_container_width=True)

    st.download_button(
        "⬇️ Download predictions (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="flight_risk_predictions.csv",
        mime="text/csv"
    )

    st.success("Done.")
