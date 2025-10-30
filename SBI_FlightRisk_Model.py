import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Employee Flight Risk Prediction", layout="wide")

st.title("✈️ Employee Flight Risk Prediction Dashboard")

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload Employee CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    st.markdown("---")

    # --- Feature Selection ---
    default_features = [
        'gender', 'age', 'age_group', 'years_of_experience',
        'department_name', 'position', 'job_group', 'business_division_name',
        'hourly_flag', 'ft_pt', 'tenure_years', 'promotion_count', 'move_count',
        'annual_salary', 'percentage_change', 'previous_salary',
        'rating', 'engagement_level', 'appraisal_flag', 'location_state'
    ]

    target_col = st.selectbox("🎯 Select Target Column (e.g., flight_risk)", df.columns)
    feature_cols = st.multiselect("🧩 Select Features", df.columns, default=default_features)

    if st.button("🚀 Train Model"):
        X = df[feature_cols].copy()
        y = df[target_col]

        # Encode categorical columns
        le_dict = {}
        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # XGBoost Model
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        st.subheader("📊 Model Performance")
        acc = accuracy_score(y_test, y_pred)
        st.write(f"✅ **Accuracy:** {acc*100:.2f}%")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("🔢 Confusion Matrix")
        st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

        # --- Feature Importance ---
        st.subheader("🔥 Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 5))
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        plt.bar(range(len(feature_cols)), importance[indices])
        plt.xticks(range(len(feature_cols)), [feature_cols[i] for i in indices], rotation=45, ha="right")
        plt.title("Feature Importance (XGBoost)")
        st.pyplot(fig)

        # --- Predict Entire Dataset ---
        df['Predicted_Risk'] = model.predict(X)
        df['Predicted_Risk_Prob'] = model.predict_proba(X)[:, 1]

        # Categorize bands
        df['Risk_Band'] = pd.cut(df['Predicted_Risk_Prob'],
                                 bins=[0, 0.3, 0.6, 1],
                                 labels=['Low', 'Medium', 'High'])

        st.subheader("📋 Predicted Results")
        st.dataframe(df[['Predicted_Risk', 'Predicted_Risk_Prob', 'Risk_Band']].head())

        # --- Download Results ---
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predicted CSV", data=csv, file_name="flight_risk_predictions.csv", mime="text/csv")

else:
    st.info("👆 Please upload your employee dataset (CSV) to get started.")
