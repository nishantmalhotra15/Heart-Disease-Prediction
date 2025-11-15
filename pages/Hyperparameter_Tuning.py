import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

st.title("🎛️ Advanced Hyperparameter Tuning")
st.write("Tune multiple Random Forest hyperparameters and see performance update instantly.")

# ---------------------- LOAD DATA ----------------------
df = pd.read_csv("Heart_disease_cleveland_new.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------- HYPERPARAMETERS ----------------------
st.subheader("⚙️ Choose Hyperparameters")

col1, col2 = st.columns(2)

with col1:
    n_est = st.slider("Number of Trees (n_estimators)", 10, 500, 100, 10)
    max_d = st.slider("Max Depth", 1, 50, 10)

    min_split = st.slider("Min Samples Split", 2, 20, 2)
    min_leaf = st.slider("Min Samples Leaf", 1, 20, 1)

with col2:
    max_feat = st.selectbox("Max Features", ["auto", "sqrt", "log2"])
    criterion = st.selectbox("Criterion", ["gini", "entropy", "log_loss"])
    bootstrap = st.selectbox("Bootstrap Sampling", [True, False])

# ---------------------- TRAIN MODEL ----------------------
if st.button("🔄 Retrain Model with Selected Parameters"):

    model = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=max_d,
        min_samples_split=min_split,
        min_samples_leaf=min_leaf,
        max_features=max_feat,
        criterion=criterion,
        bootstrap=bootstrap,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)

    # ---------------------- METRICS ----------------------
    st.subheader("📊 Performance Metrics")
    st.write(f"**Accuracy:** {acc:.3f}")
    st.write(f"**Precision:** {prec:.3f}")
    st.write(f"**Recall:** {rec:.3f}")
    st.write(f"**ROC-AUC:** {roc:.3f}")

    # ---------------------- FEATURE IMPORTANCE ----------------------
    st.subheader("📌 Feature Importance")

    importances = model.feature_importances_

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=importances, y=X.columns, ax=ax)
    st.pyplot(fig)
