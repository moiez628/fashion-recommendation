import subprocess, sys

# Force scikit-learn installation at runtime
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Fashnify - Fashion Recommendation", layout="centered")

st.title("👗 Fashnify: Your Look, Your Data, Your Style")
st.markdown("Upload your fashion dataset and get style predictions using machine learning.")

# Upload Excel file
uploaded_file = st.file_uploader("📤 Upload your Excel dataset (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ Dataset loaded successfully!")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.stop()

    # Process multi-label fields
    multi_label_columns = ['suitable_body_type', 'suitable_skin_tone', 'suitable_face_shape']
    for col in multi_label_columns:
        df[col] = df[col].apply(lambda x: x.split('|') if isinstance(x, str) else [])

    # Encode multi-label columns
    mlb = MultiLabelBinarizer()
    encoded_features = []
    for col in multi_label_columns:
        mlb_result = pd.DataFrame(mlb.fit_transform(df[col]), columns=[f"{col}_{cls}" for cls in mlb.classes_])
        encoded_features.append(mlb_result)

    # Encode single-label columns
    label_columns = ['color', 'style', 'price_range', 'suitable_height_range']
    for col in label_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Combine all features
    X = pd.concat([df[label_columns]] + encoded_features, axis=1)

    # Encode target
    y = LabelEncoder().fit_transform(df['category'])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)

    st.subheader("📊 Model Performance")
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**✅ Accuracy: {acc * 100:.2f}%**")

    # Classification report
    st.code(classification_report(y_test, y_pred), language="text")

    # Confusion Matrix
    st.subheader("🧩 Confusion Matrix")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

else:
    st.info("👆 Please upload an Excel file to start.")
