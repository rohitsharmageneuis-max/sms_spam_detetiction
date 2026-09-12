# app/main.py

import streamlit as st
import joblib
import numpy as np

# Load vectorizer and model 
vectorizer = joblib.load(
    "C:/Users/rohit/OneDrive/Desktop/SMS-Spam-Detection-using-SVM/tfidf_vectorizer.pkl"
)

model = joblib.load("C:/Users/rohit/OneDrive/Desktop/SMS-Spam-Detection-using-SVM/svm_model.pkl"
)

# App UI
st.set_page_config(page_title="SMS Spam Classifier", layout="centered")
st.title("📩 SMS Spam Detection using SVM")

st.markdown("Enter a message below to check if it's spam or not.")

# Text input
user_input = st.text_area("✍️ Enter SMS Text Here:")

# Predict button
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Transform input
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        
        # Display result
        if prediction == "spam":
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT SPAM.")
