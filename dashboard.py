# dashboard.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="IoT Device Identifier 🚀", layout="wide")
st.title("IoT Device Identifier 🚀")
st.write("Upload a CSV with IoT network traffic features OR use the default dataset to get predicted device categories!")

# Load model and encoders
model = joblib.load("iot_identifier_model.pkl")
target_encoder = joblib.load("target_encoder.pkl")
feature_encoders = joblib.load("feature_encoders.pkl")  # dict of encoders for categorical features

# Upload CSV (optional)
uploaded_file = st.file_uploader("Choose CSV file (optional)", type="csv")

# Load data
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Using uploaded dataset:")
else:
    data = pd.read_csv("iot_device_test.csv")  # default dataset in repo
    st.write("Using default dataset from repo:")

st.dataframe(data.head())

# Encode categorical features if needed
for col, le in feature_encoders.items():
    if col in data.columns:
        data[col] = le.transform(data[col].astype(str))

# Predict device categories
predictions = model.predict(data)
pred_labels = target_encoder.inverse_transform(predictions)
data['Predicted_Device'] = pred_labels

st.success("Prediction done ✅")
st.write("Predicted Device Categories:")
st.dataframe(data[['Predicted_Device']])

# Optional: download results
csv = data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download predictions as CSV",
    data=csv,
    file_name='predicted_devices.csv',
    mime='text/csv',
)

st.markdown("---")
st.write("Built with Python, Pandas, scikit-learn & Streamlit 🚀")
