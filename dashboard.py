# =====================================================
# dashboard.py
# IoT Device Identification Dashboard
# Student Project - Streamlit Version
# =====================================================

import streamlit as st
import pandas as pd
import joblib

st.title("🛰️ IoT Device Identification Dashboard")
st.write("Predict IoT device/attack type based on network traffic features.")

# ---------------------------
# Load Model & Encoders
# ---------------------------
model = joblib.load('iot_identifier_model.pkl')     # trained Random Forest
target_encoder = joblib.load('target_encoder.pkl') # LabelEncoder for attack_cat
feature_encoders = joblib.load('feature_encoders.pkl') # Encoders for proto/service

# ---------------------------
# User Inputs
# ---------------------------
st.subheader("Enter Traffic Features:")

dur = st.number_input("Duration", min_value=0.0, step=0.1)
proto = st.selectbox("Protocol", options=list(feature_encoders['proto'].classes_))
service = st.selectbox("Service", options=list(feature_encoders['service'].classes_))
spkts = st.number_input("Source Packets", min_value=0)
dpkts = st.number_input("Destination Packets", min_value=0)
sbytes = st.number_input("Source Bytes", min_value=0)
dbytes = st.number_input("Destination Bytes", min_value=0)
rate = st.number_input("Rate", min_value=0.0, step=0.1)
sttl = st.number_input("Source TTL", min_value=0)
dttl = st.number_input("Destination TTL", min_value=0)
sload = st.number_input("Source Load", min_value=0.0, step=0.1)
dload = st.number_input("Destination Load", min_value=0.0, step=0.1)
sloss = st.number_input("Source Loss", min_value=0)
dloss = st.number_input("Destination Loss", min_value=0)
smean = st.number_input("Source Mean", min_value=0.0, step=0.1)
dmean = st.number_input("Destination Mean", min_value=0.0, step=0.1)

# ---------------------------
# Predict Button
# ---------------------------
if st.button("🔍 Predict Device / Attack Type"):
    # Create input DataFrame
    df = pd.DataFrame([{
        'dur': dur,
        'proto': feature_encoders['proto'].transform([proto])[0],
        'service': feature_encoders['service'].transform([service])[0],
        'spkts': spkts,
        'dpkts': dpkts,
        'sbytes': sbytes,
        'dbytes': dbytes,
        'rate': rate,
        'sttl': sttl,
        'dttl': dttl,
        'sload': sload,
        'dload': dload,
        'sloss': sloss,
        'dloss': dloss,
        'smean': smean,
        'dmean': dmean
    }])

    # Predict
    pred = model.predict(df)
    label = target_encoder.inverse_transform(pred)[0]

    st.success(f"🧠 Predicted Category: **{label}**")

st.caption("Built by Team <Your Team Name> | UNSW-NB15 Dataset")
