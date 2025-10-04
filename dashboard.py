# dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import joblib

st.set_page_config(page_title="IoT Device Identifier", layout="wide")
st.title("📊 IoT Device Identifier Dashboard")


# --- Load Model & Encoders ---
@st.cache_resource
def load_model():
    model = joblib.load("iot_identifier_model.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    feature_encoders = joblib.load("feature_encoders.pkl")
    return model, target_encoder, feature_encoders


model, target_encoder, feature_encoders = load_model()


# --- Load CSV ---
@st.cache_data
def load_data():
    data = pd.read_csv("iot_device_test.csv")  # CSV from your project
    return data


data = load_data()

# --- Data Overview ---
st.sidebar.header("Data Overview")
if st.sidebar.checkbox("Show Raw Data"):
    st.write(data.head(50))

# --- Device Category Bar Chart ---
st.subheader("Device Category Distribution")
device_counts = data['device_category'].value_counts()
fig_bar = px.bar(
    x=device_counts.index,
    y=device_counts.values,
    labels={"x": "Device Category", "y": "Count"},
    title="Number of Devices by Category"
)
fig_bar.update_traces(marker_color='skyblue')
fig_bar.update_layout(yaxis_title="Count", xaxis_title="Device Category")
st.plotly_chart(fig_bar, use_container_width=True)

# --- Feature Selection ---
st.sidebar.header("Feature Selection")
numeric_features = data.select_dtypes(include='number').columns.tolist()
selected_features = st.sidebar.multiselect(
    "Choose features for analysis", numeric_features, default=numeric_features[:5]
)

# --- Heatmap of Selected Features ---
if selected_features:
    st.subheader("Feature Correlation Heatmap")
    corr = data[selected_features].corr()
    fig_heatmap = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.columns),
        colorscale="Viridis",
        showscale=True
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

# --- Feature Histograms ---
st.subheader("Feature Histograms by Device Category")
for feature in selected_features:
    st.write(f"**{feature}**")
    fig_hist = px.histogram(
        data,
        x=feature,
        color="device_category",
        barmode="group",
        nbins=30,
        opacity=0.7,
        labels={feature: feature}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# --- Device Category Prediction ---
st.subheader("Predict Device Category")
st.write("Enter feature values to predict the device category:")

# Collect inputs for all numeric features
input_data = {}
for feature in numeric_features:
    val = st.number_input(f"{feature}", value=float(data[feature].mean()))
    input_data[feature] = [val]

# Predict button
if st.button("Predict Device Category"):
    input_df = pd.DataFrame(input_data)

    # Encode categorical features if any (using feature_encoders)
    for feature, encoder in feature_encoders.items():
        if feature in input_df.columns:
            input_df[feature] = encoder.transform(input_df[feature])

    pred = model.predict(input_df)
    pred_label = target_encoder.inverse_transform(pred)

    st.success(f"Predicted Device Category: **{pred_label[0]}**")
