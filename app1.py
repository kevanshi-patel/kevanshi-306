import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import os

# Set page config
st.set_page_config(page_title="Delivery Time Estimator", page_icon="📦", layout="centered")

# Header with styling
st.markdown("""
    <h1 style='text-align: center;'>🚀 Delivery Time Estimator 📦</h1>
    <p style='text-align: center; color: grey;'>Predict how long your order will take to arrive!</p>
""", unsafe_allow_html=True)

# Load the Model
model_path = r"C:\Users\91743\OneDrive\Desktop\First-Streamlit-app-delivery-time-prediction-main\First-Streamlit-app-delivery-time-prediction-main\delivery_time_model.pkl"

if not os.path.exists(model_path):
    st.error("⚠️ Model file 'delivery_time_model.pkl' not found! Please check the directory.")
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"⚠️ Failed to load model: {e}")
    st.stop()

# Sidebar Inputs
st.sidebar.header("🔧 Order Details")

product_category = st.sidebar.selectbox("📦 Select Product Category", ["Electronics", "Clothing", "Furniture", "Books", "Others"])
customer_location = st.sidebar.selectbox("📍 Select Customer Location", ["Urban", "Suburban", "Rural"])
shipping_method = st.sidebar.selectbox("🚚 Select Shipping Method", ["Standard", "Express", "Same-Day"])
order_quantity = st.sidebar.number_input("🛒 Enter Order Quantity", min_value=1, step=1, value=1)

# Use Day Names Instead of Numbers
days_dict = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
             "Friday": 4, "Saturday": 5, "Sunday": 6}
order_day = st.sidebar.selectbox("📅 Purchased Day", list(days_dict.keys()))  
order_hour = st.sidebar.slider("⏰ Purchased Hour (0-23)", min_value=0, max_value=23, value=12)

# Use Numeric Input for Distance Instead of Dropdown
distance = st.sidebar.number_input("📏 Shipping Distance (in km)", min_value=1, step=1, value=10)

# Styled Button with Light Shadow Effect
st.sidebar.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        background-color: #f9f9f9;
        color: black;
        padding: 12px 20px;
        font-size: 18px;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: 0.3s;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.15);
    }
    div.stButton > button:hover {
        background-color: #f1f1f1;
        box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

if st.sidebar.button("🚀 **Predict Delivery Time**"):
    # Convert Inputs to Model-Compatible Format
    feature_dict = {"Electronics": 0, "Clothing": 1, "Furniture": 2, "Books": 3, "Others": 4,
                    "Urban": 0, "Suburban": 1, "Rural": 2,
                    "Standard": 0, "Express": 1, "Same-Day": 2}

    input_features = np.array([
        feature_dict[product_category], 
        feature_dict[customer_location], 
        feature_dict[shipping_method],
        order_quantity, 
        days_dict[order_day],  
        order_hour,
        distance  
    ]).reshape(1, -1)  
    
    try:
        predicted_time = model.predict(input_features)[0]
        st.success(f"✅ Estimated Delivery Time: **{predicted_time:.2f} days**")
        
        # Visualization
        chart_data = pd.DataFrame({"Delivery Estimate": ["Predicted Delivery Time"], "Days": [predicted_time]})
        
        chart = alt.Chart(chart_data).mark_bar(size=40).encode(
            x=alt.X("Days:Q", title="Estimated Days", scale=alt.Scale(domain=(0, predicted_time + 2))),
            y=alt.Y("Delivery Estimate:N", title=""),
            color=alt.value("#4C72B0"),
            tooltip=["Days"]
        ).properties(
            title="📊 Estimated Delivery Time",
            width=600,
            height=200
        )
        
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center;'>🔎 This tool provides an estimated delivery time based on your input. Results may vary.</p>
""", unsafe_allow_html=True)