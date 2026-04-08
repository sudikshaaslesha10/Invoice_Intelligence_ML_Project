import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag
def explain_flag(input_data):
    reasons = []

    qty = input_data.get("invoice_quantity", [0])[0]
    dollars = input_data.get("invoice_dollars", [0])[0]
    freight = input_data.get("Freight", [0])[0]
    total_qty = input_data.get("total_item_quantity", [0])[0]
    total_dollars = input_data.get("total_item_dollars", [0])[0]

    # Rule 1: Value mismatch
    if abs(dollars - total_dollars) > 5:
        reasons.append("💰 Invoice amount does not match total item dollars")

    # Rule 2: High cost per unit
    if qty > 0 and (dollars / qty) > 3000:
        reasons.append("📈 Unusually high cost per item")

    # Rule 3: High freight
    if dollars > 0 and (freight / dollars) > 0.1:
        reasons.append("🚚 Freight cost is unusually high")

    # Rule 4: Quantity mismatch
    if qty != total_qty:
        reasons.append("📦 Invoice quantity differs from total quantity")

    return reasons
# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📄",
    layout="wide"
)

# ------------------------------
# Header Section
# ------------------------------
st.markdown("""
# 📦 Vendor Invoice Intelligence Portal
### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

This internal analytics portal leverages machine learning to:
- **Forecast freight costs accurately**
- **Detect risky or abnormal vendor invoices**
- **Reduce financial leakage and manual workload**
""")

st.divider()

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.title("🔍 Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "📦 Freight Cost Prediction",
        "🚨 Invoice Manual Approval Flag",
        "📊 Both"
    ]
)

# ✅ Business Impact moved to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 💼 Business Impact")
st.sidebar.markdown("""
- 📈 Improved cost forecasting  
- 🔍 Reduced invoice fraud & anomalies  
- ⚡ Faster finance operations  
""")

# ---------------------------------------------------
# Freight Cost Prediction
# ---------------------------------------------------
if selected_model == "📦 Freight Cost Prediction":
    st.subheader("📊 Freight Cost Prediction")
    st.markdown("""
    **Objective:**
    Predict freight cost for a vendor invoice using **Quantity** and **Invoice Dollars**
    to support budgeting, forecasting, and vendor negotiations.
    """)

    with st.form("freight_form"):
        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "Quantity",
                min_value=1,
                value=1200
            )
        with col2:
            dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=18500.0
            )

        submit_freight = st.form_submit_button("🚀 Predict Freight Cost")

    if submit_freight:
        input_data = {
            "Quantity": [quantity],
            "Dollars": [dollars]
        }

        prediction = predict_freight_cost(input_data)['Predicted_freight']

        st.success("Prediction completed successfully")

        st.metric(
            label="📊 Estimated Freight Cost",
            value=f"${prediction[0]:,.2f}"
        )

# ---------------------------------------------------
# Invoice Flag Prediction
# ---------------------------------------------------
elif selected_model == "🚨 Invoice Manual Approval Flag":
    st.subheader("🚨 Invoice Manual Approval Prediction")
    st.markdown("""
    **Objective:**
    Predict if a vendor invoice should be **flagged for manual approval**
    based on abnormal cost, freight or delivery patterns.
    """)

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=50
            )
            freight = st.number_input(
                "Freight Cost",
                min_value=0.0,
                value=1.73
            )
            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=100
            )
        with col2:
            invoice_dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=352.95
            )
        with col3:
            total_item_dollars = st.number_input(
                "Total Item Dollars",
                min_value=1.0,
                value=2476.0
            )

        submit_flag = st.form_submit_button("🔍 Evaluate Invoice Risk")

    if submit_flag:
        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }

        flag_prediction = predict_invoice_flag(input_data)['Predicted Flag']

        is_flagged = bool(flag_prediction[0])

        if is_flagged:
            st.error("🚨 Invoice requires **MANUAL APPROVAL**")
             # ✅ Add explanation
        reasons = explain_flag(input_data)
    
        st.subheader("🔍 Why this was flagged:")
        for r in reasons:
            st.write(f"- {r}")

        else:
            st.success("✅ Invoice is **SAFE FOR Auto-Approval**")

# ---------------------------------------------------
# Both Option
# ---------------------------------------------------
elif selected_model == "📊 Both":
    st.info("Use the sidebar to switch between Freight Cost Prediction and Invoice Flag Prediction.")
