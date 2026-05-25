import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# -----------------------------------
# Load Model & Scaler
# -----------------------------------

model = tf.keras.models.load_model("titanic_ann_model.h5")

# Load scaler used during training
scaler = joblib.load("scaler.pkl")

# -----------------------------------
# Custom Styling
# -----------------------------------

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.title {
    font-size:40px;
    font-weight:bold;
    color:#0f172a;
}

.subtitle {
    font-size:20px;
    color:#475569;
}

.card {
    background-color:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.1);
}

.metric-card {
    background-color:#ffffff;
    padding:15px;
    border-radius:10px;
    text-align:center;
    box-shadow:0px 2px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# SECTION 1 — Header
# -----------------------------------

col1, col2 = st.columns([1, 5])

with col1:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3050/3050525.png",
        width=100
    )

with col2:
    st.markdown(
        '<div class="title">🚢 Titanic Survival Prediction System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Deep Learning Based Passenger Survival Prediction</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# -----------------------------------
# SECTION 2 — Project Description
# -----------------------------------

with st.container():

    st.markdown("## 📘 Project Description")

    st.info("""
This application predicts whether a passenger is likely to survive the Titanic disaster using an Artificial Neural Network (ANN).

🔹 Built using TensorFlow & Keras  
🔹 Uses Deep Learning for classification  
🔹 Deployed with Streamlit  
🔹 Includes real-time prediction and probability visualization
""")

# -----------------------------------
# SECTION 3 — Passenger Input Form
# -----------------------------------

st.markdown("## 🧾 Passenger Input Form")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        1,
        80,
        25
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=50.0
    )

# -----------------------------------
# SECTION 4 — Prediction Button
# -----------------------------------

st.markdown("")

predict_btn = st.button("🔮 Predict Survival")

# -----------------------------------
# Prediction Logic
# -----------------------------------

if predict_btn:

    # Create DataFrame
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Age": [age],
        "Fare": [fare]
    })

    # -----------------------------------
    # SECTION 5 — Data Preprocessing
    # -----------------------------------

    scaled_input = scaler.transform(input_data)

    # -----------------------------------
    # SECTION 6 — Model Prediction
    # -----------------------------------

    prediction = model.predict(scaled_input)

    probability = float(prediction[0][0])

    survival_prob = probability * 100
    nonsurvival_prob = (1 - probability) * 100

    st.markdown("---")

    st.markdown("## 🎯 Prediction Output")

    col1, col2, col3 = st.columns(3)

    # Final Prediction
    with col1:

        if probability > 0.5:
            st.success("✅ Survived")
        else:
            st.error("❌ Not Survived")

    # Probability
    with col2:
        st.metric(
            label="Survival Probability",
            value=f"{survival_prob:.2f}%"
        )

    # Confidence Score
    with col3:

        confidence = max(survival_prob, nonsurvival_prob)

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

    # -----------------------------------
    # SECTION 7 — Visualization
    # -----------------------------------

    st.markdown("## 📊 Probability Visualization")

    chart_data = pd.DataFrame({
        "Category": ["Survived", "Not Survived"],
        "Probability": [survival_prob, nonsurvival_prob]
    })

    st.bar_chart(
        chart_data.set_index("Category")
    )

    # Progress Meter
    st.markdown("### Survival Meter")

    st.progress(float(probability))

    st.write(f"Passenger Survival Chance: {survival_prob:.2f}%")

# -----------------------------------
# Footer
# -----------------------------------

st.markdown("---")

st.caption("Built with ❤️ using Streamlit, TensorFlow, NumPy & Deep Learning")