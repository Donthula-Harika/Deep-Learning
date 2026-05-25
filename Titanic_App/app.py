import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model('titanic_ann_model.h5')

# Load scaler
scaler = joblib.load('scaler.pkl')

st.title("Titanic Survival Prediction System")

st.subheader("Deep Learning Based Passenger Survival Prediction")

# Inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

age = st.slider(
    "Age",
    1,
    80,
    24
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=100.0
)

if st.button("Predict Survival"):

    data = np.array([[pclass, age, fare]])

    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)[0][0]

    if prediction > 0.5:
        result = "Survived"
    else:
        result = "Not Survived"

    st.success(f"Prediction: {result}")

    st.write(f"Survival Probability: {prediction:.2f}")

    st.write(f"Confidence Score: {prediction*100:.2f}%")

    # Visualization
    labels = ['Survived', 'Not Survived']
    values = [prediction, 1-prediction]

    fig, ax = plt.subplots()

    ax.bar(labels, values)

    st.pyplot(fig)