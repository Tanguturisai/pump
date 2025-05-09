import streamlit as st
import numpy as np
import joblib

# Load model and scaler once
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Predictive Maintenance - HVAC Pump")

st.write("Enter sensor values below:")

# Initialize the list to hold the sensor values
sensor_values = []

# Collect sensor values from the user (example: 5 sensors)
for i in range(1, 6):  # Collecting 5 sensor values from the user
    val = st.number_input(f'Sensor {i}', value=0.0)
    sensor_values.append(val)

# When the user presses the "Predict" button
if st.button("Predict"):
    # Pad the sensor values to match the 50 features that the model expects
    sensor_values_padded = [0.0] * (50 - len(sensor_values)) + sensor_values  # Pad with zeros to the start

    # Convert the list to a numpy array and reshape it to match the expected input shape
    arr = np.array(sensor_values_padded).reshape(1, -1)

    # Scale the input data using the scaler
    arr_scaled = scaler.transform(arr)

    # Make a prediction using the trained model
    prediction = model.predict(arr_scaled)

    # Show the result to the user
    if prediction[0] == 1:
        st.error("⚠️ Warning: Maintenance Required!")
    else:
        st.success("✅ No maintenance required.")
