import streamlit as st
import numpy as np
import joblib

# Load model and scaler once
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Predictive Maintenance - HVAC Pump")

st.write("Enter sensor values below:")

# Example: If your model expects 5 features
sensor_values = [0.0] * (50- len(sensor_values))+sensor_values
arr = np.array(sensor_values).reshape(1, -1)
arr_scaled = scaler.transform(arr)
prediction = model.predict(arr_scaled)
for i in range(1, 6):  # adjust range if your model has more/less features
    val = st.number_input(f'Sensor {i}', value=0.0)
    sensor_values.append(val)

if st.button("Predict"):
    arr = np.array(sensor_values).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prediction = model.predict(arr_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Warning: Maintenance Required!")
    else:
        st.success("✅ Pump is Operating Normally.")
