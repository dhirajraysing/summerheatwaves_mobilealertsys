import streamlit as st
import pandas as pd

import summermodel
from summermodel import model

# Streamlit interface
st.title("Summer Heat Waves Mobile Alert System")

# User input for the current weather
temp = st.number_input("Current Temperature (°C)", min_value=-30.0, max_value=50.0, value=30.0)
humidity = st.number_input("Current Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
wind_speed = st.number_input("Current Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0)

# Predict heatwave
input_data = pd.DataFrame([[temp, humidity, wind_speed]], columns=['Temperature', 'Humidity', 'WindSpeed'])
heatwave_prediction = summermodel.model.predict(input_data)[0]

if heatwave_prediction == 1:
    st.warning("Alert: Potential Heatwave Detected! Take precautions.")
else:
    st.success("No heatwave predicted. Stay safe!")

# To run the app, use:
# python -m streamlit run heatwave_alertapp.py