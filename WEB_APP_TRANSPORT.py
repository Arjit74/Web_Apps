import numpy as np
import joblib
import streamlit as st
import pandas as pd

# Load the model
model = joblib.load('E:\\Both\\transport_model.pkl')

# Set page configuration (optional)
st.set_page_config(page_title='Transportation Demand Predictor', page_icon='🚗')

# Styled Title with Background and Foreground Colors
st.markdown(
    "<div style='background-color: #FFD662; padding: 10px;'>"
    "<h1 style='text-align: center; color: #01539D;'>🚗 Transportation Demand Predictor 🚗</h1>"
    "</div>", 
    unsafe_allow_html=True
)

# Add spacing (margin) between the title and columns
st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)

# Layout: Input fields side by side using Streamlit's columns
col1, col2 = st.columns(2)

with col1:
    # Population Density input with emoji
    population_density = st.number_input('🏙️ Population Density (people/km²)', min_value=0)
    
    # Economic Activity input with emoji
    economic_activity = st.number_input('📊 Economic Activity (units)', min_value=0)
    
with col2:
    # Infrastructure Score input with emoji
    infrastructure_score = st.number_input('🏗️ Infrastructure Score (1-10)', min_value=1, max_value=10)
    
    # Average Trip Length input with emoji
    average_trip_length = st.number_input('🚶‍♂️ Average Trip Length (km)', min_value=0.0, format="%.2f")

# Transport Mode dropdown with emoji
transport_mode = st.selectbox('🛣️ Transport Mode', ['Taxi', 'Car', 'Walking', 'Bicycle', 'Ride-sharing', 'Bus', 'Motorcycle', 'Train'])

# Create input data for prediction
input_data = pd.DataFrame({
    'population_density': [population_density],
    'economic_activity': [economic_activity],
    'infrastructure_score': [infrastructure_score],
    'average_trip_length': [average_trip_length],
    'transport_mode': [transport_mode],
})

# One-Hot Encoding the transport_mode
input_data_encoded = pd.get_dummies(input_data, columns=['transport_mode'])

# Get the feature names that the model expects
expected_columns = model.feature_names_in_

# Check for missing columns and add them with zeros
for col in expected_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Reorder columns to match the order the model was trained on
input_data_encoded = input_data_encoded[expected_columns]

# Center the prediction button
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

# Prediction Button at the center
if st.button('🚀 Predict Demand'):
    try:
        # Make prediction
        demand_forecast = model.predict(input_data_encoded)

        # Correctly round the forecast value using numpy
        if isinstance(demand_forecast, np.ndarray) and demand_forecast.ndim == 1:
            forecast_value = np.round(demand_forecast[0],2)
        else:
            forecast_value = np.round(demand_forecast,2)

        # Access and round the forecast value
        if demand_forecast.ndim > 1:
            forecast_value = demand_forecast[0, 0]
        else:
            forecast_value = demand_forecast[0]

        forecast_value = round(forecast_value, 2)


        # Display the result with a colorful and formatted output
        st.markdown(
            f"<h2 style='text-align: center; color: #FA9490;'>✨ Predicted Demand Forecast</h2>"
            f"<h3 style='text-align: center; color: #ADF0D1;'>{forecast_value} units</h3>",
            unsafe_allow_html=True
        )
    except ValueError as e:
        st.error(f"Prediction Error: {e}")

# Close the div for centering
st.markdown("</div>", unsafe_allow_html=True)
