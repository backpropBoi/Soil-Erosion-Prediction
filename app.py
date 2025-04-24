import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Soil Erosion Predictor",
    page_icon="🌱",
    layout="wide"
)

# Title and description
st.title("Soil Erosion Prediction System")
st.markdown("""
This application helps predict soil erosion based on various soil and environmental parameters.
Enter the details below to get a prediction.
""")

# Create input form
st.subheader("Input Parameters")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    # Climate parameters
    mat = st.number_input("Mean Annual Temperature (°C)", min_value=-50.0, max_value=50.0, value=20.0)
    map_value = st.number_input("Mean Annual Precipitation (mm)", min_value=0.0, max_value=5000.0, value=1000.0)
    elevation = st.number_input("Elevation (m)", min_value=0.0, max_value=9000.0, value=100.0)
    
    # Location parameters
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0)
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0)
    slope = st.number_input("Slope (%)", min_value=0.0, max_value=90.0, value=5.0)

with col2:
    # Soil composition parameters
    soil_sand = st.number_input("Soil Sand Content (%)", min_value=0.0, max_value=100.0, value=40.0)
    soil_silt = st.number_input("Soil Silt Content (%)", min_value=0.0, max_value=100.0, value=30.0)
    soil_clay = st.number_input("Soil Clay Content (%)", min_value=0.0, max_value=100.0, value=30.0)

# Add a predict button
if st.button("Predict Soil Erosion"):
    # Create input data
    input_data = pd.DataFrame({
        'MAT': [mat],
        'MAP': [map_value],
        'Elevation': [elevation],
        'Latitude': [latitude],
        'Longitude': [longitude],
        'Slope': [slope],
        'Soil_sand': [soil_sand],
        'Soil_silt': [soil_silt],
        'Soil_clay': [soil_clay]
    })
    
    try:
        # Load the model and scaler
        model = joblib.load('soil_erosion_model.joblib')
        scaler = joblib.load('soil_erosion_scaler.joblib')
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display results
        st.subheader("Prediction Results")
        
        # Create a gauge chart for the prediction
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            title = {'text': "Predicted Soil Organic Carbon (%)"},
            gauge = {
                'axis': {'range': [0, 5]},  # Adjusted range for SOC percentage
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 1], 'color': "darkred"},
                    {'range': [1, 2], 'color': "red"},
                    {'range': [2, 3], 'color': "orange"},
                    {'range': [3, 4], 'color': "yellow"},
                    {'range': [4, 5], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prediction
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display interpretation
        st.subheader("Interpretation")
        if prediction < 2.0:
            st.error("High erosion risk. The soil has low organic carbon content, making it more susceptible to erosion.")
        else:
            st.success("Low erosion risk. The soil has good organic carbon content, providing better stability against erosion.")
            
        # Display recommendations
        st.subheader("Recommendations")
        if prediction < 2.0:
            st.markdown("""
            Based on the high erosion risk, consider the following measures:
            1. Implement vegetative cover to protect the soil
            2. Use terracing techniques to reduce slope effects
            3. Apply mulch or ground cover to protect soil surface
            4. Consider contour plowing to reduce water runoff
            5. Install erosion control structures
            6. Add organic matter to improve soil structure
            7. Implement conservation tillage practices
            """)
        else:
            st.markdown("""
            To maintain current soil conditions:
            1. Continue current soil management practices
            2. Monitor soil conditions regularly
            3. Maintain vegetative cover
            4. Practice sustainable farming methods
            5. Consider crop rotation to maintain soil health
            6. Avoid over-tilling to preserve soil structure
            """)
            
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the model is trained and saved first.")
        st.info("To train the model, run the training script first.") 