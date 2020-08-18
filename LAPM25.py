import streamlit as st
import pandas as pd
import pickle

st.title("Air Quality Index of Los Angeles, CA")
st.title("  ")

# Images
from PIL import Image
img = Image.open("la.jpg")
st.image(img, width=700, caption="Air Pollution of Los Angeles. Photo: USA Today.")

st.header("About the App")

st.markdown("### This application has been built based on machine learning algorithm. All the input features have to be inserted in correct units (mentioned within parentheses beside each predictor) to get a result. The main pollutant of the Los Angeles air is PM 2.5 and the objective of this app is to predict the concentration of PM 2.5 in micro-gram/cubic-meter. The accuracy of the model is 82.62%.")

st.sidebar.header('User Input parameters')

def user_input_features():
    avg_temp = st.sidebar.slider('Average Temperature (°C)', -20.0, 60.0, 25.0)
    max_temp = st.sidebar.slider('Maximum Temperature (°C)', -20.0, 60.0, 38.0)
    min_temp = st.sidebar.slider('Minimum Temperature (°C)', -20.0, 60.0, -1.0)
    sealevel_pressure = st.sidebar.slider('Atmospheric Sea Level Pressure (hPa)', 900.0, 2000.0, 1015.0)
    avg_humidity = st.sidebar.slider('Average Relative Humidity (%)', 0.0, 200.0, 73.0)
    rainfall_snowmelt = st.sidebar.slider('Total Rainfall and/or Snowmelt (mm)', 0.0, 150.0, 40.0)
    visibility = st.sidebar.slider('Average Visibility (km)', 0.0, 70.0, 9.3)
    avg_windspeed = st.sidebar.slider('Average Wind Speed (km/h)', 0.0, 400.0, 100.0)
    max_windspeed = st.sidebar.slider('Maximum Sustained Wind Speed (km/h)', 0.0, 400.0, 185.0)

    data = {'avg_temp': avg_temp, 
            'max_temp': max_temp, 
            'min_temp': min_temp, 
            'sealevel_pressure': sealevel_pressure, 
            'avg_humidity': avg_humidity, 
            'rainfall_snowmelt': rainfall_snowmelt, 
            'visibility': visibility, 
            'avg_windspeed': avg_windspeed, 
            'max_windspeed': max_windspeed} 
   
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

model = pickle.load(open('RF_regression_LA.pkl', 'rb'))  # get the model

prediction = model.predict(df)

st.success(f'Concentration of PM 2.5 is : {round(prediction[0], 4)} micro-gram/cubic-meter')

