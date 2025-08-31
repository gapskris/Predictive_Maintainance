# %%
import streamlit as st
import joblib
import numpy as np
from PIL import Image

model_bin = joblib.load("LGBMClassifier_binary.pkl")
model_multi = joblib.load("GradientBoostingClassifier.pkl")
# Load models on button click


# Load the model
# App title
image1 = Image.open('image1.png')
st.image(image1, width=800)  
image = Image.open('image.png')
st.image(image, width=800)  

st.title("🛠️ Predictive Maintainance App")
st.write("This app predicts the **Fail/No Fail as well as Failure_Type** of the machine based on input parameters.")

FailureType = ['Heat Dissipation Failure', 'No Failure', 'Overstrain Failure','Power Failure','Random Failure',"Tool Wear Failure"]
Target = ['No Failure', 'Failure']

#Heat Dissipation Failure → 0
#No Failure → 1
#Overstrain Failure → 2
#Power Failure → 3
#Random Failures → 4
#Tool Wear Failure → 5

# Input fields
Type = st.selectbox("Choose type:", ["L", "M", "H"]) 
st.write(f"Selected type: {Type}") #size = st.radio("Select Type:", ["L", "M", "S"]) 
type_mapping = {"H": 0, "L": 1, "M": 2}
Type_encoded = type_mapping[Type]

Air_Temperature = st.number_input('Air_Temperature', min_value=2.0, max_value=400, value=10.0, step=5, format="%.2f") 
st.write("Air_Temperature:", f"{Air_Temperature:.2f}")

Process_Temperature = st.number_input('Process_Temperature', min_value=2.0, max_value=400, value=10.0, step=5, format="%.2f") 
st.write("Process_Temperature:", f"{Process_Temperature:.2f}")

Roational_Temperature = st.number_input('Roational_Temperature', min_value=2.0, max_value=3000, value=3.0, step=5, format="%.2f") 
st.write("Roational_Temperature:", f"{Roational_Temperature:.2f}")

Torque_Temperature = st.number_input('Torque_Temperature', min_value=2.0, max_value=100, value=10.0, step=5, format="%.2f") 
st.write("Torque_Temperature:", f"{Torque_Temperature:.2f}")

Tool_Wear_Min = st.number_input('Tool_Wear_Min', min_value=2.0, max_value=200, value=10.0, step=10.0, format="%.2f") 
st.write("Tool_Wear_Min:", f"{Tool_Wear_Min:.2f}")

# Predict button
if st.button('Predict Binary Classification'):
    input_data = np.array([[Type_encoded, Air_Temperature, Process_Temperature, Roational_Temperature, Torque_Temperature, Tool_Wear_Min]])
    model = model_bin
    if(model_bin is None):
        st.error("Model not loaded. Please load the model first.")
    prediction = model.predict(input_data)
    st.success(prediction[0])
    st.success(f'Predicted Target: **{Target[prediction[0]]}**')   
        
        
if st.button('Predict - Multi Classification'):
    input_data = np.array([[Type_encoded, Air_Temperature, Process_Temperature, Roational_Temperature, Torque_Temperature, Tool_Wear_Min]])   
    model = model_multi
    if(model_multi is None):
        st.error("Model not loaded. Please load the model first.")
    prediction = model.predict(input_data)
    st.success(prediction[0])
    st.success(f'Predicted Failure_Type: **{FailureType[prediction[0]]}**')
    


