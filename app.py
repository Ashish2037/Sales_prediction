import streamlit as st
import joblib
import numpy as np

trained_model = joblib.load('model.pkl')


st.title('Sales Prediction')

st.write("NOTE: Please give input based on the changes we made in the data")

# Get numerical input features from the user
numerical_features = ['Item_Weight', 'Item_Fat_Content', 'Item_Type', 'Item_MRP', 'Outlet_Size','Outlet_Location_Type','Outlet_Type']
input_features = []

for feature_name in numerical_features:
    value = st.number_input(f"Enter value for {feature_name}: ")
    input_features.append(value)
    
if st.button('Predict'):
    # Convert input features to a NumPy array
    input_features_array = np.array(input_features).reshape(1, -1)

    # Make predictions using the loaded model
    predicted_sales = trained_model.predict(input_features_array)

    st.write(f"Predicted class: {predicted_sales}")