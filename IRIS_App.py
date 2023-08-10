import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

st.subheader('By Solomon Kibon')
# Load the trained model
model = joblib.load("best_model.pkl")

# Define the app title and layout
st.title('Iris Flower Classification')
st.write('Enter the values of iris features to predict the flower species.')

# Define input fields for features
sepal_length = st.number_input("sepal_length", min_value=4.0, max_value=8.0, value=5.1, step=0.1)
sepal_width = st.number_input("sepal_width", min_value=2.0, max_value=4.5, value=3.5, step=0.1)
petal_length = st.number_input("petal_length", min_value=1.0, max_value=7.0, value=1.4, step=0.1)
petal_width = st.number_input("petal_width", min_value=0.1, max_value=2.5, value=0.2, step=0.1)
# Create a button for making predictions
if st.button("Predict"):
    # Process input values
    input_data = pd.DataFrame(
        {
            "sepal_length": [sepal_length],
            "sepal_width": [sepal_width],
            "petal_length": [petal_length],
            "petal_width": [petal_width]
        }
    )

 # Scale input data using the same scaler used during training
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Make a prediction using the trained model
prediction = model.predict(input_data_scaled)
#display prediction
if prediction==0:
    st.success(f'Predicted Species: Iris-setosa')
elif prediction==1:
    st.success(f'Predicted Species: Iris-versicolor')
else:
    st.wrsuccess(f'Predicted Species: Iris-virginica')
 
    