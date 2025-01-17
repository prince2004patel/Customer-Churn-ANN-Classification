import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

@st.cache_resource
def load_encoders():
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('one_hot_encoder_geography.pkl', 'rb') as file:
        one_hot_encoder_geography = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return label_encoder_gender, one_hot_encoder_geography, scaler

model = load_model()
label_encoder_gender, one_hot_encoder_geography, scaler = load_encoders()

st.title('Customer Churn Prediction App')

# User input with tooltips
geography = st.selectbox('Geography (Country of residence)', one_hot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age (Years)', 18, 92)
balance = st.number_input('Balance (Account balance in dollars)', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=0, max_value=1000)
estimated_salary = st.number_input('Estimated Salary (Annual salary in dollars)', min_value=0.0)
tenure = st.slider('Tenure (Number of years with the bank)', 0, 10)
num_of_products = st.slider('Number of Products (Products held at the bank)', 1, 4)
has_cr_card = st.selectbox('Has Credit Card (Whether the customer has a credit card)', [0, 1])
is_active_member = st.selectbox('Is Active Member (Active status)', [0, 1])

# Add a submit button at the bottom
if st.button('Predict'):
    # Prepare the input data
    try:
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        geo_encoded = one_hot_encoder_geography.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geography.get_feature_names_out(['Geography']))
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        input_data_scaled = scaler.transform(input_data)

        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        st.markdown(
            f"""
            <div style='background-color: #f4f4f4; padding: 10px; border-radius: 5px;'>
                <h3 style='color: #333;'>Churn Probability: {prediction_proba:.2f}</h3>
                <h4 style='color: #555;'>{'The customer is likely to churn.' if prediction_proba > 0.5 else 'The customer is not likely to churn.'}</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Add the footer with your name and GitHub link
st.markdown(
    """
    <p style="text-align:center; font-size:20px; color:#00bfff;">
        Made by <a href="https://github.com/prince2004patel" style="color:#00bfff; text-decoration:none;"><b>Prince Patel</b></a>
    </p>
    """, unsafe_allow_html=True
)