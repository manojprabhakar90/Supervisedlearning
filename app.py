import streamlit as st # type: ignore
import pickle
import pandas as pd # type: ignore
from joblib import load# type: ignore

@st.cache_data 
def load_model():
    model = load('RandomForestModel.pkl')
    return model

model = load_model()

st.title('Website Phishing Detection Model')
google_index = st.number_input('Google Index', min_value=0)  
page_rank = st.number_input('Page Rank', min_value=0.0, step=0.1)
nb_hyperlinks = st.number_input('Number of Hyperlinks', min_value=0)
nb_www = st.number_input('Number of WWW', min_value=0)
domain_age = st.number_input('Domain Age', min_value=0)
if st.button('Predict'):
    input_data = pd.DataFrame([[google_index, page_rank, nb_hyperlinks, nb_www, domain_age]],
                              columns=['google_index', 'page_rank', 'nb_hyperlinks', 'nb_www', 'domain_age'])

    prediction = model.predict(input_data)
    if prediction==0:
        st.write("Phishing Website. Please be careful")
    else:
        st.write("Genuine Website")
