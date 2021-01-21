import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB

st.write("""
# Twitter User Type Prediction App

This app predicts whether a Twitter user is a **Programmer** or a **Gamer**.

Data collected from Twitter using Tweepy API with keywords of programmer and gamer: 

[python, java, c++] and [PokemonGo, AnimalCrossing, ACNH] 
""")


st.subheader('Input your Twitter Description')

description = st.text_input('Input your description here:') 

# Reads in saved classification model
load_mnb = pickle.load(open('mnb_model.pkl', 'rb'))

if description:
    st.write(my_model.predict(description))

# Apply model to make predictions
prediction = load_mnb.predict(description)
prediction_proba = load_mnb .predict_proba(description)

st.subheader('Prediction')
user_type = np.array(['Programmer','Gamer'])
st.write(user_type [prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)