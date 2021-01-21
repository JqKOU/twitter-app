import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB

st.write("""
# Twitter User Type Prediction App

This app predicts whether a Twitter user is a **Programmer** or a **Gamer** based on his/her user description.

If a user posts tweets with keywords python, java, and c++, this user is more likely to be a programmer.

If a user posts tweets with keywords PokemonGo, AnimalCrossing, and ACNH, this user is more likely to be a gamer.

Model is based on data collected from Twitter with keywords of programmer (python, java, c++) and gamer (PokemonGo, AnimalCrossing, ACNH)
""")

# Reads in saved classification model
load_mnb = pickle.load(open('mnb_model.pkl', 'rb'))

st.subheader('Input your Twitter Description')

description = st.text_input('Input your description here:') 
List = list(description.split())

# Apply model to make predictions
prediction = load_mnb.predict(List)[0]
prediction_proba = load_mnb.predict_proba(List)[0]

st.subheader('Prediction')
user_type = np.array(['Programmer','Gamer'])
st.write(user_type[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
