import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
import nltk

st.write("""
# Twitter User Type Prediction App

This app predicts whether a Twitter user is a **Programmer** or a **Gamer** based on his/her user description.
""")

# Reads in saved classification model
load_mnb = pickle.load(open('mnb_model.pkl', 'rb'))

st.subheader("Input your Twitter Description")
description = st.text_input("input your description here")
Token = description.lower() 

# Apply model to make predictions
prediction = load_mnb.predict([Token])
prediction_proba = load_mnb.predict_proba([Token])

st.subheader('Prediction')
user_type = np.array(['Programmer','Gamer'])
st.write(user_type[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
