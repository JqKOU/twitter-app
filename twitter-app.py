import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
import nltk

import re
from nltk import word_tokenize
#from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
word_lemmatizer = WordNetLemmatizer()

def preprocess(row):

    # lowercasing, tokenization, and keep only alphabetical tokens
    tokens = [word for word in word_tokenize(row.lower())\
              if word.isalpha()]
    # filtering out tokens that are not all alphabetical
    tokens = [word for word in re.findall(r'[A-Za-z]+', ' '.join(tokens))]
    # remove all stopwords
    #no_stop = [word for word in tokens\
              # if word not in stopwords.words('english')]
    # lematizing all tokens
    lemmatized = [word_lemmatizer.lemmatize(word) for word in tokens]
    # convert tokens back to a sentense as the input for CountVectorizer later
    processed = ' '.join(lemmatized)

    # return the clean sentense
    return processed


st.write("""
# Twitter User Type Prediction App

This app predicts whether a Twitter user is a **Programmer** or a **Gamer** based on his/her user description.
""")

# Reads in saved classification model
load_mnb = pickle.load(open('mnb_model.pkl', 'rb'))

st.subheader("Input your Twitter Description")
description = st.text_input("input your description here")
Token = preprocess(description)

# Apply model to make predictions
prediction = load_mnb.predict([Token])
prediction_proba = load_mnb.predict_proba([Token])

st.subheader('Prediction')
user_type = np.array(['Programmer','Gamer'])
st.write(user_type[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
