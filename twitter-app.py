import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
import re
swords = stopwords.words("english")
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def token(text):
    text = text.lower()
    text = re.sub('[^a-z]', ' ', str(text))
    text = stemSentence(text)
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in swords]
    text = ' '.join(text)
    return text

st.write("""
# Twitter User Type Prediction App

This app predicts whether a Twitter user is a **Programmer** or a **Gamer** based on his/her user description.

If a user posts tweets with keywords python, java, and c++, this user is more likely to be a programmer.

If a user posts tweets with keywords PokemonGo, AnimalCrossing, and ACNH, this user is more likely to be a gamer.

Model is based on data collected from Twitter with keywords of programmer (python, java, c++) and gamer (PokemonGo, AnimalCrossing, ACNH)
""")

# Reads in saved classification model
load_mnb = pickle.load(open('mnb_model.pkl', 'rb'))

st.subheader("Input your Twitter Description")

description = st.text_input("input your description here")
Token = token(description)

# Apply model to make predictions
prediction = load_mnb.predict(Token)
prediction_proba = load_mnb.predict_proba(Token)

st.subheader('Prediction')
user_type = np.array(['Programmer','Gamer'])
st.write(user_type[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
