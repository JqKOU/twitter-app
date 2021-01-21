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

# lemma sentence --> games --> game
porter=PorterStemmer()
def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

#preprocessing
def token(text):
    text = text.lower() #lowercase: Have a nice day! --> have a nice day!
    text = re.sub('[^a-z]', ' ', str(text)) #only keep a-z: have a nice day! --> have a nice day
    text = stemSentence(text) #lemma: games --> game
    text = nltk.word_tokenize(text) # have a nice day -- have, a, nice, day 
    text = [word for word in text if word not in swords] # remove stopwords: have, a, nice, day --> nice, day
    text = ' '.join(text) # nice day
    return text

# Reads in saved classification model
load_mnb = pickle.load(open('mnb_model.pkl', 'rb'))

#### APP #####
st.write("""
# Twitter User Type Prediction App

This app predicts whether a Twitter user is a **Programmer** or a **Gamer** based on his/her user description.
""")

st.subheader("Input your Twitter Description")
description = st.text_input("input your description here")

# Apply model to make predictions
prediction = load_mnb.predict([token(description)])
prediction_proba = load_mnb.predict_proba([token(description)])

st.subheader('Prediction')
user_type = np.array(['Programmer','Gamer'])
st.write(user_type[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba[prediction])
