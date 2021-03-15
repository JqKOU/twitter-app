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
    text = nltk.word_tokenize(text) # have a nice day -- have, a, nice, day 
    text = stemSentence(text) #lemma: games --> game
    text = [word for word in text if word not in swords] # remove stopwords: have, a, nice, day --> nice, day
    text = ' '.join(text) # nice day
    return text

# prediction 
def user_type(description):
    Type = load_mnb.predict(Token)
    prediction_proba = load_mnb.predict_proba(Token)
    if Type == 0:
        st.write ('This user is a **programmer** ')
        st.write ('The prediction probability is ', prediction_proba[0][0])
    elif Type == 1:
        st.write('This user is a **gamer**' )
        st.write ('The prediction probability is ', prediction_proba[0][1])
    else:
        st.write('invalid prediction')
        st.write ('The prediction probability is NA')

# Reads in saved classification model
load_mnb = pickle.load(open('mnb_model.pkl', 'rb'))

#### APP #####
st.write("""
# Twitter User Tagging

This app predicts whether a Twitter user is a **Programmer** or a **Gamer** based on user's description profile.


""")

st.subheader("Input the Twitter User Description")
description = st.text_input(" ")

st.write('👇 click to show/hide')
Token = [token(description)]

st.subheader('Prediction')
user_type(description)     

st.subheader(" ")
st.subheader(" ")
st.subheader("                          🌴🧸                      ")

st.subheader("About")
st.write("""
This model were trained on Twitter users' profiles who posted tweets with keywords in [python, java, C++, PokemonGo, AnimalCrossing, ACNH]. 

The first 3 keywords **[python, java, C++]** capture users that are more likely to be **programmers**.

The last 3 keywords **[PokemonGo, AnimalCrossing, ACNH]** captures users that are more likely to be **gamers**.

For more detail of this app, please check: https://github.com/JqKOU/twitter-app.  

""")
