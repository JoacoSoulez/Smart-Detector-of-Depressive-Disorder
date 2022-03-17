from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from MHFA.preprocessing_willyedit import clean_text, embed_sentence,embedding
import nltk
#nltk.download('all')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


import numpy as np
import pandas as pd
import joblib

import uvicorn

from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Welcome to our depression detection API"}


@app.get("/predict")
def predict(text):
    print('bringing vectorizer')
    vectorizer = joblib.load('tokenizer_transcripciones.joblib')



    print('bringing naive bayes model')
    naive_bayes = joblib.load('model.joblib')


    print('cleaning text')
    clean = list(clean_text(pd.Series(text)))

    print('counting words')
    word_count = word_tokenize(str(clean[0]))


    if len(word_count) > 50:

        print('bringing deeplearning model')
        model = joblib.load('transcripciones_finalizado_model.joblib')

        print('tokenizing clean text')
        #X_pred = embedding(vectorizer, X_pred)
        X_pred =  vectorizer.texts_to_sequences(clean)



        print('padding data with length hardcoded')
        X_pred = pad_sequences(X_pred, maxlen = 1348,value=-1000, dtype='float64')


        print('reshaping data')
        X_pred = np.reshape(X_pred, (X_pred.shape[0], 1, X_pred.shape[1]))

        vocab_size = len(vectorizer.word_index) + 1
        print("Total words", vocab_size)



        print('predicting deep learning on vector')
        depressed= model.predict(X_pred)[0]
        #probability = model.predict_proba(X_test_pad)[0][1]
        print('returning dict')
        return {'depressed': int(depressed),
                'warning of text length': 'your text is long: neural network has made an evaluation instead'

                }

    if len(word_count) < 50:

        print('predicting Naive Bayes on vector')
        depressed= naive_bayes.predict(clean)[0]
        probability = naive_bayes.predict_proba(clean)[0][1]
        print('returning dict')
        return {'depressed': int(depressed),
                'probability of depression': float(probability)

                }
