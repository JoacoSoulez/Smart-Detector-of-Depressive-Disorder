from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from MHFA.preprocessing import clean_text, vectorize

import numpy
import pandas as pd
import joblib



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
    return {"greeting": "Hello world"}


@app.get("/predict")
def predict(text):

    print('bringing model')
    model = joblib.load('model.joblib')

    print('cleaning text')
    clean = clean_text(pd.Series(text))


    print('predicting on vector')
    depressed= model.predict(clean)[0]
    probability = model.predict_proba(clean)[0][1]
    print('returning dict')
    return {'depressed': int(depressed),
            'probability of depression': float(probability)

            }
