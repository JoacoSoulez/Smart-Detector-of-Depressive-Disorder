# Mental Health First Aid

An Intelligent depression detector


## Motivation

In the world there are more than 280 million people diagnosed with depression. If you add them all together, it would be the fourth most populated country in the world. Today, access to quality mental health diagnosis and follow-up is taboo and difficult to access. The idea is to be able to create a therapeutic assistant that not only detects depression, but also supports the process of improvement.


## Method and results

In order to achieve this, and following the line of several researchers, we have used a twitter database with (2345 tweets) information of people who have a tendency to suffer from this disease, and contrasted it with a selection of positive tweets extracted from a twitter database with more than 1.6 million tweets analyzed according to their sentiment (positive negative).
Then, we also used a reddit database of posts in a subforum where people with depression write.


On the other hand, we have made a neural network that we trained with a database (Avec2017) of transcripts of therapy sessions of diagnosed people and people without depression. I am currently working on incorporating audio.


## Repository overview

├── Code
│   ├── Depression Detection via SocialMedia
│   │   ├── Clean_Tweets_preprocessing.ipynb
│   │   ├── data.ipynb
│   │   └── naive_bayes_basemodel.ipynb
│   ├── Depression Detection via Therapy
│   │   ├── Audio
│   │   ├── Text
│   │   │   └── Depresion_Modelo_RNN.ipynb
│   │   └── Text and Audio
│   └── naive_bayes.ipynb
├── Dockerfile
├── Draft_notebooks
├── MANIFEST.in
├── MHFA
│   ├── __init__.py
│   ├── bayes.py
│   ├── data
│   ├── data.py
│   └── preprocessing.py
├── Makefile
├── Procfile
├── README.md
├── api
│   ├── __init__.py
│   └── api.py
├── model.joblib
├── raw_data
├── requirements
├── requirements.txt
├── scripts
│   └── MHFA-run
├── setup.py
├── setup.sh
├── streamlit
│   ├── 0-4495_mental-health-icon-png-mental-health-icon-transparent.png:Zone.Identifier
│   ├── app.py
│   ├── icon-removebg-preview.png
│   ├── icon-removebg-preview.png:Zone.Identifier
│   └── icon.png
└── tests


## More resources

I am currently testing a RandomForest model and an SVC model, and will try a neural network soon.

## About

This project was done for the final project of Le Wagon's Data Science bootcamp delivered on March 18, with the help of my teammates Daniel Riojas, Lucas Pancotto and Leonardo Asad and is currently being updated.
