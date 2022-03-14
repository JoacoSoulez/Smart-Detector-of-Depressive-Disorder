import streamlit as st
import requests

url = 'https://depressiondetection-rdm72uggnq-ew.a.run.app/predict'


st.title("Mental Health First Aid")
st.write("Sentiment Analyse posts in social media")

col1, col2= st.columns([10,1])


with col1:

    text = st.text_input("")
    #st.write(type(text))


with col2:
    st.text("")
    st.text("")
    result = st.button("Analyse")


if result and text != '':
    # Get the prediction from the api
    y_pred = 0.8
    params = {'text': text}
    response = requests.get(url, params=params)
    #st.write(response.status_code)

    if response.status_code == 200:
        proba = response.json()['probability of depression']
        if proba >= 0.5:
            st.write("Depressed")
            st.write('Probability of having depression: ', f'{proba:.0%}')
        else:
            st.write("Not Depressed")
            st.write('Probability of having depression: ', f'{proba:.0%}')
    else:
        st.write("Failed conection with the API")

else:
    st.write("Please insert a post to be analysed")
