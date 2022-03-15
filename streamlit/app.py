import streamlit as st
import requests

url = 'https://depressiondetection-rdm72uggnq-ew.a.run.app/predict'

st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)

st.markdown("""
<style>
.title {
    font-size:50px !important;
}
.sub-title {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p class="title"> Mental Health First Aid', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sub-title"> Sentiment Analyse posts in social media', unsafe_allow_html=True)


col1, col2 = st.columns([10,2])

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
