import streamlit as st
import requests

st.markdown(
    """
<style>
.reportview-container .markdown-text-container {
    font-family: monospace;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
.Widget>label {
    color: white;
    font-family: monospace;
}
[class^="st-b"]  {
    color: white;
    font-family: monospace;
}
.st-bb {
    background-color: transparent;
}
.st-at {
    background-color: #0c0080;
}
footer {
    font-family: monospace;
}
.reportview-container .main footer, .reportview-container .main footer a {
    color: #0c0080;
}
header .decoration {
    background-image: none;
}

</style>
""",
    unsafe_allow_html=True,
)

url = 'https://depressiondetection-rdm72uggnq-ew.a.run.app/predict'



st.title("Anhedonia")
st.write("Mental Health First Aid - Depression Evaluation")
st.markdown("""# Depression Evaluation
## from small texts (social media!)"""
)
col1, col2= st.columns([10,1])
st.markdown("""
## from large texts"""
)
col3, col4 = st.columns([10,1])


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



with col3:
    text2 = st.text_input("Type here", key = "texto_largo")
with col4:
    st.text("")
    st.text("")
    resultados = st.button("analyse")

if resultados and text2 != '':
    y_pred = 0.8
    params = {'text': text2}
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
