import streamlit as st
from streamlit_option_menu import option_menu

import requests
from PIL import Image

st.set_page_config(
     page_title="Mental Health First Aid",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded"
 )

image = Image.open('streamlit/icon-removebg-preview.png')

url = 'https://depressiondetection-rdm72uggnq-ew.a.run.app/predict'

st.sidebar.image(image)
st.sidebar.markdown('<p class="title"> Mental Health First Aid', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sub-title"> Text based detector of depressive disorders', unsafe_allow_html=True)

st.markdown("""
<style>
.title {
    font-family:Verdana;
    font-size:40px !important;
    text-align: center;
}
.sub-title {
    font-size:20px !important;
    text-align: center;
}
.title_left {
    font-family:Verdana;
    font-size:40px !important;
}
.sub_title_left {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

with st.container():
    selected = option_menu("Main Menu", ["App", 'Statistics'],
        icons=['app', 'clipboard-data'], menu_icon="cast", default_index=0,
        orientation="horizontal",
        styles={
        "container": {"padding": "0!important"},
        "icon": {"font-size": "25px"},
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#ede868"},
    })


if selected == 'Statistics':

    st.markdown(
        """
        <p class="title_left"> Depression
        """
    , unsafe_allow_html=True)

    st.markdown(
        """
        <p class="sub_title_left"> Key facts
        """
        , unsafe_allow_html=True
    )

    st.markdown(
        """
        - Depression is a common mental disorder. Globally, it is estimated that 5.0% of adults suffer from depression.\n
        - Depression is a leading cause of disability worldwide and is a major contributor to the overall global burden of disease.\n
        - More women are affected by depression than men.\n
        - Depression can lead to suicide.\n
        - There is effective treatment for mild, moderate, and severe depression.\n
        """
    )


    st.components.v1.html(
        """
        <iframe src="https://ourworldindata.org/grapher/share-with-depression" loading="lazy" style="width: 100%; height: 600px; border: 0px none;"></iframe>
        """,
        width=900,
        height=700)

    st.components.v1.html(
        """
        <iframe src="https://ourworldindata.org/grapher/number-of-people-with-depression" loading="lazy" style="width: 100%; height: 600px; border: 0px none;"></iframe>
        """,
        width=900,
        height=700)

    st.markdown(
        """
        - References
            - WHO: https://www.who.int/news-room/fact-sheets/detail/depression\n
            - Our World in Data: https://ourworldindata.org/
        """
    )

else:

    st.info("Use this tool to get the sentiment score of any text sequence")


    col1, col2 = st.columns([10,2])

    with col1:

        text = st.text_area("")
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
                #st.write("Depressed")
                #st.write('Probability of having depression: ', f'{proba:.0%}')
                st.warning("""
                        This text might indicate that you're going through a deppresive episode. Don't worry, try logging out of social media for a while. If you need help, call:\n
                        From Mexico - IMSS: 800-2222-668 Option 4\n
                        From Spain - Psicólogos sin fronteras: 960450230\n
                        From Argentina: 0800-222-5462\n
                        """
                        )
                st.write("We also recommend you check out this [quick evaluation](https://www.psychologytoday.com/us/tests/health/mental-health-assessment)")
            else:
                #st.write("Not Depressed")
                #st.write('Probability of having depression: ', f'{proba:.0%}')
                st.success("This text don't show any signs of depression.")
        else:
            st.write("Failed conection with the API")

    else:
        st.write("Please insert a post to be analysed")
