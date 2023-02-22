import joblib
import numpy as np
import streamlit as st
import pandas as pd

num_pipeline = joblib.load('num_pipeline.pkl')
full_pipeline = joblib.load('full_pipeline.pkl')
model = joblib.load('wr_regressor.pkl')

advertisers = ['Private','Estate agency']
buildings = ['Apartment building', 'Block of flats', 'Tenement house', 'Other']
districts = ['Bemowo', 'Białołęka', 'Bielany', 'Mokotów', 'Ochota', 'Praga-południe', 'Praga-północ',
             'Rembertów', 'Targówek', 'Ursus', 'Ursynów', 'Wawer', 'Wesoła', 'Wilanów', 'Wola',
             'Włochy', 'Śródmieście', 'Żoliborz']



def description():
    link = 'https://github.com/placeholder2'
    st.header('Description')
    st.markdown('This app predicts rant for apartment in Warsaw')
    st.markdown('Predictions are made by Gradient Boosting Regressor with Huber loss function trained'
                ' on data scrapped from otodom.pl in July 2022.')
    st.subheader('Links')
    st.markdown(f'Github page: {link}')





def predict_rent():

    st.subheader('Predict Rent')

    area = st.number_input(label='Area', help='Area in squared meters', step=1, min_value=5, max_value=350, value=30)
    advertiser = st.selectbox(label='Advertiser', options=advertisers)
    building = st.selectbox(label='Building', options=buildings)
    district = st.selectbox(label='District', options=districts)
    elevator = st.checkbox('Elevator')
    parking = st.checkbox('Parking space')
    wmachine = st.checkbox('Washing machine')
    oven = st.checkbox('Oven')
    stove = st.checkbox('Stove')
    ref = st.checkbox('Refrigerator')
    tv = st.checkbox('Cable TV')
    terrace = st.checkbox('Terrace')

    data = {'Type of building': building, "Area": area, "Advertiser": advertiser,
            'Elevator': elevator, 'Parking space': parking, 'District': district, 'Washing machine': wmachine,
            'Oven': oven,
            'Stove': stove, 'Refrigerator': ref, 'Cable TV': tv, 'Terrace': terrace}
    df = pd.DataFrame(data, index=[0])
    X = full_pipeline.transform(df)
    y = model.predict(X)
    n = num_pipeline.inverse_transform(y.reshape(-1, 1))
    rent = int(np.round(n))


    if st.button("Predict"):
        st.markdown(f'### **Predicted rent is {rent} zł**')


st.title('**Warsaw Rent Prediction**')

st.sidebar.title('Menu')

options = st.sidebar.radio('Select page',options = ['Description', 'Predict Rent'])

if options == 'Predict Rent':
    predict_rent()
else:
    description()
