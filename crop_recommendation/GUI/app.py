import streamlit as st
import pandas as pd
import numpy as np
import pycaret.clustering as pc
# Create a text input field
# input_text = st.text_input('Enter your text:',type='default')

# print(input_text)
col1, col2, col3,col4 = st.columns(4)
col5, col6, col7 = st.columns(3)
data = pd.read_csv('kmeans_result2.csv')
n = float(col1.number_input('enter value of Nitrogen:',key='n', min_value=round(data['N'].min()), max_value=round(data['N'].max())))
p = float(col2.number_input('enter value of Phorphorus:', key='p', min_value=round(data['P'].min()), max_value=round(data['P'].max())))
k = float(col3.number_input('enter value of Potassium:', key='k', min_value=round(data['K'].min()), max_value=round(data['K'].max())))
ph = float(col4.number_input('enter value of soil ph:', key='ph', min_value=round(data['ph'].min()), max_value=round(data['ph'].max())))
rainfall = float(col5.number_input('enter rainfall value:',key='rainfall', min_value=round(data['rainfall'].min()), max_value=round(data['rainfall'].max())))
temperature = float(col6.number_input('enter temperature value:', key='temperature', min_value=round(data['temperature'].min()), max_value=round(data['temperature'].max())))
humidity = float(col7.number_input('enter humidity value:', key='humidity', min_value=round(data['humidity'].min()), max_value=round(data['humidity'].max())))


def process(n,p,k,rainfall,humidity,temperature,ph):
    unknown_data = pd.DataFrame([{'N':round(n),'P':round(p),'K':round(k),'rainfall':round(rainfall),'humidity':round(humidity),'ph':round(ph),'temperature':round(temperature)}])
    kmeans_result2 = pd.read_csv('kmeans_result2.csv')
    kmeans_result2['Cluster'] = kmeans_result2['Cluster'].str.replace("Cluster",'').apply(int)
    saved_model = pc.load_model("group2-model")
    pred = pc.predict_model(saved_model,unknown_data)
    pred = int(pred['Cluster'][0][-1])
    kmeans_result=kmeans_result2[kmeans_result2['Cluster']==pred]
    crops = set(kmeans_result['label'])
    
    return st.write('crops: ',crops)
    
if st.button('predict'):
    process(n,p,k,humidity,temperature,ph,rainfall)
