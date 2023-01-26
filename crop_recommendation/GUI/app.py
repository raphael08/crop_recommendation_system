import streamlit as st
import pandas as pd
import numpy as np
import pycaret.classification as pc
# Create a text input field
# input_text = st.text_input('Enter your text:',type='default')

# print(input_text)
with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title(" CROP RECOMMENDATION ")
    choice = st.radio("Navigation", ["Predict","Upload"])
    st.info("""We use state-of-the-art machine learning and deep learning technologies to help you
					guide through
					the entire farming process. Make informed decisions to understand the demographics of your area,
					understand the
					factors that affect your crop and keep them healthy for a super awesome successful yield.""")


if choice == "Predict":
    st.title("Predict")
    col1, col2, col3,col4 = st.columns(4)
    col5, col6, col7 = st.columns(3)
    #data = pd.read_csv('../dataset/kmeans_result2.csv')
    n = (col1.text_input('enter value of Nitrogen:',key='n'))
    p = (col2.text_input('enter value of Phorphorus:', key='p'))
    k = (col3.text_input('enter value of Potassium:', key='k'))
    ph = (col4.text_input('enter value of soil ph:', key='ph'))
    rainfall = (col5.text_input('enter rainfall value:',key='rainfall'))
    temperature = (col6.text_input('enter temperature value:', key='temperature'))
    humidity = (col7.text_input('enter humidity value:', key='humidity'))

    def process(n,p,k,rainfall,humidity,temperature,ph):
        unknown_data = pd.DataFrame([{'N':float(n),'P':float(p),'K':float(k),'rainfall':float(rainfall),'humidity':float(humidity),'ph':float(ph),'temperature':float(temperature)}])
        saved_model = pc.load_model("../model/crop-model-classification")
        pred = pc.predict_model(saved_model,unknown_data)
        pred = (pred['Label'][0])
        
        
        return st.success(f"crop recommended for particular variables is :{pred}")       
        
    if st.button('predict'):
        
        if n.isalpha():
            st.error("only number required in NItrogen field")
        elif n.strip()=="":
            st.error("NItrogen Field required")
        elif p.isalpha():
            st.error("only number required in Phorphorus Field")
        elif p.strip()=="":
            st.error("Phorphorus Field required")
        elif k.isalpha():
            st.error("only number required in Pottasium Field")
        elif k.strip()=="":
            st.error("Pottasium Field required")
        else:
            process(n,p,k,humidity,temperature,ph,rainfall)
if choice == "Upload":
    def process(file):
        
        # unknown_data = pd.DataFrame([{'N':float(n),'P':float(p),'K':float(k),'rainfall':float(rainfall),'humidity':float(humidity),'ph':float(ph),'temperature':float(temperature)}])
        saved_model = pc.load_model("../model/crop-model-classification")
        pred = pc.predict_model(saved_model,file)
        pred = pd.DataFrame(pred,index=None)
        return pred
        
        
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset", type=['csv'])
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        if st.dataframe(df):
            if st.button('predict'):
                
                st.write(process(df))
                
                # if st.download_button('Download',process(df),'text/CSV'):
                #     st.success('date saved')
                 