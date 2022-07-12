import sklearn
import streamlit as st
import pandas as pd
import pickle

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

model = pickle.load(open('churn_pipe.pkl', 'rb'))

with st.sidebar:
    gender = st.radio('Gender', options=['Male', 'Female'], horizontal=True)
    SeniorCitizen = st.checkbox('Senior Citizen')
    Partner = st.radio('Partner', options=['Yes', 'No'], horizontal=True)
    Dependents = st.radio('Dependents', options=['Yes', 'No'], horizontal=True)
    tenure = st.number_input('tenure', min_value=0,max_value=72)
    PhoneService = st.radio('Phone Service', options=['Yes', 'No'], horizontal=True)
    MultipleLines = st.selectbox('Multiple Phone Lines', options=['No phone service', 'No', 'Yes'])
    InternetService = st.selectbox('Internet Service', options=['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.radio('Online Security', options=['Yes', 'No'], horizontal=True)
    OnlineBackup = st.radio('Online Backup', options=['Yes', 'No'], horizontal=True)
    DeviceProtection = st.radio('Device Protection', options=['Yes', 'No'], horizontal=True)
    TechSupport = st.radio('Tech Support', options=['Yes', 'No'], horizontal=True)
    StreamingTV	= st.radio('Streaming TV', options=['Yes', 'No'], horizontal=True)
    StreamingMovies = st.radio('Streaming Movies', options=['Yes', 'No'], horizontal=True)
    Contract = st.selectbox('Contract', options=['Month-to-month', 'One year', 'Two year'])	
    PaperlessBilling = st.radio('Paperless Billing', options=['Yes', 'No'], horizontal=True)
    PaymentMethod = st.selectbox('Payment Method', options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    MonthlyCharges = st.number_input('Monthly Charges')
    TotalCharges = st.number_input('Total Charges')

columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']

df = pd.DataFrame(data=[[gender ,  SeniorCitizen ,  Partner ,  Dependents ,  tenure ,
        PhoneService ,  MultipleLines ,  InternetService ,  OnlineSecurity ,
        OnlineBackup ,  DeviceProtection ,  TechSupport ,  StreamingTV ,
        StreamingMovies ,  Contract ,  PaperlessBilling ,  PaymentMethod ,
        MonthlyCharges ,  TotalCharges ]], columns=columns)

cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       ]

df[cols] = df[cols].astype('category')



st.header('Customer Churn')
st.markdown('Please fill in the customer data for a churn prediction.')

prediction = model.predict(df)
if prediction[0] == 'No':
       st.success('This customer will not churn!')
else:
       st.error('This customer will churn!')


proba = round(model.predict_proba(df)[0][1],1) * 100


from streamlit_echarts import st_echarts



option = {
  'series': [
    {
      'type': 'gauge',
      'progress': {
       'show': True,
       'width': 30,
       'itemStyle': {
              'color': '#ffdbdb'
        },       
      },
      'pointer': {
              'show':False
        },

      'axisLine': {
        'lineStyle': {
        'width': 30,
        'color':[
            [0, '#ceeed8'],
            [0.51, '#ceeed8'],
            [1, '#ceeed8']
          ]
        }
      },
      'axisTick': {
        'show': False
      },
      'splitLine': {
        'length': 10,
        'lineStyle': {
          'width': 1,
          'color': '#fff'
        }
      },
      'axisLabel': {
        'distance': 15,
        'color': '#999',
        'fontSize': 10
      },
      'anchor': {
        'show': False,
        'showAbove': True,
        'size': 20,
        'itemStyle': {
          'borderWidth': 5,
          'color': '#fff'
        }
      },
      'title': {
        'show': True
      },
      'detail': {
        'valueAnimation': True,
        'fontSize': 40,
        'color': '#ffdbdb',
        'offsetCenter': [0, 0],
        'formatter': '{value}%',
        'valueAnimation': True,
      },
      'data': [
        {
          'value': proba
        }
      ]
    }
  ]
};


st_echarts(options=option, width='500px')






# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)