import streamlit as st
import pandas as pd
#import matplotlib as plt
import joblib
import warnings
warnings.filterwarnings('ignore')
#import plotly as px


data = pd.read_csv('Accountdata.csv')
data.head()

st.markdown("<h1 style = 'color: #DD5746; text-align: center; font-size: 60px; font-family: Monospace'>LOAN PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FFC470; text-align: center; font-family: Serif '>Built by KufreKing</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)

st.image('pngwing.com (2).png', caption= 'Built by KufreKing')


st.markdown("<h2 style = 'color: #F7C566; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)
st.markdown("Lending institutions struggle with inefficient loan approval processes, high default rates, and subjective decision-making. Traditional methods are slow and prone to bias. This study proposes a Machine Learning-based Loan Prediction Model to automaterisk assessment, enhance accuracy, and streamline approvals, ensuring better financial accessibility and reduced losses")

#sidebar design
st.sidebar.image('user icon.png')


st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.divider() #seperates the background of study from the project data
st.header('Project Data')
st.dataframe(data, use_container_width= True)

#user input section
app_income = st.sidebar.number_input('Applicant Income', data['ApplicantIncome'].min(), data['ApplicantIncome'].max())
loan_amt = st.sidebar.number_input('Loan Amount', data['LoanAmount'].min(), data['LoanAmount'].max())
coapp_income = st.sidebar.number_input('CoApplicant Income', data['CoapplicantIncome'].min(), data['CoapplicantIncome'].max())
dep = st.sidebar.selectbox('Dependents', data['Dependents'].unique())
prop_area = st.sidebar.selectbox('Property Area', data['Property_Area'].unique())
cred_hist = st.sidebar.number_input('Credit History', data['Credit_History'].min(), data['Credit_History'].max())
loan_amt_term = st.sidebar.number_input('Loan Amount Term', data['Loan_Amount_Term'].min(), data['Loan_Amount_Term'].max())


#user input linked to be the same with what is on the dataFrame

#users input
input_var = pd.DataFrame()
input_var['ApplicantIncome'] = [app_income]
input_var['LoanAmount'] = [loan_amt]
input_var['CoapplicantIncome'] = [coapp_income]
input_var['Dependents'] = [dep]
input_var['Property_Area'] = [prop_area]
input_var['Credit_History'] = [cred_hist]
input_var['Loan_Amount_Term'] = [loan_amt_term]

#Table to display the results of the users input
st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.subheader('User Inputs')
st.dataframe(input_var, use_container_width= True)

#Load selected encoded and scaled columns
app_income = joblib.load('ApplicantIncome_scaler.pkl')
coapp_income = joblib.load('CoapplicantIncome_scaler.pkl')
prop_area= joblib.load('Property_Area_encoder.pkl')


#transform the users input with the imported scalers
input_var['ApplicantIncome']= app_income.transform(input_var[['ApplicantIncome']])
input_var['CoapplicantIncome']= coapp_income.transform(input_var[['CoapplicantIncome']])
input_var['Property_Area']= prop_area.transform(input_var[['Property_Area']])


#Bringing the model for prediction
model = joblib.load('Loandmodel.pk1')
predict = model.predict(input_var)

if st.button('Check Your Loan Approval Status'):
    if predict[0] == 0:
        st.error(f'Unfortunately.... Your loan request was denied')
        st.image('denied.jpg', width= 300)
    else:
        st.success(f'Congratulations.... Your loan request has be garanted. please come to the office to process your loan')
        st.image('approved.jpg', width= 300)
        st.balloons()
