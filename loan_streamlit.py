import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

st.write("""
# Loan Default Probability Prediction

This app predicts the **Loan Default Probability**  Rifat Saeed
""")

# Displaying images on the front end
from PIL import Image
image = Image.open('webPic.jpg')

st.image(image, caption='Loan Default Probability')
st.sidebar.header('User Input Parameters')

#define multiple columns, add two graphs
col1, col2 = st.columns(2)

with col1:
# radio widget to take inputs from mulitple options
 genre1 = st.radio("What is term of the loan?",('36 months', '60 months'))
 if genre1 == '36 months':
    term= 0 
    st.write('You selected 36 months.')
 else: 
     term= 1
     st.write("You selected 60 months.")

with col2:
 genre2 = st.radio("Home Ownership?",('Mortgage', 'Rent','Own'))
 if genre2 == 'Mortgage':
    home_ownership= 0 
    st.write('You have mortage.')
 elif genre2 == 'Own':
    home_ownership= 2 
    st.write('You are owner.')
 else:
    home_ownership= 3 
    st.write('You are renting.')

loan = st.sidebar.slider('loan_amount', 0.0, 35000.0,5000.0)
annual_income = st.sidebar.slider('annual_income', 0.0,700000.0,100000.0)
purpose = st.sidebar.slider('purpose', 0,9,1)
grade = st.sidebar.slider('grade', 0.0,7.0,1.0)
emp_length=st.sidebar.slider('emp_length', 0, 11,1)
borrower_score=st.sidebar.slider('borrower_score',0.0,1.0,0.1)

data = {'loan': loan,
            'term': term,
            'annual_income': annual_income,
            'purpose': purpose,
	    'home_ownership':home_ownership,
            'grade':grade,
	    'emp_length':emp_length,
            'borrower_score':borrower_score}
features = pd.DataFrame(data,index=[0])
	 
st.subheader('User Input parameters')
st.write(features)

dataset = pd.read_csv('loan_data.csv.gz', compression='gzip',low_memory=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label_encoded_df = dataset.copy()
for col in label_encoded_df.select_dtypes(include='O').columns:
    label_encoded_df[col]=le.fit_transform(label_encoded_df[col])

X=label_encoded_df.loc[:,('loan_amnt','term','annual_inc','purpose','home_ownership','grade','emp_length','borrower_score')]
y=label_encoded_df['status']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
model = GradientBoostingClassifier(max_depth= 5, n_estimators= 180)
model.fit(X_train, y_train)

prediction_proba = model.predict_proba(features)
prediction = model.predict(features)

st.write("######")
st.write("######")
col1, col2= st.columns([1,1])
with col1:
  st.subheader('Evaluation Report')
with col2:
  if prediction==0:
   st.markdown('<p style="font-size:30px; color:red;">Not Approved </p>', unsafe_allow_html=True)
  elif prediction==2:
    st.markdown('<p style="font-size:30px; color:green;">Approved </p>', unsafe_allow_html=True)
  else:
    st.markdown('<p style="font-size:30px; color:red;">Default Risk </p>', unsafe_allow_html=True)
 

st.write("######")
st.write("######")

new_title = '<p color:Black; font-size: 35px;">Probabilty of Customer Loan Approval</p>'
st.markdown(new_title, unsafe_allow_html=True)

pred=prediction_proba.flatten()
labels=['Not Safe','Default','Safe Loan']

sns.set()
fig = plt.figure(figsize=(10,5))
sns.barplot(x=labels,y=pred*100)

#df = pd.DataFrame(prediction_proba, columns = labels)
#sns.barplot(data=df*100)

sns.set(rc={'figure.figsize':(10,5)}, font_scale=1.0)
sns.set_style({'axes.facecolor':'white', 'grid.color': '.8', 'font.family':'Times New Roman'})

# Add figure in streamlit app
st.pyplot(fig)
