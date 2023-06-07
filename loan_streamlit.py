import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st 

dataset = pd.read_csv('loan_data.csv.gz', compression='gzip',low_memory=True)
y=dataset['Status'].count_values()
labels=['Not Safe','Default','Safe Loan']
fig = plt.figure(figsize=(15,10))
df = pd.DataFrame(prediction_proba, columns = labels)
sns.barplot(data=y,order = ['Safe Loan', 'Not Safe','Default'])
# Add figure in streamlit app
st.pyplot(fig)
































