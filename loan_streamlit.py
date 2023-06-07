import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st 


titanic = sns.load_dataset("titanic")

fig = plt.figure(figsize=(10, 4))
sns.countplot(x="class", data=titanic)

st.pyplot(fig)



































