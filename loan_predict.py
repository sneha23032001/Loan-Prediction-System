#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[2]:


df=pd.read_csv("C:\Users\hp\Downloads")
df


# In[3]:


df.isnull().sum()


# In[4]:


df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)


# In[5]:


df.isnull().sum()


# In[6]:


df['Married'].fillna(df['Married'].mode()[0],inplace=True)


# In[7]:


df.isnull().sum()


# In[8]:


df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)


# In[9]:


df.isnull().sum()


# In[10]:


df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)


# In[11]:


df.isnull().sum()


# In[12]:


df.LoanAmount=df.LoanAmount.fillna(df.LoanAmount.mean())


# In[13]:


df.isnull().sum()


# In[16]:


df.Loan_Amount_Term=df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mean())


# In[17]:


df.isnull().sum()


# In[20]:


df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)


# In[21]:


df.isnull().sum()


# In[22]:


#label Encoding
df.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)


# In[23]:


df.head(4)


# In[24]:


df['Dependents'].value_counts()


# In[25]:


#replacing value of 3+ to 4
df=df.replace(to_replace='3+',value=4)


# In[26]:


df


# In[29]:


df.replace({"Married":{'No':0,'Yes':1}},inplace=True)


# In[30]:


df


# In[31]:


df.replace({"Gender":{'Male':1,'Female':0}},inplace=True)


# In[32]:


df


# In[33]:


df.replace({"Self_Employed":{'No':0,'Yes':1}},inplace=True)


# In[34]:


df.replace({"Property_Area":{'Rural':0,'Semiurban':1,'Urban':2}},inplace=True)


# In[35]:


df


# In[36]:


df.replace({"Education":{'Graduate':1,'Not Graduate':0}},inplace=True) 


# In[38]:


#seperating the data & label
X=df.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y=df['Loan_Status']


# In[39]:


#train test split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)


# In[40]:


#training the model SVM
classifier=svm.SVC(kernel='linear')


# In[41]:


classifier.fit(X_train,Y_train)


# In[43]:


#model evaluation
X_train_predict=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_predict,Y_train)


# In[44]:


print(training_data_accuracy)


# In[45]:


X_test_predict=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_predict,Y_test)


# In[46]:


print(test_data_accuracy)


# In[ ]:




