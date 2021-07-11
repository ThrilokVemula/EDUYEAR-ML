#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv("C:/Users/THRILOKNATH VEMULA/OneDrive/Desktop/coders eduyear/Machine_Learning/Minor_Project2/heart.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.mean()


# In[6]:


data.describe() #gives the statistical information of all columns


# In[7]:


sns.set(font_scale=1.5)
data.hist(edgecolor='black',linewidth=1.2,figsize=(20,20));


# In[8]:


sns.countplot(x='target',data=data,palette='cool')


# In[9]:


#to get the exact values we use value count
data['target'].value_counts()


# In[10]:


sns.countplot(x='sex',data=data)


# In[11]:


data['sex'].value_counts()


# In[12]:


sns.countplot(x='sex',hue='target',data=data,palette='autumn')


# In[13]:


data.groupby('sex')['target'].value_counts(normalize=True)


#  # Assignment-1: Check how many are older(>50) males and females are there and out of them how many had a heart attack?

# In[14]:


older=data[data['age']>50]
older['sex'].value_counts()


# In[15]:


older.groupby('sex')['target'].value_counts()


# In[16]:


older.groupby('sex')['target'].value_counts(normalize=True)


# # Assignment-2: What is the probability of getting a heart attackto a person greater than age 50 and had no cp ever and no fasting blood sugar?

# In[17]:


healthy=older[(older['cp']==0) & (older['fbs']==0)]
healthy['target'].value_counts()


# In[18]:


healthy['target'].value_counts(normalize=True)


# In[19]:


plt.figure(figsize=(12,5))
data.age.hist(bins=10)


# In[20]:


print(f"The most of the paients have a mean age of: {data.age.mean()}")


# In[21]:


plt.figure(figsize=(25,10))
sns.countplot(x='age',hue='target',data=data,palette='cool')


# In[22]:


#copying the whole dataset into another data frame
df=data.copy()
df.head()


# In[26]:


import plotly 
import plotly.express as px
import plotly.graph_objects as go
#plt.rcParams['figure.figsize']=17,8
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot


# In[27]:


#COUNT PLOT
def target_count():
    trace = go.Bar( x = df['target'].value_counts().values.tolist(),
                y=['Heart Disease','Healthy'],
                orientation='h',
                text=df['target'].value_counts().values.tolist(),
                textfont=dict(size=15),
                textposition='auto',
                opacity=1,marker=dict(
                color=['lightskyblue','gold'],
                line=dict(color='#000000',width=1.5)))
    layout=dict(title='count of target variable')
    fig=dict(data=[trace],layout=layout)
    iplot(fig)

#PERCENTAGE PLOT
def target_percent():
    trace=go.Pie(labels=['Heart Disease','Healthy'],
                values=df['target'].value_counts(),
                textfont=dict(size=15),opacity=0.8,
                marker=dict(colors=['lightskyblue','gold'],
                line=dict(color='#000000',width=1.5)))
    layout=dict(title='Distribution of target variable')
    fig=dict(data=[trace],layout=layout)
    iplot(fig)


# In[28]:


target_count()
target_percent()


# In[29]:


import plotly.figure_factory as ff
def plot_distribution(data_select,size_bin):
    #2 datasets
    tmp1=df[data_select]
    tmp2=df[data_select]
    hist_data=[tmp1,tmp2]
    group_labels=['Heart Disease','Healthy']
    colors=['gold','blue']
    fig = ff.create_distplot(hist_data, group_labels, colors = colors, show_hist = True, bin_size = size_bin, curve_type='kde')
    fig['layout'].update(title=data_select)
    iplot(fig,filename='Density plot')


# In[30]:


plot_distribution('chol',10)


# In[31]:


plot_distribution('thalach',0)


# In[32]:


categorical_values=['sex','cp','fbs','restecg','exang','slope','ca','thal','target']


# In[33]:


plt.figure(figsize=(30,30))
for i,column in enumerate(categorical_values,1):
    plt.subplot(3,3,i)
    sns.barplot(x=f"{column}",y='target',data=data)
    plt.ylabel('Possibilty to have heart disease')
    plt.xlabel(f'{column}')


# In[34]:


plt.figure(figsize=(25,10))
sns.heatmap(df.corr(),annot=True,cmap='rainbow')


# # MODEL BUILDING

# In[35]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[36]:


df.head()


# In[38]:


X=df.drop('target',axis=1)
y=df['target']


# In[39]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[41]:


#Model
DT=DecisionTreeClassifier()
#fitting
DT.fit(X_train,y_train)
#prediction
y_pred=DT.predict(X_test)

DT_accuracy=accuracy_score(y_test,y_pred)*100
#Accuracy
print("Accuracy:",DT_accuracy)


# In[45]:


from sklearn.linear_model import LogisticRegression
#Model
LR=LogisticRegression()
#fitting 
LR.fit(X_train,y_train)
#prediction
y_pred=LR.predict(X_test)

LR_accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",LR_accuracy*100)


# In[46]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=5, shuffle=True, random_state=20)


# In[62]:


cross_val_score_Decision = cross_val_score(DT, X, y, cv=k_fold, scoring="accuracy")
cross_val_score_Decision


# In[63]:


cross_val_score_Decision.mean()


# In[64]:


cross_val_score_logistic = cross_val_score(LR, X, y, cv=k_fold, scoring="accuracy")
cross_val_score_logistic


# In[65]:


cross_val_score_logistic.mean()

