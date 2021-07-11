#!/usr/bin/env python
# coding: utf-8

# # Titanic Data Analysis

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


df=pd.read_csv('C:/Users/THRILOKNATH VEMULA/OneDrive/Desktop/coders eduyear/Machine_Learning/Major_Project1/train.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[12]:


sns.heatmap(df.isnull(),cmap='viridis')


# # How many people survived and how many were dead?

# In[13]:


df['Survived'].value_counts()


# In[14]:


sns.countplot(x='Survived',data=df)


# In[15]:


#How many males and females are there on the ship
df['Sex'].value_counts()


# # How many males and females survived?

# In[16]:


df.groupby('Sex')['Survived'].value_counts()


# In[17]:


#first parameter in groupby function becomes first column in output
df.groupby('Survived')['Sex'].value_counts()


# In[18]:


#Hue parameter-This parameter take column name for colour encoding
sns.countplot(x='Sex',hue='Survived',data=df)


# # Survival rate of males

# In[19]:


#survival rate=No.of males survived/Total no. of males
df.groupby('Sex')['Survived'].value_counts()


# In[20]:


df['Sex'].value_counts()


# In[22]:


df.groupby('Sex')['Survived'].value_counts()[3]/df['Sex'].value_counts()[0]*100


# In[23]:


109/577*100


# # Survival rate of females

# In[25]:


#By applying the same formula for females we can get survival rate of females
df.groupby('Sex')['Survived'].value_counts()[0]/df['Sex'].value_counts()[1]*100


# In[26]:


233/314*100


# # So,by this analysis we know that survival rate of females was way more higher than males

# #    

# # How many were travelling alone?

# In[27]:


alone=df[(df['SibSp']==0) & (df['Parch']==0)]
not_alone=df[(df['SibSp']!=0) | (df['Parch']!=0)]


# In[28]:


alone.head()


# In[29]:


not_alone.head()


# In[31]:


alone.shape


# In[32]:


not_alone.shape


# In[33]:


#To know how many were travelling alone,we count the parameters
alone.shape[0]


# # Survival rate of people travelling alone

# In[34]:


alone['Survived'].value_counts()


# # 1.Whose survival rate is more?(people travelling alone or people who are not alone)

# In[35]:


alone['Survived'].value_counts(normalize=True)


# In[36]:


not_alone['Survived'].value_counts(normalize=True)


# In[37]:


sns.countplot(x='Survived',data=alone)


# In[38]:


sns.countplot(x='Survived',data=not_alone)


# # From above analysis,survival rate of people travelling alone is less than people travelling with someone

# # Whose survival rate is more?(People travelling with parents only(Parch) or people travelling with siblings only(SibSp))

# In[40]:


df[(df['SibSp']==0) & (df['Parch']!=0)]['Survived'].value_counts() #travelling with parents and children only


# In[41]:


df[(df['SibSp']==0) & (df['Parch']!=0)]['Survived'].value_counts(normalize=True)


# In[42]:


df[(df['SibSp']!=0) & (df['Parch']==0)]['Survived'].value_counts()#travelling only with siblings


# In[43]:


df[(df['SibSp']!=0) & (df['Parch']==0)]['Survived'].value_counts(normalize=True)


# # People travelling with parents and children(Parch) have more survival rate than travelling with Siblings

# # How many people were survived from different Pclass based on gender

# In[44]:


df.groupby(['Pclass','Sex'])['Survived'].value_counts()


# In[45]:


sns.barplot(x='Sex',y='Pclass',hue='Survived',data=df)


# In[46]:


plt.figure(figsize=(15,5))
df.groupby(['Pclass','Sex'])['Survived'].value_counts().plot(kind='bar')


# In[47]:


df.head()


# In[48]:


df.groupby('Pclass')['Fare'].mean()


# # Which passenger class contains more fare on the basis of gender?

# In[49]:


df.groupby(['Pclass','Sex'])['Fare'].mean()


# In[50]:


df.groupby(['Pclass','Sex'])['Fare'].mean().plot(kind='bar')


# In[51]:


sns.barplot(x='Sex',y='Fare',hue='Pclass',palette='CMRmap',data=df)


# # DATA CLEANING

# In[52]:


df.isnull().sum()


# In[56]:


#Age,Cabin,Embarked columns have null values
df.head()


# In[54]:


#Dropping the cabin column
df.drop(['Cabin'],axis=1,inplace=True)


# In[55]:


df.head()


# In[57]:


df[df['Embarked'].isnull()]


# In[58]:


#Only 2 null values in embarked column
df['Embarked'].value_counts()


# In[59]:


df['Embarked'].fillna('S',inplace=True)


# In[60]:


df[df['Embarked'].isnull()]


# # Null values in Embarked Column are filled with S

# In[61]:


df.isnull().sum()


# In[62]:


df


# In[63]:


df['Age'].mean()


# In[64]:


df['Age'].median()


# In[65]:


df.groupby('Pclass')['Age'].mean()


# In[66]:


df.groupby('Pclass')['Age'].median()


# In[67]:


#Filling the null values in Age column with mean values
def input_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 38.23
        elif Pclass==2:
            return 29.87
        elif Pclass==3:
            return 25.14
       
    else:
        return Age
    
#Another method-df[(df['Pclass']==1) & (df.isnull())].fillna(38.23,inplace=True)


# In[69]:


df['Age']=df[['Age','Pclass']].apply(input_age,axis=1) #calling the above function


# In[70]:


df.isnull().sum()


# # ONE HOT ENCODING

# # Converting Categorical into Numerical ones

# In[71]:


df.head()


# In[72]:


#Using replacing method
df['Sex'].replace(to_replace=['male','female'],value=[0,1],inplace=True)


# In[73]:


df.head()


# In[74]:


#label encoder
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df['Embarked']=lb.fit_transform(df['Embarked'])


# In[76]:


df.head() #Embarked column changes to 0,1,2 for C,Q,S respectively


# # FEATURE ENGINEERING

# In[77]:


df.corr()


# In[78]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,cmap='rainbow')


# In[79]:


df.groupby('Embarked')['Survived'].value_counts()


# In[80]:


df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)


# In[81]:


df.head()


# # MODEL BUILDING

# In[87]:


X=df.drop('Survived',axis=1)
y=df['Survived']


# In[88]:


X.head()


# In[84]:


y


# In[85]:


from sklearn.model_selection import train_test_split


# In[89]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=20)


# In[91]:


from sklearn.linear_model import LogisticRegression


# In[92]:


model=LogisticRegression()


# In[96]:


model.fit(X_train,y_train)


# In[94]:


y_pred=model.predict(X_test)


# In[95]:


y_pred


# In[97]:


from sklearn.metrics import accuracy_score


# In[98]:


accuracy_score(y_test,y_pred)*100


# In[ ]:




