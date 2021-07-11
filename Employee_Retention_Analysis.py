#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
#plt.rcParams['figure.figsize']=17,8
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot


# In[2]:


df=pd.read_csv('C:/Users/THRILOKNATH VEMULA/OneDrive/Desktop/coders eduyear/Machine_Learning/Major_Project2/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.isnull().sum()


# In[8]:


sns.countplot(df.Attrition)
plt.title("Attrition",fontsize=15)


# # Gender Analysis

# In[9]:


df['Gender'].value_counts()


# In[10]:


df['Gender'].value_counts(normalize=True)


# In[11]:


df['Age'].mean()


# # Gender Biased or not,if yes then by how much

# In[12]:


print(df.Gender.value_counts())
total=df.Gender.value_counts()[0]+df.Gender.value_counts()[1]
men=(df.Gender.value_counts()[0]/total)*100
women=(df.Gender.value_counts()[1]/total)*100
plt.figure(figsize=(7,7))
plt.pie([men,women],labels=['Men','Women'],autopct='%1.01f%%')
plt.title('Gender Wise Data Biasness',fontsize=20)
plt.show()


# In[14]:


sns.countplot(x='Gender',data=df)


# In[15]:


#age groups based on gender
df.groupby('Gender')['Age'].mean()


# In[16]:


df.groupby('Gender')['Attrition'].value_counts(normalize=True)


# In[17]:


sns.countplot(x='Gender',hue='Attrition',data=df)
plt.show()


# In[19]:


plt.figure(figsize=(8,8))
ax=sns.countplot(x='Gender',hue='Attrition',data=df)
ax.set_xticklabels(('Women','Men'))
plt.title('Gender wise Attrition Rate',fontsize=20)
plt.show()


# In[20]:


df.columns


# # Work life balance

# In[21]:


df['WorkLifeBalance'].value_counts()


# In[23]:


df.groupby('Gender')['WorkLifeBalance'].value_counts()


# In[24]:


sns.countplot(x='Gender',hue='WorkLifeBalance',data=df)


# # Work life balance of female is less than male

# In[25]:


plt.figure(figsize=(10,5))
ax=sns.barplot(x=df.Gender,y=df.WorkLifeBalance,estimator=np.sum,hue=df.Attrition)
ax.set_xticklabels(('Women','Men'))
plt.title('How Work Life Balance affects Attrition Rate')
plt.show()


# # Marriage and Attrition rate

# In[26]:


sns.countplot(x='MaritalStatus',hue='Attrition',data=df)


# In[27]:


df.groupby('Gender')['MaritalStatus'].value_counts()


# In[28]:


df.groupby(['Gender','MaritalStatus'])['Attrition'].value_counts(normalize=True)


# In[29]:


plt.figure(figsize=(8,5))
ax=sns.barplot(x=df.Gender,y=df.WorkLifeBalance,estimator=np.sum,hue=df.MaritalStatus)
ax.set_xticklabels(('Women','Men'))
plt.show()


# In[30]:


df['BusinessTravel'].value_counts()


# In[31]:


df.groupby('Gender')['BusinessTravel'].value_counts()


# In[32]:


plt.figure(figsize=(8,5))
ax=sns.countplot(x=df.Gender,hue=df.BusinessTravel,data=df)
ax.set_xticklabels(('Women','Men'))
plt.title('Business Travel Women vs Men')
plt.show()


# In[34]:


plt.figure(figsize=(8, 5))
sns.countplot(x = df.BusinessTravel , hue = df.Attrition,data=df)
plt.title('How Business Travels Affects the Attrition')
plt.show()


# # Job Satisfaction

# In[35]:


df['JobSatisfaction'].value_counts()


# In[36]:


df.groupby('Gender')['JobSatisfaction'].value_counts(normalize=True)


# In[37]:


sns.countplot(x='Gender',hue='JobSatisfaction',data=df)


# In[38]:


sns.countplot(x='JobSatisfaction',hue='Attrition',data=df)


# In[39]:


plt.figure(figsize=(8, 5))
ax = sns.barplot(x = df.Gender , y = df.JobSatisfaction, estimator = np.sum, hue = df.Attrition)
ax.set_xticklabels(('Women', 'Men'))
plt.show()


# In[40]:


plt.figure(figsize=(10, 6))
plt.title('How Distance from Home affects Attrition Rate')
ax = sns.barplot(x = df.BusinessTravel , y = df.DistanceFromHome, estimator = np.median, hue = df.Attrition, palette='Set1')
ax.set_xticklabels(('Non-Travel', 'Travel_Rarely', 'Travel_Frequently'))
plt.show()


# # Job satisfaction among different departments

# In[41]:


df['Department'].value_counts()


# In[42]:


df['JobSatisfaction'].value_counts()


# In[43]:


df.groupby('Department')['JobSatisfaction'].value_counts()


# In[44]:


fig = px.histogram(
    df, 
    "DailyRate", 
    nbins=80, 
    title ='DailyRate', 
    width=800,
    height=500
)

fig.show()


# # Salary Hike

# In[45]:


fig=px.pie(df,names="PercentSalaryHike",title="Percent Salary Hike")
fig.show()


# In[46]:


fig = px.bar(df, x="MonthlyIncome", y="Attrition",
              barmode='group',
             height=600)
fig.show()


# In[47]:


fig = px.bar(df, x="JobRole", y="Attrition",
              barmode='group',
             height=600)
fig.show()


# In[48]:


fig = px.bar(df, x="YearsSinceLastPromotion", y="Attrition",
              barmode='group',
             height=600)
fig.show()


# # Model Building

# In[49]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[50]:


df['StandardHours']


# In[53]:


plt.figure(figsize=(25,10))
sns.heatmap(df.corr(),annot=True)


# In[54]:


df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours','EmployeeNumber','Over18','StandardHours','EmployeeCount'], axis="columns", inplace=True)


# In[64]:


df.Attrition.replace({'Yes': 1, 'No': 0}, inplace=True)

df.BusinessTravel.replace({'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}, inplace=True)

df.Department.replace({'Sales': 0, 'Research & Development': 1, 'Human Resources': 2}, inplace=True)

df.Gender.replace({'Female': 0, 'Male': 1}, inplace=True)

df.MaritalStatus.replace({'Single': 0,'Married': 1, 'Divorced': 2}, inplace=True)

df.OverTime.replace({'No': 0, 'Yes': 1}, inplace=True)

df.EducationField.replace({'Life Sciences': 0, 'Medical': 1, 'Marketing': 2, 'Technical Degree': 3, 'Human Resources': 4, 'Other': 5}, inplace=True)

df.JobRole.replace({
'Sales Executive': 0, 'Research Scientist': 1, 'Laboratory Technician': 2,'Manufacturing Director': 3,'Healthcare Representative': 4,'Manager': 5,
    'Sales Representative': 6,'Research Director': 7,'Human Resources': 8
}, inplace=True)


# In[72]:


X=df.drop(columns=["Attrition"])
y=df["Attrition"]


# In[73]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44,shuffle =True) 


# In[86]:


DT=DecisionTreeClassifier()
DT.fit(X_train,y_train)
y_pred=DT.predict(X_test)
DT_accuracy=accuracy_score(y_test,y_pred)*100
print("Accuracy",DT_accuracy)


# In[87]:


from sklearn.linear_model import LogisticRegression
#Model
LR = LogisticRegression()

#fiting the model
LR.fit(X_train, y_train)

#prediction
y_pred = LR.predict(X_test)

LR_accuracy=accuracy_score(y_test,y_pred)

#Accuracy
print("Accuracy ", LR_accuracy*100)

#Plot the confusion matrix
sns.set(font_scale=1.5)
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True, cmap='PuBu')
plt.show()


# In[ ]:




