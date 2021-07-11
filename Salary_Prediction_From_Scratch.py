#!/usr/bin/env python
# coding: utf-8

# # Data Collection
# # Data Exploration
# # Data Cleaning
# # Data Binning
# # Data Visualization
# # One hot Encoding
# # Feature Engineering
# # Model Building

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('C:/Users/THRILOKNATH VEMULA/OneDrive/Desktop/coders eduyear/Machine_Learning/Minor_Project1/Salary_Data.csv')
df.head()


# In[3]:


df.tail()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.plot(x='YearsExperience',y='Salary')


# In[7]:


df.plot(kind='scatter',x='YearsExperience',y='Salary')


# # From Scratch

# In[8]:


#declaring x and y
x=df.iloc[:,:-1].values #Except last column-YearsExperience
y=df.iloc[:,-1].values #Last column-Salary
print(x) #x Array
print(y) #y Array


# # w0=y_mean-(w1*x_mean)
# # w1=sigma(x-x_mean)*(y-y_mean)/sigma(x-x_mean)^2

# In[9]:


#Calculating mean_x and mean_y
x_mean=np.mean(x)
y_mean=np.mean(y)
print(x_mean,y_mean)


# In[10]:


#Total length of dataset
n=len(x)

#Calculating w0 and w1
numerator=0
denominator=0
for i in range(n):
    numerator+=(x[i]-x_mean)*(y[i]-y_mean)
    denominator+=(x[i]-x_mean)**2
    
w1=numerator/denominator
w0=y_mean-(w1*x_mean)

print("The coefficients are",w0,w1)


# In[11]:


plt.scatter(x,y,color='r')

#Calculation of y_pred values
y_pred=w0+(w1*x)

#plotting the regression line
plt.plot(x,y_pred,color='b')

#putting the labels
plt.xlabel('Years of Experience in a company')
plt.ylabel('Salary of the Employee')
plt.title('Salary vs Years of Experience')

#Show the plot
plt.show()


# In[12]:


y


# In[13]:


y_pred


# In[14]:


from sklearn.metrics import r2_score
r2_score(y,y_pred)*100


# # By using libraries

# In[15]:


from sklearn.linear_model import LinearRegression


# In[16]:


model=LinearRegression()


# In[17]:


model.fit(x,y)


# In[18]:


y_pred_1=model.predict(x)


# In[19]:


y


# In[20]:


y_pred_1


# In[21]:


plt.scatter(x,y,color='r')
#regression line
plt.plot(x,y_pred_1,color='b')
#labels
plt.xlabel('Years of Experience in a company')
plt.ylabel('Salary of the Employee')
plt.title('Salary vs Years of Experience')

plt.show()


# In[22]:


from sklearn.metrics import r2_score
r2_score(y,y_pred_1)*100


# In[23]:


x=[[12]]
model.predict(x)


# In[24]:


w0+w1*x


# In[ ]:




