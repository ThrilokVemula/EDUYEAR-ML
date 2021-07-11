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


# In[7]:


df=pd.read_csv('C:/Users/THRILOKNATH VEMULA/OneDrive/Desktop/coders eduyear/Machine_Learning/Minor_Project1/Salary_Data.csv')
df.head()


# In[8]:


df.tail()


# In[9]:


df.info()


# In[11]:


df.isnull().sum()


# In[12]:


df.plot(x='YearsExperience',y='Salary')


# In[13]:


df.plot(kind='scatter',x='YearsExperience',y='Salary')


# # From Scratch

# In[17]:


#declaring x and y
x=df.iloc[:,:-1].values #Except last column-YearsExperience
y=df.iloc[:,-1].values #Last column-Salary
print(x) #x Array
print(y) #y Array


# # w0=y_mean-(w1*x_mean)
# # w1=sigma(x-x_mean)*(y-y_mean)/sigma(x-x_mean)^2

# In[23]:


#Calculating mean_x and mean_y
x_mean=np.mean(x)
y_mean=np.mean(y)
print(mean_x,mean_y)


# In[26]:


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


# In[27]:


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


# In[ ]:




