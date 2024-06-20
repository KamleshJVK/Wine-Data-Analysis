#!/usr/bin/env python
# coding: utf-8

# ##### Importing all the libraries required for the project.

# In[1]:


import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ##### Reading the data from .data file and adding the column names to the dataset.

# In[2]:


df = pd.read_csv("wine.data",names=["Class","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"])


# #####   EDA - Exploratory Data Analysis

# In[3]:


df.head()


# In[4]:


for col in df.columns:
    print(col)


# In[5]:


df.shape


# ##### Checking the data types

# In[6]:


df.info()


# ##### from the above cell we know that all values are numerical in nature with feature "Class" being discrete and all other features being continious.

# In[7]:


df.isnull().sum()


# ##### As we can see that there is no null values present in our data set

# In[8]:


df.describe()


# ##### From the analysis so far we can conclude that feature "Class" is our dependent variable while all others are independent variables . 
# ##### The code below gives us a count of how many values a particular class has.

# In[9]:


df["Class"].value_counts()


# In[10]:


plt.figure(figsize = (5,5))
plt.bar(list(df["Class"].value_counts().keys()),list(df["Class"].value_counts()),color="g")


# ##### Outlier Analysis

# In[11]:


plt.figure(figsize=(30,50))
plt.subplot(7,7,1)
sns.boxplot(df['Alcohol'])
plt.title("Alcohol")
plt.subplot(7,7,2)
sns.boxplot(df['Malic acid'])
plt.title("Malic acid")

plt.subplot(7,7,3)
sns.boxplot(df['Ash'])
plt.title("Ash")

plt.subplot(7,7,4)
sns.boxplot(df['Alcalinity of ash'])
plt.title("Alcalinity of ash")

plt.subplot(7,7,5)
sns.boxplot(df['Magnesium'])
plt.title("Magnesium")

plt.subplot(7,7,6)
sns.boxplot(df['Total phenols'])
plt.title("Total phenols")

plt.subplot(7,7,7)
sns.boxplot(df['Flavanoids'])
plt.title("Flavanoids")

plt.subplot(7,7,8)
sns.boxplot(df['Nonflavanoid phenols'])
plt.title("Nonflavanoid phenols")


plt.subplot(7,7,9)
sns.boxplot(df['Proanthocyanins'])
plt.title("Proanthocyanins")
plt.subplot(7,7,10)
sns.boxplot(df['Color intensity'])
plt.title("Color intensity")

plt.subplot(7,7,11)
sns.boxplot(df['Hue'])
plt.title("Hue")

plt.subplot(7,7,12)
sns.boxplot(df['OD280/OD315 of diluted wines'])
plt.title("OD280/OD315 of diluted wines")

plt.subplot(7,7,13)
sns.boxplot(df['Proline'])
plt.title("Proline")

plt.subplot(7,7,14)
sns.boxplot(df['Class'])
plt.title("Class")




# ##### From the above graph we can understand that features " Alcohol , Total phenols , Flavanoids , Nonflavanoid phenols , OD280/OD315 of diluted wines  and Proline do not contain any outliers.

# In[12]:


df['Malic acid']=df['Malic acid'].clip(lower=df['Malic acid'].quantile(0.05),upper=df['Malic acid'].quantile(0.95))
df['Ash']=df['Ash'].clip(lower=df['Ash'].quantile(0.05), upper=df['Ash'].quantile(0.95))
df['Alcalinity of ash']=df['Alcalinity of ash'].clip(lower=df['Alcalinity of ash'].quantile(0.05), upper=df['Alcalinity of ash'].quantile(0.95))
df['Magnesium']=df['Magnesium'].clip(lower=df['Magnesium'].quantile(0.05), upper=df['Magnesium'].quantile(0.95))
df['Proanthocyanins']=df['Proanthocyanins'].clip(lower=df['Proanthocyanins'].quantile(0.05), upper=df['Proanthocyanins'].quantile(0.95))
df['Color intensity']=df['Color intensity'].clip(lower=df['Color intensity'].quantile(0.05), upper=df['Color intensity'].quantile(0.95))
df['Hue']=df['Hue'].clip(lower=df['Hue'].quantile(0.05), upper=df['Hue'].quantile(0.95))


# ##### Using the clip method we try to fix the outliers

# In[13]:


plt.figure(figsize=(30,50))
plt.subplot(7,7,1)
sns.boxplot(df['Alcohol'])
plt.title("Alcohol")
plt.subplot(7,7,2)
sns.boxplot(df['Malic acid'])
plt.title("Malic acid")

plt.subplot(7,7,3)
sns.boxplot(df['Ash'])
plt.title("Ash")

plt.subplot(7,7,4)
sns.boxplot(df['Alcalinity of ash'])
plt.title("Alcalinity of ash")

plt.subplot(7,7,5)
sns.boxplot(df['Magnesium'])
plt.title("Magnesium")

plt.subplot(7,7,6)
sns.boxplot(df['Total phenols'])
plt.title("Total phenols")

plt.subplot(7,7,7)
sns.boxplot(df['Flavanoids'])
plt.title("Flavanoids")

plt.subplot(7,7,8)
sns.boxplot(df['Nonflavanoid phenols'])
plt.title("Nonflavanoid phenols")


plt.subplot(7,7,9)
sns.boxplot(df['Proanthocyanins'])
plt.title("Proanthocyanins")
plt.subplot(7,7,10)
sns.boxplot(df['Color intensity'])
plt.title("Color intensity")

plt.subplot(7,7,11)
sns.boxplot(df['Hue'])
plt.title("Hue")

plt.subplot(7,7,12)
sns.boxplot(df['OD280/OD315 of diluted wines'])
plt.title("OD280/OD315 of diluted wines")

plt.subplot(7,7,13)
sns.boxplot(df['Proline'])
plt.title("Proline")

plt.subplot(7,7,14)
sns.boxplot(df['Class'])
plt.title("Class")




# ##### From the above box plots we can see that there are no more outliers present in our data set.

# In[14]:


df.corr()["Class"]


# In[15]:


plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(),annot=True,square=True,cmap=sns.diverging_palette(220, 10, as_cmap=True))


# In[16]:


X = df.drop('Class', axis=1)
y = df['Class']


# In[17]:


X.head()


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


model = GaussianNB()


# In[20]:


model.fit(X_train, y_train)


# In[21]:


y_pred = model.predict(X_test)


# In[22]:


accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


# In[23]:


print(f'Accuracy: {accuracy}')
print('Confusion Matrix:\n', confusion_mat)
print('Classification Report:\n', classification_rep)


# In[24]:


model = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)


# In[25]:


y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


# In[26]:


print(f'Accuracy: {accuracy}')
print('Confusion Matrix:\n', confusion_mat)
print('Classification Report:\n', classification_rep)


# In[ ]:




