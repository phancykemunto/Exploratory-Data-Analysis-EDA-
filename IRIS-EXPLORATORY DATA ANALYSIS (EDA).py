#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')


# # Loading Dataset

# In[2]:


data = pd.read_csv('C:/PROJECT/Iris.csv')
print(data)


# # Exploratory Data Analysis(EDA)

# In[3]:


data.head()


# In[4]:


# shape of the data
data.shape


# In[5]:


#data information 
data.info()


# In[6]:


# describing the data
data.describe()


# In[7]:


#column to list 
data.columns.tolist()


# In[8]:


# check for missing values:
data.isnull().sum()


# In[9]:


#checking duplicate values 
data.nunique()


# In[10]:


duplicate_rows_data = data[data.duplicated()]
print("number of duplicate rows: ", duplicate_rows_data.shape)


# In[11]:


data = data.drop_duplicates()
data.head(5)


# In[12]:


data.count() 


# In[13]:


#Check the count for each category in the "gender" column
data["Species"].value_counts()


# In[14]:


data.Species.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("PetalWidthCm by Species")
plt.ylabel('PetalWidthCm')
plt.xlabel('Species');


# In[15]:


data.Species.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("PetalLengthCm by Species")
plt.ylabel('PetalLengthCm')
plt.xlabel('Species');


# # Splitting the dataset

# In[16]:


X=data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
X
Y = data['Species']
print(Y)


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(X,Y,test_size = 0.25, random_state= 0)


# # Scatter Plot

# In[18]:


fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(data['SepalWidthCm'], data['SepalLengthCm'])
ax.set_xlabel('SepalWidthCm')
ax.set_ylabel('SepalLengthCm')
plt.show()


# # Box plot for each attribute

# In[19]:


sns.boxplot(x=data['SepalLengthCm'])


# In[20]:


sns.boxplot(x=data['SepalWidthCm'])


# In[21]:


sns.boxplot(x=data['PetalLengthCm'])


# In[22]:


sns.boxplot(x=data['PetalWidthCm'])


# In[23]:


for col in data.columns:
    if data[col].dtypes != "object":
        sns.boxplot(data['Species'],data[col])     #Hence the features with Species has linear realtionship
        plt.show()


# # Subplot

# In[24]:


# Set Seaborn style
sns.set_style("darkgrid")
 
# Identify numerical columns
numerical_columns = data.select_dtypes(include=["int64", "float64"]).columns
 
# Plot distribution of each numerical feature
plt.figure(figsize=(14, len(numerical_columns) * 3))
for idx, feature in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, idx)
    sns.histplot(data[feature], kde=True)
    plt.title(f"{feature} | Skewness: {round(data[feature].skew(), 2)}")
 
# Adjust layout and show plots
plt.tight_layout()
plt.show()


# # Evaluating the model using a correlation matrix

# In[25]:


plt.figure(figsize=(10,5))
c= data.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c


# # Pairplot

# In[26]:


# Set the color palette
sns.set_palette("Pastel1")
 
# Assuming 'data' is your DataFrame
plt.figure(figsize=(10, 6))
 
# Using Seaborn to create a pair plot with the specified color palette
sns.pairplot(data)
 
plt.suptitle('Pair Plot for DataFrame')
plt.show()


# In[27]:


#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()  
X_train= st_x.fit_transform(X_train)    
X_test= st_x.transform(X_test)    


# # Building Decision Tree Model

# In[28]:


# Create Decision Tree classifer object
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,Y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# # Evaluating the Model

# In[29]:


# Model Accuracy, how often is the classifier correct?
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))


# In[76]:


#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(Y_test, y_pred) 
print(cm)


# # Visualizing Decision Trees

# In[79]:


#conda install python-graphviz
#pip install graphviz

#pip install pydotplus


# In[30]:


clf_tree = DecisionTreeClassifier( max_depth = 4, max_features=2)
clf_tree.fit(X_train, Y_train)
from sklearn import tree
fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (15,10), dpi=300)
tree.plot_tree(clf_tree,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')

