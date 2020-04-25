#!/usr/bin/env python
# coding: utf-8

# ### Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ### Importing the dataset

# In[2]:


dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[3]:


dataset.head()


# ### Feature Scaling

# In[4]:


#before scaling
print(X)


# In[5]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[6]:


#after scaling
print(X)


# ### Splitting the dataset into the Training set and Test set

# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[8]:


print(X_train)


# In[9]:


print(X_test)


# ### Applying LDA

# In[10]:


#get the variances of all the features

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = None)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
explained_variance = lda.explained_variance_ratio_
print(explained_variance)


# In[11]:


#plotting 2D relation here thus choosing top 2 lda

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
explained_variance = lda.explained_variance_ratio_
print(explained_variance)


# In[12]:


#2 features in X_train
print(X_train)


# In[13]:


#2 features in X_test

print(X_test)


# ### Training the Logistic Regression model on the Training set

# In[14]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# ### Predicting the Test set results

# In[15]:


y_pred = classifier.predict(X_test)


# ### Making the Confusion Matrix

# In[16]:


#confusion matrix for test data
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# ### Accuracy

# In[17]:


from sklearn.metrics import accuracy_score
print('Accuracy of logistic regression classifier on test set: {:.2f}%'.format(accuracy_score(y_test, y_pred)*100))


# ### Visualising the Training set results

# In[18]:


from matplotlib.colors import ListedColormap

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize']=12,8

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()


# ### Visualising the Test set results

# In[19]:


from matplotlib.colors import ListedColormap

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize']=12,8

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('LD')
plt.ylabel('LD2')
plt.legend()
plt.show()

