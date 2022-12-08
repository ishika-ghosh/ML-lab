#!/usr/bin/env python
# coding: utf-8

# ### Use sklearn. datasets import load_iris use k-neighbour classifier to classify the three flowers to setosa, vesicolor and Virginica.

# In[1]:


from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = load_iris().data
Y = load_iris().target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
cf = KNeighborsClassifier(n_neighbors=5)
cf.fit(X_train, Y_train)
Y_pred = cf.predict(X_test)
print("Accuracy: ", (accuracy_score(Y_test, Y_pred) * 100), "%")
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))


# In[ ]:




