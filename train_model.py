
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset from sklearn datasets

dataset=load_iris()

# Getting Feature Names

names=dataset.feature_names

# Loading features and labels from the dataset

features=dataset.data
labels=dataset.target

# Splitting labels and features to training and testing sets

feature_train,feature_test,label_train,label_test=train_test_split(features,labels,test_size=0.2,random_state=42)

# Initialising Logistic Regression Model with maximum iterations as 500

model=LogisticRegression(max_iter=500)

# Fitting Model or Training Model with training features and labels

model.fit(feature_train,label_train)

# Predicting the labels for the testing features

label_pred=model.predict(feature_test)

# Finding the accuracy score for predicted ones vs testing ones

from sklearn.metrics import accuracy_score
accuracy_score(label_pred,label_test)

# # Importing the pickle library

# import pickle

# # Dumping the model object to save it as model.pkl file

# pickle.dump(model,open('model2.pkl','wb+'))