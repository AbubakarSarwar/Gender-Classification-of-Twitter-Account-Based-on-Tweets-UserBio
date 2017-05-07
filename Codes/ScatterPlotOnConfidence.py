# -*- coding: utf-8 -*-
#libraries
import numpy as np # linear algebra
# we'll want this for plotting
import matplotlib.pyplot as plt
# we'll want this for text manipulation
import re
# for quick and dirty counting
from collections import defaultdict
# the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
# function to split the data for cross-validation
from sklearn.cross_validation import train_test_split
# function for transforming documents into counts
from sklearn.feature_extraction.text import CountVectorizer
# function for encoding categories
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

print("Scatterplot based on range")

df = pd.read_csv("GenderClassification.csv",encoding='ISO-8859-1')
df.gender = df.gender.map({'male':0,'female':1,'brand':2,'unknown':3})
print(df["gender"])

x = np.array([0,1,2])
y = np.array([0,0.2,0.4])
Cluster = np.array([0, 1, 1, 1, 3, 2, 2, 3, 0, 2])

plt.scatter(df["gender"],df["gender:confidence"])
plt.show()

print(x)