import pandas as pd
import numpy as np
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
import re
import nltk
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from collections import defaultdict

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

class_names=['acoustic guitar','trumpet','violin','electric guitar']
data = pd.read_csv('Instrument-4.csv',encoding = "ISO-8859-1")

print (data.instrument.value_counts())
print(data.shape)
