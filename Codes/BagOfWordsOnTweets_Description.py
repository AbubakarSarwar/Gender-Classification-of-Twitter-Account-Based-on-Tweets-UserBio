import pandas as pd
import numpy as np
import itertools
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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import string
import matplotlib.pyplot as plt
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
# for quick and dirty counting
from collections import defaultdict

# the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
# function to split the data for cross-validation
from sklearn.model_selection import train_test_split
# function for transforming documents into counts
from sklearn.feature_extraction.text import CountVectorizer
# function for encoding categories
from sklearn.preprocessing import LabelEncoder

class_names=['Male','Female','Brand','Unknown']
d = defaultdict(LabelEncoder)
# matplotlib.get_backend()
data = pd.read_csv('GenderClassification.csv',encoding = "ISO-8859-1")
# data = pd.read_csv("data.csv",usecols= [0,5,19,17,21,10,11],encoding='latin1')
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def cleaning(s):
    s = str(s)
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    return s
data['Tweets'] = [cleaning(s) for s in data['text']]

data['Description'] = [cleaning(s) for s in data['description']]

stop = set(nltk.corpus.stopwords.words('english'))

print(stop)
print ("We also have unknowns")
print (data.gender.value_counts())
print(data.shape)

data_confident = data[data['gender:confidence']==1]
data_new = data_confident.filter(['Description'], axis=1)
data_new2 = data_confident.filter(['Tweets'], axis=1)
data_new['Description'] = data_new['Description'].str.lower().str.split()
data_new2['Tweets'] = data_new2['Tweets'].str.lower().str.split()

data_new['Description'] = data_new['Description'].apply(lambda x : [item for item in x if item not in stop])

data_new2['Tweets'] = data_new2['Tweets'].apply(lambda x : [item for item in x if item not in stop])
data_new['Concatenated'] = data_new2['Tweets'].astype(str) + data_new['Description'].astype(str)
#Male_Words.plot(kind='bar',stacked=True, colormap='plasma')
#Female_Words.plot(kind='bar',stacked=True, colormap='OrRd')
#Brand_words.plot(kind='bar',stacked=True, colormap='Paired')


print(data_new['Description'])
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(data_new['Concatenated'])
vocab = vectorizer.get_feature_names()
print (vocab)
#x = vectorizer.fit_transform(data_confident['Description'])
encoder = LabelEncoder()
y = encoder.fit_transform(data_confident['gender'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print('<---------- Results with K Nearest Neighbours ---------->')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)
prediction = neigh.predict(x_test)
print("Accuracy with (k=3) =" + str(accuracy_score(y_test,prediction)))
print
print('<------------------------------------------------------->\n')

print('<---------- Results with K Nearest Neighbours ---------->')
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(x_train, y_train)
prediction = neigh.predict(x_test)
print("Accuracy with (k=5) =" + str(accuracy_score(y_test,prediction)))
print('<------------------------------------------------------->\n')

print('<---------- Results with K Nearest Neighbours ---------->')
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(x_train, y_train)
prediction = neigh.predict(x_test)
print("Accuracy with (k=10) =" + str(accuracy_score(y_test,prediction)))
print('<------------------------------------------------------->\n')

print('<---------- Results with DecisionTree ---------->')
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
prediction = dt.predict(x_test)
print("Accuracy with Decision Tree=" + str(accuracy_score(y_test,prediction)))
print('<------------------------------------------------------->\n')

print('<---------- Results with RandomForest ---------->')
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
prediction = rf.predict(x_test)
print("Accuracy with Random Forest=" + str(accuracy_score(y_test,prediction)))
print('<------------------------------------------------------->\n')

print('<---------- Results with Multinomial NB ---------->')
nb = MultinomialNB()
nb.fit(x_train, y_train)
prediction = nb.predict(x_test)
print("Accuracy =" + str(nb.score(x_test, y_test)))
print('<------------------------------------------------------->\n')
cnf_matrix = confusion_matrix(y_test, prediction)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix of Multinomial NB')
print (data_confident.gender.value_counts())

plt.figure()