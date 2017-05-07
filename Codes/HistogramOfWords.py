# -*- coding: utf-8 -*-
"""
Created on Sun April  30 20:16:19 2017

@author: Shahriyar
"""

import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords



#used for cleaning textual data, cleans all the special characters
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



print ("Histograms based on text")
df = pd.read_csv("GenderClassification.csv",encoding='ISO-8859-1')
df = df[df.gender != 'unknown']
#print(df)

df['Tweets'] = [cleaning(s) for s in df['text']]
df['Description'] = [cleaning(s) for s in df['description']]



#We get a set of English stop words using the line:
stop = set(stopwords.words('english'))

print(stop)


df['Tweets'] = df['Tweets'].str.lower().str.split()
# Gives us all the items which are not in the listed stop words
df['Tweets'] = df['Tweets'].apply(lambda x : [item for item in x if item not in stop])





Male = df[df['gender'] == 'male']
MaleWords = pd.Series(' '.join(Male['Tweets'].astype(str)).lower().split(" ")).value_counts()[:20]

Female = df[df['gender'] == 'female']
FemaleWords = pd.Series(' '.join(Female['Tweets'].astype(str)).lower().split(" ")).value_counts()[:20]

Brand = df[df['gender'] == 'brand']
BrandWords = pd.Series(' '.join(Brand['Tweets'].astype(str)).lower().split(" ")).value_counts()[:10]


#BrandWords.plot(stacked=True)
#FemaleWords.plot(color='red',stacked=True)
#MaleWords.plot(color='green',stacked=True)
print(stop)
MaleWords.plot(kind='bar',stacked=True, colormap='Paired')

#BrandWords.plot(kind='bar',stacked=True, colormap='Paired')


#FemaleWords.plot(kind='bar',stacked=True, colormap='Paired')