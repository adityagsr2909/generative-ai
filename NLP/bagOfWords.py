import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

corpus = """
He is a Good boy.
She is a Good girl.
Boy and Girl are good
"""
stopWordList = stopwords.words('english')
stopWordList.remove('not')

sentList = nltk.sent_tokenize(corpus.lower())
modSentList = []
LemmatizerObj = WordNetLemmatizer()
for sent in sentList:
    localList = []
    for word in nltk.word_tokenize(sent):
        if word not in localList and word not in stopWordList:
            localList.append(word.replace(".",""))
    modSentList.append(" ".join(localList))
    print(modSentList)
 

cv = CountVectorizer(binary=True)
x =  cv.fit_transform(modSentList)
print(x)
print(cv.vocabulary_)
print(x[0])