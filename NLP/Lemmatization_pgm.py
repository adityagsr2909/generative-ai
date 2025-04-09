import pandas as pd
import numpy as np
import nltk

from nltk.corpus import stopwords
#- Tokenization
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize, regexp_tokenize
#- Stemming
from nltk.stem import PorterStemmer, RegexpStemmer, SnowballStemmer
#- Lemmatization
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')

words = ["eating", "eats", "eaten", "writing", "writes", 
         "programming", "programs", "history", "finally","finalize"]

snowball_stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
for word in words:
    print(f"{word} :: {snowball_stemmer.stem(word)} :: {lemmatizer.lemmatize(word)} :: {lemmatizer.lemmatize(word, pos='v')} ")

"""
pos : 'n' - noun  'v' - verb  'a' - adjective  'r' - adverb
"""