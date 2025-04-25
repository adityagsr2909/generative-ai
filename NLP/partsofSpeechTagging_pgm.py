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
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

corpus = "Tajmahal is a beautiful monument"

words = nltk.word_tokenize(corpus)
pos_tag = nltk.pos_tag(words)
print(pos_tag)
