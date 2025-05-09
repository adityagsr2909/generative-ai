import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re

#- Tokenization
from nltk.tokenize import word_tokenize, sent_tokenize
#- Stemming
from nltk.stem import PorterStemmer, RegexpStemmer, SnowballStemmer
#- Lemmatization
from nltk.stem import WordNetLemmatizer
#- Bag Of words
from sklearn.feature_extraction.text import CountVectorizer
#- TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


corpus_df = pd.read_csv("C:/Users/SouryaDevansh/Downloads/GenAI/NLP/SMSSpamClassification.csv", names=['label', 'message'])
print(corpus_df.head(2))

stopWordList = stopwords.words('english')
stopWordList.remove('not')

vocabulary = []
lemmatizer = WordNetLemmatizer()


def data_cleaning(row):
    clean_sent = []
    for sent in nltk.sent_tokenize(row['message'].lower()):
        #- Remove Special Characters
        sent = re.sub('[^a-zA-Z]',' ', sent)
        formated_sentence = " ".join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(sent)  if word not in stopWordList])
        clean_sent.append(formated_sentence)
    return " ".join(clean_sent)


corpus_df['formated_message'] = corpus_df.apply(data_cleaning, axis=1)
#print(corpus_df.head(2))

tfidf_vect_obj = TfidfVectorizer(max_features=100, ngram_range=(1,3))
X = tfidf_vect_obj.fit_transform(corpus_df['formated_message']).toarray()

print(tfidf_vect_obj.vocabulary_)
print(X)
