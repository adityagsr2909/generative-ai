import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

#- Tokenization
from nltk.tokenize import sent_tokenize, word_tokenize
#- Stemming
from nltk.stem import PorterStemmer, SnowballStemmer, RegexpStemmer
#- Lemmatization
from nltk.stem import WordNetLemmatizer
#- BOW
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

corpus_df = pd.read_csv("C:/Users/SouryaDevansh/Downloads/GenAI/NLP/SMSSpamClassification.csv", names=['label', 'message'])
print(corpus_df.head(2))

#-Data Cleaning 
"""
1. Lower Corpus
2. Tokenization
3. Remove Stopwords
4. Lemmatization
"""

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

#- BOW
countvec_obj = CountVectorizer(binary=True,  #- Binary Bag of words ( freq= 1 or >1 value will be 1 )
                               lowercase=True, 
                               stop_words=stopWordList, 
                               max_features=2500, #- pick top 2500 words with highest frequency
                               ngram_range=(1,1))

X = countvec_obj.fit_transform(corpus_df['formated_message']).toarray()
print(X.shape)
print(countvec_obj.get_feature_names_out())

#- CountVectorizer.get_feature_names_out() returns words in the order they were encountered and sorted internally, not sorted by frequency.
#- The frequency count (X.toarray().sum(axis=0)) is just a NumPy array aligned positionally with the features (words), not ordered by value.

vocab = countvec_obj.vocabulary_
word_frequencies = X.sum(axis=0)
word_to_freq = dict(zip(countvec_obj.get_feature_names_out(), word_frequencies))
print(word_to_freq)


output_df = pd.DataFrame(X)
output_df.columns = countvec_obj.get_feature_names_out()
output_df['target'] = corpus_df['label']
print(output_df.head(2))