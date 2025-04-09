import pandas as pd
import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize, regexp_tokenize
from nltk.tokenize import WhitespaceTokenizer, TreebankWordTokenizer

from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

words = ["eating", "eats", "eaten", "writing", "writes", 
         "programming", "programs", "history", "finally","finalize"]

print(words)

#- PorterStemmer
stemming = PorterStemmer()
for word in words:
    print(f"{word} => {stemming.stem(word)}")
print(stemming.stem('Congratulations')) #- congratul (lost meaning)

#- RegexpStemmer Class - Regular Expression
from nltk.stem import RegexpStemmer

words = ['eating', 'eats', 'eye', 'eatable']
reg_stemmer = RegexpStemmer('ing$|s$|e$|able$', min=4) #- atleast 4 characters
for word in words:
    print(word, " => ", reg_stemmer.stem(word))


#- Snowball Stemmer
from nltk.stem import SnowballStemmer
words = ["eating", "eats", "eaten", "writing", "writes", 
         "programming", "programs", "history", "finally","finalize"]
snow_stemmer = SnowballStemmer(language='english')
for word in words:
    print(f"{word} :: {stemming.stem(word)}  :: {snow_stemmer.stem(word)}")



print(stemming.stem('fairly'), stemming.stem('sportingly'))
print(snow_stemmer.stem('fairly'), snow_stemmer.stem('sportingly'))
