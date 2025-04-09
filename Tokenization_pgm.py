import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer, WhitespaceTokenizer

nltk.download('punkt')
nltk.download('stopwords')

corpus = """Hello world! Generative AI is booming these days.
    Lets understand how important it is
"""
#- Word Tokenizer
vocabulary = []
tokens = nltk.tokenize.sent_tokenize(corpus.lower())
for idx, token in enumerate(tokens):
    print(idx, token)
    for wd in nltk.tokenize.word_tokenize(token):
        if wd not in vocabulary:
            vocabulary.append(wd)
print(f"word_tokenize vocabulary = ", vocabulary)

#- TreebankWordToenizer
tb_tokenizer = TreebankWordTokenizer()
vocabulary = []
tokens = nltk.tokenize.sent_tokenize(corpus.lower())
for idx, token in enumerate(tokens):
    for wd in tb_tokenizer.tokenize(token):
        if wd not in vocabulary:
            vocabulary.append(wd)
print(f"TreebankWordTokenizer vocabulary = ", vocabulary)

#- whitespaceTokenizer
ws_tokenizer = WhitespaceTokenizer()
vocabulary = []
tokens = nltk.tokenize.sent_tokenize(corpus.lower())
for idx, token in enumerate(tokens):
    for wd in ws_tokenizer.tokenize(token):
        if wd not in vocabulary:
            vocabulary.append(wd)
print(f"WhitespaceTokenizer vocabulary = ", vocabulary)

"""
----------
output:
----------
0 hello world!
1 generative ai is booming these days.
2 lets understand how important it is
word_tokenize vocabulary =  ['hello', 'world', '!', 'generative', 'ai', 'is', 'booming', 'these', 'days', '.', 'lets', 'understand', 'how', 'important', 'it']
TreebankWordTokenizer vocabulary =  ['hello', 'world', '!', 'generative', 'ai', 'is', 'booming', 'these', 'days', '.', 'lets', 'understand', 'how', 'important', 'it']
WhitespaceTokenizer vocabulary =  ['hello', 'world!', 'generative', 'ai', 'is', 'booming', 'these', 'days.', 'lets', 'understand', 'how', 'important', 'it']
 
"""