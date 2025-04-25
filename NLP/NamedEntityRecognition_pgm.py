import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

corpus = """
The Eiffel Tower was built from 1887 to 1889 by French Engineer Gustave Eiffel, whose company specialized in building metal frameworks and structures.
"""

"""
Person Eg: Aditya
Place or Location Eg:India
Date Eg: September, 29-09-1988
Time Eg: 4:30pm
Money Eg: 1 million dollar
Organization Eg: Verizon
Percent Eg: 20%, twenty percent
"""

wordList = nltk.word_tokenize(corpus)
postTagElements = nltk.pos_tag(wordList)

nltk.ne_chunk(postTagElements).draw()