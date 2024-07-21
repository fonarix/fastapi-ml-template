
# Links:
# https://blog.gopenai.com/text-pre-processing-pipeline-using-nltk-3c086d04ad4e


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


import nltk

import os
import numpy as np
import pandas as pd
#from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import regex as re
from string import punctuation
import math

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

#nltk.download()


from nltk.corpus import reuters
#from nltk.corpus import imdb

import string



sents = ['coronavirus is a highly infectious disease',
'coronavirus affects older people the most',
'older people are at high risk due to this disease']

cv = CountVectorizer()


X = cv.fit_transform(sents)
X = X.toarray()
print(X)



voc = sorted(cv.vocabulary_.keys())
print(voc)


cv = CountVectorizer(ngram_range=(2,2))
X = cv.fit_transform(sents)
X = X.toarray()
print(X)


voc = sorted(cv.vocabulary_.keys())
print(voc)

###############################################################################
# tfidf

tfidf = TfidfVectorizer()

transformed = tfidf.fit_transform(sents)
import pandas as pd

df = pd.DataFrame(transformed[0].T.todense(),
index=tfidf.get_feature_names_out(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print(df)


# 1) Skip-Gram
# 2) CBOW - Continuous Bag of Word



#from gensim import models
#from google.colab import drive
#drive.mount('/content/gdrive/')
#w2v = models.KeyedVectors.load_word2vec_format('/content/gdrive/GoogleNews-vectors-negative300.

#BERT Google
#Universal Sentence Encode

def GetDefaultSettings():
    #settings = {'RemoveHtlmTags' : True}
    settings = {}
    settings['sklearn.RemoveHtlmTags'] = False
    settings['sklearn.RemoveUrls'] = False
    settings['sklearn.RemovePunct'] = True
    return settings







