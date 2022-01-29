import math

import gensim
import pandas as pd
from gensim.models import Word2Vec
from sklearn import svm
#import warnings
import numpy as np
from sklearn import *
import os
import gensim
import pandas as pd
import matplotlib.pyplot as plt


#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 github token
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def embed(uid):
    #warnings.filterwarnings('ignore')


    ratings_df = pd.read_csv('BX-Book-Ratings.csv')
    ratings_df = ratings_df.loc[ratings_df['uid']==uid]
    ratings_df['isbn'] = ratings_df['isbn'].map(lambda x: x.strip())

    books_df = pd.read_csv('BX-Books.csv')
    books_df = books_df.drop(['book_title', 'book_author', 'year_of_publication', 'publisher', 'category'], axis='columns')
    books_df['isbn'] = books_df['isbn'].map(lambda x: x.strip())

    summaries = []
    mtuple = []
    ratings = []
    i = 0
    for isbn in ratings_df.isbn:
        try:
            i += 1
            summaries.append(listToString(books_df[books_df['isbn'] == str(isbn)].summary.values))
            if summaries[len(summaries) - 1] == "":
                continue
            else:
                # ola 8a exoun mpei me thn idia seira
                mtuple.append((isbn, summaries[len(summaries) - 1]))
                ratings.append(ratings_df[ratings_df['isbn'] == str(isbn)].rating.values[0])
        except:
            ratings_df.pop(i)

    ztuples = []
    rtuples = []
    books_rated = []  # o va8mos tou antistoixou vivliou
    # ksexwrizw ti exei va8mologh8ei ke ti oxi
    for j in range(0, len(mtuple)):
        if ratings[j] == 0:  # prwta 8a valw ta 0
            ztuples.append(mtuple[j])
        else:
            rtuples.append(mtuple[j])
            books_rated.append(ratings[j])

    mtuple = ztuples + rtuples  # enwnw ta va8mologhmena me ta mh va8mologhmena

    model = Word2Vec.load('readyvocab.model')

    processed_sentences = []
    for sentence in mtuple:
        processed_sentences.append(gensim.utils.simple_preprocess(sentence[1]))

    vectors = {}
    i = 0
    for v in processed_sentences:
        vectors[str(i)] = []
        for k in v:
            try:
                vectors[str(i)].append(model.wv[k].mean())
            except:
                vectors[str(i)].append(np.nan)
        i += 1  # posa dianusmata exw

    df_input = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in vectors.items()]))
    df_input.fillna(value=0.0, inplace=True)  # oti einai nan paei 0

    for j in range(0, len(vectors)):
        df_input[str(j)].replace(to_replace=0, value=df_input[str(j)].mean(), inplace=True)

    df_input = df_input.transpose()


    svm_mod = svm.SVR()
    svm_mod.fit(df_input[i - len(books_rated):], books_rated)
    res = svm_mod.predict(df_input[:i - len(books_rated)])

    y_tuple = []

    for h in ztuples:
        for re in res:
            y_tuple.append((h[0], np.floor(re)))

    y_tuple = tuple(set(y_tuple))

    return y_tuple

def listToString(s):
    str1 = ''

    for ele in s:
        str1 += ele

    return str1
