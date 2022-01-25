
import gensim
import gensim.downloader as api
from gensim.models import Word2Vec as w2v

import inspect

import logging

import warnings

import numpy as np

from sklearn import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances

import os


import pandas as pd

import matplotlib.pyplot as plt


def embed(uid):
    warnings.filterwarnings('ignore')


    ratings_df = pd.read_csv('/home/dagkla/Documents/BX-Book-Ratings.csv')
    ratings_df = ratings_df.loc[ratings_df['uid']==uid]
    ratings_df['isbn'] = ratings_df['isbn'].map(lambda x: x.strip())
    # print(len(ratings_df.isbn))

    books_df = pd.read_csv('/home/dagkla/Documents/BX-Books.csv')
    # print(books_df.columns)
    books_df = books_df.drop(['book_title', 'book_author', 'year_of_publication', 'publisher', 'category'], axis='columns')
    books_df['isbn'] = books_df['isbn'].map(lambda x: x.strip())

    summaries = []
    mtuple= []
    myratings = []
    i = 0
    for isbn in ratings_df.isbn:
        try:
            i+=1
            summaries.append(list2string(books_df[books_df['isbn']==str(isbn)].summary.values))
            if summaries[len(summaries)-1] == "":
                continue
            else:
                mtuple.append( (isbn , summaries[len(summaries)-1] ))
                myratings.append( ratings_df[ratings_df['isbn']==str(isbn)].rating.values[0])
        except:
            ratings_df.pop(i)
    i = 0
    for j in mtuple:
        print(j)
        print("Va8mos : " + str(myratings[i]))
        i+=1


    '''logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    corpus = api.load('text8')
    print(inspect.getsource(corpus.__class__))
    print(inspect.getfile(corpus.__class__))
    model = w2v(corpus)
    model.save('readyvocab.model')'''

    model = w2v.load('readyvocab.model')
    processed_sentences = []
    for sentence in mtuple:
        processed_sentences.append(gensim.utils.simple_preprocess(sentence[1]))

        # print(*processed_sentences, sep='\n')
    vectors = {}
    i = 0
    for v in processed_sentences:
        vectors[str(i)] = []
        for k in v:
            try:
                vectors[str(i)].append(model.wv[k].mean())
            except:
                vectors[str(i)].append(np.nan)
        i+=1

    df_input =  pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in vectors.items() ]))
    for i in range(0,len(vectors)):
        print(df_input[str(i)])
        df_input.fillna(value=0.0,inplace=True)
        df_input[str(i)].replace(to_replace=0,value=df_input[str(i)].mean(),inplace=True )
        print(df_input[str(i)])

    my_kmeans(df=df_input,k=2)
    #np.any(np.isnan(df_input))
    #np.all(np.isfinite(df_input))
    #X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=False)
    '''ta exw ola ws ka8eta dianusmata'''


def my_kmeans(df,k = 2):
    #arxikopoihsh kentrweidwn
    print(df)
    choose = np.random.randint(df.shape[1] , size=k)
    my_centroids = []
    print()
    for i in range(0,k):
        my_centroids.append( df[str(choose[i])] )
    print(my_centroids)
    i = 0
    for i in range(0,df.shape[1]):
        if i in choose:
            pass
        else:
            similarities = []
            for j in range(0,len(my_centroids)):
                #edw paizei 8ema
                similarities.append( cosine_similarity([my_centroids[j] ,df[str(i)]] ) )
            print(similarities)

def list2string(s):
    strl = ""
    for e in s :
        strl +=e
    return strl

def main():
    embed(8)

if __name__ == "__main__":
    main()
