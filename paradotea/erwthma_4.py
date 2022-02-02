
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

from elasticsearch import Elasticsearch, helpers
import json
import time

''' To query '''
matching_query = { "query_string": {
                    "query" : None
                    }
                 }

def main():
    global matching_query
    muser = None #o user gia ton opoio 8a exw to id tou
    #sundesh
    es = Elasticsearch(host = "localhost", port = 9200)
    #results["hits"]["hits"][0]["_score"] to score ka8e document
    #results[0]["_source"] periexontai ta kleidia ths plhroforias
    while 1:
        #pairnw eisodo
        mvar = str(input("Give a string : "))
        res_sz = str(input(" *** \n(1 < #results && 10 000 > #results)n\ *** \nNumber of results : "))
        mykappa = int(input("Number of clusters for kmeans : "))
        maxIterations = int(input("Number of iterations for kmeans : "))
        matching_query["query_string"]["query"] = str(mvar)
        #searching ...
        results = es.search(index="bx_books_2",query=matching_query,size = int(res_sz))
        mcounter = 0 #gia na apari8mhsw to plh8os
        results = results["hits"]["hits"]#ta apotelesmata moy
        #pairnw ta kleidia
        try :
            lst = list(results[0]["_source"].keys())
        except IndexError : #an paizei index error den exw parei apotelesmata, afou prospa8w na parw to 0 ke den to vriskw eimai se empty list
            print("No results.\nSearch again.")

        summaries = []
        for res in results :
            summaries.append(res["_source"]["summary"])
        print(str(summaries))
        warnings.filterwarnings('ignore')


    #ratings_df = pd.read_csv('BX-Book-Ratings.csv')
    #ratings_df = ratings_df.loc[ratings_df['uid']==uid]
    #ratings_df['isbn'] = ratings_df['isbn'].map(lambda x: x.strip())
    # print(len(ratings_df.isbn))

    #books_df = pd.read_csv('BX-Books.csv')
    # print(books_df.columns)
    #books_df = books_df.drop(['book_author', 'year_of_publication', 'publisher', 'category'], axis='columns')
    #books_df['isbn'] = books_df['isbn'].map(lambda x: x.strip())
        mtuple= []
        '''logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        corpus = api.load('text8')
        print(inspect.getsource(corpus.__class__))
        print(inspect.getfile(corpus.__class__))
        model = w2v(corpus)
        model.save('readyvocab.model')'''

    #myratings = []
    #i = 0
    #for isbn in ratings_df.isbn:
    #    try:
    #        i+=1
    #        summaries.append(list2string(books_df[books_df['isbn']==str(isbn)].summary.values))
    #        if summaries[len(summaries)-1] == "":
    #            continue
    #        else:
    #            mtuple.append( (isbn , summaries[len(summaries)-1] ))
    #            myratings.append( ratings_df[ratings_df['isbn']==str(isbn)].rating.values[0])
    #    except:
    #        ratings_df.pop(i)


        model = w2v.load('readyvocab.model')
        processed_sentences = []
        for sentence in summaries:
            processed_sentences.append(gensim.utils.simple_preprocess(sentence))

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
            df_input.fillna(value=0.0,inplace=True)
            df_input[str(i)].replace(to_replace=0,value=df_input[str(i)].mean(),inplace=True )
        processed = my_kmeans(df=df_input,k=mykappa,maxIterations=maxIterations)

    #np.any(np.isnan(df_input))
    #np.all(np.isfinite(df_input))
    #X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=False)
        '''pairnw tous titlous'''
        titles = []
        for res in results:
            try:
                titles.append(res["_source"]["book_title"])
            #titles.append(list2string(books_df[books_df['isbn']==str(isbn)].book_title.values))
            except:
                pass#ratings_df.pop(i)
        '''print tis klaseis'''
        for myint in range(0,mykappa):#poses klaseis exw
            mcounter = -1
            print('\n'+5*"*"+" Klash : "+str(myint+1)+ " "+5*"*"+'\n')
            for j in processed[1]:
                mcounter+=1
                if myint == j:#einai sthn idia klash
                    print(titles[mcounter])
                else:
                    pass


def my_kmeans(df,k = 2,maxIterations=10):
    #arxikopoihsh kentrweidwn
    always_centroid = []
    c1 = None
    c2 = None
    choose = np.random.randint(df.shape[1] , size=k)
    my_centroids = []
    for i in range(0,k):
        my_centroids.append( df[str(choose[i])].values.tolist() )
        always_centroid.append( df[str(choose[i])] )
    #ta exw kanei lista
    i = 0
    to_centroid = []
    for i in range(0,df.shape[1]):
        if i in choose:
            pass
        else:
            similarities = []
            for j in range(0,len(my_centroids)):
                #vazw tis omoiothtes se lista ke pairnw thn megaluterh apoluth timh
                similarities.append( my_cosine_similarity(np.squeeze( np.asarray(my_centroids[j] ) ) ,np.squeeze( np.asarray(df[str(i)].values.tolist() ) ) ) )
            #dialegw to megalutero similarity
            best = 0
            for j in range(0,len(similarities)):
                if abs(similarities[j]) > best:
                    best = similarities[j]
                    #prepei na kanw ke ena pop
                    if len(to_centroid)-1 == i:#to plh8os twn stoixeiwn einai iso me to i panta!1 kentroeides gia ka8e perilhpsh
                        to_centroid.pop(len(to_centroid) -1)
                    #to dianusma 8a paei sto kentroeides tade
                    to_centroid.append(j)
    iterations = -1
    while iterations < maxIterations:
        c1 = always_centroid#prin allaksei to kentroeides
        iterations+=1
        kappa = 0
        #update centroids
        for i in range(0,len(my_centroids)):#gia ka8e kedroeides
            for j in range(0,len(to_centroid)):
                #an eimai sto katallhlo kanw summ
                if to_centroid[j] == i:
                    #kane sum
                    always_centroid[i] = always_centroid[i]+df[str(j)]
                else:
                    pass
            #sto telos pollaplasiazw ola ta stoixeia
            always_centroid[i] = always_centroid[i]*(1/len(always_centroid[i]))
        #ksanakanw thn diadikasia ?
        my_centroids = []
        for i in range(0,k):
            my_centroids.append( always_centroid[i].values.tolist() )
        #ta exw kanei lista
        i = 0
        to_centroid = []
        for i in range(0,df.shape[1]):
            if i in choose:
                pass
            else:
                similarities = []
                for j in range(0,len(my_centroids)):
                    #vazw tis omoiothtes se lista ke pairnw thn megaluterh apoluth timh
                    similarities.append( my_cosine_similarity(np.squeeze( np.asarray(my_centroids[j] ) ) ,np.squeeze( np.asarray(df[str(i)].values.tolist() ) ) ) )
                #dialegw to megalutero similarity
                best = 0
                for j in range(0,len(similarities)):
                    if abs(similarities[j]) > best:
                        best = similarities[j]
                        #prepei na kanw ke ena pop
                        if len(to_centroid)-1 == i:#to plh8os twn stoixeiwn einai iso me to i panta!1 kentroeides gia ka8e perilhpsh
                            to_centroid.pop(len(to_centroid) - 1)
                        #to dianusma 8a paei sto kentroeides tade
                        #print(csimilarity)
                        to_centroid.append(j)
        c2 = my_centroids
        #an ta kedroeidh idia tote break
        p = True
        for i in range(0,k):
            #print(str(c1[i]))
            #print(str(c2[i]))
            print("Finished in : "+ str(iterations) +" iterations .")
            if c1[i].equals(c2[i]):
                pass
            else:
                p = False

    return (choose, to_centroid)


def my_cosine_similarity(arr1,arr2):
    dot = sum(a*b for a,b in zip(arr1,arr2) )
    norm_arr1 = sum(a*a for a in arr1) ** 0.5
    norm_arr2 = sum(b*b for b in arr2) ** 0.5
    csimilarity = dot/(norm_arr1*norm_arr2)
    return csimilarity

def list2string(s):
    strl = ""
    for e in s :
        strl +=e
    return strl


if __name__ == "__main__":
    main()
