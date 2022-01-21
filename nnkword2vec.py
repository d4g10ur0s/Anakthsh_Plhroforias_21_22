from gensim.models import Word2Vec as w2v
import warnings
#from keras import *
import numpy as np
#import tensorflow
from sklearn import *
#from keras.preprocessing.text import *
#from keras.layers import *
#from keras.models import *
#from keras.layers.embeddings import *
#from keras.preprocessing.sequence import *
import os
import gensim
import pandas as pd
import matplotlib.pyplot as plt


#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 github token

def embed(uid):
    warnings.filterwarnings('ignore')

    import gensim
    import pandas as pd
    from gensim.models import Word2Vec

    ratings_df = pd.read_csv('/home/dagkla/Documents/BX-Book-Ratings.csv')
    ratings_df = ratings_df.loc[ratings_df['uid']==uid]
    ratings_df['isbn'] = ratings_df['isbn'].map(lambda x: x.strip())
    # print(len(ratings_df.isbn))

    books_df = pd.read_csv('/home/dagkla/Documents/BX-Books.csv')
    # print(books_df.columns)
    books_df = books_df.drop(['book_title', 'book_author', 'year_of_publication', 'publisher', 'category'], axis='columns')
    books_df['isbn'] = books_df['isbn'].map(lambda x: x.strip())
    # print(books_df.head(5))

    # print(ratings_df.isbn)
    print(str(ratings_df))
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
            print(myratings[i])
            i+=1

    # import logging
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # import gensim.downloader as api
    # corpus = api.load('text8')
    # import inspect
    # print(inspect.getsource(corpus.__class__))
    # print(inspect.getfile(corpus.__class__))
    # model = Word2Vec(corpus)
    # model.save('./readyvocab.model')

    #model = Word2Vec.load('readyvocab.model')

    # processed_sentences = []
    # for sentence in summaries:
    #     processed_sentences.append(gensim.utils.simple_preprocess(sentence))
    #
    # vectors = []
    # for v in processed_sentences:
    #     try:
    #         vectors.append(model.wv[v].mean())
    #     except:
    #         vectors.append(np.nan)
    #
    # df_input = pd.DataFrame(vectors)
    #
    # df_input = df_input.fillna(df_input.mean())
    #
    # # print(len(vectors), len(df.overall), sep='\n')
    #
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(df_input, ratings_df.rating, test_size=0.2, random_state=2)
    #
    #
    # from sklearn.linear_model import LogisticRegression
    # log = LogisticRegression()
    # log.fit(X_train, y_train)
    # print(log.score(X_test, y_test))

def list2string(s):
    strl = ""
    for e in s :
        strl +=e
    return strl

def main():
    embed(8)

if __name__ == "__main__":
    main()
