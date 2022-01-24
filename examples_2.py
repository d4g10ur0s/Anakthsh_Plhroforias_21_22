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

    ratings_df = pd.read_csv('BX-Book-Ratings.csv')
    ratings_df = ratings_df.loc[ratings_df['uid']==uid]
    ratings_df['isbn'] = ratings_df['isbn'].map(lambda x: x.strip())
    # print(len(ratings_df.isbn))
    # print(ratings_df)

    books_df = pd.read_csv('BX-Books.csv')
    # print(books_df.columns)
    books_df = books_df.drop(['book_title', 'book_author', 'year_of_publication', 'publisher', 'category'], axis='columns')
    books_df['isbn'] = books_df['isbn'].map(lambda x: x.strip())
    # print(books_df.head(5))

    summaries = []
    # ids = []
    i = 0
    mtuple = []
    ratings = []
    for isbn in ratings_df.isbn:
        if (ratings_df[ratings_df['isbn']==str(isbn)].rating.values[0]) > 0:
            try:
                i+=1
                # ids.append(isbn)
                summaries.append(listToString(books_df[books_df['isbn'].astype(str).str.contains(isbn)].summary.values))
                if summaries[len(summaries)-1] == '':
                    continue
                else:
                    mtuple.append((isbn, summaries[len(summaries)-1], ratings_df[ratings_df['isbn']==str(isbn)].rating.values[0]))
                    ratings.append(ratings_df[ratings_df['isbn']==str(isbn)].rating.values[0])
            except:
                ratings_df.pop(i)
        else:
            pass

    # print(ratings)

    # import logging
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # import gensim.downloader as api
    # corpus = api.load('text8')
    # import inspect
    # print(inspect.getsource(corpus.__class__))
    # print(inspect.getfile(corpus.__class__))
    # model = Word2Vec(corpus)
    # model.save('./readyvocab.model')

    model = Word2Vec.load('readyvocab.model')

    processed_sentences = []
    for sentence in mtuple:
        processed_sentences.append(gensim.utils.simple_preprocess(sentence[1]))

    # print(*processed_sentences, sep='\n')
    # vectors = []
    # for v in processed_sentences:
    #     try:
    #         vectors.append(model.wv[v].mean())
    #     except:
    #         vectors.append(np.nan)
    vectors = {}
    i = 0
    for v in processed_sentences:
        vectors[str(i)] = []
        # print(v)
        for k in v:
            try:
                vectors[str(i)].append(model.wv[k].mean())
            except:
                vectors[str(i)].append(np.nan)
        # print(vectors[str(i)])
        i += 1

    # df_input = pd.DataFrame(vectors)
    df_input = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in vectors.items()]))
    # print(df_input)
    for i in range(0, len(vectors)):
        # print(df_input[str(i)])
        df_input.fillna(value=0.0, inplace=True)
        df_input[str(i)].replace(to_replace=0, value=df_input[str(i)].mean(), inplace=True)
        # print(df_input[str(i)])

    # print(df_input.index)

    inp = []
    for column in df_input:
        # print(df_input[column].values)
        inp.append(df_input[column].values.mean())

    X = pd.DataFrame(inp)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, ratings, test_size=0.2, random_state=2)



    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import LassoLars
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm
    from sklearn import tree
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import (NearestNeighbors, NeighborhoodComponentsAnalysis, KNeighborsClassifier)
    nca = NeighborhoodComponentsAnalysis(random_state=2)
    knn = KNeighborsClassifier(n_neighbors=10)
    log = svm.SVR()
    log.fit(X, ratings)
    # print(log.score(X_test, y_test))
    # print(log.predict(X_test))

    p = []
    b = []

    for w in books_df.summary:
        try:
            test = gensim.utils.simple_preprocess(w)
            test = model.wv[test].mean()
            res = log.predict(test.reshape(1, -1))
            p.append(test)
            b.append(res)
            # print(test)
            # if res > 0:
            #     print(res)
        except:
            pass

    plt.scatter(p, b, color='blue', marker='+')
    plt.show()

    # test = gensim.utils.simple_preprocess('artificial intelligence is taking over the market')
    # test = model.wv[test].mean()
    # print(test.reshape(1, -1))
    # print(log.predict(test.reshape(1, -1)))




def listToString(s):
    str1 = ''

    for ele in s:
        str1 += ele

    return str1

def main():
    embed(8)

if __name__ == "__main__":
    main()
