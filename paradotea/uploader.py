import warnings

from elasticsearch import Elasticsearch, helpers
import pandas as pd
import csv
import os

def es_cnfg():

    es = Elasticsearch(HOST='http://localhost', PORT=9200)
    es = Elasticsearch()                                        #configures elasticsearch

    warnings.filterwarnings('ignore')

    # df = pd.read_csv('BX-Books.csv', delimiter=',', encoding="utf-8", skipinitialspace=True)

    if es.indices.exists(index="books"):
        pass
    else:
        with open('BX-Books.csv') as f:
            reader = csv.DictReader(f)
            helpers.bulk(es, reader, index='books')             #bulk upload books to elasticsearch
