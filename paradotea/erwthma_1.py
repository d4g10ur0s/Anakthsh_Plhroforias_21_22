import warnings

from elasticsearch import Elasticsearch, helpers
import pandas as pd
import csv
import os

def es_cnfg():

    es = Elasticsearch(HOST='http://localhost/', PORT=9200)
    es = Elasticsearch()

    warnings.filterwarnings('ignore')

    # es.indices.create(index='first_index', ignore=400)

    # df = pd.read_csv('BX-Books.csv', delimiter=',', encoding="utf-8", skipinitialspace=True)
    # print(df)

    if es.indices.exists(index="books"):
        pass
    else:
        with open('BX-Books.csv') as f:
            reader = csv.DictReader(f)
            helpers.bulk(es, reader, index='books')

    # if es.indices.exists(index='books'):
        # print(es.indices.get_alias())
