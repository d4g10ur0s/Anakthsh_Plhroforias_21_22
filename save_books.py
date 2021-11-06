from elasticsearch import Elasticsearch, helpers
import csv
import json
import time

mvar = "clara"

matching_query = { "query_string": {
                    "query": mvar
                    }
                 }

def main():
    #sundesh
    es = Elasticsearch(host = "localhost", port = 9200)
    #anagnwsh arxeiou
    with open('BX-Books.csv',"r+",encoding="utf8") as f:
        reader = csv.DictReader(f)
        #pairnw ws list o,ti paizei mesa se reader
        lst = list(reader)
        #dhmiourgeia arxeiou ann auto den yparxei
        helpers.bulk(es, reader, index="bx_books_2")

if __name__ == "__main__":
    main()
