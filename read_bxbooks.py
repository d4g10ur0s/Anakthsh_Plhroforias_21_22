from elasticsearch import Elasticsearch, helpers
import csv
import json
import time

matching_query = { "query_string": {
                    "query" : None
                    }
                 }
class mBook:
    isbn = None
    book_title = None
    book_author = None
    year_pub = None
    pub = None
    summ = None
    metric = 0

    def __init__(self,isbn,book_title,book_author,year_pub,pub,summ):
        self.isbn = isbn
        self.book_title = book_title
        self.book_author = book_author
        self.year_pub = year_pub
        self.pub = pub
        self.summ = summ
    #olh h plhroforia se mia lista
    def get_info(self):
        return [self.isbn,self.book_title,self.book_author,self.year_pub,self.pub,self.summ,self.metric]
    #to metriko vasei tou user pou exw id
    def set_metric(self,metric):
        self.metric = metric

def main():

    global matching_query
    #sundesh
    es = Elasticsearch(host = "localhost", port = 9200)
    #results["hits"]["hits"][0]["_score"] to score ka8e document
    #results[0]["_source"] periexontai ta kleidia ths plhroforias
    while 1:
        #pairnw eisodo
        mvar = str(input("Give a string : "))
        matching_query["query_string"]["query"] = str(mvar)
        #searching ...
        results = es.search(index="bx_books_2",query=matching_query)
        mcounter = 0 #gia na apari8mhsw to plh8os
        results = results["hits"]["hits"]#ta apotelesmata moy
        #pairnw ta kleidia
        try :
            lst = list(results[0]["_source"].keys())
        except IndexError : #an paizei index error den exw parei apotelesmata, afou prospa8w na parw to 0 ke den to vriskw eimai se empty list
            print("No results.\nSearch again.")
        #emfanish apotelesmatwn +fwna me metrikh?
        for i in results:
            print("Score : "+str(i["_score"]))
            ln = "Number : " +str(mcounter) + " \n"
            for k in lst:
                ln = ln + "   " + str(k)+ " : "+str(i["_source"][str(k)]) + "\n"
                #endfor
            print(ln)#emfanish apotelesmatos
            mcounter+=1
        #endfor
    #endwhile


if __name__ == "__main__":
    main()
