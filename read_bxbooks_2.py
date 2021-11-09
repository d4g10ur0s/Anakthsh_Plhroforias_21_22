from elasticsearch import Elasticsearch, helpers
import csv
import json
import pandas
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
    muser = None #o user gia ton opoio 8a exw to id tou
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

        #edw paizei h fash me user
        with open('BX-Book-Ratings.csv',"r+",encoding="utf8") as f:
            reader = pandas.read_csv(f)
            ''' uid | isbn | rating '''
            print(str(reader.columns))
            #pairnw to info
            uids = reader.get("uid")
            isbns = reader.get("isbn")
            ratings = reader.get("rating")
            while 1:
                print(str(uids))
                #vres user
                try :
                    uid = int(input("Give user id : "))
                    muser = reader.loc[reader["uid"] == uid ][["uid","isbn","rating"]]
                    print(str(muser.values[0]))
                    break
                except IndexError :
                    print("User not found.")
                except TypeError :
                    print("Give integer.")

            #loop gia evaluation
            mcounter = 0
            '''Exei meinei h sumperilhpsh _score ke my_score'''
            for res in results:
                try :
                    if str(muser["isbn"]) == str(res["_source"]["isbn"]):
                        eval = reader.loc[(reader["isbn"] == isbns[0]) & (reader["uid"] != uids[0] )][["uid","isbn","rating"]]
                        end_score = eval[rating].mean() + int(muser.values[2])
                        print("End score : " + str(end_score))
                    else:
                        #pairnw users ke mean value
                        eval = reader.loc[reader["isbn"] == res["_source"]["isbn"] ]["rating"]
                        mean = eval["rating"].mean() #vriskw meso oro
                        print("Mean  : " +str(mean))
                except KeyError :
                    #yparxei key error an den exw parei apotelesmata apo csv arxeio
                    print("No ratings for book with isbn : "+ str(res["_source"]["isbn"]))
                    #emfanish apotelesmatwn +fwna me metrikh?
                    print("Score : "+str(res["_score"]))
                    ln = "Number : " +str(mcounter) + " \n"
                    for k in lst:
                        ln = ln + "   " + str(k)+ " : "+str(res["_source"][str(k)]) + "\n"
                    #endfor
                    print(ln)#emfanish apotelesmatos
                    mcounter+=1
            #endfor
        #endwith
    #endwhile


if __name__ == "__main__":
    main()
