from elasticsearch import Elasticsearch, helpers
import csv
import json
import pandas
import time

''' To query '''
matching_query = { "query_string": {
                    "query" : None
                    }
                 }

def my_merge(arr1,arr2):
    ret = []
    if len(arr1) <= 1 and len(arr2) == 0:
        ret = arr1
    elif len(arr2) <= 1 and len(arr1) == 0:
        ret = arr2
    else:
        #gia ka8e ena se arr1
        for i in arr1 :
            c = 0
            for j in arr2:
                if i["_score"] >= j["_score"]:
                    ret.append(i)
                    break       #paw sto epomeno i
                else:
                    ret.append(j)
                    c+=1
            for j in range(c):  #vgainoun ola osa exw valei
                arr2.pop(0)
            if len(arr2)==0:    #teleiwsa me arr2, arr1 einai taksinomhmenh alla den exw teleiwsei
                ret+=arr1
                break
        if len(arr2) > 0:       #teleiwsa me arr1 alla exei meinei arr2
            ret+=arr2
    return ret

def my_order(arr):
    if len(arr) <= 1:
        return arr
    elif len(arr)%2==0:
        a = my_merge( my_order(arr[:int(len(arr)/2)]), my_order(arr[int(len(arr)/2):]) )
        return a
    else:
        a = my_merge( my_order ( arr[ : int( (len(arr)-1)/2 ) ] ), my_order( arr[ int( (len(arr)-1)/2) : ] ) )
        return a

#den eimai sigouros an 8a to xrhsimopoihsw en telei
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
        res_sz = str(input(" *** \n(1 < #results && 10 000 > #results)n\ *** \nNumber of results : "))
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

        #edw paizei h fash me user
        with open('BX-Book-Ratings.csv',"r+",encoding="utf8") as f:
            reader = pandas.read_csv(f) #diavasma arxeiou
            ''' uid | isbn | rating '''
            #pairnw to info
            uids = reader.get("uid") #to kanw gia na kserw poia id paizoun , apla gia debugging...
            while 1:
                print(str(uids))
                #vres user
                try :
                    uid = int(input("Give user id : "))
                    muser = reader.loc[reader["uid"] == uid ][["uid","isbn","rating"]]
                    print("O user pu dialeksa : " + str(muser.values[0]))
                    break
                except IndexError :
                    print("User not found.")
                except TypeError :
                    print("Give integer.")

            #loop gia evaluation
            mcounter = 0
            '''Exei meinei h sumperilhpsh _score ke my_score'''
            for res in results:
                #######
                if str(muser["isbn"]) == str(res["_source"]["isbn"]):#an o user exei valei va8mologia
                    #8elw idia isbns ke 8elw diaforetika uid apo ton user mu                                                   #pairnw uid isbn ke rating
                    eval = reader.loc[(reader["isbn"] == res["_source"]["isbn"]) & (reader["uid"] != muser.values[0]["uid"] )][["uid","isbn","rating"]]
                    end_score = eval["rating"].mean() + int(muser.values[0]["rating"])
                    res["_score"] += end_score
                else:
                    print("Den einai edw")
                    #pairnw users ke mean value
                    eval = reader.loc[reader["isbn"] == res["_source"]["isbn"] ]["rating"]
                    end_score = eval.mean() #vriskw meso oro
                    res["_score"] += end_score
            #endfor
            #taksinomhsh ke print
            results = my_order(results)

            pr = []                             #sthn lista pr vazw ola ta dipla, grammes 136-139
            for res in results:
                if res["_source"]["isbn"] in pr:
                    continue
                else:
                    pr.append(res["_source"]["isbn"])
                #emfanish apotelesmatwn +fwna me metrikh?
                print("Score : "+str(res["_score"]))
                ln = "Number : " +str(mcounter) + " \n"
                for k in lst:
                    ln = ln + "   " + str(k)+ " : "+str(res["_source"][str(k)]) + "\n"
                #endfor
                print(ln)#emfanish apotelesmatos
                mcounter+=1
        #endwith
    #endwhile


if __name__ == "__main__":
    main()
