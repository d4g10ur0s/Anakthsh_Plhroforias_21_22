from elasticsearch import Elasticsearch, helpers
import csv
import json
import time
import pandas

mvar = "clara"

matching_query = { "query_string": {
                    "query": mvar
                    }
                 }

def main():
    f = open('BX-Books.csv',"r",encoding="utf8")
    reader = pandas.read_csv(f)
    f.close()
    #pairnw ws list o,ti paizei mesa se reader
    #lst = list(reader)
    #dhmiourgeia arxeiou ann auto den yparxei
    vivlia = []
    ids = reader.get("isbn")
    kappa = 0
    for i in reader.get("summary"):
        #print(i)
        #print(ids[kappa])
        vivlia.append((ids[kappa],i))
        kappa+=1
        #print(vivlia)
    for i in vivlia:
        #inp=input()
        f = open('summary_dir\\'+i[0]+".txt","w+",encoding="utf8")
        f.write(i[1])
        f.close()




if __name__ == "__main__":
    main()
