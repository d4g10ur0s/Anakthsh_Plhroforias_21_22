import wordemb
from uploader import *
import json
from array import *
import numpy as np

def search(indx, search_for):
    print('\nSearching for',search_for, '...\n')

    es = Elasticsearch(HOST='http://localhost', PORT=9200)
    es = Elasticsearch()

    res = es.search(index=indx, body={"from":0, "size":20, "query":{"match":{"summary":search_for}}})

    # creates an array with all the results' isbns
    books_found = []
    for i in range(len(res['hits']['hits'])):
        books_found.append(res['hits']['hits'][i]['_source']['isbn'])

    # for books in books_found:
    #     print(books)

    return books_found              #all books' isbn that came up with search

def users():
    es = Elasticsearch(HOST='http://localhost', PORT=9200)
    es = Elasticsearch()

    if es.indices.exists(index="ratings"):
        pass
    else:
        with open('BX-Book-Ratings.csv') as f:
            reader = csv.DictReader(f)
            helpers.bulk(es, reader, index='ratings')       #bulk upload ratings to elasticsearch

    if es.indices.exists(index="users"):
        pass
    else:
        with open('BX-Users.csv') as f:
            reader = csv.DictReader(f)
            helpers.bulk(es, reader, index='users')         #bulk upload users to elasticsearch

def userid(id):

    es = Elasticsearch(HOST='http://localhost', PORT=9200)
    es = Elasticsearch()

    res = es.search(index='ratings', body={"from": 0, "size": 20,                                           #searches for all ratings of specific user
                    "sort" : [{"rating.keyword": {"order" : "desc"}}], "query": {"match": {"uid": id}}})

    user_books = []
    for i in range(len(res['hits']['hits'])):
        user_books.append(res['hits']['hits'][i]['_source']['isbn'])               #appends isbns that have a rating from user

    return user_books


def first(arr, low, high, x, n):
    if (high >= low):
        mid = low + (high - low) // 2;  # (low + high)/2;
        if ((mid == 0 or x > arr[mid - 1]) and arr[mid] == x):
            return mid
        if (x > arr[mid]):
            return first(arr, (mid + 1), high, x, n)
        return first(arr, low, (mid - 1), x, n)

    return -1

def sortAccording(books_found, user_books, m, n):           #sortarei mia lista analoga me mia allh
    temp = [0] * m
    visited = [0] * m

    for i in range(0, m):
        temp[i] = books_found[i]
        visited[i] = 0

    temp.sort()

    ind = 0

    for i in range(0, n):
        f = first(temp, 0, m - 1, user_books[i], m)

        if (f == -1):
            continue

        j = f
        while (j < m and temp[j] == user_books[i]):
            books_found[ind] = temp[j]
            ind = ind + 1
            visited[j] = 1
            j = j + 1

    for i in range(0, m):
        if (visited[i] == 0):
            books_found[ind] = temp[i]
            ind = ind + 1

    return books_found


def book_sorter(id, user_books, books_found):       #id of user, list with isbns that have rating from specific user, all book isbns results

    es = Elasticsearch()

    print('\nFound books:\n')                       #book results with no order
    for books in books_found:
        res = es.search(index='books', body={"from": 0, "size": 20, "query": {"match": {"isbn.keyword": books}}})

        for i in range(len(res['hits']['hits'])):
            print('~ ', res['hits']['hits'][i]['_source']['book_title'], ', ', res['hits']['hits'][i]['_source']['book_author'], sep='')

    m = len(books_found)
    n = len(user_books)

    books_found2 = sortAccording(books_found, user_books, m, n)

    print('\nResults sorted by user rating: \n')
    for books in books_found2:
        res = es.search(index='books', body={"from": 0, "size": 20, "query": {"match": {"isbn.keyword": books}}})

        for i in range(len(res['hits']['hits'])):
            print('~ ', res['hits']['hits'][i]['_source']['book_title'], ', ', res['hits']['hits'][i]['_source']['book_author'], ', ', res['hits']['hits'][i]['_source']['isbn'], sep='')

    user_avg = []
    all_users_books = []

    # print('\nBook Average Ratings: \n')
    for isbn in user_books:
        res = es.search(index='ratings', body={"from": 0, "size": 20, "query": {"match": {"isbn.keyword": isbn}}})

        for i in range(len(res['hits']['hits'])):
            if res['hits']['hits'][i]['_source']['isbn'] not in all_users_books:        #bazei ta isbns poy den exei bathmologisei o xristis se lista
                all_users_books.append(res['hits']['hits'][i]['_source']['isbn'])

        book_avg = 0
        users_n = 0

        for i in range(len(res['hits']['hits'])):
            users_n = users_n + 1                   #briskei posoi einai oi xristes sinolo poy exoyn bathmologisei ayto to biblio
            # print('~ ', isbn, ': ', res['hits']['hits'][i]['_source']['rating'], sep='')
            book_avg = book_avg + int(res['hits']['hits'][i]['_source']['rating'])


        user_avg.append(int(book_avg/users_n))

    zipped_lists = zip(user_avg, all_users_books)   #bazei mazi to avg rating kai to isbn gia kathe biblio
    sorted_zipped_lists = sorted(zipped_lists)      #sortarei tin zipped list ascending
    book_avg = [element for _, element in sorted_zipped_lists]
    book_avg = book_avg[::-1]                       #antistrefei to sort se descending

    m = len(books_found2)
    n = len(book_avg)

    books_found3 = sortAccording(books_found2, book_avg, m, n)

    u_a_books = []

    print('\nResults sorted by average user rating + individual user rating: \n')
    for books in books_found3:
        res = es.search(index='books', body={"from": 0, "size": 20, "query": {"match": {"isbn.keyword": books}}})

        for i in range(len(res['hits']['hits'])):
            print('~ ', res['hits']['hits'][i]['_source']['book_title'], ', ',
                  res['hits']['hits'][i]['_source']['book_author'], ', ',
                  res['hits']['hits'][i]['_source']['isbn'],
                  sep='')
            u_a_books.append(res['hits']['hits'][i]['_source']['isbn'])

    predictions, cl = wordemb.embed(int(id))
    m = len(books_found3)
    n = len(predictions)

    books_found4 = sortAccording(books_found3, predictions, m, n)

    print('\nResults sorted by average user rating + individual user rating + predictions rating: \n')
    for books in books_found4:
        res = es.search(index='books', body={"from": 0, "size": 20, "query": {"match": {"isbn.keyword": books}}})

        for i in range(len(res['hits']['hits'])):
            print('~ ', res['hits']['hits'][i]['_source']['book_title'], ', ',
                  res['hits']['hits'][i]['_source']['book_author'], ', ',
                  res['hits']['hits'][i]['_source']['isbn'],
                  sep='')

    print('\n', cl)