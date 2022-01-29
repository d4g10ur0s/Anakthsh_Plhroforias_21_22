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

    # print('Found ', len(res['hits']['hits']), ' results.', sep='')
    # for i in range(len(res['hits']['hits'])):
    #     print('~ ', res['hits']['hits'][i]['_source']['book_title'], ', ', res['hits']['hits'][i]['_source']['book_author'], sep='')

    # creates an array with all the results
    books_found = []
    for i in range(len(res['hits']['hits'])):
        books_found.append(res['hits']['hits'][i]['_source']['isbn'])

    # for books in books_found:
    #     print(books)

    return books_found

def users():
    es = Elasticsearch(HOST='http://localhost', PORT=9200)
    es = Elasticsearch()

    if es.indices.exists(index="ratings"):
        pass
    else:
        with open('BX-Book-Ratings.csv') as f:
            reader = csv.DictReader(f)
            helpers.bulk(es, reader, index='ratings')

    if es.indices.exists(index="users"):
        pass
    else:
        with open('BX-Users.csv') as f:
            reader = csv.DictReader(f)
            helpers.bulk(es, reader, index='users')

def userid(id):

    es = Elasticsearch(HOST='http://localhost', PORT=9200)
    es = Elasticsearch()

    res = es.search(index='ratings', body={"from": 0, "size": 20,
                    "sort" : [{"rating.keyword": {"order" : "desc"}}], "query": {"match": {"uid": id}}})

    user_books = []
    for i in range(len(res['hits']['hits'])):
        user_books.append(res['hits']['hits'][i]['_source']['isbn'])

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

def sortAccording(books_found, user_books, m, n):
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


def book_sorter(id, user_books, books_found):

    es = Elasticsearch()

    # print('\nbooks found: \n')
    # for books in books_found:
    #     print(books)
    #
    # print('\nsorted books: \n')
    # for user_books in user_books:
    #     print(user_books)

    print('\nFound books:\n')
    for books in books_found:
        res = es.search(index='books', body={"from": 0, "size": 20, "query": {"match": {"isbn.keyword": books}}})

        for i in range(len(res['hits']['hits'])):
            print('~ ', res['hits']['hits'][i]['_source']['book_title'], ', ', res['hits']['hits'][i]['_source']['book_author'], sep='')

    m = len(books_found)
    n = len(user_books)

    sortAccording(books_found, user_books, m, n)

    print('\nResults sorted by user rating: \n')
    for books in books_found:
        res = es.search(index='books', body={"from": 0, "size": 20, "query": {"match": {"isbn.keyword": books}}})

        for i in range(len(res['hits']['hits'])):
            print('~ ', res['hits']['hits'][i]['_source']['book_title'], ', ', res['hits']['hits'][i]['_source']['book_author'], ', ', res['hits']['hits'][i]['_source']['isbn'], sep='')

    user_avg = []
    all_users_books = []

    # print('\nBook Average Ratings: \n')
    for isbn in user_books:
        res = es.search(index='ratings', body={"from": 0, "size": 20, "query": {"match": {"isbn.keyword": isbn}}})

        for i in range(len(res['hits']['hits'])):
            if res['hits']['hits'][i]['_source']['isbn'] not in all_users_books:
                all_users_books.append(res['hits']['hits'][i]['_source']['isbn'])

        book_avg = 0
        users = 0

        for i in range(len(res['hits']['hits'])):
            users = users + 1
            # print('~ ', isbn, ': ', res['hits']['hits'][i]['_source']['rating'], sep='')
            book_avg = book_avg + int(res['hits']['hits'][i]['_source']['rating'])


        user_avg.append(int(book_avg/users))
        # print(isbn, ': ', int(book_avg/users))

    # print(all_users_books, user_avg)

    zipped_lists = zip(user_avg, all_users_books)
    sorted_zipped_lists = sorted(zipped_lists)
    book_avg = [element for _, element in sorted_zipped_lists]
    book_avg = book_avg[::-1]

    # for books in book_avg:
    #     print(books)

    m = len(books_found)
    n = len(book_avg)

    sortAccording(books_found, book_avg, m, n)


    print('\nResults sorted by average user rating + individual user rating: \n')
    for books in books_found:
        res = es.search(index='books', body={"from": 0, "size": 20, "query": {"match": {"isbn.keyword": books}}})
        res2 = es.search(index='ratings', body={"from": 0, "size": 20, "query": {"match": {"isbn.keyword": books}}})

        for i in range(len(res['hits']['hits'])):
            print('~ ', res['hits']['hits'][i]['_source']['book_title'], ', ',
                  res['hits']['hits'][i]['_source']['book_author'], ', ',
                  res['hits']['hits'][i]['_source']['isbn'],
                  sep='')

    try:
        print(wordemb.embed(int(id)))
    except:
        pass

    user_avg2= []
    user_avg2.append(int(book_avg / users))

    zipped_lists = zip(user_avg, all_users_books)
    sorted_zipped_lists = sorted(zipped_lists)
    book_avg = [element for _, element in sorted_zipped_lists]
    book_avg = book_avg[::-1]

    m = len(books_found)
    n = len(book_avg)

    sortAccording(books_found, book_avg, m, n)
