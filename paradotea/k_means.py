import numpy as np

def my_kmeans(df,k = 2,maxIterations=10):
    #arxikopoihsh kentrweidwn
    always_centroid = []
    c1 = None
    c2 = None
    choose = np.random.randint(df.shape[1] , size=k)
    my_centroids = []
    for i in range(0,k):
        my_centroids.append( df[str(choose[i])].values.tolist() )
        always_centroid.append( df[str(choose[i])] )
    #ta exw kanei lista
    i = 0
    to_centroid = []
    for i in range(0,df.shape[1]):
        if i in choose:
            pass
        else:
            similarities = []
            for j in range(0,len(my_centroids)):
                #vazw tis omoiothtes se lista ke pairnw thn megaluterh apoluth timh
                similarities.append( my_cosine_similarity(np.squeeze( np.asarray(my_centroids[j] ) ) ,np.squeeze( np.asarray(df[str(i)].values.tolist() ) ) ) )
            #dialegw to megalutero similarity
            best = 0
            for j in range(0,len(similarities)):
                if abs(similarities[j]) > best:
                    best = similarities[j]
                    #prepei na kanw ke ena pop
                    if len(to_centroid)-1 == i:#to plh8os twn stoixeiwn einai iso me to i panta!1 kentroeides gia ka8e perilhpsh
                        to_centroid.pop(len(to_centroid) -1)
                    #to dianusma 8a paei sto kentroeides tade
                    to_centroid.append(j)
    iterations = -1
    while iterations < maxIterations:
        c1 = always_centroid#prin allaksei to kentroeides
        iterations+=1
        kappa = 0
        #update centroids
        for i in range(0,len(my_centroids)):#gia ka8e kedroeides
            for j in range(0,len(to_centroid)):
                #an eimai sto katallhlo kanw summ
                if to_centroid[j] == i:
                    #kane sum
                    always_centroid[i] = always_centroid[i]+df[str(j)]
                else:
                    pass
            #sto telos pollaplasiazw ola ta stoixeia
            always_centroid[i] = always_centroid[i]*(1/len(always_centroid[i]))
        #ksanakanw thn diadikasia ?
        my_centroids = []
        for i in range(0,k):
            my_centroids.append( always_centroid[i].values.tolist() )
        #ta exw kanei lista
        i = 0
        to_centroid = []
        for i in range(0,df.shape[1]):
            if i in choose:
                pass
            else:
                similarities = []
                for j in range(0,len(my_centroids)):
                    #vazw tis omoiothtes se lista ke pairnw thn megaluterh apoluth timh
                    similarities.append( my_cosine_similarity(np.squeeze( np.asarray(my_centroids[j] ) ) ,np.squeeze( np.asarray(df[str(i)].values.tolist() ) ) ) )
                #dialegw to megalutero similarity
                best = 0
                for j in range(0,len(similarities)):
                    if abs(similarities[j]) > best:
                        best = similarities[j]
                        #prepei na kanw ke ena pop
                        if len(to_centroid)-1 == i:#to plh8os twn stoixeiwn einai iso me to i panta!1 kentroeides gia ka8e perilhpsh
                            to_centroid.pop(len(to_centroid) - 1)
                        #to dianusma 8a paei sto kentroeides tade
                        #print(csimilarity)
                        to_centroid.append(j)
        c2 = my_centroids
        #an ta kedroeidh idia tote break
        p = True
        for i in range(0,k):
            #print(str(c1[i]))
            #print(str(c2[i]))
            print("Finished in : "+ str(iterations) +" iterations .")
            if c1[i].equals(c2[i]):
                pass
            else:
                p = False

    return (choose, to_centroid)


def my_cosine_similarity(arr1,arr2):
    dot = sum(a*b for a,b in zip(arr1,arr2) )
    norm_arr1 = sum(a*a for a in arr1) ** 0.5
    norm_arr2 = sum(b*b for b in arr2) ** 0.5
    csimilarity = dot/(norm_arr1*norm_arr2)
    return csimilarity
