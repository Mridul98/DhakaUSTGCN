
import numpy as np
import pprint
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from itertools import filterfalse
device = torch.device

def apply_kmeans(np_array,n_clusters):

    cluster_index = [i for i in range(n_clusters)]
    cluster_members = dict()

    print('starting....')
    print('applying k_means....')
    k_means = KMeans(n_clusters=n_clusters,random_state=824,init='random').fit(np_array)
    # print(k_means.labels_)
    labels = enumerate(k_means.labels_)
    for i in cluster_index:

        cluster_members[str(i)] = list(

            filterfalse(
                lambda x : x[1] != i,
                labels
            )
        )
    
    return cluster_members

def binary_search(a,length,x):
    
    if length == 1:
        if a[0] == x:
            return True
        return False

    mid_point = length // 2
    print(f'length {length}')
    left_len = len(a[:mid_point])
    right_len = len(a[mid_point:])

    return (binary_search(a[:mid_point],left_len,x) | binary_search(a[mid_point:],right_len,x))


    

if __name__ == '__main__':
   
    # a = torch.FloatTensor(size=(154,96),).cuda()
    # a.requires_grad = True
    # b = a.clone().detach().cpu().numpy()

    # x = torch.FloatTensor(size = (154,8)).repeat(12,1)
    # print(x.shape)
   
    # # print(apply_kmeans(b,2))
    # # print(list(filterfalse(lambda x : x == 5, [1,2,3,4,5,6])))


    a = [1,2,4,52,52,12,34,5]

    print(binary_search(a,len(a),0))