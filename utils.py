import numpy as np 
import json
import networkx as nx
import pandas as pd 
import matplotlib.pyplot as plt 
import re
import torch
from networkx.algorithms.centrality import (
    betweenness_centrality,
    closeness_centrality
)
def get_model_config(path_to_json):
    model_config = None
    with open(path_to_json,'r') as config:
        model_config = json.load(config)
    
    return model_config

def get_bw_centrality(path_to_dataset):

    resampled,resampled_numpy ,adj = get_dataset_with_adjacency_mat(path_to_dataset,'mean')
    list_of_tuples = get_edges_from_adj(adj)
    G = nx.Graph()
    G.add_edges_from(list_of_tuples)
    node_dict = betweenness_centrality(G,normalized=True)
    bc = pd.DataFrame(
        sorted(node_dict.items(),
        key= lambda x : x[1],reverse=True)
    )
    bc.rename(
        columns={
            1:'betweenness centrality'
        },
        inplace=True
    )
    return bc

def get_closeness_centrality(path_to_dataset):

    resampled,resampled_numpy ,adj = get_dataset_with_adjacency_mat(path_to_dataset,'mean')
    list_of_tuples = get_edges_from_adj(adj)
    G = nx.Graph()
    G.add_edges_from(list_of_tuples)
    node_dict = closeness_centrality(G)
    cc = pd.DataFrame(
        sorted(node_dict.items(),
        key= lambda x : x[1],reverse=True)
    )
    cc.rename(
        columns={
            1:'closeness centrality'
        },
        inplace=True
    )
    return cc




def get_edges_from_adj(adj):
    """build a list of 
    tuple(source node,destination node) from the adjacency matrix

    Args:
        adj (numpy array): adjacency matrix in numpy format

    Returns:
        list: list of adjacent node tuples of size 2
    """

    list_of_node_tuples = list()
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i,j] == 1.0:
                list_of_node_tuples.append((i,j))
    
    return list_of_node_tuples


def get_loss_weights(path_to_new_road,same,weight,tierwise):
    resampled,resampled_numpy ,adj = get_dataset_with_adjacency_mat(path_to_new_road,'mean')
    list_of_tuples = get_edges_from_adj(adj)
    G = nx.Graph()
    G.add_edges_from(list_of_tuples)
    node_dict = betweenness_centrality(G,normalized=True)
    bc = pd.DataFrame(
        sorted(node_dict.items(),
        key= lambda x : x[1],reverse=True)
    )
    bc.rename(
        columns={
            1:'betweenness centrality'
        },
        inplace=True
    )
    loss_weights = torch.logspace(1.3090,0,154,10).numpy()
    loss_weights = np.sqrt(loss_weights)
    bc['weights'] = loss_weights
    bc.sort_values(0,ascending=True,inplace=True)
    if same:
        assert weight != None
        return_weights = torch.full((154,1),fill_value = weight)
        return return_weights
    if tierwise:
        tier_info = pd.read_excel('Road_tiers.xlsx')



    return torch.Tensor(bc['weights'].values.reshape((154,1)))

def get_loss_weights_tierwise(path_to_tier_info,tier_1_weight,tier_2_weight):
    
    road_tiers = pd.read_excel('Road_tiers.xlsx')
    road_tiers = road_tiers[['Road Number','Tier']]
    road_tiers['loss_weight'] = np.where(road_tiers['Tier'] == 1 , tier_1_weight,tier_2_weight)
    return torch.sqrt(torch.Tensor(road_tiers['loss_weight'].values).reshape((154,1)))


def get_dataset_with_adjacency_mat(path_to_csv_data , resampling_method):
    '''
    returns 
        pandas dataframe, numpy_converted , adjacency_matrix

    returns timeseries dataset of dhaka traffic jam sampled in 5 min interval and adjacency matrix for connected edges
    path_to_csv_data: 'str'
    resampling_method: either 'max' or 'min' or 'mean' or 'median'
   
       
    '''
    methodlist = ['max','min','mean','median']
    assert resampling_method in methodlist , 'unidentified resampling method'
    
    street_data = pd.read_csv(path_to_csv_data)
    
    #splitting edges into two points
    splitted_endpoints = street_data.Intersection.str.split('-')
    for i in range(len(splitted_endpoints)):
        assert len(splitted_endpoints[i]) == 2 ,'error at index'+str(splitted_endpoints[i])+str(i)
    
    road_to_index = {i:j for i,j in enumerate(splitted_endpoints)}
    total_edges = len(road_to_index)
    #initialized adjacency matrix with zeros
    adjacency_matrix = np.zeros((total_edges,total_edges),dtype = np.float32) 
    
    #populating adjacency matrix
    visited  = list()
    for i in range(len(road_to_index)):
        dest = road_to_index[i]
        if dest not in visited:
            for j in range(len(road_to_index)):
                if road_to_index[j][0] == dest[1]:
                    adjacency_matrix[i][j] = 1
            visited.append(dest)
    street_data_transposed = street_data.T
    street_data_transposed = street_data_transposed[1:]
    indexes = street_data_transposed.index.values
    
    #converting timestamp data to suitable format
    for i in range(len(indexes)):
        pos = re.finditer('_',indexes[i])
        matched_pos = [m.start() for m in pos]
        for j in matched_pos:
            if j == 2 or j==5:
                indexes[i] = indexes[i][:j] + '/' + indexes[i][j + 1:]
            else: 
                indexes[i] = indexes[i][:j] + ':' + indexes[i][j + 1:]
    street_data_transposed.index = pd.to_datetime(street_data_transposed.index,format='%d/%m/%Y %H:%M:%S')
    
    col = street_data_transposed.columns
    street_data_transposed[col] = street_data_transposed[col].apply(pd.to_numeric)
    
    #resampling using preferred method
    if resampling_method == 'max':
        resampled = street_data_transposed.resample('5T').max().dropna()
    elif resampling_method == 'min':
        resampled = street_data_transposed.resample('5T').min().dropna()
    elif resampling_method == 'mean':
        resampled = street_data_transposed.resample('5T').mean().dropna()
    else:
        resampled = street_data_transposed.resample('5T').median().dropna()
        
    resampled_numpy = resampled.to_numpy() 
    resampled.reset_index(inplace=True)
    resampled.to_csv('saved_csv_dhaka.csv')  
    return resampled , resampled_numpy, adjacency_matrix




def create_graph(adj):
    rows,cols = np.where(adj==1)
    edge_coord = zip(rows.tolist(),cols.tolist())
    plt.figure(figsize=(40,40))
    dhaka_graph = nx.Graph()
    dhaka_graph.add_edges_from(edge_coord)
    nx.draw(dhaka_graph, with_labels=True)
    plt.savefig('dhaka_network.png',format = 'png')
    plt.show() 