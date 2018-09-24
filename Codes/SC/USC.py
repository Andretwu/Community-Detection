# Unnormalized spectral clustering (using kmeans) 
# paper: A Tutorial on Spectral Clustering
from collections import defaultdict
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import networkx as nx
import random
import argparse
from time import time

# Unnomalised Matrix
def generate_Laplacian(G):# Input G(Graph)
    G_edges=G.edges()
    G_nodes=G.nodes()
    nodes_num=len(G_nodes)

    A=np.zeros([nodes_num,nodes_num])
    for row_id, column_id in G_edges:
        A[row_id, column_id]=A[column_id,row_id]=1

    D=np.diag(np.sum(A,axis=1)) #axis=0: sum(every column) axis=1: sum(every raw)
    L=D-A
    return L #return Laplacian Matrix

def cal_Eigen(L):
    eigen_value,eigen_vector=sp.linalg.eig(L)
    eigen_vector=eigen_vector.T #from column to row

    sort_eivalueid=np.argsort(eigen_value) #argsort函数返回的是数组值从小到大的索引值 (sort values return indexes)
   
    return eigen_value[sort_eivalueid],eigen_vector[sort_eivalueid]

def means_Euclidean(v1,v2):    #calculate mean
    return np.sqrt(sum(np.power(v1 - v2, 2)))

# init centroids with random samples  
def initCentroids(dataSet, k):  
    numSamples, dim = dataSet.shape   
    centroids = np.zeros((k, dim))         
    for i in range(k):  
        index = int(random.uniform(0, numSamples))  
        centroids[i, :] = dataSet[index, :]  
    return centroids  
  
# k-means cluster 
def k_Means(dataSet, k):  
    dataSet=dataSet.T
    numSamples = dataSet.shape[0]  
    # first column stores which cluster this sample belongs to,  
    # second column stores the error between this sample and its centroid  
    clusterAssment = np.mat(np.zeros((numSamples, 2)))  
    clusterChanged = True  
  
    # step 1: init centroids  
    centroids = initCentroids(dataSet, k)  
  
    while clusterChanged:  
        clusterChanged = False  

        # for each sample  
        for i in range(numSamples):  #range
            minDist  = 100000.0  
            minIndex = 0  

            # for each centroid  
            # step 2: find the centroid who is closest  
            for j in range(k):  
                distance = means_Euclidean(centroids[j, :], dataSet[i, :])  
                if distance < minDist:  
                    minDist  = distance  
                    minIndex = j  
              
            # step 3: update its cluster 
            if clusterAssment[i, 0] != minIndex:  
                clusterChanged = True  
                clusterAssment[i, :] = minIndex, minDist**2  
  
        # step 4: update centroids  
        for j in range(k):  
            
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]] 
            centroids[j, :] = np.mean(pointsInCluster, axis = 0)
  
    #print ('cluster complete!')  
    #print(clusterAssment)
    return centroids, clusterAssment  

def as_Partition(result):
    partition={}
    for i in range(len(result)):
        partition[i]=int((result[i].T)[0])
    
    return partition

def add_Group(G_original,partition):
    num=0
    nodegroup={} #dict
    for part in partition:
        for node in part:
            nodegroup[node]={'group':num}
        num=num+1
    #print(nodegroup)
    nx.set_node_attributes(G_original,nodegroup)
    return nodegroup

def save_Gml(G,gmal_name):
    nx.write_gml(G, gmal_name)

def draw_Network(G,partition,path):
    plt.figure()
    sp=nx.spring_layout(G)
    plt.axis('off')
    values = [partition.get(node)['group'] for node in G.nodes()]
    nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=80, with_labels=True,font_size=10)
    plt.savefig(path, dpi=100, format = "PNG")
    #plt.show()
    plt.clf() #claer figure

def save_CSV(nodegroup,path):
    nodegroup=sorted(nodegroup.items())
    f=open(path,'w')
    for item in nodegroup:
        f.write(str(item[0])+','+str(list(item[1].values())[0])+'\n')
    f.close()

def dict_to_List(partition):
    p_dict = defaultdict(list)
    for node, group_id in partition.items():
        p_dict[group_id].append(node)
    
    nodelist=[]
    for group, nodes in p_dict.items():
        nodelist.append(nodes)

    return nodelist

def save_Time(time_str,path):
    f=open(path,'a')
    f.write(time_str+'\n')
    f.close()

def USC(G):
    partitions=[]
    for i in range(11): # The range of k is from 10 to 20
        k=i+10
        L_matrix=generate_Laplacian(G)
        eigen_value,eigen_vector=cal_Eigen(L_matrix)

        result=k_Means(eigen_vector[0:k],k)[1]
        

        #save_Data(result,'result/eigenvectors_USC.csv')
        
        partition=as_Partition(result)

        nodelist=dict_to_List(partition)
        partitions.append(nodelist)
    opt_k=[]
    for j in range(len(partitions)):
        opt_k.append(calculate_Q(partitions[j],G))
    #print(opt_k.index(max(opt_k))+10)
    return partitions[opt_k.index(max(opt_k))]

def calculate_Q(partition,G):
    m=len(G.edges())
    e=[]
    a=[]
    q=0.0 
    # calculate e (ls/m)
    for community in partition:
        ls=0.0
        for i in range(len(community)):
            for j in range(len(community)):
                if (G.has_edge(community[i],community[j])):
                    ls=ls+1.0
        e.append(ls/(2*m))
        # in the function denominator is m, but in reality #egdes calculated twice so it is 2*m

    # calculate a (ds/2m)
    for community in partition:
        ds=0.0
        for node in community:
            ds=ds+len(list(G.neighbors(node)))
        a.append(ds/(2*m))

    for ei,ai in zip(e,a):
        q=q+(ei-ai**2)      

    return q

def USC_main(G,path_str,time_path):
    start = time()
        
    nodelist=USC(G)
    
    nodegroup=add_Group(G,nodelist)
    

    draw_Network(G,nodegroup,'result/resultofUSC.png')

    #save_Gml(G,'result/outputofUSC.gml')

    save_CSV(nodegroup,path_str)
    
    stop = time()
    time_str=str(stop-start) + " second"
    
    save_Time(time_str,time_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-path',type=str,default='result/outputofUSC.csv')
    parser.add_argument('-data',type=str,default='../DataSet/football.txt')

    args = parser.parse_args()
    # different type
    #G=nx.read_gml('../Dataset/football.gml',label='id')

    G=nx.read_edgelist(args.data,create_using=nx.Graph(),nodetype=int)
    time_path='result/time.txt'

    USC_main(G,args.path,time_path)
    



