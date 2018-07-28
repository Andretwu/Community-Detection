# Unnormalized spectral clustering (using kmeans) 
# paper: A Tutorial on Spectral Clustering
from collections import defaultdict
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import networkx as nx
import random

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

# Save Eigenvctors
def save_Data(data,path):
    data=np.array(data)
    # one column is a eigenvector
    num_row, num_column = data.shape
    #print(num_row, num_column) #115 2
    
    f = open(path,'w')
    for i in range(num_row):
        f.write(str(i))
        for j in range(num_column):
            f.write(str(','+'%.8f'%data[i][j].round(8)))
        f.write('\n')
    f.close()

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
    values = [partition.get(node) for node in G.nodes()]
    nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=80, with_labels=True,font_size=10)
    plt.savefig(path, dpi=100, format = "PNG")
    plt.show()
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

if __name__ == '__main__':

    G=nx.read_edgelist('../DataSet/football.txt',create_using=nx.Graph(),nodetype=int)
    
    L_matrix=generate_Laplacian(G)
    eigen_value,eigen_vector=cal_Eigen(L_matrix)

    k=10 # set the number of communities

    result=k_Means(eigen_vector[0:k],k)[1]
    #print(result)

    #save_Data(result,'result/eigenvectors_USC.csv')
    
    partition=as_Partition(result)

    nodelist=dict_to_List(partition)

    nodegroup=add_Group(G,nodelist)
    print(nodegroup)

    #draw_Network(G,partition,'result/resultofUSC.png')

    #save_Gml(G,'result/outputofUSC.gml')

    #save_CSV(nodegroup,'result/outputofUSC.csv')
    

    



