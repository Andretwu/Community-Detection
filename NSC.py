#paper: A Tutorial on Spectral Clustering
#(using kmeans) 

#Normalized spectral clustering according to Ng, Jordan, and Weiss (2002)

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict


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
    #print(data)
    # one column is a eigenvector
    num_row, num_column = data.shape
    #print(num_row, num_column)
    
    f = open(path,'w')
    for i in range(num_row):
        f.write(str(i))
        for j in range(num_column):
            f.write(str(','+'%.8f'%data[i][j].round(8)))
        f.write('\n')
    f.close()


def normalised_Laplacian(G):
    G_edges=G.edges()
    G_nodes=G.nodes()
    nodes_num=len(G_nodes)

    A=np.zeros([nodes_num,nodes_num])
    for row_id, column_id in G_edges:
        A[row_id, column_id]=A[column_id,row_id]=1
    
    D=np.diag(np.sum(A,axis=1)) #axis=0: sum(every column) axis=1: sum(every raw)
    L=D-A

    normalised_D=np.diag(np.power(np.sum(A,axis=1),-0.5))
    
    normalised_L=np.dot(normalised_D,L)
    
    normalised_L=np.dot(L,normalised_D)
    #print((normalised_L))
    return normalised_L

# normalising rows of normalised_U(k eigenvectors)
def normalised_T(normalised_U):
    
    T=normalised_U.T #eigenvecor as a column
    num_row, num_column = T.shape
    normalised_T=[]
    for i in range(num_row):
        line=[]
        sum_row=np.power(sum(np.power(T[i],2)),0.5)

        for j in range(num_column):
            #print(sum_row)
            line.append(T[i][j]/sum_row)
        normalised_T.append(line)
    #print(np.array(normalised_T).shape)
    return np.array(normalised_T).T

# KMEANS START
def means_Euclidean(v1,v2):    #calculate mean

    return np.sqrt(sum(np.power(v1 - v2, 2)))

# Initial k centers  
def initial_Centers(data, k):  
    num_row,num_column=data.shape
    centers = np.zeros((k, num_column))        
    for i in range(k):  
        index=np.random.randint(0,num_row)
        centers[i, :] = data[index, :]  
    return centers  
  
#kmeans cluster 
#dataSet is a matrix
def k_Means(dataSet, k):  
    dataSet=dataSet.T
    num_samples = dataSet.shape[0]  #the number of samples
    # first column stores which cluster this sample belongs to,  
    # second column stores the error between this sample and its centers  
    cluster_label = np.mat(np.zeros((num_samples, 2))) 
    cluster_mark = True  
  
    ## step 1: init centers  
    centers = initial_Centers(dataSet, k)  #initial data center
  
    while cluster_mark:  
        cluster_mark = False  
        ## for each sample  
        for i in range(num_samples):  
            min_distance  = 1000000.0  
            min_index = 0  
            ## for each centers  
            ## step 2: find the centers who is closest  
            for j in range(k):  
                distance = means_Euclidean(centers[j, :], dataSet[i, :])  
                if distance < min_distance:  
                    min_distance  = distance  
                    min_index = j  
              
            ## step 3: update its cluster 
            
            if cluster_label[i, 0] != min_index:  
                cluster_mark = True  
                cluster_label[i, :] = min_index, np.power(min_distance,2) #min_distance**2 
  
        ## step 4: update centers  
        for j in range(k):  
            #cluster_label[:,0].A==j -> find a array which is in the j-th of dataset
            data_in_cluster= dataSet[np.nonzero(cluster_label[:, 0].A == j)[0]] 
            centers[j, :] = np.mean(data_in_cluster, axis = 0)  
  
    #print ('cluster complete!')  
    #print(cluster_label)
    return centers, cluster_label  

def as_Partition(result):
    partition={}
    for i in range(len(result)):
        partition[i]=int((result[i].T)[0])
    
    return partition

def save_Gml(G,gmal_name):
    nx.write_gml(G, gmal_name)

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

def draw_Network(G,partition,path):
    plt.figure()
    sp=nx.spring_layout(G)
    plt.axis('off')
    values = [partition.get(node) for node in G.nodes()]
    nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=80, with_labels=True,font_size=10)
    #plt.savefig(path, dpi=100, format = "PNG")
    plt.show()
    plt.clf() #claer figure


if __name__ == '__main__':
   
    G=nx.read_edgelist('../DataSet/football.txt',create_using=nx.Graph(),nodetype=int)

    normalised_L_matrix=normalised_Laplacian(G)
    normalised_eigen_value,normalised_eigen_vector=cal_Eigen(normalised_L_matrix)
   
    k=10

    normalised_T_matrix=normalised_T(normalised_eigen_vector[0:k])
    normalised_result=k_Means(normalised_T_matrix[0:k],k)[1]
  
    #save_Data(normalised_result,'result/eigenvectors_NSC.csv')
   
    partition=as_Partition(normalised_result)
    nodelist=dict_to_List(partition)

    nodegroup=add_Group(G,nodelist)
    print(nodegroup)

    #save_Gml(G,'result/outputofNSC.gml')

    #save_CSV(nodegroup,'result/outputofNSC.csv')

    #draw_Network(G,partition,'result/resultofNSC.png')
   
   

