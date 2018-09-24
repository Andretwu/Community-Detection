#paper: A Tutorial on Spectral Clustering
#(using kmeans) 

#Normalized spectral clustering according to Ng, Jordan, and Weiss (2002)

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
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

    sort_eivalueid=np.argsort(eigen_value) #argsort(sort values return indexes)
   
    return eigen_value[sort_eivalueid],eigen_vector[sort_eivalueid]

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
  
    # step 1: init centers  
    centers = initial_Centers(dataSet, k)  #initial data center
  
    while cluster_mark:  
        cluster_mark = False  
        # for each sample  
        for i in range(num_samples):  
            min_distance  = 1000000.0  
            min_index = 0  
            # for each centers  
            # step 2: find the centers who is closest  
            for j in range(k):  
                distance = means_Euclidean(centers[j, :], dataSet[i, :])  
                if distance < min_distance:  
                    min_distance  = distance  
                    min_index = j  
            
            # step 3: update its cluster 
            
            if cluster_label[i, 0] != min_index:  
                cluster_mark = True  
                cluster_label[i, :] = min_index, np.power(min_distance,2) #min_distance**2 
  
        # step 4: update centers  
        for j in range(k):  
            #cluster_label[:,0].A==j -> find a array which is in the j-th of dataset
            data_in_cluster= dataSet[np.nonzero(cluster_label[:, 0].A == j)[0]] 
            centers[j, :] = np.mean(data_in_cluster, axis = 0)  
  
    
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
    values = [partition.get(node)['group'] for node in G.nodes()]
    nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=80, with_labels=True,font_size=10)
    plt.savefig(path, dpi=100, format = "PNG")
    #plt.show()
    plt.clf() #claer figure

def save_Time(time_str,path):
    f=open(path,'a')
    f.write(time_str+'\n')
    f.close()

def NSC(G):
    partitions=[]
    for i in range(11): # The range of k is from 10 to 20
        k=i+10

        normalised_L_matrix=normalised_Laplacian(G)
        normalised_eigen_value,normalised_eigen_vector=cal_Eigen(normalised_L_matrix)

        normalised_T_matrix=normalised_T(normalised_eigen_vector[0:k])
        normalised_result=k_Means(normalised_T_matrix[0:k],k)[1]
      
        partition=as_Partition(normalised_result)
        nodelist=dict_to_List(partition)

        partitions.append(nodelist)
    opt_k=[]
    for j in range(len(partitions)):
        opt_k.append(calculate_Q(partitions[j],G))
    
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

def NSC_main(G,path_str,time_path):
    start = time()
    nodelist=NSC(G)
    
    nodegroup=add_Group(G,nodelist)
    

    #save_Gml(G,'result/outputofNSC.gml')

    save_CSV(nodegroup,path_str)

    draw_Network(G,nodegroup,'result/resultofNSC.png')
   
    stop = time()
    time_str=str(stop-start) + " second"
    
    save_Time(time_str,time_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-path',type=str,default='result/outputofNSC.csv')
    parser.add_argument('-data',type=str,default='../DataSet/football.txt')

    args = parser.parse_args()
    # different type
    #G=nx.read_gml('../Dataset/football.gml',label='id')
    G=nx.read_edgelist(args.data,create_using=nx.Graph(),nodetype=int)
    time_path='result/time.txt'
    
    NSC_main(G,args.path,time_path)



   

