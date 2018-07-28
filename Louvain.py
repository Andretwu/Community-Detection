import networkx as nx
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt

# Read data from file as a Graph
def G_From_File(path):
    G = nx.Graph()
    with open(path) as f:
        for line in f.readlines():
            nodes_list=line.strip().split()
            node_1=int(nodes_list[0])
            node_2=int(nodes_list[1])
            G.add_edge(node_1,node_2,weight=1.0)
            G.add_edge(node_2,node_1,weight=1.0)
    return G

def G_to_Dict(G):
    nodes_dict={}
    # weights_dict -> like a "two-tired" defaultdict
    weights_dict=defaultdict(lambda: defaultdict(float))
   
    for partition,node in enumerate(G.node()):
        nodes_dict[node]=partition
        for edge in G[node].items():
            weights_dict[node][edge[0]]=edge[1]['weight'] # edge[1]['weight'] -> the weight of edge
    # in weights_dict: keys are nodes; values are groups
    return nodes_dict,weights_dict

# Calculate all weights of all edges in the graph, the sum is called m
def cal_M(weights_dict):
    '''
    all_weights=[]
    for i in weights_dict.keys():
        for j,weight in weights_dict[i].items():
            all_weights.append(weight)
    all_weights=sum(all_weights)/2
    #print(len(all_weights))
    '''
    all_weights=sum([weight for i in weights_dict.keys() for j,weight in weights_dict[i].items()])/2
    
    return all_weights

# Calculate the degrees of nodes, can be considered as the weights of nodes, return a dict
def cal_Node_Degree(weights_dict):
    node_degrees=defaultdict(float)
    for node in weights_dict.keys():
        # Calculate the degrees in each node
        node_degrees[node]=sum(weights_dict[node].values())
    #print(node_degrees)
    return node_degrees

def get_Neighbours(node,weights_dict):
    if node not in weights_dict:
        return defaultdict(float).items()
    else:
        return weights_dict[node].items()

def cal_Delta_Q(node,nodes_dict,weights_dict,node_degrees):
    
    nodes_list=[]
    for item,group in nodes_dict.items():
        if group==nodes_dict[node] and item!=node:
            nodes_list.append(item)
    #print(nodes_list) 
    #nodes_list = [n for n, group_id in nodes_dict.items() if group_id == nodes_dict[node] and node != n]
    
    # The sum of weights of links incident to nodes in community
    sum_tot=[]
    for n in nodes_list:
        sum_tot.append(sum(weights_dict[n].values()))
    sum_tot=float(sum(sum_tot))
    #print(sum_tot)

    # The sum of weights of links from node(i) to other nodes in community
    ki_in=[]
    neighbours=get_Neighbours(node,weights_dict)
    #print(neighbours)
    nodes=nodes_dict[node]
    for neighbour in neighbours:
        if nodes==nodes_dict[neighbour[0]]:
            ki_in.append(neighbour[1])
    #print(ki_in)
    ki_in=float(sum(ki_in))*2

    # The sum of weights of links incident to node i
    ki=node_degrees[node]

    m=cal_M(weights_dict)

    delta_q=ki_in-(sum_tot*ki)/m
    #print(1,delta_q)
    return delta_q

def first_Phase(nodes_dict,weights_dict):
    m=cal_M(weights_dict)
    node_degrees=cal_Node_Degree(weights_dict)

    run=True
    #print(nodes_dict)
    while run:
        group_changed=[]
        for node,group_id in nodes_dict.items():
            max_q=0.0
            best_group_id=group_id

            neighbours=get_Neighbours(node,weights_dict)
            neighbours_list=[]
            for item in neighbours:
                neighbours_list.append(item[0])
            #print(neighbours_list)

            for neighbour in neighbours_list:
                nodes_copy=nodes_dict.copy()
                
                nodes_copy[node]=nodes_copy[neighbour]
                delta_q=cal_Delta_Q(node,nodes_copy,weights_dict,node_degrees)

                if delta_q>max_q:
                    max_q=delta_q
                    best_group_id=nodes_copy[neighbour]
            nodes_dict[node]=best_group_id
            group_changed.append(group_id!=best_group_id)
        #print(group_changed)
        if sum(group_changed) == 0:
            break
    #print(nodes_dict)
  
    return nodes_dict

# Calculate Q using degrees
def cal_Q(nodes_dict,weights_dict):
    q=[]
    m=cal_M(weights_dict)

    groups_dict=defaultdict(list) 
    for node, group_id in nodes_dict.items():
        groups_dict[group_id].append(node)
    #print(groups_dict)

    for group_id, group in groups_dict.items():

        nodes_combinations = list(itertools.combinations(group, 2)) + [(node, node) for node in group]
        
        sum_in=[]
        for nodes_pair in nodes_combinations:
            sum_in.append(weights_dict[nodes_pair[0]][nodes_pair[1]])
        sum_in=float(sum(sum_in))
        #print(sum_in)
        
        sum_tot=[]
        for n in group:
            sum_tot.append(sum(weights_dict[n].values()))
        sum_tot=float(sum(sum_tot))
        
        #print(sum_tot)
        q.append((sum_in/(2*m)-(sum_tot/(2*m))**2))
    #print(q)
    q=float(sum(q))

    return q

def second_Phase(nodes_dict,weights_dict):
    group_dict = defaultdict(list)
    new_nodes_dict = {}
    new_weights_dict = defaultdict(lambda : defaultdict(float))

    for node, group_id in nodes_dict.items():
        group_dict[group_id].append(node)
        #print(node, group_id)
        if group_id not in new_nodes_dict:
            new_nodes_dict[group_id] = group_id

    #print(group_dict)

    nodes = list(nodes_dict.keys())
    nodes_pairs = list(itertools.permutations(nodes, 2)) + [(node, node) for node in nodes]
    #print(nodes_pairs)

    for edge in nodes_pairs:
        new_weights_dict[new_nodes_dict[nodes_dict[edge[0]]]][new_nodes_dict[nodes_dict[edge[1]]]] += weights_dict[edge[0]][edge[1]]
    #print(new_weights_dict)
    
    return new_nodes_dict,new_weights_dict
    
def new_Partition(new_nodes_dict, partition):
    new_partition = defaultdict(list)
    for node,group_id in partition.items():
        new_partition[group_id].append(node)

    for old_group_id, new_group_id in new_nodes_dict.items():
        for old_group in new_partition[old_group_id]:
            partition[old_group] = new_group_id
    return partition

def Louvain(G):
    nodes_dict,weights_dict=G_to_Dict(G)
    nodes_dict=first_Phase(nodes_dict,weights_dict)
    best_modularity=cal_Q(nodes_dict,weights_dict)
    
    partition=nodes_dict.copy()
    new_nodes_dict,new_weights_dict= second_Phase(nodes_dict,weights_dict)
    
    while True:
        new_nodes_dict=first_Phase(new_nodes_dict,new_weights_dict)
        modularity=cal_Q(new_nodes_dict,new_weights_dict)

        # RUN or STOP
        if best_modularity==modularity:
            break
        best_modularity=modularity
        
        partition=new_Partition(new_nodes_dict,partition)
        new_nodes_dict,new_weights_dict=second_Phase(new_nodes_dict,new_weights_dict)

    #print(best_modularity) #0.6045695626834571
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

def save_Gml(G_original,gmal_name):
    nx.write_gml(G_original, gmal_name)

def draw_G(G_original,partition,path):
    spring_pos = nx.spring_layout(G_original)
    plt.axis("off")

    values = [partition.get(node) for node in G_original.nodes()]
    print(values)
    nx.draw_spring(G_original, cmap = plt.get_cmap('jet'), node_color = values, node_size=80, with_labels=True, font_size=10)
    
    #plt.savefig(path, dpi = 100, format = "PNG") # Save picture
    plt.show()

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


if __name__ == "__main__":
    G = G_From_File('../Dataset/football.txt')
    partition=Louvain(G)
    print(partition)
    #print(sorted(partition.items(), key=lambda d: d[1]))
    
    nodelist=dict_to_List(partition)

    G_original=nx.read_edgelist('../DataSet/football.txt',create_using=nx.Graph(),nodetype=int)
    nodegroup=add_Group(G_original,nodelist)
    #print(nodegroup)

    #save_Gml(G_original,'result/outputofLouvain.gml')

    #save_CSV(nodegroup,'result/outputofLouvain.csv')

    #draw_G(G_original,partition,"result/resultofLouvain.png")



