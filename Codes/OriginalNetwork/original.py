import networkx as nx             
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import community
from collections import defaultdict
import argparse

# read real communities from .dat; return two-dimentional list with lenth 115
def read_Group(path):
	data_comm = np.array(pd.read_table(path,sep = '::',header=None,engine='python')).tolist()

	group_list=[]
	
	for line in data_comm:
		
		item=[]
		item.append(int(line[0].split(' ')[0]))
		item.append(int(line[0].split(' ')[-1]))
		group_list.append(item)

	return group_list

# set group in G; return two-dimentional dict
def set_Group(G_original,group_list):
	nodegroup={} # dict

	for line in group_list:	
		nodegroup[line[0]]={'group':line[1]} # dict

	nx.set_node_attributes(G_original,nodegroup)
	return nodegroup

# save as .gml used in Gephi
def save_Gml(G_original,gmal_name):
	
	nx.write_gml(G_original, gmal_name)

# draw using matplotlib
def draw_Network(G,pic_name,nodegroup):
	sp=nx.spring_layout(G)
	plt.axis('off')

	values =[nodegroup.get(node)['group'] for node in G.nodes()]
	nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=260, with_labels=True, font_size=10)

	# save picture
	plt.savefig(pic_name, dpi = 100, format = "PNG")

	#plt.show()
	plt.clf() # claer figure	

# save communities to evaluate
def save_CSV(nodegroup,path):
	nodegroup=sorted(nodegroup.items())
	f=open(path,'w')
	for item in nodegroup:
		f.write(str(item[0])+','+str(list(item[1].values())[0]-1)+'\n')
	f.close()

def dict_to_List(partition):
    p_dict = defaultdict(list)
    for node, group_id in partition.items():
        #print(node, group_id.values())
        #print(list(group_id.values())[0])
        p_dict[list(group_id.values())[0]].append(node)
    
    nodelist=[]
    for group, nodes in p_dict.items():
        nodelist.append(nodes)

    return nodelist



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='manual to this script')
	parser.add_argument('-data',type=str,default='../Dataset/football.txt')
	parser.add_argument('-comm',type=str,default='../DataSet/football_comm.dat')
	args = parser.parse_args()
	
	G_original=nx.read_edgelist(args.data,create_using=nx.Graph(),nodetype=int)
	#G_original=nx.read_gml('../Dataset/football.gml',label='id')
	#print(nx.info(G_original))
	#print(G_original.edges())
	groups=read_Group(args.comm)
	#print(groups)
	
	nodegroup=set_Group(G_original,groups)
	#print(sorted(dict_to_List(nodegroup) ))
	
	save_CSV(nodegroup,'result/original0.csv')
	
	#save_Gml(G_original,'result/original.gml')
	
	draw_Network(G_original,"result/originalnetwork.png",nodegroup)

	
