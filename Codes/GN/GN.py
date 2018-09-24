# paper: Community structure in social and biological networks
# Algorithm: GN algorithm

import networkx as nx             
import matplotlib.pyplot as plt
import argparse
from time import time

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

def GN_Algorithm(G,G_original):

	partition=[[n for n in G.nodes()]]
	max_q=0.0

	while len(G.edges()) != 0:
		# calculate betweenness
		betweenness_dict=nx.edge_betweenness(G)
		betweenness_max_edge=max(betweenness_dict.items(),key = lambda item: item[1])[0]

		G.remove_edge(betweenness_max_edge[0],betweenness_max_edge[1])
		components = [list(c) for c in list(nx.connected_components(G))]
		
		if(len(components)!=len(partition)):
			cal_q=calculate_Q(components,G_original)
			
			# We want to fine the biggest modularity value!
			if cal_q>max_q:
				max_q=cal_q
				partition=components
		
		# if do not use Q
		'''
		if len(components)==num_groups:
			partition=components
			max_q=calculate_Q(components,G_original)
			break
		'''

	return partition

def add_Group(G_original,partition):
	num=0
	nodegroup={} #dict
	for part in partition:
		for node in part:
			nodegroup[node]={'group':num}
		num=num+1

	nx.set_node_attributes(G_original,nodegroup)
	return nodegroup

def save_Gml(G_original,gmal_name):
	nx.write_gml(G_original, gmal_name)

# drawing
def draw_Network(G,pic_name,partition):
	sp=nx.spring_layout(G)

	plt.axis('off')

	values =[partition.get(node)['group'] for node in G.nodes()]
	nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=80, with_labels=True, font_size=10)

	plt.savefig(pic_name, dpi = 100, format = "PNG")

	#plt.show()

	plt.clf() #claer figure
	
# save communities to evaluate
def save_CSV(nodegroup,path):
	nodegroup=sorted(nodegroup.items())
	f=open(path,'w')
	for item in nodegroup:
		f.write(str(item[0])+','+str(list(item[1].values())[0])+'\n')
	f.close()

def save_Time(time_str,path):
	f=open(path,'a')
	f.write(time_str+'\n')
	f.close()

def GN_main(G,G_original,path_str,time_path):
	start = time()
		
	partition=GN_Algorithm(G,G_original)
	
	nodegroup=add_Group(G_original,partition)
	
	
	#save_Gml(G_original,'result/outputofGN.gml')

	save_CSV(nodegroup,path_str)
	
	draw_Network(G_original,"result/footballnetworkofGN.png",nodegroup)

	stop = time()
	time_str=str(stop-start) + " second"

	save_Time(time_str,time_path)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='manual to this script')
	parser.add_argument('-path',type=str,default='result/outputofGN.csv')
	parser.add_argument('-data',type=str,default='../Dataset/football.txt')

	args = parser.parse_args()
	# different type
	#G=nx.read_gml('../Dataset/football.gml',label='id')
	G=nx.read_edgelist(args.data,create_using=nx.Graph(),nodetype=int)
	G_original=nx.read_edgelist(args.data,create_using=nx.Graph(),nodetype=int) # cannot use G_original=G
	time_path='result/time.txt'

	path_str=args.path
	GN_main(G,G_original,path_str,time_path)

