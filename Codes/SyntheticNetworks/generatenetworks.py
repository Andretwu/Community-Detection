# paper: Benchmark graphs for testing community detection algorithms”
import sys
sys.path.append('../GN')
sys.path.append('../OriginalNetwork')  
sys.path.append('../Evaluation')
sys.path.append('../Louvain') 
import GN
import original
import internal
import Louvain
from networkx.algorithms.community import LFR_benchmark_graph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def draw_Network(G,pic_name,nodegroup):
	sp=nx.spring_layout(G)
	plt.axis('off')

	values =[nodegroup.get(node)['group'] for node in G.nodes()]
	nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=80, with_labels=True, font_size=10)

	# save picture
	plt.savefig(pic_name, dpi = 100, format = "PNG")
	plt.clf() # Clear figure
	#plt.show()
	#plt.clf() # claer figure	

def add_Group(partition):
	num=1
	nodegroup={} #dict
	for part in partition:
		for node in part:
			nodegroup[node]={'group':num}
		num=num+1
	
	return nodegroup

def write_network(path,edges):
	f=open(path,'a')
	for i,item in enumerate(edges):
		if i!=len(edges)-1:
			f.write(str(item[0])+' '+str(item[1])+'\n')
		else:
			f.write(str(item[0])+' '+str(item[1]))

def write_community(path,nodegroup):
	f=open(path,'a')
	for i,item in enumerate(nodegroup):
		
		if i!=len(nodegroup)-1:
			f.write(str(item)+'   '+str(nodegroup.get(item)['group'])+'\n')
		else:
			f.write(str(item)+'   '+str(nodegroup.get(item)['group']))

def generating_main(G_edge,path_net='synthetic1.txt',path_comm='synthetic1_comm.dat',path_pic='synthetic1.png',muid=0.1):
	p_copy=path_net
	pc_copy=path_comm
	pi_copy=path_pic
	m_copy=muid
	G_copy=G_edge
	
	# generating synthetic networks
	
	n = 200
	tau1 = 2
	tau2 = 1.5
	mu = muid
	G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=6,max_degree=20,min_community=20,seed=10)
	
	communities = {frozenset(G.nodes[v]['community']) for v in G}

	partition_original=[]
	
	g_e=len(G.edges())
	if g_e in G_edge:
		generating_main(G_copy,p_copy,pc_copy,pi_copy,m_copy+0.003)
		
	else:
		G_edge.append(g_e)
		print(G_edge)
		print(len(communities))
		print(nx.info(G))
		
		for item in communities:
			partition_original.append(list(sorted(item)))
		partition_original=sorted(partition_original)
		nodegroup=add_Group(partition_original)
		

		draw_Network(G,path_pic,nodegroup)

		edges=sorted(G.edges())
		write_network(path_net,edges)
		write_community(path_comm,nodegroup)


if __name__ == '__main__':
	
	G_edge=[]

	for i in range(30):

		path_net='synthetic'+str(i+1)+'.txt'
		path_comm='synthetic'+str(i+1)+'_comm.dat'
		path_pic='synthetic'+str(i+1)+'.png'
		generating_main(G_edge,path_net,path_comm,path_pic,((i+1)*0.01)+0.05) # 0.001 is good for pics
		
		print("one done "+str(i+1)+' '+str(((i+1)*0.01)+0.05))
	
	