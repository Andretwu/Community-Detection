# paper GA-Net: A Genetic Algorithm for Community Detection in Social Networks (Clara Pizzuti)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import argparse
from time import time

def initial_Population(lam,G):
	nodes_num=G.number_of_nodes()

	population=[]
	for i in range(lam):
		ind=[]
		for j in range(nodes_num):
			
			ind.append(np.random.choice(list(G[j])))
		population.append(ind)

	return population

def uniform_Crossover(p1,p2,c_rate=0.8):
	if np.random.random()<=c_rate:
		
		# Create a random binary vector/list
		bi_vector=[]
		vector_len=len(p1)

		for i in range(vector_len):
			bi_vector.append(np.random.randint(0,2))
		
		# Create a child
		child=[]
		for i in range(vector_len):
			if bi_vector[i]==1:
				child.append(p1[i])
			else:
				child.append(p2[i])

		return child
	else:
		if np.random.random()<=0.5:
			return p1
		else:
			return p2

def Mutation(ind,G,m_rate=0.5):

	if np.random.random()<=m_rate:
		mutated_ind=[]
		for i in range(len(ind)):
			mutated_ind.append(np.random.choice(list(G[i])))
		
		return mutated_ind
	else:
		return ind

def cal_CS(ind,r=1.5): # fitness function

	G_ind=nx.Graph()
	for i in range(len(ind)):
		G_ind.add_edge(i,ind[i])
	all_parts=sorted(nx.connected_components(G_ind))
	
	CS=0.0
	for i in range(len(all_parts)):
		
		A=np.zeros((len(ind),len(ind)))
		part=list(all_parts[i])

		for j in range(len(part)):
			edges=list(G_ind.edges(part[j]))
			for item in edges:			
				A[item[0],item[1]]=1.0
			
		num_row_I, num_column_J = A.shape
		a_iJ=np.mean(A,axis=1) # the mean value of the ith row, it is a vector
		a_Ji=np.mean(A,axis=0) # the mean value of the jth column, it is a vector
		
		M=np.power(np.sum(a_iJ),r)/num_row_I
		
		Vs=sum(sum(A))
	
		Q=M*Vs
	
		CS=CS+Q
	
	return CS

def roulette_Wheel(cs_list):

	sum_fitness = sum(cs_list)
	# generate a random number
	rand_num = np.random.uniform(0, sum_fitness)
	# Calculate the index: O(N)
	sum_value = 0.0

	for index, value in enumerate(cs_list):
		sum_value += value
		if sum_value >= rand_num:

			return index # return index

def cal_Fitnesses(pop):
	pop_cs=[]

	for i in range(len(pop)):
		pop_cs.append(cal_CS(pop[i]))
		
	return pop_cs # return a list of fitness values 

def GA_Net(pop,G,lam):
	generation=0
	while 1:
		# Calculate the fitness values
		pop_cs=cal_Fitnesses(pop)
		
		pop_new=[]
		for j in range(lam):
			p1=roulette_Wheel(pop_cs) # index of p1
			p2=roulette_Wheel(pop_cs) # index of p2

			one_child=uniform_Crossover(pop[p1],pop[p2])
			mutated_child=Mutation(one_child,G)
			
			mutated_child_copy=mutated_child

			child_cs=cal_CS(mutated_child_copy)

			# terminal condition
			max_p=max(pop_cs[p1],pop_cs[p2])

			G_best=nx.Graph()
			for i in range(len(mutated_child)):
				G_best.add_edge(i,mutated_child[i])
			all_parts=sorted(nx.connected_components(G_best))
		
			parts_list=[]
			for mutated_child in all_parts:
				parts_list.append(list(mutated_child))
			
			termial=calculate_Q(parts_list,G)
			
			if  (termial>=0.3 and child_cs>max_p) or generation>=20:
				
				return parts_list
			else:
				
				pop_new.append(mutated_child_copy)

		
		generation=generation+1
		pop=pop_new
		

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

def draw_Network(G,pic_name,partition):
	sp=nx.spring_layout(G)
	plt.axis('off')
	
	values =[partition.get(node)['group'] for node in G.nodes()]

	nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=50, with_labels=True, font_size=10)

	plt.savefig(pic_name, dpi=100, format = "PNG")

	#plt.show()
	plt.clf() # Clear figure

def read_Partition(path):
    partition={}
    with open(path) as f:

        for line in f.readlines():
            nodes_list=line.strip().split(',')
            
            partition[int(nodes_list[0])]=int(float(nodes_list[1]))

    return partition
    
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

def calculate_Q(partitions,G):
	partition=[]
	for item in partitions:
		partition.append(list(item))
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

def GANet_main(G,path_str,time_path):
	start = time()
	
	lam=100
	pop=initial_Population(lam,G)

	parts_list=GA_Net(pop,G,lam)
	
	nodegroup=add_Group(G,parts_list)
	

	#save_Gml(G,'result/outputofGA.gml')
	
	save_CSV(nodegroup,path_str)
	
	draw_Network(G,"result/footballnetworkofGA.png",nodegroup)

	stop = time()
	time_str=str(stop-start) + " second"
	save_Time(time_str,time_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='manual to this script')
	parser.add_argument('-path',type=str,default='result/outputofGA.csv')
	parser.add_argument('-data',type=str,default='../DataSet/football.txt')

	args = parser.parse_args()

	time_path='result/time.txt' 
	# different type
	#G=nx.read_gml('../Dataset/football.gml',label='id')
	G=nx.read_edgelist(args.data,create_using=nx.Graph(),nodetype=int)
	
	GANet_main(G,args.path,time_path)
	


