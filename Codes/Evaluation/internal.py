import networkx as nx  
import numpy as np
import argparse


def read_Results(path):
	
	partition={}
	with open(path) as f:

		for line in f.readlines():
			nodes_list=line.strip().split(',')
			partition[int(nodes_list[0])]=int(nodes_list[1])
	
	num_groups=(max(partition.values()))+1

	partition=sorted(partition.items(), key=lambda d: d[1])

	all_groups=[[] for i in range(num_groups)]

	for item in partition:
		for j in range(len(all_groups)):
			if item[1]==j:
				all_groups[j].append(item[0])
	
	all_groups=sorted(all_groups)
	return all_groups


# Modularity
# Finding and evaluating community structure in networks	
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


# Internal edge density 
# paper: Defining and identifying communities in networks
def internal_Density(partition,G):
	inden_all=[]
	for group in partition:
		
		num_edges=0 # 2*ms
		len_group=len(group) # ns
		for i in range(len_group):
			for j in range(len_group):
				if G.has_edge(group[i],group[j]):
					
					num_edges=num_edges+1
		
		try:
			in_den=(num_edges/2.0)/(len_group*(len_group-1)/2.0)	
		except ZeroDivisionError: 
			in_den=0.0
		finally:
			inden_all.append(in_den)

	return sum(inden_all)/len(inden_all),np.cov(inden_all)


# paper: Defining and identifying communities in networks.+Defining and Evaluating Network Communities based on Ground-truth
def average_Embeddedness(partition,G):
	avem_all=[]
	for group in partition:
		
		num_edges=0 # 2*ms
		len_group=len(group) # ns
		for i in range(len_group):
			for j in range(len_group):
				if G.has_edge(group[i],group[j]):
					
					num_edges=num_edges+1
		
		try:
			av_em=(num_edges)/(len_group)
		except ZeroDivisionError: 
			av_em=0.0
		finally:
			avem_all.append(av_em)

	return sum(avem_all)/len(avem_all),np.cov(avem_all)


# Conductance shows the Connectedness between one community to other communities, less->better
# paper: Defining and Evaluating Network Communities based on Ground-truth
def Conductance(partition,G):
	con_all=[]
	for group in partition:
		
		num_edges=0 # 2*ms
		len_group=len(group) # ns
		all_degrees=0 	
		for i in range(len_group):
			for j in range(len_group):
				if G.has_edge(group[i],group[j]):
					
					num_edges=num_edges+1
			
			all_degrees=all_degrees+G.degree(group[i])

		outer_edges=all_degrees-num_edges # cs

		try:
			con=(outer_edges)/(num_edges+outer_edges)
		except ZeroDivisionError: 
			con=0.0
		finally:
			con_all.append(con)
	
	return sum(con_all)/len(con_all),np.cov(con_all)


# paper: Normalized Cuts and Image Segmentation
# Cut Ratio
def cut_Ratio(partition,G):
	cr_all=[]
	num_nodes=(max(max(item) for item in partition))+1 # n
	#print(num_nodes)
	for group in partition:
		
		num_edges=0 # 2*ms
		len_group=len(group) # ns
		all_degrees=0 	
		for i in range(len_group):
			for j in range(len_group):
				if G.has_edge(group[i],group[j]):
					
					num_edges=num_edges+1
			
			all_degrees=all_degrees+G.degree(group[i])

		outer_edges=all_degrees-num_edges # cs

		try:
			cr=(outer_edges)/(len_group*(num_nodes-len_group))
		except ZeroDivisionError: 
			cr=0.0
		finally:
			cr_all.append(cr)
	
	return sum(cr_all)/len(cr_all),np.cov(cr_all)


# paper: Normalized Cuts and Image Segmentation
# Normalized Cut
def normalised_Cut(partition,G):
	nc_all=[]
	m=G.size() # m
	for group in partition:
		
		num_edges=0 # 2*ms
		len_group=len(group) # ns
		all_degrees=0 	
		for i in range(len_group):
			for j in range(len_group):
				if G.has_edge(group[i],group[j]):
					
					num_edges=num_edges+1
			
			all_degrees=all_degrees+G.degree(group[i])

		outer_edges=all_degrees-num_edges # cs

		try:
			nc=(outer_edges)/(num_edges+outer_edges)+(outer_edges)/(2*m-num_edges+outer_edges)
		except ZeroDivisionError: 
			nc=0.0
		finally:
			nc_all.append(nc)
	
	return sum(nc_all)/len(nc_all),np.cov(nc_all)


# main function
def internal(G,path,path_in):
	f=open(path_in,'a')

	f.write(path[3:-4]+'\n')


	partition=read_Results(path)
		
	Q=calculate_Q(partition,G)
	f.write('Modularity: '+str(Q)+'\n')

	inden_mean,inden_cov=internal_Density(partition,G)
	f.write('Internal Edge Density: '+str(inden_mean)+' '+str(inden_cov)+'\n') #'%.2f'%internal_density

	avem_mean,avem_cov=average_Embeddedness(partition,G)
	f.write('Average Embeddedness: '+str(avem_mean)+' '+str(avem_cov)+'\n')

	con_mean,con_cov=Conductance(partition,G)
	f.write('Conductance: '+str(con_mean)+' '+str(con_cov)+'\n')

	cr_mean,cr_cov=cut_Ratio(partition,G)
	f.write('Cut Ratio: '+str(cr_mean)+' '+str(cr_cov)+'\n')

	nc_mean,nc_cov=normalised_Cut(partition,G)
	f.write('Normalised Cut: '+str(nc_mean)+' '+str(nc_cov)+'\n')
	f.write('\n')
	f.close()


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='manual to this script')
	parser.add_argument('-times',type=int,default=10)
	parser.add_argument('-data',type=str,default='../DataSet/football.txt')

	args = parser.parse_args()

	times_len=args.times 

	G=nx.read_edgelist(args.data,create_using=nx.Graph(),nodetype=int)
	
	path_in='result/result_in.txt'
	internal(G,'../OriginalNetwork/result/original0.csv',path_in)		
	
	for i in range(times_len):
		internal(G,'../GN/result/outputofGN'+str(i+1)+'.csv',path_in)

	for i in range(times_len):
		internal(G,'../Louvain/result/outputofLouvain'+str(i+1)+'.csv',path_in)

	for i in range(times_len):
		internal(G,'../SC/result/outputofUSC'+str(i+1)+'.csv',path_in)

	for i in range(times_len):
		internal(G,'../SC/result/outputofNSC'+str(i+1)+'.csv',path_in)

	for i in range(times_len):
		internal(G,'../GA-Net/result/outputofGA'+str(i+1)+'.csv',path_in)

		


