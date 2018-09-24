import networkx as nx  
import numpy as np
from scipy.special import comb
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


# paper: Comparing community structure identification
# Normalised Mutual Information
def cal_NMI(A,B):

	num_group_A=len(A)
	num_group_B=len(B)

	N=np.zeros([num_group_A,num_group_B]) # num_row, num_column = 16(i), 10(j)
	
	for i in range(num_group_A):
		for j in range(num_group_B):
			#print(set(A[i]),set(B[j]))
			n_ij=len(set(A[i])&set(B[j]))
			N[i][j]=n_ij
	
	N_size=(max(max(item) for item in A))+1 # N
	eps = 1.4e-45
	
	mi=[]
	for i in range(num_group_A):
		for j in range(num_group_B):
			mi.append(N[i][j]*np.log2((N[i][j]*N_size)/(sum(N[i])*sum(N.T[j]))+eps))
	mi=sum(mi)
	
	ni=[]
	for i in range(num_group_A):
		sum_Ni=sum(N[i])
		ni.append(sum_Ni*np.log2((sum_Ni/N_size)))
	ni=sum(ni)

	nj=[]
	for j in range(num_group_B):
		sum_Nj=sum(N.T[j])
		nj.append(sum_Nj*np.log2((sum_Nj/N_size)))
	nj=sum(nj)

	nmi=((-2.0)*mi)/(ni+nj)
	
	return nmi


# paper: ARI_On the Use of the Adjusted Rand Index
# Agjusted Rand Index
def cal_ARI(A,B):
	
	num_group_A=len(A)
	num_group_B=len(B)

	N=np.zeros([num_group_A,num_group_B]) # num_row, num_column = 16(A), 10(B)
	
	for i in range(num_group_A):
		for j in range(num_group_B):
			
			n_ij=len(set(A[i])&set(B[j]))
			N[i][j]=n_ij
	
	N_size=(max(max(item) for item in A))+1 # n
	
	sum_ab=[]
	for a in range(num_group_A):
		for b in range(num_group_B):
			sum_ab.append(comb(N[a][b],2))
	sum_ab=sum(sum_ab)

	sum_a=[]
	for a in range(num_group_A):
		sum_a.append(comb(sum(N[a]),2))

	sum_a=sum(sum_a)

	sum_b=[]
	for b in range(num_group_B):
		sum_b.append(comb(sum(N.T[b]),2))

	sum_b=sum(sum_b)
	
	n_2=comb(N_size,2)

	ari=(n_2*sum_ab-sum_a*sum_b)/(0.5*n_2*(sum_a+sum_b)-sum_a*sum_b)
	
	return ari
	

# paper: Comparative evaluation of community detection algorithms- a topological approach
# fraction of correctly classified nodes (FCC)
# A is the control group(reference/real communities); B is the experimental group(estimated/found communities)
def cal_FCC(A,B): 	
	N_size=(max(max(item) for item in A))+1 # N
	count=0
	for i in range(len(B)):
		for j in range(len(B[i])):

			# for every node in the data set
			for item in A:
				if B[i][j] in item:
					index=A.index(item)
					
					if len(B[i])/2 <= len(set(B[i])&set(A[index])):
						
						count=count+1
	
	return 	count/N_size


def external(pathA,pathB):
	f=open('result/result_ex.txt','a')

	f.write(pathB[3:-4]+'\n')

	partition_A=read_Results(pathA)

	partition_B=read_Results(pathB)
	
	nmi_AB=cal_NMI(partition_A,partition_B)
	f.write('Normalised Mutual Information: '+str(nmi_AB)+'\n')

	ari_AB=cal_ARI(partition_A,partition_B)
	f.write('Adjusted Rand Index: '+str(ari_AB)+'\n')

	fcc_AB=cal_FCC(partition_A,partition_B)
	f.write('Fraction of Correctly Classified Nodes: '+str(fcc_AB)+'\n')
	f.write('\n')
	f.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='manual to this script')
	parser.add_argument('-times',type=int,default=10)

	args = parser.parse_args()

	times_len=args.times 
	pathA='../OriginalNetwork/result/original0.csv'

	for i in range(times_len):
		external(pathA,'../GN/result/outputofGN'+str(i+1)+'.csv')

	for i in range(times_len):
		external(pathA,'../Louvain/result/outputofLouvain'+str(i+1)+'.csv')

	for i in range(times_len):
		external(pathA,'../SC/result/outputofUSC'+str(i+1)+'.csv')

	for i in range(times_len):
		external(pathA,'../SC/result/outputofNSC'+str(i+1)+'.csv')

	for i in range(times_len):
		external(pathA,'../GA-Net/result/outputofGA'+str(i+1)+'.csv')
	
	
	
	






