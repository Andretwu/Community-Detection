import networkx as nx             

def open_File(path):

	dataSet = []
	file = open(path)

	for line in file.readlines():
		dataSet.append(line.strip().split(','))
	#print(dataSet)
	return dataSet

def save_Gml(G_original,gmal_name):
	
	nx.write_gml(G_original, gmal_name)

def set_Group(G_original,group_list):
	nodegroup={} #dict
	#print(group_list)
	for line in group_list:
		
		nodegroup[int(line[0])]={'group':int(float(line[1]))}
	
	#print(nodegroup)
	nx.set_node_attributes(G_original,nodegroup)
	
	


if __name__ == '__main__':
	
	
	#G=nx.read_gml("../DataSet/football.gml") 
	G_original=nx.read_edgelist('../DataSet/football.txt',create_using=nx.Graph(),nodetype=int)
	
	name1='result/result_normalised.csv'
	name2='result/result_unnormalised.csv'

	file1='result/outputofSC.gml'
	file2='result/outputofUSC.gml'

	partition=open_File(name2)
	
	set_Group(G_original,partition)
	#print(G_original[0])
	save_Gml(G_original,file2)
	
	

