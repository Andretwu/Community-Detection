import networkx as nx  
import matplotlib.pyplot as plt
import numpy as np
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

if __name__ == '__main__':
	'''
	G = nx.Graph()
	#example=[('1', '2'),('1','3'),('2','5'), ('1','4'), ('1', '5'), ('2', '4'),('3', '4'), ('3', '5'), ('4', '5'), ('4', '7'),('6','7'),('6','8'),('7','8'),('8','9'),('9','10'),('9','11'),('10','11'),('2','11'),('11','12'),('10','12'),('9','12')]
	#example=[('1','2'),('2','3'),('3','1'),('4','6'),('5','4'),('6','2'),('7','4')]
	example=[('1','2'),('2','3'),('3','2'),('4','7'),('5','4'),('6','4'),('7','4')]
	G.add_edges_from(example)
	spring_pos = nx.spring_layout(G)
	plt.axis("off")
	#values =[i for i in range(7)]
	
	#nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=800, with_labels=True, font_size=20)
	nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_size=800, with_labels=True, font_size=20)

	plt.show()
	#plt.clf()
	'''
	#G = nx.karate_club_graph()
	'''
	a=nx.edge_betweenness(G) # same as .edge_betweenness_centrality
	print(sorted(a.items(), key=lambda k: k[1], reverse=True)) #('2', '11'): 0.3068181818181818 biggest
	print('\n')
	b=nx.betweenness_centrality(G) # '2': 0.2863636363636364 biggest
	print(sorted(b.items(), key=lambda k: k[1], reverse=True))
	print('\n')
	#c=nx.betweenness_centrality_subset(G,['1'],['2'])
	#print(sorted(c.items(), key=lambda k: k[1], reverse=True))
	print('\n')
	#d=nx.edge_betweenness_centrality_subset(G,'2','1')
	#print(sorted(d.items(), key=lambda k: k[1], reverse=True))

	partition=[['1','2','3','4','5'],['6','7','8'],['9','10','11','12']]
	print(calculate_Q(partition,G))
	print(len(example))
	'''
	G = nx.Graph()
	#example=[('1', '2'),('2','3'),('2','5'), ('2','4'), ('4', '5')]
	example=[('1','2'),('2','3'),('3','1'),('4','6'),('5','4'),('6','2'),('7','4')]

	G.add_edges_from(example)
	nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_size=800, with_labels=True, font_size=20)

	plt.show()

'''
	li_usc=[1.3455862998962402,1.0130419731140137,1.0770900249481201,1.122995138168335,1.038801908493042,1.048557996749878,1.0793538093566895,1.0238010883331299,1.3100450038909912,1.2133889198303223]
	li_nsc=[1.1321990489959717,1.0107288360595703,1.1164758205413818,0.9831960201263428,1.049015760421753,1.134221076965332,1.1282308101654053,1.0757431983947754,1.0514378547668457,1.1144189834594727]
	li_ga=[6.0915210247039795,
6.198554039001465,
6.642816781997681,
6.195428133010864,
5.973688840866089,
6.706628084182739,
6.471612930297852,
7.340868234634399,
7.439681053161621,
6.600854158401489]
	li_ga2=[0.37499284744262695,
1.383770227432251,
1.0112881660461426,
0.747514009475708,
1.060448169708252,
0.6287059783935547,
5.922415256500244,
0.3926839828491211,
3.8600871562957764,
1.4088518619537354]
	print(np.mean(li_ga2))
'''




