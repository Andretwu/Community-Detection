import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def Modularity():
	data = pd.read_csv('result/in_all.csv')
	
	all_data=[]
	
	all_data.append(data.M2)
	all_data.append(data.M3)
	all_data.append(data.M4)
	all_data.append(data.M5)
	all_data.append(data.M6)
	all_data.append(data.M1) #original
	
	plt.subplot(2,3,1)
	medianprops = dict(linestyle='-', linewidth=1.2, color='black')

	meanpointprops = dict(marker='^', markeredgecolor='blue',
	                  markerfacecolor='blue',markersize=8)
	
	bplot = plt.boxplot(all_data,
	            notch=False,  # notch shape
	            vert=True,   # vertical box aligmnent
	            meanline=False,
	            showmeans=True,
	            
	            meanprops=meanpointprops,
	            medianprops=medianprops,
	            
	            patch_artist=True)   # fill with color
	plt.grid(True,axis="y",ls='--') 
	
	colors=['#FF4500','#FFB6C1','#87CEFA','#32CD32','#808080','whitesmoke']
	for patch, color in zip(bplot['boxes'], colors):
	    patch.set_facecolor(color)
	plt.xticks([y+1 for y in range(len(all_data))], ['GN','Louvain','USC','NSC','GA-Net','Original'])

	t = plt.title('Modularity')


def Internal_edge_density():
	data = pd.read_csv('result/in_all.csv')
	
	all_data=[]
	
	all_data.append(data.IED2)
	all_data.append(data.IED3)
	all_data.append(data.IED4)
	all_data.append(data.IED5)
	all_data.append(data.IED6)
	all_data.append(data.IED1) #original
	
	plt.subplot(2,3,2)
	medianprops = dict(linestyle='-', linewidth=1.2, color='black')

	meanpointprops = dict(marker='^', markeredgecolor='blue',
	                  markerfacecolor='blue',markersize=8)
	
	bplot = plt.boxplot(all_data,
	            notch=False,  # notch shape
	            vert=True,   # vertical box aligmnent
	            meanline=False,
	            showmeans=True,
	            
	            meanprops=meanpointprops,
	            medianprops=medianprops,
	            
	            patch_artist=True)   # fill with color
	plt.grid(True,axis="y",ls='--') 
	
	colors=['#FF4500','#FFB6C1','#87CEFA','#32CD32','#808080','whitesmoke']
	for patch, color in zip(bplot['boxes'], colors):
	    patch.set_facecolor(color)
	plt.xticks([y+1 for y in range(len(all_data))], ['GN','Louvain','USC','NSC','GA-Net','Original'])
	
	t = plt.title('Internal Edge Density')


def Inverse_conductance():
	data = pd.read_csv('result/in_all.csv')
	
	all_data=[]
	
	all_data.append(data.IC2)
	all_data.append(data.IC3)
	all_data.append(data.IC4)
	all_data.append(data.IC5)
	all_data.append(data.IC6)
	all_data.append(data.IC1) #original
	
	plt.subplot(2,3,3)
	medianprops = dict(linestyle='-', linewidth=1.2, color='black')

	meanpointprops = dict(marker='^', markeredgecolor='blue',
	                  markerfacecolor='blue',markersize=8)
	
	bplot = plt.boxplot(all_data,
	            notch=False,  # notch shape
	            vert=True,   # vertical box aligmnent
	            meanline=False,
	            showmeans=True,
	            
	            meanprops=meanpointprops,
	            medianprops=medianprops,
	            
	            patch_artist=True)   # fill with color
	plt.grid(True,axis="y",ls='--') 
	
	colors=['#FF4500','#FFB6C1','#87CEFA','#32CD32','#808080','whitesmoke']
	for patch, color in zip(bplot['boxes'], colors):
	    patch.set_facecolor(color)
	plt.xticks([y+1 for y in range(len(all_data))], ['GN','Louvain','USC','NSC','GA-Net','Original'])
	
	t = plt.title('Inverse Conductance')


def Normalised_mutual_information():
	data = pd.read_csv('result/ex_all.csv')
	
	all_data=[]
	all_data.append(data.NMI1)
	all_data.append(data.NMI2)
	all_data.append(data.NMI3)
	all_data.append(data.NMI4)
	all_data.append(data.NMI5)
	
	plt.subplot(2,3,4)
	medianprops = dict(linestyle='-', linewidth=1.2, color='black')

	meanpointprops = dict(marker='^', markeredgecolor='blue',
	                  markerfacecolor='blue',markersize=8)
	
	bplot = plt.boxplot(all_data,
	            notch=False,  # notch shape
	            vert=True,   # vertical box aligmnent
	            meanline=False,
	            showmeans=True,
	            
	            
	            meanprops=meanpointprops,
	            medianprops=medianprops,
	            
	            patch_artist=True)   # fill with color
	plt.grid(True,axis="y",ls='--') 
	
	colors=['#FF4500','#FFB6C1','#87CEFA','#32CD32','#808080']
	for patch, color in zip(bplot['boxes'], colors):
	    patch.set_facecolor(color)
	plt.xticks([y+1 for y in range(len(all_data))], ['GN','Louvain','USC','NSC','GA-Net'])
	
	t = plt.title('Normalised Mutual Information')


def Adjusted_rand_index():
	data = pd.read_csv('result/ex_all.csv')
	
	all_data=[]
	all_data.append(data.ARI1)
	all_data.append(data.ARI2)
	all_data.append(data.ARI3)
	all_data.append(data.ARI4)
	all_data.append(data.ARI5)
	
	plt.subplot(2,3,5)
	medianprops = dict(linestyle='-', linewidth=1.2, color='black')

	meanpointprops = dict(marker='^', markeredgecolor='blue',
	                  markerfacecolor='blue',markersize=8)
	flierprops = dict(marker='+', markerfacecolor='black', markersize=12,
                  linestyle='none') #outliers
	bplot = plt.boxplot(all_data,
	            notch=False,  # notch shape
	            vert=True,   # vertical box aligmnent
	            meanline=False,
	            showmeans=True,
	            
	            meanprops=meanpointprops,
	            medianprops=medianprops,
	            
	            patch_artist=True)   # fill with color
	plt.grid(True,axis="y",ls='--') 
	
	colors=['#FF4500','#FFB6C1','#87CEFA','#32CD32','#808080']
	for patch, color in zip(bplot['boxes'], colors):
	    patch.set_facecolor(color)

	plt.xticks([y+1 for y in range(len(all_data))], ['GN','Louvain','USC','NSC','GA-Net'])
	
	t = plt.title('Adjusted Rand Index')


def Fraction_correct_classified():
	data = pd.read_csv('result/ex_all.csv')
	
	all_data=[]
	all_data.append(data.FCC1)
	all_data.append(data.FCC2)
	all_data.append(data.FCC3)
	all_data.append(data.FCC4)
	all_data.append(data.FCC5)
	
	plt.subplot(2,3,6)
	
	medianprops = dict(linestyle='-', linewidth=1.2, color='black')

	meanpointprops = dict(marker='^', markeredgecolor='blue',
	                  markerfacecolor='blue',markersize=8)
	
	bplot = plt.boxplot(all_data,
	            notch=False,  # notch shape
	            vert=True,   # vertical box aligmnent
	            meanline=False,
	            showmeans=True,
	            
	            
	            meanprops=meanpointprops,
	            medianprops=medianprops,
	            
	            patch_artist=True)   # fill with color
	plt.grid(True,axis="y",ls='--') 
	
	colors=['#FF4500','#FFB6C1','#87CEFA','#32CD32','#808080']
	for patch, color in zip(bplot['boxes'], colors):
	    patch.set_facecolor(color)
	
	plt.xticks([y+1 for y in range(len(all_data))], ['GN','Louvain','USC','NSC','GA-Net'])
	
	t = plt.title('Fraction Correct Classified')


if __name__ == '__main__':

	fig, axes = plt.subplots(nrows=2, ncols=3)
	
	fig.tight_layout()
	#plt.subplots_adjust(wspace =0.5, hspace =0.5)

	Modularity()
	Internal_edge_density()
	Inverse_conductance()
	Normalised_mutual_information()
	Adjusted_rand_index()
	Fraction_correct_classified()
	#plt.savefig('result/out.png', dpi=600)
	plt.show()
	
